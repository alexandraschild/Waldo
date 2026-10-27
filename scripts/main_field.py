#!/usr/bin/env python3
"""
Distributed VLM inference on benchmarks across all available GPUs for added field transformation.
"""

# Resolve shared project code and default outputs independently of the invoking directory.
import sys as _sys
from pathlib import Path as _Path
_REPOSITORY_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))
if __name__ == "__main__":
    import os as _os
    _os.chdir(_REPOSITORY_ROOT)


import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset
import json
import time
import argparse
from pathlib import Path
import threading
from typing import Dict, List, Optional, Union
import logging
from contextlib import contextmanager
import warnings
from datasets import load_dataset
from src.benchmarks import (
    MMVetDataset, MMEvalDataset, MathVerseDataset, MMStar, CharXiv,
    VQAv2Dataset, GQADataset, CVBenchDataset, TextVQADataset, ChartQADataset,
    DocVQADataset, InfographicVQADataset, VStarBenchDataset, HRBenchDataset,
    MMERealWorldLiteDataset,
)
from src.models import (
    LLavaWorker, QwenWorker, LLamaWorker, PixtralWorker, InternVLWorker, PaliGemmaWorker,
    FastVLMWorker, MolmoWorker
)
from src.utils import collect_results

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Benchmark configuration mapping
BENCHMARK_CLASSES = {
    "mmvet": MMVetDataset,
    "mathverse": MathVerseDataset,
    "mmstar": MMStar,
    "charxiv": CharXiv,
    "mmeval": MMEvalDataset,
    "vqav2": VQAv2Dataset,
    "gqa": GQADataset,
    "cvbench": CVBenchDataset,
    "textvqa": TextVQADataset,
    "chartqa": ChartQADataset,
    "docvqa": DocVQADataset,
    "infographicvqa": InfographicVQADataset,
    "vstarbench": VStarBenchDataset,
    "hrbench": HRBenchDataset,
    "mmerealworldlite": MMERealWorldLiteDataset,
}

BENCHMARK_DATASETS = {
    "mmvet": "whyu/mm-vet",
    "mathverse": "AI4Math/MathVerse",
    "mmstar": "Lin-Chen/MMStar",
    "charxiv": "princeton-nlp/charxiv",
    "mmeval": "darkyarding/MME",
    "vqav2": "lmms-lab/VQAv2",
    "gqa": "lmms-lab/GQA",
    "cvbench": "nyu-visionx/CV-Bench",
    "textvqa": "lmms-lab/TextVQA",
    "chartqa": "lmms-lab/ChartQA",
    "docvqa": "lmms-lab/DocVQA",
    "infographicvqa": "lmms-lab/DocVQA",
    "vstarbench": "craigwu/vstar_bench",
    "hrbench": "DreamMr/HR-Bench",
    "mmerealworldlite": "yifanzhang114/MME-RealWorld-Lite",
}

# Answer-format instruction appended to every question for a given benchmark, matching the
# convention each benchmark's own paper / standard eval harness (lmms-eval, VLMEvalKit) scores
# against. Without it, chat-tuned models tend to answer in full sentences instead of the
# short/letter/yes-no form these benchmarks expect, which skews grading regardless of model
# quality. Benchmarks not listed here (mmvet, charxiv, cvbench, vstarbench) are intentionally
# left unsuffixed: mmvet/charxiv are open-ended by design, and cvbench/vstarbench's own
# "prompt"/"text" fields may already bake in an equivalent instruction - verify before adding
# one here to avoid stacking two.
BENCHMARK_ANSWER_FORMAT_SUFFIXES = {
    "mmeval": "\nPlease answer yes or no.",
    "mathverse": "\nAnswer the question and then provide your final answer clearly at the end, prefixed with 'Final Answer:'.",
    "mmstar": "\nAnswer with the option's letter from the given choices directly.",
    "vqav2": "\nAnswer the question using a single word or phrase.",
    "gqa": "\nAnswer the question using a single word or phrase.",
    "textvqa": "\nAnswer the question using a single word or phrase.",
    "chartqa": "\nAnswer the question using a single word or phrase.",
    "docvqa": "\nAnswer the question using a single word or phrase.",
    "infographicvqa": "\nAnswer the question using a single word or phrase.",
    "hrbench": "\nAnswer with the option's letter from the given choices directly.",
    "mmerealworldlite": "\nAnswer with the option's letter from the given choices directly.",
}


class _SuffixedDataset(Dataset):
    """Wraps a benchmark dataset to append a fixed answer-format instruction to every question."""

    def __init__(self, inner: Dataset, suffix: str):
        self.inner = inner
        self.suffix = suffix

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        item["question"] = item["question"] + self.suffix
        return item

def worker_process(
    gpu_id: int,
    model_name: str,
    worker_class,
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    batch_size: int
):
    """Worker process function."""
    worker = worker_class(gpu_id, model_name, work_queue, result_queue, stop_event, batch_size=batch_size)
    worker.run()


def save_results(results: List[Dict], output_file: Path):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

def get_worker_class(model_name: str):
    """Select appropriate worker class based on model name."""
    model_lower = model_name.lower()
    
    if "llama" in model_lower:
        return LLamaWorker
    elif "qwen" in model_lower:
        return QwenWorker
    elif "llava" in model_lower:
        return LLavaWorker
    elif "pixtral" in model_lower:
        return PixtralWorker
    elif "internvl" in model_lower:
        return InternVLWorker
    elif "paligemma" in model_lower:
        return PaliGemmaWorker
    elif "fastvlm" in model_lower:
        return FastVLMWorker
    elif "molmo" in model_lower:
        return MolmoWorker

    raise ValueError(f"Unsupported model: {model_name}")

def create_dataset(
    benchmark: str,
    field: Optional[float] = None,
    position: str = 'center'
) -> Dataset:
    """Create benchmark dataset with specified transformation."""
    benchmark_lower = benchmark.lower()
    
    if benchmark_lower not in BENCHMARK_CLASSES:
        raise ValueError(
            f"Unsupported benchmark: {benchmark}. Please, add new benchmark to benchmarks.py"
            f"Available: {list(BENCHMARK_CLASSES.keys())}"
        )
    
    dataset_class = BENCHMARK_CLASSES[benchmark_lower]
    dataset_name = BENCHMARK_DATASETS[benchmark_lower]
    
    kwargs = {"dataset_name": dataset_name}
    kwargs["field"] = field
    kwargs["position"] = position  # Default position for added field

    dataset = dataset_class(**kwargs)

    suffix = BENCHMARK_ANSWER_FORMAT_SUFFIXES.get(benchmark_lower)
    if suffix:
        dataset = _SuffixedDataset(dataset, suffix)

    return dataset

def get_model_short_name(model_name: str) -> str:
    """Extract short model name from full path."""
    return model_name.split('/')[-1] if '/' in model_name else model_name

def generate_output_filename(
    benchmark: str,
    model_name: str,
    param_value: Union[int, float],
    experiment: str = "field",
    partial: bool = False
) -> str:
    """Generate standardized output filename.

    ``experiment`` distinguishes the three result sets produced per field
    value: ``field`` (original embedded in a larger blank canvas), ``resized``
    (plain original downscaled by the same model-preprocessing coefficient),
    and ``bicubic`` (``resized`` scaled back up to the original resolution).
    Filenames are deterministic (no timestamp) so a rerun can tell whether a
    given (benchmark, model, field value) combination is already done - see
    ``find_existing_result``."""
    short_name = get_model_short_name(model_name)
    suffix = "_partial" if partial else ""

    return f"{benchmark}_{short_name}_{experiment}{param_value}{suffix}.json"


# Maps an output experiment tag to the result key holding its answer.
EXPERIMENTS = {
    "field": "generated_answer",
    "resized": "generated_answer_resized",
    "bicubic": "generated_answer_bicubic",
}
_ANSWER_KEYS = set(EXPERIMENTS.values())


def split_experiment_results(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Split combined per-sample results into one result set per experiment.

    Each collected result carries an answer per experiment (``generated_answer``
    for field, ``generated_answer_resized``, ``generated_answer_bicubic``).
    Downstream evaluation expects the answer under ``generated_answer``, so we
    emit one list per experiment with the relevant answer under that key while
    preserving the size metadata for analysis."""
    split: Dict[str, List[Dict]] = {name: [] for name in EXPERIMENTS}
    for r in results:
        base = {k: v for k, v in r.items() if k not in _ANSWER_KEYS}
        for name, answer_key in EXPERIMENTS.items():
            item = dict(base)
            item["generated_answer"] = r.get(answer_key, "")
            split[name].append(item)
    return split


def find_existing_result(
    output_path: Path,
    benchmark: str,
    model_name: str,
    param_value: Union[int, float]
) -> Optional[Path]:
    """Look for previously completed result files for this benchmark/model/param_value
    combination, so a rerun can skip work that's already done. A field value only counts
    as done once all three experiment files (field/resized/bicubic) exist - a run
    interrupted partway through would otherwise be silently treated as complete."""
    candidates = [
        output_path / generate_output_filename(benchmark, model_name, param_value, experiment=experiment)
        for experiment in EXPERIMENTS
    ]
    if all(candidate.exists() for candidate in candidates):
        return candidates[0]
    return None

@contextmanager
def multiprocess_environment(num_gpus: int, batch_size: int):
    """Context manager for multiprocessing resources."""
    work_queue = mp.Queue(maxsize=num_gpus * batch_size * 2)
    result_queue = mp.Queue()
    stop_event = mp.Event()
    processes = []
    
    try:
        yield work_queue, result_queue, stop_event, processes
    finally:
        # Cleanup
        stop_event.set()
        
        # Terminate all processes
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
        
        # Clear queues
        while not work_queue.empty():
            try:
                work_queue.get_nowait()
            except:
                break
        
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except:
                break

def log_gpu_utilization(results: List[Dict]):
    """Log GPU utilization statistics."""
    gpu_counts = {}
    for result in results:
        gpu_id = result.get("gpu_id", "unknown")
        gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1

    logger.info("GPU utilization:")
    for gpu_id, count in sorted(gpu_counts.items()):
        logger.info(f"  GPU {gpu_id}: {count} samples")


def run_benchmark_iteration(
    dataset: Dataset,
    worker_class,
    model_name: str,
    num_gpus: int,
    output_path: Path,
    benchmark: str,
    param_value: Union[int, float],
    model_load_path: Optional[str] = None,
    batch_size: int = 8
) -> List[Dict]:
    """Run a single benchmark iteration with given parameters."""

    with multiprocess_environment(num_gpus, batch_size) as (work_queue, result_queue, stop_event, processes):
        # Start worker processes
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, model_load_path or model_name, worker_class, work_queue, result_queue, stop_event, batch_size)
            )
            p.start()
            processes.append(p)
        
        # Start result collector thread
        results = []
        collector_thread = threading.Thread(
            target=collect_results,
            args=(result_queue, results, dataset, stop_event)
        )
        collector_thread.start()
        
        # Feed work to queue
        start_time = time.time()
        total_samples = len(dataset)
        
        try:
            for i, sample in enumerate(dataset):
                work_queue.put(sample)
                if (i + 1) % 100 == 0:
                    logger.info(f"Queued {i + 1}/{total_samples} samples")
            
            # Add sentinel values to stop workers
            for _ in range(num_gpus):
                work_queue.put(None)
            
            # Wait for all processes to complete
            for p in processes:
                p.join()
            
            # Stop result collector
            result_queue.put(None)
            collector_thread.join()
            
            # Calculate timing
            elapsed_time = time.time() - start_time
            
            logger.info(f"Inference completed in {elapsed_time:.2f} seconds")
            if results:
                logger.info(f"Average time per sample: {elapsed_time/len(results):.2f} seconds")
                logger.info(f"Throughput: {len(results)/elapsed_time:.2f} samples/second")
            
            # Split into the three experiments and save one file each
            for experiment, exp_results in split_experiment_results(results).items():
                save_results(exp_results, output_path / generate_output_filename(
                    benchmark, model_name, param_value, experiment=experiment
                ))
            log_gpu_utilization(results)

            return results

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            stop_event.set()

            # Save partial results
            if results:
                for experiment, exp_results in split_experiment_results(results).items():
                    save_results(exp_results, output_path / generate_output_filename(
                        benchmark, model_name, param_value,
                        experiment=experiment, partial=True
                    ))

            raise

def parse_list_argument(value: str) -> List[Union[int, float]]:
    """Parse comma-separated list argument."""
    if isinstance(value, list):
        return value
    
    try:
        # Remove brackets if present
        value = value.strip('[]')
        # Split and convert, dropping empty tokens from leading/trailing/doubled commas
        # (e.g. "0, " or "0,,1") instead of choking on them
        parts = [x.strip() for x in value.split(',') if x.strip()]
        
        # Try converting to int first, then float
        result = []
        for part in parts:
            try:
                result.append(int(part))
            except ValueError:
                result.append(float(part))
        
        return result
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid list format: {value}. Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Distributed inference for VLMs on benchmarks"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None, #"meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local path to load model weights from (e.g. a local Hugging Face snapshot). "
             "Defaults to --model_name, which is otherwise used for HF Hub lookups, "
             "worker selection, and output filenames."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./VLM_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples to batch per generate() call on each GPU"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None, #"MMVet",
        help=f"Benchmark to test. Available: {list(BENCHMARK_CLASSES.keys())}"
    )
    parser.add_argument(
        "--added_fields",
        type=str,
        default="0",#,0.5,1,2,3,4,5
        help="Comma-separated field sizes in percent (e.g., '0, 1, 2')"
    )
    parser.add_argument(
        "--position",
        type=str,
        default="center",
        help="Position of the image within the added white field: center, top, or bottom"
    )
    
    args = parser.parse_args()
    
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
    
        # Convert string arguments to lists
        if isinstance(args.added_fields, str):
            args.added_fields = parse_list_argument(args.added_fields)
        
        # Setup multiprocessing
        mp.set_start_method('spawn', force=True)
        
        # Initialize
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get worker class
        worker_class = get_worker_class(args.model_name)
        
        # Determine number of GPUs
        num_gpus = args.num_gpus or torch.cuda.device_count()
        num_gpus = min(num_gpus, torch.cuda.device_count())
        
        # Log setup information
        logger.info(f"Using {num_gpus} GPUs for inference, batch_size={args.batch_size}")
        logger.info(f"Model: {args.model_name}")
        if args.model_path:
            logger.info(f"Loading weights from: {args.model_path}")
        logger.info(f"Benchmark: {args.benchmark}")
        
        parameters = args.added_fields

        # Run benchmarks for each parameter
        for param_value in parameters:
            existing = find_existing_result(output_path, args.benchmark, args.model_name, param_value)
            if existing is not None:
                logger.info(f"Results already exist for field={param_value} at {existing} - skipping inference")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Running benchmark with field: {param_value}")
            logger.info(f"{'='*60}\n")

            # Create dataset
            dataset = create_dataset(
                args.benchmark,
                field=param_value,
                position=args.position
            )
            
            # Run benchmark
            run_benchmark_iteration(
                dataset=dataset,
                worker_class=worker_class,
                model_name=args.model_name,
                num_gpus=num_gpus,
                output_path=output_path,
                benchmark=args.benchmark,
                param_value=param_value,
                model_load_path=args.model_path,
                batch_size=args.batch_size
            )
        
        logger.info("\n" + "="*60)
        logger.info("All benchmark iterations completed successfully!")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


