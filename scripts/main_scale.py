#!/usr/bin/env python3
"""
Distributed VLM inference on benchmarks across all available GPUs for upscaling transformation.
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


import os
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

def worker_process(
    gpu_id: int,
    model_name: str,
    worker_class,
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event
):
    """Worker process function."""
    worker = worker_class(gpu_id, model_name, work_queue, result_queue, stop_event)
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
    scale: Optional[float] = None,
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
    kwargs["scale"] = scale
    
    return dataset_class(**kwargs)

def get_model_short_name(model_name: str) -> str:
    """Extract short model name from full path."""
    return model_name.split('/')[-1] if '/' in model_name else model_name

def generate_output_filename(
    benchmark: str,
    model_name: str,
    param_value: Union[int, float],
    timestamp: str,
    partial: bool = False
) -> str:
    """Generate standardized output filename."""
    short_name = get_model_short_name(model_name)
    suffix = "_partial" if partial else ""
    
    return f"{benchmark}_{short_name}_{param_value}_{timestamp}{suffix}.json"

@contextmanager
def multiprocess_environment(num_gpus: int):
    """Context manager for multiprocessing resources."""
    work_queue = mp.Queue(maxsize=num_gpus * 2)
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
    timestamp: str
) -> List[Dict]:
    """Run a single benchmark iteration with given parameters."""
    
    with multiprocess_environment(num_gpus) as (work_queue, result_queue, stop_event, processes):
        # Start worker processes
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, model_name, worker_class, work_queue, result_queue, stop_event)
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
            
            # Save results
            output_file = output_path / generate_output_filename(
                benchmark, model_name, param_value, timestamp
            )
            save_results(results, output_file)
            log_gpu_utilization(results)
            
            return results
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            stop_event.set()
            
            # Save partial results
            if results:
                partial_file = output_path / generate_output_filename(
                    benchmark, model_name, param_value, timestamp, partial=True
                )
                save_results(results, partial_file)
            
            raise

def parse_list_argument(value: str) -> List[Union[int, float]]:
    """Parse comma-separated list argument."""
    if isinstance(value, list):
        return value
    
    try:
        # Remove brackets if present
        value = value.strip('[]')
        # Split and convert
        parts = [x.strip() for x in value.split(',')]
        
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
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Model name or path"
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
        "--benchmark",
        type=str,
        default="MMVet",
        help=f"Benchmark to test. Available: {list(BENCHMARK_CLASSES.keys())}"
    )
    parser.add_argument(
        "--added_scales",
        type=str,
        default="5,6",#1.5, 2, 3, 4",
        help="Comma-separated scales in percent (e.g., '0, 1, 2')"
    )
    
    args = parser.parse_args()
    
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
    
        # Convert string arguments to lists
        if isinstance(args.added_scales, str):
            args.added_scales = parse_list_argument(args.added_scales)
        
        # Setup multiprocessing
        mp.set_start_method('spawn', force=True)
        
        # Initialize
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get worker class
        worker_class = get_worker_class(args.model_name)
        
        # Determine number of GPUs
        num_gpus = args.num_gpus or torch.cuda.device_count()
        num_gpus = min(num_gpus, torch.cuda.device_count())
        
        # Log setup information
        logger.info(f"Using {num_gpus} GPUs for inference")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Benchmark: {args.benchmark}")
        
        parameters = args.added_scales

        # Run benchmarks for each parameter
        for param_value in parameters:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running benchmark with scales: {param_value}")
            logger.info(f"{'='*60}\n")
            
            # Create dataset
            dataset = create_dataset(
                args.benchmark,
                scale=param_value,
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
                timestamp=timestamp
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


