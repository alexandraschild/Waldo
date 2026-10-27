"""
Script for evaluation of VLM predictions across multiple benchmarks.
Optimized with async API calls and caching.
"""

import json
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import glob
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

try:
    import orjson
    def load_json(filepath):
        with open(filepath, "rb") as f:
            return orjson.loads(f.read())
    def dump_json(data, filepath):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
except ImportError:
    def load_json(filepath):
        with open(filepath) as f:
            return json.load(f)
    def dump_json(data, filepath):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

from custom_parcing import mmeval_parse_answer, mathverse_parse_answer

load_dotenv()

# ==================== Configuration ====================

# Max concurrent API requests (tune based on your rate limit)
MAX_CONCURRENT_REQUESTS = 50

class EvalMethod(Enum):
    EXACT_MATCH = "exact_match"
    PARSED_MATCH = "parsed_match"
    LLM_EVAL = "llm_eval"
    OUT_OF_MEMORY = "out of memory"


@dataclass
class EvaluatorConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 10
    temperature: float = 0.0


@dataclass
class BenchmarkConfig:
    name: str
    valid_answers: Optional[List[str]] = None
    parse_func: Optional[Callable] = None
    original: Optional[List[float]] = None
    fields: Optional[List[float]] = None
    scales: Optional[List[float]] = None
    interpolation: Optional[List[float]] = None

    @property
    def is_closed_form(self) -> bool:
        return self.valid_answers is not None


# ==================== Evaluation Prompts ====================

CORRECTNESS_PROMPT = """Compare two answers to the question and output 1 if the predicted answer contains the same information as the ground truth. Output 0 if some of the information is missing or the answer is completely wrong. 
ex1: 
Ground truth: "a"
Prediction: "the style depicted in this image is a: impressionism."
Score: 1
ex2: 
Ground truth: "$\\frac(1)(2000)$"
Prediction: "the average rate of change in the graph is 5."
Score: 0

Is the predicted answer correct?
Question: {question}
Ground truth: {ground_truth}
Prediction: {prediction}
Output only the numeric score (0 or 1)"""


# ==================== Async LLM Evaluator ====================

class AsyncLLMEvaluator:
    """Handles LLM-based evaluation using async OpenAI API with concurrency control."""

    def __init__(self, config: EvaluatorConfig, max_concurrent: int = MAX_CONCURRENT_REQUESTS):
        self.config = config
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def evaluate_single(self, question: str, ground_truth: str, prediction: str) -> Optional[float]:
        """Evaluate a single prediction with rate limiting."""
        prompt = CORRECTNESS_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return float(response.choices[0].message.content.strip())
            except Exception as e:
                print(f"API error: {e}")
                return None

    async def evaluate_batch(self, items: List[Dict]) -> List[Dict]:
        """Evaluate a batch of items concurrently."""
        if not items:
            return items

        tasks = [
            self.evaluate_single(item["question"], item["ground_truth"], item["generated_answer"])
            for item in items
        ]

        scores = await tqdm_asyncio.gather(*tasks, desc=f"  LLM eval ({len(items)} items)")

        for item, score in zip(items, scores):
            item["score"] = score
            item["eval_method"] = EvalMethod.LLM_EVAL.value

        return items


# ==================== Answer Matcher (no changes needed, already fast) ====================

class AnswerMatcher:
    @staticmethod
    def exact_match(prediction: str, ground_truth: str, valid_answers: List[str]) -> Optional[float]:
        pred_clean = prediction.replace(".", "").lower()
        if pred_clean in valid_answers:
            return 1.0 if pred_clean == ground_truth.lower() else 0.0
        return None

    @staticmethod
    def parsed_match(prediction: str, ground_truth: str,
                     valid_answers: List[str], parse_func: Callable) -> Optional[float]:
        try:
            parsed_pred = parse_func(prediction)
            if parsed_pred in valid_answers:
                return 1.0 if parsed_pred == ground_truth else 0.0
        except Exception:
            pass
        return None


# ==================== Core evaluation logic ====================

def evaluate_item_local(item: Dict, config: BenchmarkConfig) -> Tuple[Dict, bool]:
    """
    Try to evaluate an item without LLM. Returns (item, needs_llm).
    
    If needs_llm is True, the item still needs LLM evaluation.
    """
    gt = item["ground_truth"].lower().strip()
    pred = item["generated_answer"].lower().strip()
    item["ground_truth"] = gt
    item["generated_answer"] = pred

    # OOM check
    if "error: cuda out of memory." in pred:
        item["score"] = 0.0
        item["eval_method"] = EvalMethod.OUT_OF_MEMORY.value
        return item, False

    if config.is_closed_form:
        # Exact match
        score = AnswerMatcher.exact_match(pred, gt, config.valid_answers)
        if score is not None:
            item["score"] = score
            item["eval_method"] = EvalMethod.EXACT_MATCH.value
            return item, False

        # Parsed match
        if config.parse_func:
            score = AnswerMatcher.parsed_match(pred, gt, config.valid_answers, config.parse_func)
            if score is not None:
                item["score"] = score
                item["eval_method"] = EvalMethod.PARSED_MATCH.value
                return item, False
    else:
        if pred == gt:
            item["score"] = 1.0
            item["eval_method"] = EvalMethod.EXACT_MATCH.value
            return item, False

    return item, True


# ==================== File utilities ====================

def build_glob_pattern(benchmark: str, model: str, param: float,
                       position: str, folder: str) -> str:
    position_str = f"_{position}" if position else ""
    model_folder = "Qwen_" if "qwen" in model.lower() else ""
    return f"{folder}/{model_folder}{model}/{benchmark}/{benchmark}_{model}_{param}{position_str}*.json"


def get_output_path(output_dir: str, input_folder: str, model: str,
                    benchmark: str, param: float, position: str) -> Path:
    position_suffix = f"_{position}" if position else ""
    return Path(output_dir) / input_folder / model / benchmark / f"{benchmark}_{model}_{param}{position_suffix}.json"


# ==================== Main evaluation logic ====================

async def evaluate_single_file(
    filepath: Path,
    config: BenchmarkConfig,
    output_path: Path,
    llm_evaluator: AsyncLLMEvaluator,
    param: float,
):
    """Evaluate a single result file."""
    # Skip if already evaluated
    if output_path.exists():
        return

    results = load_json(filepath)

    # Phase 1: local evaluation (exact/parsed match) — CPU bound, fast
    evaluated = []
    needs_llm = []
    for item in results:
        item, needs = evaluate_item_local(item, config)
        if needs:
            needs_llm.append(item)
        else:
            evaluated.append(item)

    # Phase 2: batch LLM evaluation — I/O bound, async
    if needs_llm:
        needs_llm = await llm_evaluator.evaluate_batch(needs_llm)

    all_results = evaluated + needs_llm

    # Save
    dump_json(all_results, output_path)

    # Print summary
    scores = [r["score"] for r in all_results if r.get("score") is not None]
    if scores:
        avg = sum(scores) / len(scores)
        exact = sum(1 for r in all_results
                    if r.get("eval_method") in [EvalMethod.EXACT_MATCH.value, EvalMethod.PARSED_MATCH.value])
        print(f"  {config.name} {param}: avg={avg:.3f}, "
              f"exact={exact}/{len(all_results)}, llm={len(needs_llm)}")


async def evaluate_model(
    model: str,
    benchmarks: Dict[str, BenchmarkConfig],
    llm_evaluator: AsyncLLMEvaluator,
    output_dir: str = "evaluated1",
    input_folder: str = "results_field",
):
    """Evaluate a single model across all benchmarks."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating model: {model}")
    print(f"{'=' * 60}\n")

    for benchmark_name, config in benchmarks.items():
        # Select parameters based on folder
        if "results0" in input_folder:
            parameters = config.original
        elif "results_field" in input_folder:
            parameters = config.fields
        elif "results_scale" in input_folder:
            parameters = config.scales
        elif "results_interpolation" in input_folder:
            parameters = config.interpolation
        else:
            parameters = config.fields

        if not parameters:
            continue

        for param in parameters:
            pattern = build_glob_pattern(config.name, model, param, "", input_folder)
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"  No files: {config.name} {model} {param}, {pattern}")
                continue

            output_path = get_output_path(output_dir, input_folder, model, config.name, param, "")
            await evaluate_single_file(Path(files[0]), config, output_path, llm_evaluator, param)


async def main():
    """Main async execution."""
    evaluator_config = EvaluatorConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    models = [
        "pixtral-12b",
        "llava-1.5-7b-hf",
        "Llama-3.2-11B-Vision-Instruct",
        #"Qwen2-VL-2B-Instruct",
        #"Qwen2-VL-7B-Instruct",
        #"Qwen2-VL-72B-Instruct",
        #"Qwen2.5-VL-3B-Instruct",
        #"Qwen2.5-VL-7B-Instruct",
        #"Qwen2.5-VL-32B-Instruct",
        #"Qwen2.5-VL-72B-Instruct",
        #"Qwen3-VL-2B-Instruct",
        #"Qwen3-VL-4B-Instruct",
        #"Qwen3-VL-8B-Instruct",
        #"Qwen3-VL-32B-Instruct",
        #"gpt-5-mini",
    ]

    llm_evaluator = AsyncLLMEvaluator(evaluator_config, max_concurrent=MAX_CONCURRENT_REQUESTS)

    all_benchmarks = {
        **BenchmarkRegistry.get_closed_form_benchmarks(),
        **BenchmarkRegistry.get_open_ended_benchmarks(),
    }

    for model in models:
        await evaluate_model(model, all_benchmarks, llm_evaluator,
                             output_dir="evaluated1", input_folder="results_field_upper_right_corner")


# ==================== Benchmark Registry ====================

class BenchmarkRegistry:
    @staticmethod
    def get_closed_form_benchmarks() -> Dict[str, BenchmarkConfig]:
        return {
            "MMEval": BenchmarkConfig(
                name="MMEval",
                valid_answers=["yes", "no"],
                parse_func=mmeval_parse_answer,
                original=[0], 
                fields=[0.5, 1, 2, 3, 4, 5],
                scales=[1.5, 2, 3, 4, 5, 6], 
                interpolation=[1.5, 2, 3, 4, 5, 6],
            ),
            "MathVerse": BenchmarkConfig(
                name="MathVerse",
                valid_answers=["a", "b", "c", "d", "a.", "b.", "c.", "d."],
                parse_func=mathverse_parse_answer,
                original=[0], 
                fields=[0.5, 1, 2, 3, 4, 5],
                scales=[1.5, 2, 3, 4, 5, 6], 
                interpolation=[1.5, 2, 3, 4, 5, 6],
            ),
            "MMStar": BenchmarkConfig(
                name="MMStar",
                valid_answers=["a", "b", "c", "d", "a.", "b.", "c.", "d."],
                parse_func=mathverse_parse_answer,
                original=[0], 
                fields=[0.5, 1, 2, 3, 4, 5],
                scales=[1.5, 2, 3, 4, 5, 6], 
                interpolation=[1.5, 2, 3, 4, 5, 6],
            ),
        }

    @staticmethod
    def get_open_ended_benchmarks() -> Dict[str, BenchmarkConfig]:
        return {
            "MMVet": BenchmarkConfig(
                name="MMVet",
                original=[0], 
                fields=[0.5, 1, 2, 3, 4, 5],
                scales=[1.5, 2, 3, 4, 5, 6], 
                interpolation=[1.5, 2, 3, 4, 5, 6],
            ),
            "CharXiv": BenchmarkConfig(
                name="CharXiv",
                original=[0], 
                fields=[0.5, 1, 2, 3, 4, 5],
                scales=[1.5, 2, 3, 4, 5, 6], 
                interpolation=[1.5, 2, 3, 4, 5, 6],
            ),
        }

    @staticmethod
    def get_all_benchmarks() -> Dict[str, BenchmarkConfig]:
        return {
            **BenchmarkRegistry.get_closed_form_benchmarks(),
            **BenchmarkRegistry.get_open_ended_benchmarks(),
        }


if __name__ == "__main__":
    asyncio.run(main())