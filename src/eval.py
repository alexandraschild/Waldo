"""
Script for evaluation of VLM predictions across multiple benchmarks 
We use exact and parcing based matching, where possible. 
For other cases we implement LLM-as-a-judge evaluation with gpt-4o-mini
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import glob

import torch
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

from custom_parcing import mmeval_parse_answer, mathverse_parse_answer

load_dotenv()

# ==================== Configuration ====================

class EvalMethod(Enum):
    """Evaluation methods supported by the system."""
    EXACT_MATCH = "exact_match"
    PARSED_MATCH = "parsed_match"
    LLM_EVAL = "llm_eval"
    OUT_OF_MEMORY = "out of memory"

@dataclass
class EvaluatorConfig:
    """Configuration for the LLM evaluator."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 10
    temperature: float = 0.0
    device: str = "cpu"


@dataclass
class BenchmarkConfig:
    """Configuration for a specific benchmark."""
    name: str
    valid_answers: Optional[List[str]] = None
    parse_func: Optional[Callable] = None
    fields: Optional[List[float]] = None
    scales: Optional[List[float]] = None
    
    @property
    def is_closed_form(self) -> bool:
        """Check if benchmark uses closed-form answers."""
        return self.valid_answers is not None


# ==================== Evaluation Prompts ====================

class EvalPrompts:
    """Centralized evaluation prompts."""
    
    CORRECTNESS = """Compare two answers to the question and output 1 if the predicted answer contains the same information as the ground truth. Output 0 if some of the information is missing or the answer is completely wrong. 
ex1: 
Ground truth: "a"
Prediction: "the style depicted in this image is a: impressionism."
Score: 1
ex2: 
Ground truth: "$\\frac(1)(2000)$"
Prediction: "the average rate of change in the graph is 5."
Score: 0

Is the predicted answer correct?
Ground truth: {ground_truth}
Prediction: {prediction}
Output only the numeric score (0 or 1)"""


# ==================== LLM Evaluator ====================

class LLMEvaluator:
    """Handles LLM-based evaluation using OpenAI API."""

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.client = self._initialize_client()

    def _initialize_client(self) -> OpenAI:
        """Initialize OpenAI client with configuration."""
        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )

    def evaluate_single(self, ground_truth: str, prediction: str) -> Optional[float]:
        """
        Evaluate a single prediction against ground truth.
        
        Args:
            ground_truth: The correct answer
            prediction: The model's prediction
            
        Returns:
            Score (0.0 or 1.0) or None if evaluation fails
        """
        prompt = EvalPrompts.CORRECTNESS.format(
            ground_truth=ground_truth,
            prediction=prediction
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            score = float(response.choices[0].message.content.strip())
            return score
        except Exception as e:
            print(f"API evaluation error: {e}")
            return None


# ==================== Answer Matcher ====================

class AnswerMatcher:
    """Handles exact and parsed matching of answers."""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str, valid_answers: List[str]) -> Optional[float]:
        """
        Check for exact match.
        
        Returns:
            Score (0.0 or 1.0) if match found in valid answers, None otherwise
        """
        pred_clean = prediction.replace(".", "").lower()
        if pred_clean in valid_answers:
            return 1.0 if pred_clean == ground_truth.lower() else 0.0
        return None
    
    @staticmethod
    def parsed_match(prediction: str, ground_truth: str, 
                    valid_answers: List[str], parse_func: Callable) -> Optional[float]:
        """
        Check for parsed match using custom parsing function.
        
        Returns:
            Score (0.0 or 1.0) if parsed match found, None otherwise
        """
        try:
            parsed_pred = parse_func(prediction)
            if parsed_pred in valid_answers:
                return 1.0 if parsed_pred == ground_truth else 0.0
        except Exception as e:
            print(f"Parsing error: {e}")
        return None


# ==================== Item Evaluator ====================

class ItemEvaluator:
    """Evaluates individual items using multiple strategies."""
    
    def __init__(self, llm_evaluator: LLMEvaluator):
        self.llm_evaluator = llm_evaluator
        self.matcher = AnswerMatcher()
    
    def evaluate_item(self, item: Dict, config: BenchmarkConfig) -> Dict:
        """
        Evaluate a single item using appropriate strategy.
        
        Args:
            item: Dictionary with 'question', 'ground_truth', 'generated_answer'
            config: Benchmark configuration
            
        Returns:
            Item with added 'score' and 'eval_method' keys
        """
        # Normalize answers
        item["ground_truth"] = self.matcher.normalize(item["ground_truth"])
        item["generated_answer"] = self.matcher.normalize(item["generated_answer"])
        
        # Check for OOM errors
        if "error: cuda out of memory." in item["generated_answer"]:
            item["score"] = 0.0
            item["eval_method"] = EvalMethod.OUT_OF_MEMORY.value
            return item
        
        # Try exact match first (for all benchmarks)
        if config.is_closed_form:
            score = self.matcher.exact_match(
                item["generated_answer"],
                item["ground_truth"],
                config.valid_answers
            )
            if score is not None:
                item["score"] = score
                item["eval_method"] = EvalMethod.EXACT_MATCH.value
                return item
            
            # Try parsed match
            if config.parse_func:
                score = self.matcher.parsed_match(
                    item["generated_answer"],
                    item["ground_truth"],
                    config.valid_answers,
                    config.parse_func
                )
                if score is not None:
                    item["score"] = score
                    item["eval_method"] = EvalMethod.PARSED_MATCH.value
                    return item
        else:
            # Open-ended: simple exact match
            if item["generated_answer"] == item["ground_truth"]:
                item["score"] = 1.0
                item["eval_method"] = EvalMethod.EXACT_MATCH.value
                return item
        
        # Fall back to LLM evaluation
        item["score"] = self.llm_evaluator.evaluate_single(
            item["ground_truth"],
            item["generated_answer"]
        )
        item["eval_method"] = EvalMethod.LLM_EVAL.value
        return item


# ==================== File Handler ====================

class ResultFileHandler:
    """Handles file operations for results."""
    
    def __init__(self, results_dir: str = "results_field"):
        self.results_dir = Path(results_dir)
    
    @staticmethod
    def _build_pattern(benchmark: str, model: str, param: float, 
                       position: str, folder: str) -> str:
        """Build file pattern for glob matching."""
        position_str = f"_{position}" if position else ""
        model_folder = "Qwen_" if "qwen" in model.lower() else ""
        return f"{folder}/{model_folder}{model}/{benchmark}/{benchmark}_{model}_{param}{position_str}*.json"
    
    def find_result_files(self, benchmark: str, model: str, param: float,
                         position: str, use_fields: bool) -> List[Path]:
        """
        Find result files matching the pattern.
        
        Args:
            benchmark: Benchmark name
            model: Model name
            param: Field or scale parameter value
            position: Position identifier (optional)
            use_fields: Whether to use fields (True) or scales (False)
            
        Returns:
            Sorted list of matching file paths
        """
        folder = "results_field" if use_fields else "results_scale"
        pattern = self._build_pattern(benchmark, model, param, position, folder)
        files = sorted(glob.glob(pattern))
        return [Path(f) for f in files]
    
    @staticmethod
    def load_results(filepath: Path) -> List[Dict]:
        """Load results from JSON file."""
        with open(filepath) as f:
            return json.load(f)
    
    @staticmethod
    def save_results(results: List[Dict], filepath: Path):
        """Save results to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)


# ==================== Benchmark Evaluator ====================

class BenchmarkEvaluator:
    """Main evaluator for benchmarks."""
    
    def __init__(self, llm_evaluator: LLMEvaluator, results_dir: str = "results_field"):
        self.item_evaluator = ItemEvaluator(llm_evaluator)
        self.llm_evaluator = llm_evaluator
        self.file_handler = ResultFileHandler(results_dir)
    
    def _separate_items(self, items: List[Dict], config: BenchmarkConfig) -> tuple:
        """
        Separate items into those needing LLM eval and those that don't.
        
        Returns:
            Tuple of (evaluated_items, llm_eval_items)
        """
        evaluated_items = []
        llm_eval_items = []
        
        for item in items:
            evaluated_item = self.item_evaluator.evaluate_item(item, config)
            
            #if evaluated_item.get("eval_method") == EvalMethod.LLM_EVAL.value:
                # Remove the score set by evaluate_item to re-evaluate in batch
                #evaluated_item.pop("score", None)
            #    llm_eval_items.append(evaluated_item)
            #else:
            evaluated_items.append(evaluated_item)
        
        return evaluated_items#, llm_eval_items
    
    def _print_summary(self, results: List[Dict], benchmark: str, param: float):
        """Print evaluation summary statistics."""
        scores = [r["score"] for r in results if r["score"] is not None]
        if not scores:
            print(f"{benchmark} {param}: No valid scores")
            return
        
        avg_score = sum(scores) / len(scores)
        exact_matches = sum(
            1 for r in results 
            if r["eval_method"] in [EvalMethod.EXACT_MATCH.value, EvalMethod.PARSED_MATCH.value]
        )
        
        print(f"{benchmark} {param}: avg={avg_score:.3f}, "
              f"exact_matches={exact_matches}/{len(results)}")
    
    def evaluate_benchmark(self, model: str, config: BenchmarkConfig,
                          position: str = "", output_dir: str = "evaluated"):
        """
        Evaluate model on a benchmark across multiple parameters.
        
        Args:
            model: Model name
            config: Benchmark configuration
            position: Position identifier (optional)
            output_dir: Output directory for results
        """
        #use_fields = config.fields is not None
        use_fields = False
        parameters = config.fields if use_fields else config.scales
        folder = "results_field" if use_fields else "results_scale"
        
        for param in parameters:
            files = self.file_handler.find_result_files(
                config.name, model, param, position, use_fields
            )
            
            if not files:
                print(f"No files found for {config.name} {model} {param} {position}")
                continue
            
            # Load and evaluate results
            results = self.file_handler.load_results(files[0])
            
            desc = f"Evaluating {config.name}: {param}"
            evaluated_items = self._separate_items( #, llm_eval_items
                tqdm(results, desc=desc), config
            )
            
            # Batch evaluate remaining items
            #if llm_eval_items:
            #    print(f"LLM evaluating {len(llm_eval_items)} items...")
            #    llm_eval_items = self.llm_evaluator.batch_evaluate(llm_eval_items)
            
            # Combine and save results
            #all_results = evaluated_items + llm_eval_items
            
            position_suffix = f"_{position}" if position else ""
            output_path = Path(output_dir) / folder / model / config.name / f"{config.name}_{model}_{param}{position_suffix}.json"
            self.file_handler.save_results(evaluated_items, output_path)
            
            self._print_summary(evaluated_items, config.name, param)


# ==================== Benchmark Registry ====================

class BenchmarkRegistry:
    """Registry of all available benchmarks."""
    
    @staticmethod
    def get_closed_form_benchmarks() -> Dict[str, BenchmarkConfig]:
        """Get closed-form benchmarks configuration."""
        return {
            "MMEval": BenchmarkConfig(
                name="MMEval",
                valid_answers=["yes", "no"],
                parse_func=mmeval_parse_answer,
                fields=[0, 0.5, 1, 2, 3, 4, 5],
                scales=[1.5, 2, 3, 4],
            ),
            "MathVerse": BenchmarkConfig(
                name="MathVerse",
                valid_answers=["a", "b", "c", "d", "a.", "b.", "c.", "d."],
                parse_func=mathverse_parse_answer,
                fields=[0, 0.5, 1, 2, 3, 4, 5],
                scales=[1.5, 2, 3, 4],
            ),
            "MMStar": BenchmarkConfig(
                name="MMStar",
                valid_answers=["a", "b", "c", "d", "a.", "b.", "c.", "d."],
                parse_func=mathverse_parse_answer,
                fields=[0, 0.5, 1, 2, 3, 4, 5],
                scales=[1.5, 2, 3, 4],
            )
        }
    
    @staticmethod
    def get_open_ended_benchmarks() -> Dict[str, BenchmarkConfig]:
        """Get open-ended benchmarks configuration."""
        return {
            "MMVet": BenchmarkConfig(
                name="MMVet",
                fields=[0, 0.5, 1, 2, 3, 4, 5],
                scales=[1.5, 2, 3, 4],
            ),
            "CharXiv": BenchmarkConfig(
                name="CharXiv",
                fields=[0, 0.5, 1, 2, 3, 4, 5],
                scales=[1.5, 2, 3, 4],
            )
        }
    
    @staticmethod
    def get_all_benchmarks() -> Dict[str, BenchmarkConfig]:
        """Get all benchmarks configuration."""
        return {
            **BenchmarkRegistry.get_closed_form_benchmarks(),
            **BenchmarkRegistry.get_open_ended_benchmarks()
        }


# ==================== Main Execution ====================

def main():
    """Main execution function."""
    # Configuration
    evaluator_config = EvaluatorConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    
    models = [
        #"pixtral-12b",
        #"llava-1.5-7b-hf",
        #"Llama-3.2-11B-Vision-Instruct",
        "Qwen2-VL-2B-Instruct",
        "Qwen2-VL-7B-Instruct",
        "Qwen2-VL-72B-Instruct",
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-32B-Instruct",
        "Qwen2.5-VL-72B-Instruct",
        "Qwen3-VL-2B-Instruct",
        "Qwen3-VL-4B-Instruct",
        "Qwen3-VL-8B-Instruct",
        "Qwen3-VL-32B-Instruct"
    ]
    
    # Initialize evaluators
    llm_evaluator = LLMEvaluator(evaluator_config)
    benchmark_evaluator = BenchmarkEvaluator(llm_evaluator)
    
    # Get benchmark configurations
    all_benchmarks = BenchmarkRegistry.get_all_benchmarks()
    
    # Run evaluations
    for model in models:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model}")
        print(f"{'='*60}\n")
        
        for benchmark_name, config in all_benchmarks.items():
            benchmark_evaluator.evaluate_benchmark(
                model=model,
                config=config,
                position="",
            )


if __name__ == "__main__":
    main()