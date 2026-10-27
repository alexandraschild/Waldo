from openai import OpenAI
import json
import os
from tqdm import tqdm
from typing import List, Dict, Optional, Callable, Union
import glob
from custom_parcing import mmeval_parse_answer, mathverse_parse_answer
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum
import torch

load_dotenv()


class EvaluatorType(Enum):
    GPT4 = "gpt-4"
    QWEN3_LOCAL = "qwen-local"


@dataclass
class EvaluatorConfig:
    evaluator_type: EvaluatorType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 10  # Increased from 3 to allow for possible extra tokens
    temperature: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LLMEvaluator:
    """Handles LLM-based evaluation with support for multiple models."""
    
    EVAL_PROMPT = """Compare the prediction against the ground truth answer and provide a correctness score.

Score 1 if the prediction is CORRECT (contains the full right answer).
Score 0 if the prediction is WRONG or INCOMPLETE.

Is prediction the same as ground truth?:
Ground truth: {ground_truth}
Prediction: {prediction}

Output only the numeric score (0 or 1)"""

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.client = None
        self.tokenizer = None
        self.model = None
        
        if config.evaluator_type == EvaluatorType.GPT4:
            self._init_openai()
        elif config.evaluator_type == EvaluatorType.QWEN3_LOCAL:
            self._init_qwen_local()

    def _init_openai(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=self.config.api_key)
        self.model_name = "gpt-4o-mini"

    def _init_qwen_local(self):
        """Initialize local Qwen3 model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = self.config.model_name
        print(f"Loading Qwen3 model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left'  # Fix for decoder-only models
        )
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map=self.config.device if self.config.device == "cuda" else None,
        )
        
        if self.config.device == "cpu":
            self.model = self.model.to(self.config.device)
        
        self.model.eval()
        print(f"Model loaded on {self.config.device}")

    def _evaluate_with_api(self, prompt_text: str) -> Optional[float]:
        """Evaluate using API (OpenAI or Qwen API)."""
        messages = [{"role": "user", "content": prompt_text}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=messages,
            )
            score = float(response.choices[0].message.content.strip())
            return score
        except Exception as e:
            print(f"API evaluation error: {e}")
            return None

    def _evaluate_with_local_qwen(self, prompt_text: str) -> Optional[float]:
        """Evaluate using local Qwen3 model."""
        messages = [{"role": "user", "content": prompt_text}]
        
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=False,  # Use greedy decoding for consistency
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only the generated tokens
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            ).strip()
            
            # Extract score - be more aggressive with parsing
            # First, try to find just "0" or "1" in the output
            if '1' in generated_text and '0' not in generated_text:
                return 1.0
            elif '0' in generated_text and '1' not in generated_text:
                return 0.0
            # If both or neither, try to parse the first character
            elif generated_text:
                first_char = generated_text[0]
                if first_char in ['0', '1']:
                    return float(first_char)
            
            # Last resort - try to convert whole string
            score = float(generated_text)
            return score
            
        except Exception as e:
            print(f"Local model evaluation error: {e}")
            print(f"Generated text was: '{generated_text if 'generated_text' in locals() else 'N/A'}'")
            return None

    def evaluate(self, question: str, ground_truth: str, prediction: str) -> Optional[float]:
        """Evaluate a single prediction against ground truth."""
        prompt_text = self.EVAL_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction
        )
        
        if self.config.evaluator_type == EvaluatorType.QWEN3_LOCAL:
            return self._evaluate_with_local_qwen(prompt_text)
        else:
            return self._evaluate_with_api(prompt_text)

    def batch_evaluate(self, items: List[Dict], batch_size: int = 1) -> List[Dict]:
        """Evaluate multiple items with progress bar.
        
        Note: batch_size > 1 only supported for local models currently.
        """
        results = []
        
        if self.config.evaluator_type == EvaluatorType.QWEN3_LOCAL and batch_size > 1:
            # Batch processing for local model
            for i in tqdm(range(0, len(items), batch_size), desc="LLM Evaluation (batched)"):
                batch = items[i:i+batch_size]
                batch_results = self._batch_evaluate_local(batch)
                results.extend(batch_results)
        else:
            # Sequential processing
            for item in tqdm(items, desc="LLM Evaluation"):
                score = self.evaluate(
                    item["question"],
                    item["ground_truth"],
                    item["generated_answer"]
                )
                item["score"] = score
                item["eval_method"] = "llm_eval"
                results.append(item)
        
        return results

    def _batch_evaluate_local(self, batch: List[Dict]) -> List[Dict]:
        """Batch evaluation for local Qwen3 model."""
        all_messages = []
        for item in batch:
            prompt_text = self.EVAL_PROMPT.format(
                question=item["question"],
                ground_truth=item["ground_truth"],
                prediction=item["generated_answer"]
            )
            all_messages.append([{"role": "user", "content": prompt_text}])
        
        try:
            # Process batch
            inputs_list = []
            for messages in all_messages:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs_list.append(inputs["input_ids"])
            
            # Pad sequences for batch processing
            from torch.nn.utils.rnn import pad_sequence
            padded_inputs = pad_sequence(
                [inp.squeeze(0) for inp in inputs_list],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            ).to(self.model.device)
            
            attention_mask = (padded_inputs != self.tokenizer.pad_token_id).long()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=padded_inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=False,  # Use greedy decoding for consistency, matching _evaluate_with_local_qwen
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode outputs
            for idx, item in enumerate(batch):
                input_length = inputs_list[idx].shape[-1]
                generated_text = self.tokenizer.decode(
                    outputs[idx][input_length:],
                    skip_special_tokens=True
                ).strip()
                
                # Parse score with robust extraction
                try:
                    if '1' in generated_text and '0' not in generated_text:
                        item["score"] = 1.0
                    elif '0' in generated_text and '1' not in generated_text:
                        item["score"] = 0.0
                    elif generated_text and generated_text[0] in ['0', '1']:
                        item["score"] = float(generated_text[0])
                    else:
                        item["score"] = float(generated_text)
                except:
                    print(f"Warning: Could not parse score from: '{generated_text}'")
                    item["score"] = None
                item["eval_method"] = "llm_eval"
        
        except Exception as e:
            print(f"Batch evaluation error: {e}")
            for item in batch:
                item["score"] = None
                item["eval_method"] = "llm_eval"
        
        return batch


class BenchmarkEvaluator:
    """Evaluates model predictions on various benchmarks."""
    
    def __init__(self, llm_evaluator: LLMEvaluator, results_dir: str = "sampled_VLM_results"):
        self.llm_evaluator = llm_evaluator
        self.results_dir = results_dir

    @staticmethod
    def find_result_files(benchmark: str, model: str, field: float, scale: float,
                         position: str = "", path: str = ".") -> List[str]:
        """Find result files matching the pattern."""
        if field: 
            position_str = f"_{position}" if position else ""
            pattern = f"{path}/{benchmark}_results_{model}_field{field}{position_str}*.json"
            return sorted(glob.glob(pattern))
        elif scale:
            position_str = f"_{position}" if position else ""
            pattern = f"{path}/{benchmark}_results_sr_{model}_scale{scale}{position_str}*.json"
            return sorted(glob.glob(pattern))

    def evaluate_closed_form(self, item: Dict, valid_answers: List[str], 
                           parse_func: Optional[Callable] = None) -> Dict:
        """Evaluate a closed-form answer item."""
        # Normalize to lowercase once
        item["ground_truth"] = item["ground_truth"].lower()
        item["generated_answer"] = item["generated_answer"].lower()

        if "error: cuda out of memory." in item["generated_answer"]:
            item["score"] = 0.0 
            item["eval_method"] = "out of memory"
            return item                     
        
        # Try exact match
        pred_clean = item["generated_answer"].replace(".", "")
        if pred_clean in valid_answers:
            item["eval_method"] = "exact_match"
            item["score"] = 1.0 if pred_clean == item["ground_truth"] else 0.0
            return item
        
        # Try parsed match if function provided
        if parse_func:
            parsed_pred = parse_func(item["generated_answer"])
            if parsed_pred in valid_answers:
                item["eval_method"] = "parsed_match"
                item["score"] = 1.0 if parsed_pred == item["ground_truth"] else 0.0
                return item
        
        # Fall back to LLM evaluation
        item["eval_method"] = "llm_eval"
        item["score"] = self.llm_evaluator.evaluate(
            item["question"],
            item["ground_truth"],
            item["generated_answer"]
        )
        return item

    def evaluate_open_ended(self, item: Dict) -> Dict:
        """Evaluate an open-ended answer item."""
        # Normalize to lowercase once
        item["ground_truth"] = item["ground_truth"].lower()
        item["generated_answer"] = item["generated_answer"].lower()
        
        if item["generated_answer"] == item["ground_truth"]:
            item["eval_method"] = "exact_match"
            item["score"] = 1.0
        elif "error: cuda out of memory." in item["generated_answer"]:
            item["score"] = 0.0 
            item["eval_method"] = "out of memory"                              
        else:
            item["eval_method"] = "llm_eval"
            item["score"] = self.llm_evaluator.evaluate(
                item["question"],
                item["ground_truth"],
                item["generated_answer"]
            )
        return item

    def evaluate_benchmark(self, model: str, benchmark: str, fields: List[float], scales: List[float],
                          valid_answers: Optional[List[str]] = None,
                          parse_func: Optional[Callable] = None,
                          position: str = "", output_dir: str = "sampled_eval_results",
                          batch_size: int = 1):
        """Evaluate model on a benchmark across multiple fields."""
        os.makedirs(output_dir, exist_ok=True)
        is_closed_form = valid_answers is not None
        
        if fields:
            for field in fields:
                files = self.find_result_files(
                    benchmark, model, field, position, self.results_dir
                )
                
                if not files:
                    print(f"No files found for {benchmark} {model} field {field} position {position}")
                    continue
                
                # Load results
                with open(files[0]) as f:
                    results = json.load(f)
                
                # Separate items that need LLM eval from those that don't
                exact_match_items = []
                llm_eval_items = []
                
                desc = f"Evaluating {benchmark} field {field}"
                
                for item in tqdm(results, desc=desc):
                    if is_closed_form:
                        # Try exact/parsed match first
                        item["ground_truth"] = item["ground_truth"].lower()
                        item["generated_answer"] = item["generated_answer"].lower()
                        pred_clean = item["generated_answer"].replace(".", "")
                        
                        if pred_clean in valid_answers:
                            item["eval_method"] = "exact_match"
                            item["score"] = 1.0 if pred_clean == item["ground_truth"] else 0.0
                            exact_match_items.append(item)
                        elif parse_func:
                            parsed_pred = parse_func(item["generated_answer"])
                            if parsed_pred in valid_answers:
                                item["eval_method"] = "parsed_match"
                                item["score"] = 1.0 if parsed_pred == item["ground_truth"] else 0.0
                                exact_match_items.append(item)
                            else:
                                llm_eval_items.append(item)
                        else:
                            llm_eval_items.append(item)
                    else:
                        # Open-ended
                        item["ground_truth"] = item["ground_truth"].lower()
                        item["generated_answer"] = item["generated_answer"].lower()
                        
                        if item["generated_answer"] == item["ground_truth"]:
                            item["eval_method"] = "exact_match"
                            item["score"] = 1.0
                            exact_match_items.append(item)
                        else:
                            llm_eval_items.append(item)
                
                # Batch evaluate LLM items
                if llm_eval_items:
                    print(f"LLM evaluating {len(llm_eval_items)} items...")
                    llm_eval_items = self.llm_evaluator.batch_evaluate(
                        llm_eval_items, batch_size=batch_size
                    )
                
                # Combine results
                eval_results = exact_match_items + llm_eval_items
                
                # Save results
                position_suffix = f"_{position}" if position else ""
                output_path = f"{output_dir}/{benchmark}_pred_eval_{model}_field{field}{position_suffix}.json"
                
                with open(output_path, 'w') as f:
                    json.dump(eval_results, f, indent=4)
                
                # Print summary statistics
                scores = [r["score"] for r in eval_results if r["score"] is not None]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    exact_matches = len([r for r in eval_results if r["eval_method"] in ["exact_match", "parsed_match"]])
                    print(f"{benchmark} field {field}: avg={avg_score:.3f}, exact_matches={exact_matches}/{len(eval_results)}")
        elif scales:
            for scale in scales:
                files = self.find_result_files(
                    benchmark, model, field = None, scale=scale, position = position, path = self.results_dir
                )
                
                if not files:
                    print(f"No files found for {benchmark} {model} field {scale} position {position}")
                    continue
                
                # Load results
                with open(files[0]) as f:
                    results = json.load(f)
                
                # Separate items that need LLM eval from those that don't
                exact_match_items = []
                llm_eval_items = []
                
                desc = f"Evaluating {benchmark} field {scale}"
                
                for item in tqdm(results, desc=desc):
                    if is_closed_form:
                        # Try exact/parsed match first
                        item["ground_truth"] = item["ground_truth"].lower()
                        item["generated_answer"] = item["generated_answer"].lower()
                        pred_clean = item["generated_answer"].replace(".", "")
                        
                        if pred_clean in valid_answers:
                            item["eval_method"] = "exact_match"
                            item["score"] = 1.0 if pred_clean == item["ground_truth"] else 0.0
                            exact_match_items.append(item)
                        elif parse_func:
                            parsed_pred = parse_func(item["generated_answer"])
                            if parsed_pred in valid_answers:
                                item["eval_method"] = "parsed_match"
                                item["score"] = 1.0 if parsed_pred == item["ground_truth"] else 0.0
                                exact_match_items.append(item)
                            else:
                                llm_eval_items.append(item)
                        else:
                            llm_eval_items.append(item)
                    else:
                        # Open-ended
                        item["ground_truth"] = item["ground_truth"].lower()
                        item["generated_answer"] = item["generated_answer"].lower()
                        
                        if item["generated_answer"] == item["ground_truth"]:
                            item["eval_method"] = "exact_match"
                            item["score"] = 1.0
                            exact_match_items.append(item)
                        else:
                            llm_eval_items.append(item)
                
                # Batch evaluate LLM items
                if llm_eval_items:
                    print(f"LLM evaluating {len(llm_eval_items)} items...")
                    llm_eval_items = self.llm_evaluator.batch_evaluate(
                        llm_eval_items, batch_size=batch_size
                    )
                
                # Combine results
                eval_results = exact_match_items + llm_eval_items
                
                # Save results
                position_suffix = f"_{position}" if position else ""
                output_path = f"{output_dir}/{benchmark}_pred_eval_{model}_scale{scale}{position_suffix}.json"
                
                with open(output_path, 'w') as f:
                    json.dump(eval_results, f, indent=4)
                
                # Print summary statistics
                scores = [r["score"] for r in eval_results if r["score"] is not None]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    exact_matches = len([r for r in eval_results if r["eval_method"] in ["exact_match", "parsed_match"]])
                    print(f"{benchmark} scale {scale}: avg={avg_score:.3f}, exact_matches={exact_matches}/{len(eval_results)}")


def main():
    # Configuration - Choose your evaluator
    EVALUATOR_TYPE = EvaluatorType.GPT4  # or EvaluatorType.GPT4
    
    if EVALUATOR_TYPE == EvaluatorType.GPT4:
        evaluator_config = EvaluatorConfig(
            evaluator_type=EvaluatorType.GPT4,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    else:  # QWEN3_LOCAL
        evaluator_config = EvaluatorConfig(
            evaluator_type=EvaluatorType.QWEN3_LOCAL,
            model_name="Qwen/Qwen2.5-14B-Instruct",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    
    # Initialize evaluator
    llm_evaluator = LLMEvaluator(evaluator_config)
    benchmark_evaluator = BenchmarkEvaluator(llm_evaluator)
    
    # Define benchmarks and their configurations
    closed_form_benchmarks = {
        "MMEval": {
            "valid_answers": ["yes", "no"],
            "parse_func": mmeval_parse_answer,
            "fields": [0, 0.5, 1, 2, 3],
            "scales": [1.5, 2, 3, 4],
        },
        "MathVerse": {
            "valid_answers": ["a", "b", "c", "d", "a.", "b.", "c.", "d."],
            "parse_func": mathverse_parse_answer,
            "fields": [0, 0.5, 1, 2, 3],
            "scales": [1.5, 2, 3, 4],
        },
        "MMStar": {
            "valid_answers": ["a", "b", "c", "d", "a.", "b.", "c.", "d."],
            "parse_func": mathverse_parse_answer,
            "fields": [0, 0.5, 1, 2, 3],
            "scales": [1.5, 2, 3, 4],
        }
    }
    
    open_ended_benchmarks = {
        "MMVet": {"fields": [0, 0.5, 1, 2, 3],
            "scales": [1.5, 2, 3, 4],},
        "CharXiv": {"fields": [0, 0.5, 1, 2, 3],
            "scales": [1.5, 2, 3, 4],}
    }
    
    # Models to evaluate
    models = ["pixtral-12b", 
              "llava-1.5-7b-hf", 
              "Llama-3.2-11B-Vision-Instruct", 
              "Qwen2.5-VL-7B-Instruct", 
              "Qwen2-VL-2B-Instruct", 
              "Qwen2-VL-7B-Instruct", 
              "Qwen3-VL-2B-Instruct",
              "Qwen3-VL-4B-Instruct",
              "Qwen3-VL-8B-Instruct",
              "Qwen3.5-VL-2B-Instruct",
              "Qwen3.5-VL-4B-Instruct",
              "Qwen3.5-VL-8B-Instruct"
              ]
    
    # Batch size for local model (set to 1 for API models)
    batch_size = 8 if EVALUATOR_TYPE == EvaluatorType.QWEN3_LOCAL else 1
    
    # Run evaluations
    for model in models:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model}")
        print(f"Using evaluator: {EVALUATOR_TYPE.value}")
        print(f"{'='*60}\n")
        
        # Evaluate open-ended benchmarks
        for benchmark, config in open_ended_benchmarks.items():
            benchmark_evaluator.evaluate_benchmark(
                model=model,
                benchmark=benchmark,
                fields=None,
                scales=config["scales"],
                position="",
                batch_size=batch_size
            )
        
        # Evaluate closed-form benchmarks
        for benchmark, config in closed_form_benchmarks.items():
            benchmark_evaluator.evaluate_benchmark(
                model=model,
                benchmark=benchmark,
                fields = None,
                scales=config["scales"],
                valid_answers=config["valid_answers"],
                parse_func=config["parse_func"],
                position="",
                batch_size=batch_size
            )


if __name__ == "__main__":
    main()