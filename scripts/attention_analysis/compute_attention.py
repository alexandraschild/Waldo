#!/usr/bin/env python3
"""
Computes everything interactive_attention_analysis.ipynb needs for one instance - the
model's answer and attention map on both the original image and the field-augmented
(bordered) image - and saves it to a single .npz artifact.

Runs as its own process on purpose: attn_implementation="eager" on a 7B model, at large
field values pushing the vision-token count toward Qwen's 16,384 cap, can peak close to
a GPU's full 80GB (see src/attention_viz.py's module docstring). Re-loading/re-running
that inside a long-lived Jupyter kernel risks leaking memory across cell reruns (e.g. if
a cell that loads the model is rerun before the old one is released) - a one-shot process
always fully releases its GPU memory on exit, so the notebook itself never has to hold a
model at all.

Usage:
    python compute_attention.py --model qwen --benchmark DocVQA --id 104 --field 3
"""
import argparse
import json
import os
import re
from pathlib import Path

# Must be set before torch initializes any CUDA context - reduces (but, per the retry
# logic below, does not eliminate) OOMs caused by allocator fragmentation rather than
# genuinely insufficient memory.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

from common import MODEL_CONFIGS, DATASET_CLASSES, pick_gpu_with_most_free_memory
from src.parse_results import score_example
from src.utils import build_field_images
from scripts.main_field import BENCHMARK_ANSWER_FORMAT_SUFFIXES


def artifact_path(output_dir: Path, model_key: str, benchmark: str, target_id, field_value: float) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9]+", "_", str(target_id)).strip("_")
    return output_dir / f"{model_key}_{benchmark}_{safe_id}_field{field_value:g}.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), required=True)
    parser.add_argument("--benchmark", required=True, help=f"one of {sorted(DATASET_CLASSES)}")
    parser.add_argument("--id", required=True, help="the benchmark's own sample id (see the notebook's selection cell)")
    parser.add_argument("--field", type=float, required=True, help="field-experiment size, e.g. 3 for a 300%% border")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    if args.benchmark not in DATASET_CLASSES:
        raise ValueError(f"Unknown benchmark {args.benchmark!r} - available: {sorted(DATASET_CLASSES)}")
    cfg = MODEL_CONFIGS[args.model]
    target_id = args.id

    # Any truthy `field` reproduces the exact same seed=42, sample_size=200 subset
    # main_field.py drew from - see sample_indices_or_all in src/benchmarks.py.
    dataset = DATASET_CLASSES[args.benchmark](field=1)
    # Compare stringified ids rather than guessing a type to coerce `target_id` to: some
    # benchmarks key samples by a real int (e.g. MMStar), others by a string that merely
    # looks numeric (e.g. GQA's question ids) - coercing "20785934" to int 20785934 would
    # silently never match GQA's actual (string) sample ids.
    sample = next((s for s in dataset if str(s["id"]) == target_id), None)
    if sample is None:
        raise ValueError(f"No {args.benchmark} sample with id={target_id!r} in the seeded 200-sample subset.")

    original_image = sample["image"].convert("RGB")
    question = sample["question"] + BENCHMARK_ANSWER_FORMAT_SUFFIXES.get(args.benchmark.lower(), "")
    ground_truth = sample["answer"]

    device = pick_gpu_with_most_free_memory()
    print(f"Loading {cfg['model_hf_name']} onto {device}")
    model, processor = cfg["load_fn"](cfg["model_hf_name"], device)
    state = {"device": device}

    def ensure_best_device():
        # Only called reactively (on OOM below), not proactively before every call: moving a
        # live multi-GB model across GPUs via .to() has empirically triggered sporadic
        # "unspecified launch failure" CUDA errors on this shared cluster (observed with both
        # Pixtral and Qwen, always right after a proactive move to a GPU another job was
        # concurrently hammering) - a crash that a fresh process is needed to recover from
        # (the CUDA context is left unusable), which is strictly worse than just staying put on
        # a device that was never actually a problem. Reserving the move for genuine OOM
        # recovery, where there's no working alternative anyway, is a much better trade.
        best = pick_gpu_with_most_free_memory()
        if best != state["device"]:
            print(f"Moving model {state['device']} -> {best} (GPU load has shifted)")
            model.to(best)
            state["device"] = best
            torch.cuda.empty_cache()

    def run(image, label):
        try:
            result = cfg["attention_fn"](model, processor, image, question, state["device"],
                                          max_new_tokens=args.max_new_tokens)
        except torch.OutOfMemoryError:
            print(f"[{label}] OOM on {state['device']} - clearing cache and retrying on the next-best GPU")
            torch.cuda.empty_cache()
            ensure_best_device()
            result = cfg["attention_fn"](model, processor, image, question, state["device"],
                                          max_new_tokens=args.max_new_tokens)
        score, eval_method = score_example(args.benchmark, {
            "generated_answer": result["generated_answer"], "ground_truth": ground_truth, "question": question,
        })
        print(f"[{label}] answer={result['generated_answer']!r} score={score} ({eval_method})")
        return result, score, eval_method

    field_img, _, _, _ = build_field_images(processor, cfg["model_hf_name"], original_image, field=args.field)

    # Blank controls: a plain white canvas at each condition's own size, asked the exact same
    # question. If the model still lands on a plausible-looking answer with zero actual image
    # content, that's a language-prior/hallucination tell rather than genuine visual grounding.
    white_orig_image = Image.new("RGB", original_image.size, (255, 255, 255))
    white_field_image = Image.new("RGB", field_img.size, (255, 255, 255))

    orig_result, orig_score, orig_eval_method = run(original_image, "original")
    field_result, field_score, field_eval_method = run(field_img, f"field={args.field:g}")
    white_orig_result, white_orig_score, white_orig_eval_method = run(white_orig_image, "white (original size)")
    white_field_result, white_field_score, white_field_eval_method = run(white_field_image, "white (field size)")

    meta = {
        "model_key": args.model, "model_hf_name": cfg["model_hf_name"], "geometry_kind": cfg["geometry_kind"],
        "benchmark": args.benchmark, "id": target_id, "field_value": args.field,
        "question": question, "ground_truth": ground_truth,
        "original_size": list(original_image.size), "field_size": list(field_img.size),
        "orig_generated_answer": orig_result["generated_answer"], "orig_score": orig_score,
        "orig_eval_method": orig_eval_method, "orig_correct": orig_score == 1.0,
        "field_generated_answer": field_result["generated_answer"], "field_score": field_score,
        "field_eval_method": field_eval_method, "field_correct": field_score == 1.0,
        "white_orig_generated_answer": white_orig_result["generated_answer"], "white_orig_score": white_orig_score,
        "white_orig_eval_method": white_orig_eval_method, "white_orig_correct": white_orig_score == 1.0,
        "white_field_generated_answer": white_field_result["generated_answer"], "white_field_score": white_field_score,
        "white_field_eval_method": white_field_eval_method, "white_field_correct": white_field_score == 1.0,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_path(args.output_dir, args.model, args.benchmark, target_id, args.field)
    np.savez(
        out_path,
        meta=np.array(json.dumps(meta)),
        original_image=np.array(original_image),
        field_image=np.array(field_img),
        orig_model_view=np.array(orig_result["model_view"]),
        field_model_view=np.array(field_result["model_view"]),
        orig_attn_grid=orig_result["attn_grid"],
        field_attn_grid=field_result["attn_grid"],
        white_orig_model_view=np.array(white_orig_result["model_view"]),
        white_field_model_view=np.array(white_field_result["model_view"]),
        white_orig_attn_grid=white_orig_result["attn_grid"],
        white_field_attn_grid=white_field_result["attn_grid"],
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
