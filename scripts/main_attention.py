#!/usr/bin/env python3
"""
Visualize where LLaVA's attention goes on an OCR task, across the same image
variants the rest of this repo uses to study resolution sensitivity:

  * original    - the raw benchmark image
  * field       - original embedded in a larger white-bordered canvas
                  (src.utils.build_field_images, same transform as main_field.py)
  * resized     - the borderless control at the field variant's effective content
                  resolution
  * bicubic     - `resized` scaled back up to the original resolution (detail
                  loss without a size change)
  * downsized   - the plain original bicubic-downscaled by a fixed factor
                  (src.benchmarks.apply_image_transform, same transform as
                  main_interpolation.py / main_scale.py, just with factor < 1)

For each variant, attention is averaged over every generated token, every
layer and every head, restricted to the image-patch key positions (see
src/attention_viz.py) - this needs attn_implementation="eager" since sdpa
doesn't materialize attention weights.

Usage:
    python scripts/main_attention.py --benchmark docvqa --sample_index 0
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


import argparse
import logging
from pathlib import Path

from src.attention_viz import load_llava, compute_attention_map, plot_attention_grid
from src.benchmarks import (
    DocVQADataset, TextVQADataset, ChartQADataset, InfographicVQADataset, apply_image_transform,
)
from scripts.main_field import BENCHMARK_ANSWER_FORMAT_SUFFIXES
from src.parse_results import score_short_answer
from src.utils import build_field_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# OCR-flavored benchmarks only - each question requires reading rendered text out of the image.
BENCHMARK_CLASSES = {
    "docvqa": DocVQADataset,
    "textvqa": TextVQADataset,
    "chartqa": ChartQADataset,
    "infographicvqa": InfographicVQADataset,
}


def build_variants(processor, model_name, image, field_values, downsize_scales):
    """Returns an ordered dict of {label: PIL.Image} for every variant to visualize."""
    variants = {"original": image}
    for f in field_values:
        field_img, resized_img, bicubic_img, _ = build_field_images(processor, model_name, image, field=f)
        variants[f"field={f:g}"] = field_img
        variants[f"resized={f:g}"] = resized_img
        variants[f"bicubic={f:g}"] = bicubic_img
    for s in downsize_scales:
        variants[f"downsized x{s:g}"] = apply_image_transform(image, interpolation=s)
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--benchmark", type=str, default="docvqa", choices=list(BENCHMARK_CLASSES))
    parser.add_argument("--sample_index", type=int, default=0, help="Index into the benchmark's seeded sample subset")
    parser.add_argument("--num_samples", type=int, default=1, help="How many consecutive samples (from sample_index) to visualize, one figure each")
    parser.add_argument("--field_values", type=float, nargs="*", default=[1.0, 3.0])
    parser.add_argument("--downsize_scales", type=float, nargs="*", default=[0.5, 0.25])
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./results_attention")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_llava(args.model_name, args.device)

    # sample_size fixed at the dataset classes' own default (200): random.sample() with the same
    # seed but a different k is not guaranteed to be a prefix of the k=200 draw, so any sample_size
    # other than the one actually used to pick a target instance can silently point at a different
    # image than intended.
    dataset = BENCHMARK_CLASSES[args.benchmark](field=1)

    for i in range(args.sample_index, args.sample_index + args.num_samples):
        sample = dataset[i]
        image, ground_truth = sample["image"], sample["answer"]
        # Must match main_field.py's actual prompt exactly (raw question + benchmark-specific
        # answer-format suffix) - without it the model answers more verbosely and the specific
        # correct/wrong split an instance was picked for may not reproduce.
        question = sample["question"] + BENCHMARK_ANSWER_FORMAT_SUFFIXES.get(args.benchmark, "")
        logger.info(f"[{args.benchmark} #{i}] question: {question!r} | ground truth: {ground_truth!r}")

        variants = build_variants(processor, args.model_name, image, args.field_values, args.downsize_scales)

        rows = []
        for label, variant_image in variants.items():
            result = compute_attention_map(model, processor, variant_image, question, args.device,
                                            max_new_tokens=args.max_new_tokens)
            correct = score_short_answer(result["generated_answer"], ground_truth) == 1.0
            logger.info(f"  {label:16s} ({variant_image.size[0]}x{variant_image.size[1]}) "
                        f"-> {result['generated_answer']!r} ({'correct' if correct else 'wrong'})")
            rows.append({
                "label": f"{label} ({variant_image.size[0]}x{variant_image.size[1]})",
                "model_view": result["model_view"],
                "attn_grid": result["attn_grid"],
                "generated_answer": result["generated_answer"],
                "correct": correct,
            })

        model_short = args.model_name.split("/")[-1]
        output_file = output_dir / f"{args.benchmark}_{model_short}_sample{i}_attention.png"
        plot_attention_grid(
            rows, output_file,
            suptitle=f"{args.benchmark} #{i}: \"{question}\"  (ground truth: {ground_truth!r})",
        )


if __name__ == "__main__":
    main()
