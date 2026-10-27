#!/usr/bin/env python3
"""Batch counterpart to interactive_attention_analysis.ipynb.

The notebook drills into one hand-picked (model, MMVet instance) pair at a time and
displays its visualizations inline. This script instead runs every one of MMVet's
seeded 200-sample subset (the same subset main_field.py/compute_attention.py draw
from - see sample_indices_or_all in src/benchmarks.py) through every configured model,
and writes every one of the notebook's per-instance visualizations to disk instead of
showing them, plus one text file per instance with all the printed numbers (including
the blank white-canvas hallucination-control results).

Each model is loaded once and reused across all MMVet examples (unlike the notebook's
compute_attention.py, which is invoked as a fresh subprocess per instance - fine for
one-off drilldowns, but reloading a 7-12B model 200 times over would dominate runtime
here).

Output layout:
    attention_analysis/visuals/<mmvet_id>/<model_key>_<NN>_<description>.png
    attention_analysis/visuals/<mmvet_id>/summary.txt

Usage:
    python run_mmvet_attention_analysis.py                       # all models, all 200 examples, field=3.0
    python run_mmvet_attention_analysis.py --models qwen llava   # subset of models
    python run_mmvet_attention_analysis.py --limit 5             # smoke test
    python run_mmvet_attention_analysis.py --ids v1_6 v1_28      # specific instances only
"""
import argparse
import re
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

notebook_dir = Path(__file__).resolve().parent   # attention_analysis/
project_root = notebook_dir.parent.parent        # repo root
for p in (notebook_dir, project_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import (
    MODEL_CONFIGS, content_box_in_field_img, content_area_fraction_grid,
    content_box_in_grid_cells, resample_grid, pick_gpu_with_most_free_memory,
)
from src.attention_viz import desink_attention_grid, upsample_grid, ATTENTION_CMAP
from src.benchmarks import MMVetDataset
from src.parse_results import score_example
from src.utils import build_field_images
from scripts.main_field import BENCHMARK_ANSWER_FORMAT_SUFFIXES

BENCHMARK = "MMVet"
DEFAULT_OUTPUT_DIR = notebook_dir / "visuals"
TOP_K_FRACTION = 0.10
SINK_MASS_THRESHOLD = 0.15


def safe_id(target_id) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(target_id)).strip("_")


def top_k_mask(grid, fraction):
    k = max(1, int(round(grid.size * fraction)))
    threshold = np.partition(grid.ravel(), -k)[-k]
    return grid >= threshold


def marker_line(model_key: str) -> str:
    # Written at the top of each model's block in summary.txt so a re-run can skip
    # (model, instance) pairs that already completed, rather than re-running expensive
    # GPU inference on every restart of a 200-example x N-model batch.
    return f"### MODEL_KEY={model_key} ###"


def already_done(txt_path: Path, model_key: str) -> bool:
    return txt_path.exists() and marker_line(model_key) in txt_path.read_text()


def append_text_block(txt_path: Path, text: str):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


# --------------------------------------------------------------------------------
# Figure savers - same visualizations as the notebook's show()/show_attention_overlay()/
# show_mask()/show_content_border_bar() helpers, but saved to disk with a descriptive
# title instead of being displayed inline.
# --------------------------------------------------------------------------------

def _finish(fig, ax, path: Path, title: str):
    ax.set_title(title, fontsize=8, wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_image(image, path, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.axis("off")
    _finish(fig, ax, path, title)


def save_grid(grid, path, title, vmax=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="viridis", vmax=vmax if vmax is not None else grid.max())
    ax.axis("off")
    _finish(fig, ax, path, title)


def save_attention_overlay(model_view, attn_grid, path, title):
    overlay = upsample_grid(attn_grid, model_view.size)
    vmax = float(np.percentile(attn_grid, 99))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(model_view)
    ax.imshow(overlay, cmap=ATTENTION_CMAP, alpha=0.55, vmin=0, vmax=vmax)
    ax.axis("off")
    _finish(fig, ax, path, title)


def save_mask(grid, mask, path, title, vmax=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="viridis", vmax=vmax if vmax is not None else grid.max())
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors="red", linewidths=1.5)
    ax.axis("off")
    _finish(fig, ax, path, title)


def save_content_border_bar(content_mass_frac, border_mass_frac, path, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar([0, 1], [content_mass_frac * 100, border_mass_frac * 100], color=["#0072B2", "#D55E00"], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["content", "border"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of attention mass")
    _finish(fig, ax, path, title)


def run_with_oom_retry(cfg, model, processor, state, image, question, max_new_tokens):
    # Same reactive (not proactive) device-move-on-OOM policy as compute_attention.py:
    # moving a live model across GPUs mid-session has empirically triggered CUDA context
    # crashes on this shared cluster, so only do it when there's no working alternative.
    try:
        return cfg["attention_fn"](model, processor, image, question, state["device"], max_new_tokens=max_new_tokens)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        best = pick_gpu_with_most_free_memory()
        if best != state["device"]:
            model.to(best)
            state["device"] = best
            torch.cuda.empty_cache()
        return cfg["attention_fn"](model, processor, image, question, state["device"], max_new_tokens=max_new_tokens)


def process_example(model_key, cfg, model, processor, state, sample, field_value, max_new_tokens, output_dir, force):
    target_id = sample["id"]
    id_dir = output_dir / safe_id(target_id)
    txt_path = id_dir / "summary.txt"

    if not force and already_done(txt_path, model_key):
        print(f"  [{model_key}] {target_id}: already processed - skipping (use --force to redo)")
        return

    question = sample["question"] + BENCHMARK_ANSWER_FORMAT_SUFFIXES.get(BENCHMARK.lower(), "")
    ground_truth = sample["answer"]
    original_image = sample["image"].convert("RGB")
    fv = field_value

    field_img, _, _, _ = build_field_images(processor, cfg["model_hf_name"], original_image, field=fv)
    white_orig_image = Image.new("RGB", original_image.size, (255, 255, 255))
    white_field_image = Image.new("RGB", field_img.size, (255, 255, 255))

    def run_scored(image, label):
        result = run_with_oom_retry(cfg, model, processor, state, image, question, max_new_tokens)
        score, eval_method = score_example(BENCHMARK, {
            "generated_answer": result["generated_answer"], "ground_truth": ground_truth, "question": question,
        })
        print(f"  [{model_key}] {target_id} [{label}] answer={result['generated_answer']!r} score={score} ({eval_method})")
        return result, score, eval_method

    orig_result, orig_score, orig_eval = run_scored(original_image, "original")
    field_result, field_score, field_eval = run_scored(field_img, f"field={fv:g}")
    white_orig_result, white_orig_score, white_orig_eval = run_scored(white_orig_image, "white (original size)")
    white_field_result, white_field_score, white_field_eval = run_scored(white_field_image, "white (field size)")

    orig_attn_grid = orig_result["attn_grid"]
    field_attn_grid = field_result["attn_grid"]

    content_box = content_box_in_field_img(original_image.size, field_img.size)
    content_frac = content_area_fraction_grid(
        content_box, field_img.size, field_attn_grid.shape, field_result["model_view"].size, cfg["geometry_kind"],
    )
    total_mass = field_attn_grid.sum()
    content_mass_frac = float((field_attn_grid * content_frac).sum() / total_mass)
    border_mass_frac = 1 - content_mass_frac
    content_area_frac = float(content_frac.mean())

    r0, r1, c0, c1 = content_box_in_grid_cells(
        content_box, field_img.size, field_attn_grid.shape, field_result["model_view"].size, cfg["geometry_kind"],
    )
    field_content_aligned = resample_grid(field_attn_grid[r0:r1, c0:c1], orig_attn_grid.shape)

    orig_topk_mask = top_k_mask(orig_attn_grid, TOP_K_FRACTION)
    new_topk_mask = top_k_mask(field_content_aligned, TOP_K_FRACTION)
    patch_survival = float((orig_topk_mask & new_topk_mask).sum() / orig_topk_mask.sum())
    mass_survival = float(orig_attn_grid[new_topk_mask].sum() / orig_attn_grid.sum())

    orig_corrected, orig_sink_mask, orig_sink_frac = desink_attention_grid(orig_attn_grid, SINK_MASS_THRESHOLD)
    field_corrected, field_sink_mask, field_sink_frac = desink_attention_grid(field_attn_grid, SINK_MASS_THRESHOLD)

    # ---- images -----------------------------------------------------------------
    header = f"{cfg['model_hf_name']} ({model_key}) | {BENCHMARK} {target_id}"

    def outcome(score):
        return "correct" if score == 1.0 else "wrong"

    def p(name):
        return id_dir / f"{model_key}_{name}.png"

    save_image(original_image, p("01_original_image"),
               f"{header} | original image ({original_image.size[0]}x{original_image.size[1]})")
    save_image(orig_result["model_view"], p("02_original_model_input"),
               f"{header} | original model input | answer: {orig_result['generated_answer']!r} ({outcome(orig_score)})")
    save_attention_overlay(orig_result["model_view"], orig_attn_grid, p("03_original_attention_overlay"),
               f"{header} | original attention overlay | answer: {orig_result['generated_answer']!r} ({outcome(orig_score)})")

    save_image(field_img, p(f"04_field{fv:g}_image"),
               f"{header} | field={fv:g} bordered image ({field_img.size[0]}x{field_img.size[1]})")
    save_image(field_result["model_view"], p(f"05_field{fv:g}_model_input"),
               f"{header} | field={fv:g} model input | answer: {field_result['generated_answer']!r} ({outcome(field_score)})")
    save_attention_overlay(field_result["model_view"], field_attn_grid, p(f"06_field{fv:g}_attention_overlay"),
               f"{header} | field={fv:g} attention overlay | answer: {field_result['generated_answer']!r} ({outcome(field_score)})")

    save_attention_overlay(white_orig_result["model_view"], white_orig_result["attn_grid"],
               p("07_white_original_attention_overlay"),
               f"{header} | blank white canvas, original size, NO image content | "
               f"answer: {white_orig_result['generated_answer']!r} ({outcome(white_orig_score)})")
    save_attention_overlay(white_field_result["model_view"], white_field_result["attn_grid"],
               p(f"08_white_field{fv:g}_attention_overlay"),
               f"{header} | blank white canvas, field={fv:g} size, NO image content | "
               f"answer: {white_field_result['generated_answer']!r} ({outcome(white_field_score)})")

    save_content_border_bar(content_mass_frac, border_mass_frac, p("09_content_vs_border_mass_bar"),
               f"{header} | field={fv:g} attention mass: content {content_mass_frac * 100:.1f}% vs "
               f"border {border_mass_frac * 100:.1f}% (content area {content_area_frac * 100:.1f}%)")

    save_mask(orig_attn_grid, orig_topk_mask, p("10_topk_mask_original"),
               f"{header} | original top-{TOP_K_FRACTION:.0%} attention mask")
    save_mask(field_content_aligned, new_topk_mask, p(f"11_topk_mask_field{fv:g}_content_aligned"),
               f"{header} | field={fv:g} top-{TOP_K_FRACTION:.0%} mask, content-aligned | "
               f"patch survival {patch_survival * 100:.1f}% | mass survival {mass_survival * 100:.1f}%")

    save_grid(orig_attn_grid, p("12_attn_grid_original_raw"), f"{header} | original raw attention grid")
    save_grid(orig_corrected, p("13_attn_grid_original_desinked"),
               f"{header} | original de-sinked attention grid | sinks: {int(orig_sink_mask.sum())} cell(s), "
               f"{orig_sink_frac * 100:.1f}% of mass")
    save_grid(field_attn_grid, p(f"14_attn_grid_field{fv:g}_raw"), f"{header} | field={fv:g} raw attention grid")
    save_grid(field_corrected, p(f"15_attn_grid_field{fv:g}_desinked"),
               f"{header} | field={fv:g} de-sinked attention grid | sinks: {int(field_sink_mask.sum())} cell(s), "
               f"{field_sink_frac * 100:.1f}% of mass")

    # ---- text -----------------------------------------------------------------
    lines = [
        marker_line(model_key),
        "=" * 88,
        f"{cfg['model_hf_name']}  (model_key={model_key})",
        "=" * 88,
        f"Benchmark: {BENCHMARK} | id: {target_id} | field value: {fv:g}",
        f"Question: {question}",
        f"Ground truth: {ground_truth!r}",
        "",
        f"Original size: {original_image.size[0]}x{original_image.size[1]}  |  "
        f"Field size: {field_img.size[0]}x{field_img.size[1]}",
        f"Model input (original): {orig_result['model_view'].size[0]}x{orig_result['model_view'].size[1]}  |  "
        f"Model input (field): {field_result['model_view'].size[0]}x{field_result['model_view'].size[1]}",
        "",
        f"Original answer: {orig_result['generated_answer']!r} - correct: {orig_score == 1.0} "
        f"(score={orig_score}, eval_method={orig_eval})",
        f"Field answer (field={fv:g}): {field_result['generated_answer']!r} - correct: {field_score == 1.0} "
        f"(score={field_score}, eval_method={field_eval})",
        f"White-canvas answer (original size, no image content): {white_orig_result['generated_answer']!r} - "
        f"correct: {white_orig_score == 1.0} (score={white_orig_score}, eval_method={white_orig_eval})",
        f"White-canvas answer (field size, no image content): {white_field_result['generated_answer']!r} - "
        f"correct: {white_field_score == 1.0} (score={white_field_score}, eval_method={white_field_eval})",
        "",
        f"Content area: {content_area_frac * 100:.1f}%  |  "
        f"Attention mass on content: {content_mass_frac * 100:.1f}%  |  on border: {border_mass_frac * 100:.1f}%",
        f"Top-{TOP_K_FRACTION:.0%} mask: patch survival {patch_survival * 100:.1f}%  |  "
        f"mass survival {mass_survival * 100:.1f}%",
        f"Attention sinks (>{SINK_MASS_THRESHOLD:.0%} of mass in one cell) - "
        f"original: {int(orig_sink_mask.sum())} cell(s), {orig_sink_frac * 100:.1f}% of mass  |  "
        f"field: {int(field_sink_mask.sum())} cell(s), {field_sink_frac * 100:.1f}% of mass",
        "",
    ]
    append_text_block(txt_path, "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS), default=list(MODEL_CONFIGS))
    parser.add_argument("--field", type=float, default=3.0, help="field-experiment size, e.g. 3 for a 300%% border")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N MMVet examples (smoke test)")
    parser.add_argument("--ids", nargs="+", default=None, help="only process these specific MMVet sample ids")
    parser.add_argument("--force", action="store_true", help="reprocess even if summary.txt already has this model's block")
    args = parser.parse_args()

    dataset = MMVetDataset(field=1)  # seeded 200-sample subset, same as compute_attention.py
    samples = list(dataset)
    if args.ids:
        wanted = set(args.ids)
        samples = [s for s in samples if str(s["id"]) in wanted]
    if args.limit:
        samples = samples[:args.limit]
    print(f"{len(samples)} MMVet example(s) queued x {len(args.models)} model(s), field={args.field:g}")
    print(f"Output directory: {args.output_dir}")

    for model_key in args.models:
        cfg = MODEL_CONFIGS[model_key]
        device = pick_gpu_with_most_free_memory()
        print(f"\n{'=' * 88}\nLoading {cfg['model_hf_name']} onto {device}\n{'=' * 88}")
        model, processor = cfg["load_fn"](cfg["model_hf_name"], device)
        state = {"device": device}
        try:
            for i, sample in enumerate(samples):
                print(f"[{model_key}] ({i + 1}/{len(samples)}) id={sample['id']}")
                try:
                    process_example(model_key, cfg, model, processor, state, sample,
                                     args.field, args.max_new_tokens, args.output_dir, args.force)
                except Exception:
                    err = traceback.format_exc()
                    print(err)
                    id_dir = args.output_dir / safe_id(sample["id"])
                    append_text_block(id_dir / "summary.txt", f"{marker_line(model_key)}\nFAILED: {err}\n")
        finally:
            del model
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
