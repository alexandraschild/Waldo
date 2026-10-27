#!/usr/bin/env python3
"""
Measure CKA (Centered Kernel Alignment) divergence between LLaVA's internal
activations for three variants of the same image:

  1. field       - the image with a white "field" of the given percent added around it
                    (ImageProcessor.add_white_field, same transform used by main_field.py)
  2. interpolation - the image upscaled by the given factor via bicubic resizing
                    (same transform used by main_interpolation.py)
  3. content     - the ORIGINAL (unpadded) image downsampled directly to the resolution
                    its content would occupy inside the model's fixed input frame once
                    `field` has been applied and the model's own preprocessing has shrunk
                    the padded canvas back down. This isolates pure resolution loss from
                    the confound of the distracting white border: same effective content
                    resolution as condition 1, but with no field pixels at all.

For each condition, every sample is run through the vision tower and the language
model with output_hidden_states=True. Per-layer activations are mean-pooled over
the token dimension (patch tokens for the vision tower, attention-mask-weighted
mean over the sequence for the language model) to get one vector per sample per
layer, and linear CKA is computed layer-by-layer between each pair of conditions.
This shows, at each depth in the network, how much the model's representation of
the "real" content vs. the distracting field vs. plain upscaling has diverged.

The CLI below (and `load_model_and_processor`/`collect_activations`) only
supports LLaVA (llava-hf/llava-1.5-7b-hf): it reaches into the model's
internal module structure (model.model.vision_tower / .language_model) which
is architecture-specific. `load_qwen_model_and_processor`/
`collect_qwen_activations` further down provide the Qwen2-VL-family
equivalent (model.model.visual / .language_model, patch-packed rather than
batched vision inputs) for notebooks that need it.
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
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, LlavaForConditionalGeneration

from src.benchmarks import ChartQADataset, apply_image_transform
from src.cka import linear_cka

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS = ["field", "interpolation", "content"]
PAIRS = [("field", "interpolation"), ("field", "content"), ("interpolation", "content")]


def load_raw_samples(num_samples: int, seed: int) -> List[Dict]:
    """Sample `num_samples` ChartQA examples (same seed/sampler as the rest of the
    repo's field/interpolation runs), returning the untransformed image + question
    so all three conditions can be built from the same source image."""
    dataset = ChartQADataset(field=1, sample_size=num_samples, seed=seed)
    return [{"image": s["image"], "question": s["question"]} for s in dataset.samples]


def build_content_image(image: Image.Image, field_percent: float, full_short_edge: int,
                         resample: int) -> Image.Image:
    """Resize the original (unpadded) image so its shortest edge equals the size the
    content would end up at inside the model's fixed input frame under `field_percent`,
    e.g. field=1.0 (100%) halves the content's share of each dimension, so a model with
    a 336px input ends up with ~168px worth of real content resolution."""
    image = image.convert("RGB")
    width, height = image.size
    content_scale = 1.0 / (1.0 + field_percent)
    target_short_edge = max(1, round(full_short_edge * content_scale))

    if width <= height:
        new_width = target_short_edge
        new_height = max(1, round(height * target_short_edge / width))
    else:
        new_height = target_short_edge
        new_width = max(1, round(width * target_short_edge / height))

    return image.resize((new_width, new_height), resample=resample)


def build_conditions(raw_samples: List[Dict], field_percent: float, interpolation_scale: float,
                      full_short_edge: int, resample: int) -> Dict[str, List[Image.Image]]:
    images = {name: [] for name in CONDITIONS}
    for sample in raw_samples:
        image = sample["image"]
        images["field"].append(apply_image_transform(image, field=field_percent))
        images["interpolation"].append(apply_image_transform(image, interpolation=interpolation_scale))
        images["content"].append(build_content_image(image, field_percent, full_short_edge, resample))
    return images


def load_model_and_processor(model_name: str, model_path: str, device: str):
    load_path = model_path or model_name
    logger.info(f"Loading {load_path} onto {device}")
    model = LlavaForConditionalGeneration.from_pretrained(
        load_path,
        dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def vision_input_short_edge(processor) -> Tuple[int, int]:
    """(short_edge, resample) the model's own image processor resizes/crops to."""
    image_processor = processor.image_processor
    crop_size = getattr(image_processor, "crop_size", None)
    if crop_size:
        short_edge = min(crop_size["height"], crop_size["width"])
    else:
        short_edge = image_processor.size["shortest_edge"]
    resample = getattr(image_processor, "resample", Image.BICUBIC)
    return short_edge, resample


@torch.inference_mode()
def extract_pooled_batch(model, processor, images: List[Image.Image], questions: List[str],
                          device: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Run one batch through the vision tower and the full model, mean-pooling each
    layer's hidden states down to one vector per sample. Returns (vision_layers,
    lm_layers), each a list indexed by depth of (batch, hidden_dim) arrays."""
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for q in questions
    ]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(
        device, dtype=model.dtype
    )

    vision_out = model.model.vision_tower(inputs["pixel_values"], output_hidden_states=True, return_dict=True)
    vision_layers = [hs[:, 1:, :].float().mean(dim=1).cpu().numpy() for hs in vision_out.hidden_states]

    lm_out = model(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )
    attn = inputs["attention_mask"].unsqueeze(-1).float()
    counts = attn.sum(dim=1).clamp(min=1)
    lm_layers = [((hs.float() * attn).sum(dim=1) / counts).cpu().numpy() for hs in lm_out.hidden_states]

    return vision_layers, lm_layers


def collect_activations(model, processor, images: List[Image.Image], questions: List[str],
                         device: str, batch_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    vision_batches: List[List[np.ndarray]] = None
    lm_batches: List[List[np.ndarray]] = None

    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        batch_questions = questions[start:start + batch_size]
        vision_layers, lm_layers = extract_pooled_batch(model, processor, batch_images, batch_questions, device)

        if vision_batches is None:
            vision_batches = [[] for _ in vision_layers]
            lm_batches = [[] for _ in lm_layers]
        for layer_idx, layer_acts in enumerate(vision_layers):
            vision_batches[layer_idx].append(layer_acts)
        for layer_idx, layer_acts in enumerate(lm_layers):
            lm_batches[layer_idx].append(layer_acts)

        logger.info(f"Processed {min(start + batch_size, len(images))}/{len(images)} samples")

    vision_acts = [np.concatenate(layer_batches, axis=0) for layer_batches in vision_batches]
    lm_acts = [np.concatenate(layer_batches, axis=0) for layer_batches in lm_batches]
    return vision_acts, lm_acts


def load_qwen_model_and_processor(model_name: str, model_path: str, device: str):
    """Qwen2-VL-family counterpart to `load_model_and_processor`. `AutoModelForImageTextToText`
    rather than a Qwen2VL-specific class so this also covers Qwen2.5-VL/Qwen3-VL checkpoints,
    which share the `.model.visual`/`.model.language_model` module layout and `image_grid_thw`
    input this file's Qwen extraction functions rely on."""
    load_path = model_path or model_name
    logger.info(f"Loading {load_path} onto {device}")
    model = AutoModelForImageTextToText.from_pretrained(
        load_path, dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(load_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


@torch.inference_mode()
def extract_qwen_pooled_batch(model, processor, images: List[Image.Image], questions: List[str],
                               device: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Qwen2-VL counterpart to `extract_pooled_batch`. Unlike LLaVA's fixed-size
    `pixel_values` (batch, C, H, W), Qwen's image processor patchifies every image
    in the batch into one flat (total_patches_across_images, patch_dim) tensor with
    no per-image batch dimension - `image_grid_thw` gives each image's (t, h, w) patch
    counts, from which per-image patch-count boundaries are derived to split the
    vision tower's hidden states back apart before mean-pooling each image's own
    patches into one vector per layer. The language-model side is unaffected by this
    (input_ids/hidden_states are still batched per-sample as usual), so it's pooled
    exactly like LLaVA's."""
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for q in questions
    ]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    grid_thw = inputs["image_grid_thw"]
    vision_out = model.model.visual(inputs["pixel_values"], grid_thw=grid_thw, output_hidden_states=True, return_dict=True)
    patch_counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    vision_layers = []
    for hs in vision_out.hidden_states:
        per_image = torch.split(hs, patch_counts, dim=0)
        pooled = torch.stack([chunk.float().mean(dim=0) for chunk in per_image], dim=0)
        vision_layers.append(pooled.cpu().numpy())

    lm_out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        image_grid_thw=grid_thw,
        output_hidden_states=True,
        return_dict=True,
    )
    attn = inputs["attention_mask"].unsqueeze(-1).float()
    counts = attn.sum(dim=1).clamp(min=1)
    lm_layers = [((hs.float() * attn).sum(dim=1) / counts).cpu().numpy() for hs in lm_out.hidden_states]

    return vision_layers, lm_layers


def collect_qwen_activations(model, processor, images: List[Image.Image], questions: List[str],
                              device: str, batch_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Unlike LLaVA's fixed ~576 vision tokens, Qwen's native-resolution processing can push a
    single image to its full 16,384-token budget - `output_hidden_states=True` then materializes
    every one of the ~29 language-model layers at that full sequence length simultaneously, which
    at a LLaVA-sized batch_size can exceed GPU memory even for a 7B model. Freeing the model
    outputs and clearing the CUDA caching allocator after every batch (rather than only at the
    end, as `collect_activations` does for LLaVA) keeps peak usage bounded to one batch's worth
    instead of accumulating reserved-but-unallocated fragments across an entire benchmark's
    worth of batches."""
    vision_batches: List[List[np.ndarray]] = None
    lm_batches: List[List[np.ndarray]] = None

    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        batch_questions = questions[start:start + batch_size]
        vision_layers, lm_layers = extract_qwen_pooled_batch(model, processor, batch_images, batch_questions, device)

        if vision_batches is None:
            vision_batches = [[] for _ in vision_layers]
            lm_batches = [[] for _ in lm_layers]
        for layer_idx, layer_acts in enumerate(vision_layers):
            vision_batches[layer_idx].append(layer_acts)
        for layer_idx, layer_acts in enumerate(lm_layers):
            lm_batches[layer_idx].append(layer_acts)

        del vision_layers, lm_layers
        torch.cuda.empty_cache()

        logger.info(f"Processed {min(start + batch_size, len(images))}/{len(images)} samples")

    vision_acts = [np.concatenate(layer_batches, axis=0) for layer_batches in vision_batches]
    lm_acts = [np.concatenate(layer_batches, axis=0) for layer_batches in lm_batches]
    return vision_acts, lm_acts


def compute_cka_table(activations: Dict[str, List[np.ndarray]], component: str) -> List[Dict]:
    num_layers = len(next(iter(activations.values())))
    rows = []
    for layer_idx in range(num_layers):
        for cond_a, cond_b in PAIRS:
            cka = linear_cka(activations[cond_a][layer_idx], activations[cond_b][layer_idx])
            rows.append({
                "component": component,
                "layer": layer_idx,
                "pair": f"{cond_a}_vs_{cond_b}",
                "cka": cka,
            })
    return rows


def plot_cka(rows: List[Dict], output_file: Path):
    import matplotlib.pyplot as plt

    colors = {
        "field_vs_interpolation": "#2a78d6",
        "field_vs_content": "#eb6834",
        "interpolation_vs_content": "#1baf7a",
    }
    components = sorted(set(r["component"] for r in rows))

    fig, axes = plt.subplots(1, len(components), figsize=(6 * len(components), 4.5), squeeze=False)
    axes = axes[0]

    for ax, component in zip(axes, components):
        comp_rows = [r for r in rows if r["component"] == component]
        for pair, color in colors.items():
            pair_rows = sorted((r for r in comp_rows if r["pair"] == pair), key=lambda r: r["layer"])
            if not pair_rows:
                continue
            xs = [r["layer"] for r in pair_rows]
            ys = [r["cka"] for r in pair_rows]
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=color,
                    label=pair.replace("_vs_", " vs "))

        ax.set_xlabel("Layer depth")
        ax.set_ylabel("Linear CKA")
        ax.set_ylim(0, 1.02)
        ax.set_title(component.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    logger.info(f"Saved plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="CKA divergence between LLaVA activations for field/interpolation/content-only image variants")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--model_path", type=str, default=None, help="Local path to load model weights from")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of ChartQA samples to analyze")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--field_percent", type=float, default=1.0, help="Field percent for condition 1 (and to derive condition 3's content resolution), e.g. 1.0 = 100%")
    parser.add_argument("--interpolation_scale", type=float, default=2.0, help="Upscale factor for condition 2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", type=str, default="./results_cka")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(args.model_name, args.model_path, args.device)
    full_short_edge, resample = vision_input_short_edge(processor)
    logger.info(f"Model input short edge: {full_short_edge}px (resample={resample})")

    raw_samples = load_raw_samples(args.num_samples, args.seed)
    questions = [s["question"] for s in raw_samples]
    logger.info(f"Loaded {len(raw_samples)} ChartQA samples")

    condition_images = build_conditions(raw_samples, args.field_percent, args.interpolation_scale,
                                         full_short_edge, resample)

    vision_activations: Dict[str, List[np.ndarray]] = {}
    lm_activations: Dict[str, List[np.ndarray]] = {}
    for condition in CONDITIONS:
        logger.info(f"Extracting activations for condition: {condition}")
        vision_acts, lm_acts = collect_activations(
            model, processor, condition_images[condition], questions, args.device, args.batch_size
        )
        vision_activations[condition] = vision_acts
        lm_activations[condition] = lm_acts

    rows = compute_cka_table(vision_activations, "vision_tower") + compute_cka_table(lm_activations, "language_model")

    results_file = output_path / f"cka_{args.model_name.replace('/', '_')}_field{args.field_percent}_interp{args.interpolation_scale}.json"
    with open(results_file, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "num_samples": len(raw_samples),
            "field_percent": args.field_percent,
            "interpolation_scale": args.interpolation_scale,
            "results": rows,
        }, f, indent=2)
    logger.info(f"Saved CKA results to {results_file}")

    plot_file = output_path / f"cka_{args.model_name.replace('/', '_')}_field{args.field_percent}_interp{args.interpolation_scale}.png"
    plot_cka(rows, plot_file)


if __name__ == "__main__":
    main()
