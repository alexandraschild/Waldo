"""Shared config/helpers for interactive_attention_analysis.ipynb and its companion
compute_attention.py script, so neither duplicates the other's MODEL_CONFIGS/
DATASET_CLASSES/geometry math.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from src.attention_viz import (
    load_llava, load_qwen, load_pixtral, load_mllama,
    compute_attention_map, compute_qwen_attention_map, compute_pixtral_attention_map, compute_mllama_attention_map,
)
from src.benchmarks import (
    MMVetDataset, MMEvalDataset, MathVerseDataset, MMStar, CharXiv,
    VQAv2Dataset, GQADataset, CVBenchDataset, TextVQADataset, ChartQADataset,
    DocVQADataset, InfographicVQADataset, VStarBenchDataset, HRBenchDataset,
    MMERealWorldLiteDataset,
)

# How to map a field-image pixel box into an attn_grid's own patch-cell space (see
# _map_content_box_to_model_view): "crop" - LLaVA's fixed 336x336 resize + center-crop;
# "resize" - Qwen/Pixtral's plain (no-crop) native-resolution resize; "tiled" - Mllama's
# fit-to-tile-canvas resize, anchored top-left with padding tiles rather than a centered crop.
MODEL_CONFIGS = {
    "llava": {
        "model_dir": "llava-hf_llava-1.5-7b-hf",
        "model_short": "llava-1.5-7b-hf",
        "model_hf_name": "llava-hf/llava-1.5-7b-hf",
        "load_fn": load_llava,
        "attention_fn": compute_attention_map,
        "geometry_kind": "crop",
    },
    "qwen": {
        "model_dir": "Qwen_Qwen2-VL-7B-Instruct",
        "model_short": "Qwen2-VL-7B-Instruct",
        "model_hf_name": "Qwen/Qwen2-VL-7B-Instruct",
        "load_fn": load_qwen,
        "attention_fn": compute_qwen_attention_map,
        "geometry_kind": "resize",
    },
    "pixtral": {
        "model_dir": "mistral-community_pixtral-12b",
        "model_short": "pixtral-12b",
        "model_hf_name": "mistral-community/pixtral-12b",
        "load_fn": load_pixtral,
        "attention_fn": compute_pixtral_attention_map,
        "geometry_kind": "resize",
    },
    "llama": {
        "model_dir": "meta-llama_Llama-3.2-11B-Vision-Instruct",
        "model_short": "Llama-3.2-11B-Vision-Instruct",
        "model_hf_name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "load_fn": load_mllama,
        "attention_fn": compute_mllama_attention_map,
        "geometry_kind": "tiled",
    },
}

DATASET_CLASSES = {
    "MMVet": MMVetDataset, "MMEval": MMEvalDataset, "MathVerse": MathVerseDataset,
    "MMStar": MMStar, "CharXiv": CharXiv, "VQAv2": VQAv2Dataset, "GQA": GQADataset,
    "CVBench": CVBenchDataset, "TextVQA": TextVQADataset, "ChartQA": ChartQADataset,
    "DocVQA": DocVQADataset, "InfographicVQA": InfographicVQADataset,
    "VStarBench": VStarBenchDataset, "HRBench": HRBenchDataset,
    "MMERealWorldLite": MMERealWorldLiteDataset,
}


def pick_gpu_with_most_free_memory():
    # This machine's GPUs are shared with other, independently-running jobs, so free
    # memory varies at runtime - pick whichever device currently has the most headroom.
    if not torch.cuda.is_available():
        return "cpu"
    free_by_device = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
    return f"cuda:{max(free_by_device, key=free_by_device.get)}"


def content_box_in_field_img(original_size, field_img_size):
    """(left, top, right, bottom) of the un-bordered content, in field-image pixel space -
    matches add_white_field's position='center' placement (src/utils.py)."""
    ow, oh = original_size
    fw, fh = field_img_size
    left, top = (fw - ow) // 2, (fh - oh) // 2
    return left, top, left + ow, top + oh


# Mllama's own fixed tile pixel size / patch size (config.vision_config.image_size /
# .patch_size) - hardcoded here the same way the "crop" branch below hardcodes LLaVA's 336.
_MLLAMA_TILE_SIZE = 560
_MLLAMA_PATCH_SIZE = 14


def _map_content_box_to_model_view(content_box_field_px, field_img_size, attn_grid_shape, model_view_size, geometry_kind):
    """(box_l, box_t, box_r, box_b, cell_w, cell_h): the content box mapped from field-image
    pixel space into the attn_grid's own patch-cell pixel space, plus that space's per-cell
    size. `geometry_kind` (MODEL_CONFIGS[...]["geometry_kind"]):
      - "crop": LLaVA's fixed 336x336 resize + center-crop.
      - "resize": Qwen/Pixtral's plain (no-crop) native-resolution resize.
      - "tiled": Mllama's fit-to-tile-canvas resize, anchored top-left with padding tiles
        (rather than field_img's own centered white border) filling any leftover canvas.
    Same mapping math as src.attention_viz.patch_grid_info, just forward (content box ->
    model-view pixels) instead of backward (patch boundaries -> original pixels). Shared by
    content_area_fraction_grid (fractional per-cell overlap) and content_box_in_grid_cells
    (integer cell bounds for cropping)."""
    gh, gw = attn_grid_shape
    fw, fh = field_img_size
    cl, ct, cr, cb = content_box_field_px

    if geometry_kind == "crop":
        model_input_size = 336
        scale = model_input_size / min(fw, fh)
        crop_left = (fw * scale - model_input_size) / 2
        crop_top = (fh * scale - model_input_size) / 2
        box_l, box_t = cl * scale - crop_left, ct * scale - crop_top
        box_r, box_b = cr * scale - crop_left, cb * scale - crop_top
        cell_w = cell_h = model_input_size / gw
    elif geometry_kind == "tiled":
        from transformers.models.mllama.image_processing_mllama import get_image_size_fit_to_canvas

        mv_w, mv_h = model_view_size
        resized_h, resized_w = get_image_size_fit_to_canvas(fh, fw, mv_h, mv_w, _MLLAMA_TILE_SIZE)
        scale_x, scale_y = resized_w / fw, resized_h / fh
        box_l, box_t, box_r, box_b = cl * scale_x, ct * scale_y, cr * scale_x, cb * scale_y
        cell_w = cell_h = _MLLAMA_PATCH_SIZE
    else:
        mv_w, mv_h = model_view_size
        scale_x, scale_y = mv_w / fw, mv_h / fh
        box_l, box_t, box_r, box_b = cl * scale_x, ct * scale_y, cr * scale_x, cb * scale_y
        cell_w, cell_h = mv_w / gw, mv_h / gh

    return box_l, box_t, box_r, box_b, cell_w, cell_h


def content_area_fraction_grid(content_box_field_px, field_img_size, attn_grid_shape, model_view_size, geometry_kind):
    """Fraction of each attn_grid cell's area that falls inside `content_box_field_px`, mapped
    into the attn_grid's own patch-cell space."""
    gh, gw = attn_grid_shape
    box_l, box_t, box_r, box_b, cell_w, cell_h = _map_content_box_to_model_view(
        content_box_field_px, field_img_size, attn_grid_shape, model_view_size, geometry_kind,
    )

    frac = np.zeros((gh, gw), dtype=np.float64)
    for i in range(gh):
        overlap_y = max(0.0, min((i + 1) * cell_h, box_b) - max(i * cell_h, box_t))
        for j in range(gw):
            overlap_x = max(0.0, min((j + 1) * cell_w, box_r) - max(j * cell_w, box_l))
            frac[i, j] = (overlap_x * overlap_y) / (cell_w * cell_h)
    return frac


def content_box_in_grid_cells(content_box_field_px, field_img_size, attn_grid_shape, model_view_size, geometry_kind):
    """(row_start, row_end, col_start, col_end): integer attn_grid cell bounds covering the
    content region, e.g. for cropping a field-image attn_grid down to just its content-covering
    cells before comparing against an attn_grid computed on the plain original image."""
    gh, gw = attn_grid_shape
    box_l, box_t, box_r, box_b, cell_w, cell_h = _map_content_box_to_model_view(
        content_box_field_px, field_img_size, attn_grid_shape, model_view_size, geometry_kind,
    )
    row_start = max(0, int(np.floor(box_t / cell_h)))
    row_end = min(gh, int(np.ceil(box_b / cell_h)))
    col_start = max(0, int(np.floor(box_l / cell_w)))
    col_end = min(gw, int(np.ceil(box_r / cell_w)))
    return row_start, row_end, col_start, col_end


def resample_grid(grid, target_shape):
    """Resample a 2D grid to `target_shape` = (rows, cols) via bicubic interpolation (up- or
    down-sampling as needed) - lets two attention grids computed at different patch-grid
    resolutions (e.g. an original-image grid and a content-cropped field-image grid) be
    compared cell-for-cell. Normalizes to uint8 for PIL's resize and rescales back
    afterwards, same approach as src.attention_viz.upsample_grid."""
    target_rows, target_cols = target_shape
    lo, hi = grid.min(), grid.max()
    normalized = (grid - lo) / (hi - lo) if hi > lo else np.zeros_like(grid)
    img = Image.fromarray((normalized * 255).astype(np.uint8)).resize((target_cols, target_rows), resample=Image.BICUBIC)
    return np.asarray(img, dtype=np.float64) / 255.0 * (hi - lo) + lo
