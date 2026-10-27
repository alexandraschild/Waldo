"""Extract and visualize image-token attention for a single sample.

LLaVA and Qwen2-VL are both supported: each inlines one placeholder token per
vision patch directly into `input_ids` (see `_vision_token_counts` in
`src/models.py`), so a plain decoder self-attention row already contains "how
much did this generated token attend to each image patch" with no further
architecture-specific plumbing beyond locating those placeholder positions and
(for Qwen) reshaping them into a non-square patch grid via `image_grid_thw`
instead of LLaVA's fixed 24x24 CLIP-ViT grid. Extracting this requires
`attn_implementation="eager"` - sdpa/flash-attention backends don't
materialize attention weights at all.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, LlavaForConditionalGeneration, MllamaForConditionalGeneration,
)

logger = logging.getLogger(__name__)

# Sequential blue ramp (references/palette.md), light->dark, for attention-weight overlays.
ATTENTION_CMAP = LinearSegmentedColormap.from_list("attention_blue", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
])


def load_llava(model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda:0"):
    """Load LLaVA with eager attention - the only backend that returns attention weights."""
    logger.info(f"Loading {model_name} onto {device} (attn_implementation=eager)")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float16, device_map={"": device},
        trust_remote_code=True, attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def _model_view_image(pixel_values: torch.Tensor, image_processor) -> Image.Image:
    """Reconstruct the exact 336x336 RGB image the vision tower saw, by inverting
    the processor's own normalization on `pixel_values` - guarantees pixel-for-pixel
    alignment with the attention grid, independent of the processor's internal
    resize/crop implementation."""
    mean = torch.tensor(image_processor.image_mean, device=pixel_values.device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, device=pixel_values.device).view(1, 3, 1, 1)
    img = (pixel_values.float() * std + mean).clamp(0, 1)[0]
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@torch.inference_mode()
def compute_attention_map(model, processor, image: Image.Image, question: str, device: str,
                           max_new_tokens: int = 24, layer_fraction: float = 0.5) -> Dict:
    """Generate an answer and return the average attention (over every generated
    token, layer and head) each generated token paid to each image patch token.

    Returns a dict with:
      * generated_answer: the decoded answer text
      * attn_grid: (H, W) raw attention weights (rows summing to ~1 each,
        averaged over generation steps/layers/heads - NOT renormalized to
        [0, 1], so magnitude is comparable across different images/variants)
      * model_view: the exact 336x336 RGB image the vision tower encoded
        (pixel-aligned with attn_grid)

    `layer_fraction` restricts averaging to the last fraction of decoder layers
    (default: the back half). Early transformer layers mix generic low-level
    features with near-uniform attention; the content-driven, task-specific
    localization this visualization is after only emerges in later layers -
    averaging in the early ones just dilutes it with noise.
    """
    image = image.convert("RGB") if image.mode != "RGB" else image
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
        add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype=model.dtype)
    prompt_len = inputs["input_ids"].shape[1]

    image_token_id = model.config.image_token_index
    vision_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    num_vision_tokens = vision_positions.shape[0]
    side = int(round(num_vision_tokens ** 0.5))
    if side * side != num_vision_tokens:
        raise ValueError(f"Vision token count {num_vision_tokens} isn't a perfect square - "
                          f"can't reshape into a patch grid.")

    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        output_attentions=True, return_dict_in_generate=True,
    )

    num_layers = len(outputs.attentions[0])
    start_layer = int(round(num_layers * (1 - layer_fraction)))

    attn_sum = torch.zeros(num_vision_tokens, dtype=torch.float32, device=device)
    num_terms = 0
    for step, step_attentions in enumerate(outputs.attentions):
        query_idx = -1 if step == 0 else 0  # step 0: full prefill row; later steps: single cached query
        for layer_attn in step_attentions[start_layer:]:
            row = layer_attn[0, :, query_idx, vision_positions]  # (heads, num_vision_tokens)
            attn_sum += row.mean(dim=0).float()
            num_terms += 1
    attn_grid = (attn_sum / num_terms).reshape(side, side).cpu().numpy()

    generated_answer = processor.tokenizer.decode(
        outputs.sequences[0, prompt_len:], skip_special_tokens=True
    ).strip()

    model_view = _model_view_image(inputs["pixel_values"], processor.image_processor)

    return {
        "generated_answer": generated_answer,
        "attn_grid": attn_grid,
        "model_view": model_view,
    }


def load_qwen(model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda:0"):
    """Load a Qwen2-VL-family model with eager attention on the *language model* only -
    the only backend that returns attention weights, needed there to read off what the
    decoder attends to over vision-key positions. The vision tower stays on sdpa: unlike
    the language model's causal decoding (where only the latest query row matters - see
    `compute_qwen_attention_map`), the vision transformer's self-attention is full
    bidirectional attention over every patch in the image, with no equivalent single-row
    shortcut, so eager there would materialize the full patch x patch matrix (before the
    spatial merge shrinks the token count - as much as ~65k x 65k patches for an image
    near the 16,384-vision-token budget) and reliably OOM regardless of image content.
    `AutoModelForImageTextToText` (rather than importing e.g. `Qwen2VLForConditionalGeneration`
    directly) so this also covers Qwen2.5-VL/Qwen3-VL checkpoints, which share the same
    `.model.visual`/`.config.image_token_id`/`image_grid_thw` conventions this module relies on."""
    logger.info(f"Loading {model_name} onto {device} (attn_implementation=eager for the language model, sdpa for vision)")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map={"": device},
        attn_implementation={"text_config": "eager", "vision_config": "sdpa", "": "sdpa"},
    )
    processor = AutoProcessor.from_pretrained(model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def _qwen_model_view_image(pixel_values: torch.Tensor, grid_thw: torch.Tensor, image_processor) -> Image.Image:
    """Reconstruct the exact resized RGB image Qwen's vision tower encoded for a
    single image, by inverting `Qwen2VLImageProcessor._preprocess`'s patch
    flatten/normalize (see transformers `image_processing_qwen2_vl.py`): patches
    are grouped into `merge_size x merge_size` blocks (matching the later spatial
    merge) rather than simple row-major patches, and the single real frame is
    duplicated `temporal_patch_size` times - both undone here before reassembling
    pixel rows/columns and inverting the mean/std normalization. Unlike LLaVA's
    single fixed-size `pixel_values` tensor, Qwen's is pre-patchified with no
    per-image spatial structure left, so this (rather than a plain reshape) is
    required to get a pixel-aligned image back at all."""
    patch_size = image_processor.patch_size
    temporal_patch_size = image_processor.temporal_patch_size
    merge_size = image_processor.merge_size
    t, h, w = (int(x) for x in grid_thw)
    channel = 3

    patches = pixel_values.reshape(
        t, h // merge_size, w // merge_size, merge_size, merge_size,
        channel, temporal_patch_size, patch_size, patch_size,
    )
    # Inverse of the processor's transpose(0, 3, 6, 4, 7, 2, 1, 5, 8): reorders
    # (t, h//m, w//m, m_h, m_w, c, tp, ph, pw) back to (t, tp, c, h//m, m_h, ph, w//m, m_w, pw).
    patches = patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
    resized_height, resized_width = h * patch_size, w * patch_size
    frames = patches.reshape(t * temporal_patch_size, channel, resized_height, resized_width)
    frame = frames[0]  # a still image's temporal-duplicated frames are identical

    mean = torch.tensor(image_processor.image_mean, device=frame.device).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std, device=frame.device).view(3, 1, 1)
    img = (frame.float() * std + mean).clamp(0, 1)
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@torch.inference_mode()
def compute_qwen_attention_map(model, processor, image: Image.Image, question: str, device: str,
                                max_new_tokens: int = 24, layer_fraction: float = 0.5) -> Dict:
    """Qwen2-VL counterpart to `compute_attention_map`. Same averaging scheme
    (every generated token, the back `layer_fraction` of decoder layers, every
    head, restricted to image-patch key positions) but the vision-token grid
    comes from `image_grid_thw` (post spatial-merge: `grid_h // merge_size` x
    `grid_w // merge_size`) rather than LLaVA's fixed sqrt(num_patches) square,
    since Qwen's native-resolution processing produces a different aspect ratio
    of patches per image rather than always a fixed square crop."""
    image = image.convert("RGB") if image.mode != "RGB" else image
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
        add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    prompt_len = inputs["input_ids"].shape[1]

    image_token_id = model.config.image_token_id
    vision_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    num_vision_tokens = vision_positions.shape[0]

    grid_thw = inputs["image_grid_thw"][0]
    merge_size = processor.image_processor.merge_size
    grid_t, grid_h, grid_w = int(grid_thw[0]), int(grid_thw[1]) // merge_size, int(grid_thw[2]) // merge_size
    if grid_t * grid_h * grid_w != num_vision_tokens:
        raise ValueError(f"Vision token count {num_vision_tokens} doesn't match "
                          f"grid_thw-derived {grid_t}x{grid_h}x{grid_w} - can't reshape into a patch grid.")

    # Deliberately NOT requesting generate(output_attentions=True) here: eager attention always
    # computes the full (heads, q_len, k_len) matrix per layer regardless of that flag (that's
    # what distinguishes "eager" from sdpa/flash, which never materialize weights at all) - the
    # flag only controls whether transformers *retains* every layer's matrix in a persistent
    # output tuple. For LLaVA's fixed ~576-token images that retention is cheap, but Qwen's
    # native-resolution processing can push a single image up to its 16,384-token budget, where
    # one layer's prefill attention matrix alone is ~14GiB - retaining all ~28 layers at once (as
    # output_attentions=True would) is hundreds of GiB and reliably OOMs. Forward hooks on each
    # decoder layer's self-attention module harvest just the one query row we need (the latest
    # token's attention over vision keys) and let each layer's full matrix be freed immediately
    # after, capping peak memory at roughly one layer's transient matrix instead of all of them.
    decoder_layers = model.model.language_model.layers
    num_layers = len(decoder_layers)
    start_layer = int(round(num_layers * (1 - layer_fraction)))

    attn_sum = torch.zeros(num_vision_tokens, dtype=torch.float32, device=device)
    term_counter = {"n": 0}

    def _harvest_hook(module, inputs_, output):
        attn_weights = output[1]
        if attn_weights is None:
            return
        # Row -1: the latest token's query - the full prefill row at step 0 (q_len > 1), or the
        # (only) cached-query row at every later decode step (q_len == 1).
        row = attn_weights[0, :, -1, vision_positions]  # (heads, num_vision_tokens)
        attn_sum.add_(row.mean(dim=0).float())
        term_counter["n"] += 1

    handles = [layer.self_attn.register_forward_hook(_harvest_hook) for layer in decoder_layers[start_layer:]]
    try:
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    attn_grid = (attn_sum / term_counter["n"]).reshape(grid_h, grid_w).cpu().numpy()

    generated_answer = processor.tokenizer.decode(
        outputs.sequences[0, prompt_len:], skip_special_tokens=True
    ).strip()

    model_view = _qwen_model_view_image(inputs["pixel_values"], grid_thw, processor.image_processor)

    return {
        "generated_answer": generated_answer,
        "attn_grid": attn_grid,
        "model_view": model_view,
    }


def load_pixtral(model_name: str = "mistral-community/pixtral-12b", device: str = "cuda:0"):
    """Load Pixtral with eager attention on the language model only. Pixtral is wrapped in the
    same LlavaForConditionalGeneration class as plain LLaVA (same image_token_index mechanism
    for locating vision-key positions in input_ids), but - like Qwen - processes images at
    (near-)native resolution rather than a fixed square crop, so vision stays on sdpa for the
    same reason as Qwen's: its bidirectional self-attention over every patch has no single-row
    shortcut, and eager there would materialize the full patch x patch matrix before the merge
    ever shrinks it."""
    logger.info(f"Loading {model_name} onto {device} (attn_implementation=eager for the language model, sdpa for vision)")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map={"": device},
        attn_implementation={"text_config": "eager", "vision_config": "sdpa", "": "sdpa"},
    )
    processor = AutoProcessor.from_pretrained(model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


@torch.inference_mode()
def compute_pixtral_attention_map(model, processor, image: Image.Image, question: str, device: str,
                                   max_new_tokens: int = 24, layer_fraction: float = 0.5) -> Dict:
    """Pixtral counterpart to compute_qwen_attention_map. Same averaging scheme (every
    generated token, the back `layer_fraction` of decoder layers, every head, restricted to
    image-patch key positions) and the same hook-based (not output_attentions=True) memory
    guard Qwen uses, since Pixtral's native-resolution processing can push a single
    field-augmented image's vision-token count up the same way Qwen's does. The patch grid
    comes from `image_sizes` (the resized image's pixel dimensions) divided by the vision
    tower's patch_size, rather than Qwen's post-merge image_grid_thw."""
    image = image.convert("RGB") if image.mode != "RGB" else image
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
        add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    prompt_len = inputs["input_ids"].shape[1]

    image_token_id = model.config.image_token_index
    vision_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    num_vision_tokens = vision_positions.shape[0]

    patch_size = model.config.vision_config.patch_size
    img_h, img_w = inputs["image_sizes"][0].tolist()
    grid_h, grid_w = img_h // patch_size, img_w // patch_size
    if grid_h * grid_w != num_vision_tokens:
        raise ValueError(f"Vision token count {num_vision_tokens} doesn't match "
                          f"image_sizes-derived {grid_h}x{grid_w} - can't reshape into a patch grid.")

    decoder_layers = model.model.language_model.layers
    num_layers = len(decoder_layers)
    start_layer = int(round(num_layers * (1 - layer_fraction)))

    attn_sum = torch.zeros(num_vision_tokens, dtype=torch.float32, device=device)
    term_counter = {"n": 0}

    def _harvest_hook(module, inputs_, output):
        attn_weights = output[1]
        if attn_weights is None:
            return
        row = attn_weights[0, :, -1, vision_positions]  # (heads, num_vision_tokens)
        attn_sum.add_(row.mean(dim=0).float())
        term_counter["n"] += 1

    handles = [layer.self_attn.register_forward_hook(_harvest_hook) for layer in decoder_layers[start_layer:]]
    try:
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    attn_grid = (attn_sum / term_counter["n"]).reshape(grid_h, grid_w).cpu().numpy()

    generated_answer = processor.tokenizer.decode(
        outputs.sequences[0, prompt_len:], skip_special_tokens=True
    ).strip()

    model_view = _model_view_image(inputs["pixel_values"], processor.image_processor)

    return {
        "generated_answer": generated_answer,
        "attn_grid": attn_grid,
        "model_view": model_view,
    }


def load_mllama(model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", device: str = "cuda:0"):
    """Load LLaMA-3.2-Vision (Mllama) with eager attention on the language model only. Unlike
    LLaVA/Qwen/Pixtral, Mllama never inlines per-patch tokens into input_ids - the image is
    instead attended to via dedicated cross-attention layers (8 of its 40 decoder layers, see
    config.text_config.cross_attention_layers), so "attention over image patches" here means
    hooking those specific layers' cross-attention weights (compute_mllama_attention_map), not
    slicing a self-attention row by vision-token position. Vision stays on sdpa for the same
    reason as the other loaders here."""
    logger.info(f"Loading {model_name} onto {device} (attn_implementation=eager for the language model, sdpa for vision)")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map={"": device},
        attn_implementation={"text_config": "eager", "vision_config": "sdpa", "": "sdpa"},
    )
    processor = AutoProcessor.from_pretrained(model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def _mllama_tile_layout(aspect_ratio_id: int, max_num_tiles: int = 4) -> Tuple[int, int]:
    """(num_tiles_h, num_tiles_w) for a 1-indexed aspect_ratio_id (0 = no image/padding), from
    transformers' own image_processing_mllama.get_all_supported_aspect_ratios - the same lookup
    table the image processor used to lay tiles out in the first place, so both
    _mllama_model_view_image and compute_mllama_attention_map arrange tiles identically."""
    from transformers.models.mllama.image_processing_mllama import get_all_supported_aspect_ratios

    if aspect_ratio_id <= 0:
        return 1, 1
    return get_all_supported_aspect_ratios(max_num_tiles)[aspect_ratio_id - 1]


def _mllama_model_view_image(pixel_values: torch.Tensor, aspect_ratio_ids: torch.Tensor,
                              aspect_ratio_mask: torch.Tensor, image_processor) -> Image.Image:
    """Reconstruct the tiled mosaic Mllama's vision tower actually saw: `pixel_values` holds
    `max_num_tiles` per-image tile crops (padded up to that count regardless of how many the
    image actually needed - `aspect_ratio_mask` marks which are real content versus pure
    padding). Valid tiles are inverse-normalized and reassembled into their row-major
    (`get_all_supported_aspect_ratios`) layout - the image is resized to fit the chosen tile
    canvas aspect-preserving and anchored at the top-left (see
    image_processing_mllama.get_image_size_fit_to_canvas), so any leftover tiles are pure
    padding at the bottom/right rather than a centered border."""
    mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1)
    tiles = pixel_values[0, 0]  # (num_tiles, C, H, W) - single image, single media slot
    mask = aspect_ratio_mask[0, 0].bool().cpu()
    num_tiles_h, num_tiles_w = _mllama_tile_layout(int(aspect_ratio_ids[0, 0]))

    imgs = (tiles.float().cpu() * std + mean).clamp(0, 1)
    tile_h, tile_w = imgs.shape[-2], imgs.shape[-1]
    mosaic = np.zeros((num_tiles_h * tile_h, num_tiles_w * tile_w, 3), dtype=np.uint8)
    for idx in range(min(len(mask), num_tiles_h * num_tiles_w)):
        if not mask[idx]:
            continue
        row, col = divmod(idx, num_tiles_w)
        arr = (imgs[idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mosaic[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = arr
    return Image.fromarray(mosaic)


@torch.inference_mode()
def compute_mllama_attention_map(model, processor, image: Image.Image, question: str, device: str,
                                  max_new_tokens: int = 24, layer_fraction: float = 0.5) -> Dict:
    """Cross-attention counterpart to compute_qwen_attention_map/compute_pixtral_attention_map.
    Mllama has no vision placeholder tokens in input_ids to slice a self-attention row by -
    instead, 8 of its 40 decoder layers (config.text_config.cross_attention_layers) are
    dedicated cross-attention layers whose key/value sequence is the vision tower's own output:
    `max_num_tiles` tiles (padded up to that count - see aspect_ratio_mask) x (patches-per-tile
    + 1 CLS token) each. Forward hooks on just the back `layer_fraction` of *those* 8 layers
    harvest each generation step's (heads, q_len, kv_len) cross-attention row (query = the
    latest generated token - same convention as every other model here), dropping each tile's
    CLS column and reshaping the rest into that tile's own square patch grid, then arranging
    valid tiles into one combined mosaic grid at the same layout `_mllama_model_view_image`
    uses - invalid (padded) tiles are left zeroed via aspect_ratio_mask so they don't skew any
    content/border or top-k analysis built on top of this grid."""
    image = image.convert("RGB") if image.mode != "RGB" else image
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
        add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    prompt_len = inputs["input_ids"].shape[1]

    aspect_ratio_ids = inputs["aspect_ratio_ids"]
    aspect_ratio_mask = inputs["aspect_ratio_mask"]
    tile_mask = aspect_ratio_mask[0, 0].bool()  # (max_num_tiles,)
    num_tiles = tile_mask.shape[0]
    num_tiles_h, num_tiles_w = _mllama_tile_layout(int(aspect_ratio_ids[0, 0]))

    patch_size = model.config.vision_config.patch_size
    tile_image_size = model.config.vision_config.image_size
    tile_side = tile_image_size // patch_size  # patches per tile edge
    patches_per_tile = tile_side * tile_side

    cross_layer_indices = model.config.text_config.cross_attention_layers
    decoder_layers = model.model.language_model.layers
    start_layer = int(round(len(decoder_layers) * (1 - layer_fraction)))
    hooked_indices = [i for i in cross_layer_indices if i >= start_layer] or cross_layer_indices[-1:]

    attn_sum = torch.zeros(num_tiles, patches_per_tile + 1, dtype=torch.float32, device=device)
    term_counter = {"n": 0}

    def _harvest_hook(module, inputs_, output):
        attn_weights = output[1]
        if attn_weights is None:
            return
        # Row -1: the latest token's query - the full prefill row at step 0 (q_len > 1), or the
        # (only) cached-query row at every later decode step (q_len == 1); the cross-attention
        # layer recomputes attention against the cached image key/values at every step either
        # way (see MllamaTextCrossAttention.forward), so this fires every generation step.
        row = attn_weights[0, :, -1, :]  # (heads, kv_len = num_tiles * (patches_per_tile + 1))
        attn_sum.add_(row.mean(dim=0).float().reshape(num_tiles, patches_per_tile + 1))
        term_counter["n"] += 1

    handles = [decoder_layers[i].cross_attn.register_forward_hook(_harvest_hook) for i in hooked_indices]
    try:
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    per_tile = (attn_sum / term_counter["n"])[:, 1:]  # drop each tile's CLS column
    per_tile = per_tile.reshape(num_tiles, tile_side, tile_side).cpu().numpy()

    attn_grid = np.zeros((num_tiles_h * tile_side, num_tiles_w * tile_side), dtype=np.float32)
    for idx in range(min(num_tiles, num_tiles_h * num_tiles_w)):
        if not tile_mask[idx]:
            continue
        row, col = divmod(idx, num_tiles_w)
        attn_grid[row * tile_side:(row + 1) * tile_side, col * tile_side:(col + 1) * tile_side] = per_tile[idx]

    generated_answer = processor.tokenizer.decode(
        outputs.sequences[0, prompt_len:], skip_special_tokens=True
    ).strip()

    model_view = _mllama_model_view_image(inputs["pixel_values"], aspect_ratio_ids, aspect_ratio_mask,
                                           processor.image_processor)

    return {
        "generated_answer": generated_answer,
        "attn_grid": attn_grid,
        "model_view": model_view,
    }


def patch_grid_info(image_size: Tuple[int, int], model_input_size: int = 336, patch_size: int = 14) -> Dict:
    """Where LLaVA's fixed 24x24 patch grid falls, in both spaces, for an input image of
    `image_size` = (width, height). LLaVA's CLIPImageProcessor resizes the shortest edge to
    `model_input_size` (aspect-preserving) then center-crops to model_input_size x
    model_input_size before patchifying - so the patch grid itself (boundaries in the
    336x336 space the model actually operates in) is IDENTICAL for every input, at
    multiples of `patch_size` regardless of the variant. What varies per variant is where
    those boundaries land back in the *original* image, and how many original pixels one
    patch spans (`patch_size_in_original_px`) - i.e. how much the variant's own
    preprocessing has already compressed its content before the model ever patchifies it.
    """
    width, height = image_size
    scale = model_input_size / min(width, height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    crop_left, crop_top = (resized_w - model_input_size) / 2, (resized_h - model_input_size) / 2

    boundaries_model_px = list(range(0, model_input_size + 1, patch_size))
    boundaries_original_x = [(b + crop_left) / scale for b in boundaries_model_px]
    boundaries_original_y = [(b + crop_top) / scale for b in boundaries_model_px]

    return {
        "original_size": (width, height),
        "resize_scale": scale,
        "resized_size": (resized_w, resized_h),
        "crop_box_in_resized_px": (crop_left, crop_top, crop_left + model_input_size, crop_top + model_input_size),
        "boundaries_model_px": boundaries_model_px,
        "boundaries_original_x_px": boundaries_original_x,
        "boundaries_original_y_px": boundaries_original_y,
        "patch_size_in_original_px": patch_size / scale,
    }


def draw_patch_grid(model_view: Image.Image, patch_size: int = 14,
                     color: Tuple[int, int, int] = (80, 80, 80)) -> Image.Image:
    """Draw the 24x24 patch boundary lines onto a copy of the (336x336) model-view image."""
    from PIL import ImageDraw

    img = model_view.copy()
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    for x in range(0, size + 1, patch_size):
        draw.line([(x, 0), (x, size)], fill=color, width=1)
    for y in range(0, size + 1, patch_size):
        draw.line([(0, y), (size, y)], fill=color, width=1)
    return img


def desink_attention_grid(attn_grid: np.ndarray, sink_mass_threshold: float = 0.15) -> Tuple[np.ndarray, np.ndarray, float]:
    """Detect and clip extreme "attention sink" cells - isolated cells holding a wildly
    disproportionate share of an image's *total* attention mass on their own. This is a
    documented artifact in some register-free ViTs/LLMs (largely content-independent -
    e.g. Pixtral exhibited a single fixed bottom-corner cell absorbing >50% of total mass
    on a real example, versus <10% for LLaVA/Qwen/Mllama's single highest cell on the same
    image), distinct from the milder, more numerous "sink patches" `plot_attention_grid`'s
    99th-percentile color clipping already tolerates - a single cell at 50%+ of total mass
    would also badly distort any downstream sum over this grid (content/border mass,
    top-k survival), not just the color scale.

    A cell is flagged if its own share of `attn_grid`'s total mass exceeds
    `sink_mass_threshold`; flagged cells are clipped down to the highest non-sink value.

    Returns ``(corrected_grid, sink_mask, sink_mass_frac)`` - `sink_mass_frac` is the
    fraction of the *original* grid's total mass the flagged cells held, before clipping.
    """
    total_mass = attn_grid.sum()
    sink_mask = (attn_grid / total_mass) > sink_mass_threshold if total_mass > 0 else np.zeros_like(attn_grid, dtype=bool)
    sink_mass_frac = float(attn_grid[sink_mask].sum() / total_mass) if sink_mask.any() else 0.0

    corrected = attn_grid.copy()
    if sink_mask.any():
        non_sink = attn_grid[~sink_mask]
        cap = non_sink.max() if non_sink.size else 0.0
        corrected[sink_mask] = cap
    return corrected, sink_mask, sink_mass_frac


def upsample_grid(grid: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Smoothly upsample a small attention grid to `size` = (width, height) pixels."""
    lo, hi = grid.min(), grid.max()
    normalized = (grid - lo) / (hi - lo) if hi > lo else np.zeros_like(grid)
    img = Image.fromarray((normalized * 255).astype(np.uint8)).resize(size, resample=Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0 * (hi - lo) + lo


def plot_attention_grid(rows: List[Dict], output_file=None, suptitle: str = None, close: bool = True):
    """rows: list of {"label": str, "model_view": PIL.Image, "attn_grid": np.ndarray,
    "generated_answer": str, "correct": Optional[bool]}. One row per image variant,
    two columns (model input | attention overlay). Color scale is shared across all
    rows so brightness is directly comparable (e.g. attention spreading out /
    weakening under corruption shows up as visibly dimmer, not just differently shaped).
    Clipped to the 99th percentile (rather than the true max) across all rows: LLaVA's
    CLIP-ViT backbone has no register tokens, so a handful of fixed high-norm "sink"
    patches absorb disproportionate attention regardless of content (a documented
    register-free-ViT artifact) - an unclipped scale would wash out everything else
    to near-black in comparison.

    `output_file` is optional (skip saving to just get the figure back). `close=False`
    returns the figure instead of closing it, e.g. for `plt.show()` in a notebook."""
    import matplotlib.pyplot as plt

    vmax = float(np.percentile(np.concatenate([r["attn_grid"].ravel() for r in rows]), 99))
    fig, axes = plt.subplots(len(rows), 2, figsize=(6.4, 3.15 * len(rows)), squeeze=False,
                              constrained_layout=True)

    for ax_row, r in zip(axes, rows):
        model_view = r["model_view"]
        overlay = upsample_grid(r["attn_grid"], model_view.size)

        ax_row[0].imshow(model_view)
        ax_row[0].set_title(r["label"], fontsize=9, loc="left")
        ax_row[0].axis("off")

        ax_row[1].imshow(model_view)
        im = ax_row[1].imshow(overlay, cmap=ATTENTION_CMAP, alpha=0.55, vmin=0, vmax=vmax)
        answer_note = r.get("generated_answer", "")
        if len(answer_note) > 40:
            answer_note = answer_note[:37] + "..."
        title = f'answer: "{answer_note}"'
        if r.get("correct") is not None:
            title += "  ✓" if r["correct"] else "  ✗"
        ax_row[1].set_title(title, fontsize=8, loc="left")
        ax_row[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[:, 1], fraction=0.025, pad=0.02)
    cbar.set_label("attention weight", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    if output_file:
        fig.savefig(output_file, dpi=160, bbox_inches="tight")
        logger.info(f"Saved attention map figure to {output_file}")
    if close:
        plt.close(fig)
        return None
    return fig
