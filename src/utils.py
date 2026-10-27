from PIL import Image
import base64
import tempfile
import logging
import queue
import torch
import numpy as np
from typing import Dict
from .RealESRGAN import RealESRGAN
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

logger = logging.getLogger(__name__)

def collect_results(result_queue, results, dataset, stop_event):
    while True:
        try:
            result = result_queue.get(timeout=1.0)
            if result is None:  # Sentinel
                break
            results.append(result)
            if len(results) % 50 == 0:
                logger.info(f"Collected {len(results)}/{len(dataset)} results")
        except queue.Empty:
            if stop_event.is_set():
                break

def encode_image(image):
  # Convert image to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_image_path = tmp_file.name
        image.save(temp_image_path, format="JPEG")

    with open(temp_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ImageProcessor:
    @staticmethod
    def pad_image_to_square(img: Image.Image, resize: bool = True, dim: int = 448, 
                           background_color: tuple = (255, 255, 255)) -> Image.Image:
        width, height = img.size
        max_dim = max(width, height)
        new_img = Image.new("RGB", (max_dim, max_dim), background_color)
        paste_position = ((max_dim - width) // 2, (max_dim - height) // 2)
        new_img.paste(img, paste_position)
        return new_img.resize((dim, dim)) if resize else new_img
    
    @staticmethod
    def add_white_field(img: Image.Image, 
                        percent: float = 0, 
                        position: str = 'center', 
                        squire: bool = False,
                        aspect: bool = True,
                        background_color: tuple = (255, 255, 255)
                        ) -> Image.Image:
        width, height = img.size

        if squire: # Make the image square
            max_dimension = max(width, height)
            background_size = int(max_dimension * (1 + percent))
            background = Image.new("RGB", (background_size, background_size), background_color)
        elif aspect: # Maintain aspect ratio
            background_width = int(width * (1 + percent))
            background_height = int(height * (1 + percent))
            background = Image.new("RGB", (background_width, background_height), background_color)
        else: # Add fields based on the larger dimension for aspect ratio sensitivity stress testing
            if width > height:
                background_width = int(width * ((1 + percent)**2))
                background_height = height
                background = Image.new("RGB", (background_width, background_height), background_color)
            else:
                background_height = int(height * ((1 + percent)**2))
                background_width = width
                background = Image.new("RGB", (background_width, background_height), background_color)
        
        if position == 'center':
            pos = ((background.size[0] - width) // 2, (background.size[1] - height) // 2)
        elif position == 'bottom':
            pos = (0, 0)
        elif position == 'top':
            pos = (background.size[0] - width, background.size[1] - height)
        else:
            raise ValueError(f"Invalid position: {position}")
            
        background.paste(img, pos)
        return background
    

def _mllama_fit_size(width: int, height: int, tile: int, max_tiles: int):
    """Replicate MllamaImageProcessor tile-canvas selection to find the pixel
    size the image is resized to (before padding/tiling).

    Enumerates every supported tile arrangement (num_tiles_h, num_tiles_w) with
    product <= max_tiles, computes the aspect-preserving scale that fits the
    image into each canvas, and picks the same scale the processor would:
    the smallest scale >= 1 if upscaling is possible, otherwise the largest
    scale < 1 (least downscaling)."""
    arrangements = [
        (nh, nw)
        for nh in range(1, max_tiles + 1)
        for nw in range(1, max_tiles + 1)
        if nh * nw <= max_tiles
    ]
    scales = [min((nw * tile) / width, (nh * tile) / height) for nh, nw in arrangements]
    upscale = [s for s in scales if s >= 1.0]
    selected = min(upscale) if upscale else max(scales)
    return max(1, round(width * selected)), max(1, round(height * selected))


def get_preprocessed_size(processor, model_name: str, image: Image.Image):
    """Return the (width, height) in pixels that the model-specific image
    preprocessing resizes ``image`` to, before patchification/tiling.

    This is ``image_resized_size_px`` used to derive the field->preprocessing
    rescaling coefficient. Falls back to the input size on any failure so a
    single quirky sample never aborts a run."""
    name = model_name.lower()
    ip = getattr(processor, "image_processor", processor)
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    w, h = rgb.size
    try:
        if "qwen" in name:
            out = ip(images=rgb, return_tensors="pt")
            grid = out["image_grid_thw"][0]  # (t, grid_h, grid_w) in patch units
            patch = int(getattr(ip, "patch_size", 14))
            return int(grid[2]) * patch, int(grid[1]) * patch
        if "pixtral" in name:
            out = ip(images=rgb, return_tensors="pt")
            rh, rw = (int(v) for v in out["image_sizes"][0][:2])
            return rw, rh
        if "llava" in name:
            cs = getattr(ip, "crop_size", None) or getattr(ip, "size", None) or {}
            dim_w = cs.get("width", cs.get("shortest_edge", 336))
            dim_h = cs.get("height", cs.get("shortest_edge", 336))
            return int(dim_w), int(dim_h)
        if "llama" in name:
            size = getattr(ip, "size", {}) or {}
            tile = int(size.get("height", size.get("width", 560)))
            max_tiles = int(getattr(ip, "max_image_tiles", 4))
            return _mllama_fit_size(w, h, tile, max_tiles)
    except Exception as e:
        logger.warning(f"Could not determine preprocessed size for {model_name}: {e}. "
                       f"Falling back to generic pixel_values probe.")

    # Generic fallback for architectures without a bespoke branch above (e.g.
    # PaliGemma - fixed single resize; InternVL - dynamic multi-tile cropping,
    # where this yields the fixed per-tile crop resolution rather than a
    # single whole-image resize target, same approximation this codebase's
    # own vision-token accounting already accepts for InternVL elsewhere).
    # Molmo's processor.process() takes a different call signature entirely
    # (no plain `images=` kwarg), so it always falls through to the raw input
    # size below - consistent with Molmo having no single "resized to WxH"
    # concept in the rest of this codebase either.
    try:
        out = ip(images=rgb, return_tensors="pt")
        pixel_values = out.get("pixel_values")
        if pixel_values is not None and pixel_values.dim() >= 4:
            return int(pixel_values.shape[-1]), int(pixel_values.shape[-2])
    except Exception as e:
        logger.warning(f"Could not determine preprocessed size for {model_name}: {e}. "
                       f"Falling back to input size {(w, h)}.")
    return w, h


def _wh(size) -> Dict:
    return {"width": size[0], "height": size[1]}


def build_field_images(processor, model_name: str, image: Image.Image,
                       field, position: str = 'center'):
    """Build the three image variants for a single field-experiment sample.

    Returns ``(field_img, resized_img, bicubic_img, info)`` where:
      * ``field_img`` is the original embedded in a larger blank canvas at the
        requested ``field`` percent (the "field" experiment input),
      * ``resized_img`` is the plain original (no border) downscaled by the
        same coefficient the field image experiences during model
        preprocessing - as if the border had been cropped back out after
        resizing (the "resized" control, isolating content downscaling from
        the added-border effect),
      * ``bicubic_img`` is ``resized_img`` scaled back up to the original
        image's resolution via bicubic interpolation (the "bicubic" control:
        same resolution as the input, but with the detail lost during the
        downscale - simulating detail loss without a size change).
      * ``info`` holds the sizes/coefficient for logging, in the same
        ``{"width":, "height":}`` shape used by ``_image_size_fields`` for
        every other (non-field) result row.
    """
    original = image.convert("RGB") if image.mode != "RGB" else image
    ow, oh = original.size

    if field:
        pos = position or 'center'
        field_img = ImageProcessor.add_white_field(original, percent=field, position=pos)
    else:
        field_img = original
    fw, fh = field_img.size

    rw, rh = get_preprocessed_size(processor, model_name, field_img)
    cx, cy = rw / fw, rh / fh

    tw, th = max(1, round(ow * cx)), max(1, round(oh * cy))
    resized_img = original.resize((tw, th), resample=Image.LANCZOS)

    # Third variant: upsample the downscaled control back to the original size.
    bicubic_img = resized_img.resize((ow, oh), resample=Image.BICUBIC)

    info = {
        "image_original_size_px": _wh((ow, oh)),
        "image_field_size_px": _wh((fw, fh)),
        "image_resized_size_px": _wh((rw, rh)),
        "rescaling_coefficient": {"width": cx, "height": cy},
        "image_resized_experiment_size_px": _wh((tw, th)),
    }
    return field_img, resized_img, bicubic_img, info


class ImageProcessor_upscaling:
    """Image super-resolution class using Real-ESRGAN model"""
    def __init__(self, scale:int):
        self.scale = int(scale)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RealESRGAN(device, scale=self.scale)
        weights_path = WEIGHTS_DIR / f"RealESRGAN_x{self.scale}.pth"
        self.model.load_weights(str(weights_path), download=False)
        #self.model.load_weights(f'weights/RealESRGAN_x{int(self.scale)}.pth', download=False)  
    
    def upscale(self, img: Image.Image) -> Image.Image:            
        return self.model.predict(img)