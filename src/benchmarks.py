from datasets import load_dataset
import logging
from torch.utils.data import Dataset, DataLoader
from .utils import ImageProcessor, ImageProcessor_upscaling
import random
from PIL import Image
import base64
import io
import json
import os
import zipfile
from collections import Counter

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def make_upscaler(scale):
    """Load the Real-ESRGAN upscaler for the requested scale, snapping up to the
    nearest supported factor (2, 4 or 8) if scale isn't one of those exactly."""
    if scale in [2, 4, 8]:
        logger.info(f"Loaded Real_ESRGAN upscaler with scale {scale}.")
        return ImageProcessor_upscaling(scale=scale)
    scale_new = min(s for s in [2, 4, 8] if s >= scale)
    logger.info(f"Loaded Real_ESRGAN upscaler with scale {scale_new}.")
    logger.info(f"Image will be downsized to {scale} to match the requirement.")
    return ImageProcessor_upscaling(scale=scale_new)


def random_sample_selector(n, sample_size, seed):
    """Default sample-selection strategy: a uniform random subset, capped at
    sample_size (up to 200 by callers' defaults)."""
    random.seed(seed)
    return random.sample(range(n), min(sample_size, n))


def sample_indices_or_all(n, field, scale, interpolation, sample_size, seed, sampler=random_sample_selector):
    """Indices to evaluate: all `n` of them for the original-resolution baseline
    (field/scale/interpolation all falsy), or `sampler(n, sample_size, seed)`
    otherwise - e.g. for upscaled-resolution (scale) runs.

    To use a different sample-selection strategy later (stratified, hardest-example,
    etc.), pass a different `sampler` callable with the same
    `(n, sample_size, seed) -> list[int]` signature - no other code needs to change.
    """
    if not (field or scale or interpolation):
        return list(range(n))
    return sampler(n, sample_size, seed)


def apply_image_transform(image, scale=None, interpolation=None, position='center', upscaler=None):
    """Apply the configured scale/interpolation augmentation to an image.

    Field (white-border) augmentation is deliberately NOT handled here: main_field.py's
    worker needs the raw, untransformed image plus the field percent/position as plain
    metadata (see the "field"/"position" keys every dataset below returns) so it can build
    the field/resized/bicubic experiment triplet itself via build_field_images(), using the
    target model's own preprocessing to compute the rescaling coefficient. Applying the
    field transform here, before the worker ever sees the image, would both double up the
    padding and give the worker no way to recover the pre-padding original."""
    if scale:
        img_orig = image.convert('RGB')
        img = upscaler.upscale(img_orig)
        if scale not in [2, 4, 8]:
            w, h = img_orig.size
            target_size = (int(round(w * scale)), int(round(h * scale)))
            img = img.resize(target_size, resample=Image.LANCZOS)
        return img
    elif interpolation:
        img_orig = image.convert('RGB')
        target_size = (
            int(round(img_orig.size[0] * interpolation)),
            int(round(img_orig.size[1] * interpolation))
        )
        return img_orig.resize(target_size, resample=Image.BICUBIC)
    else:
        return image

class MMVetDataset(Dataset):
    """Dataset class for MMVet benchmark."""
    
    def __init__(self, dataset_name: str = "whyu/mm-vet", field = None, scale = None, interpolation = None, position: str = 'center', sample_size:int = 200, seed: int = 42):
        random.seed(seed)

        full_dataset = load_dataset(dataset_name)['test']
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)
        self.samples = full_dataset.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position

        if self.scale:
            if self.scale in [2,4,8]:
                self.upscaler = ImageProcessor_upscaling(scale = int(self.scale))
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale}.")
            else:
                self.scale_new = min(s for s in [2,4,8] if s >= self.scale)
                self.upscaler = ImageProcessor_upscaling(scale = int(self.scale_new))
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale_new}.")
                logger.info(f"Image will be downsized to {self.scale} to match the requirement.")

        logger.info(f"Loaded {len(self.samples)} samples from MMVet dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.scale:
            img_orig = sample['image'].convert('RGB')
            img = self.upscaler.upscale(img_orig)
            
            if self.scale not in [2,4,8]:
                w, h = img_orig.size
                target_size = (
                    int(round(w * self.scale)),
                    int(round(h * self.scale))
                ) 
                img = img.resize(target_size, resample=Image.LANCZOS)
        elif self.interpolation: 
            img_orig = sample['image'].convert('RGB')
            target_size = (
                int(round(img_orig.size[0] * self.interpolation)),
                int(round(img_orig.size[1] * self.interpolation))
            )
            img = img_orig.resize(target_size, resample=Image.BICUBIC)
        else:
            img = sample['image']

        return {
            "id": sample["id"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["question"],
            "answer": sample["answer"]
        }
    
class MMEvalDataset(Dataset):
    """Dataset class for MMEval benchmark."""
    
    def __init__(self, dataset_name: str = "darkyarding/MME", field = None, scale = None, interpolation = None, position: str = 'center', seed:int = 42, sample_size:int = 200):
        full_dataset = load_dataset(dataset_name)['test']

        # Sample 200 items with fixed seed for reproducibility (or all of them at
        # the original-resolution baseline - see sample_indices_or_all)
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)

        # Select the sampled items
        self.samples = full_dataset.select(indices)
        self.field = field
        self.position = position
        self.scale = scale
        self.interpolation = interpolation

        if self.scale:
            if self.scale in [2,4,8]:
                self.upscaler = ImageProcessor_upscaling(scale = self.scale)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale}.")
            else: 
                self.scale_new = min(s for s in [2,4,8] if s >= self.scale) 
                self.upscaler = ImageProcessor_upscaling(scale = self.scale_new)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale_new}.")
                logger.info(f"Image will be downsized to {self.scale} to match the requirement.")

        self.position = position
        logger.info(f"Loaded {len(self.samples)} samples from MMEval dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.scale:
            img_orig = sample['image'].convert('RGB')
            img = self.upscaler.upscale(img_orig)
            
            if self.scale not in [2,4,8]:
                w, h = img_orig.size
                target_size = (
                    int(round(w * self.scale)),
                    int(round(h * self.scale))
                ) 
                img = img.resize(target_size, resample=Image.LANCZOS)
        elif self.interpolation: 
            img_orig = sample['image'].convert('RGB')
            target_size = (
                int(round(img_orig.size[0] * self.interpolation)),
                int(round(img_orig.size[1] * self.interpolation))
            )
            img = img_orig.resize(target_size, resample=Image.BICUBIC)
        else:
            img = sample['image']
    
        return {
            "id": sample["question_id"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["question"],
            "answer": sample["answer"]
        }

class MathVerseDataset(Dataset):
    """Dataset class for MathVerse benchmark."""
    
    def __init__(self, dataset_name: str = "AI4Math/MathVerse", field = None, scale = None, interpolation = None, position: str = 'center', seed:int = 42, sample_size:int = 200):
        full_dataset = load_dataset(dataset_name, 'testmini')['testmini']

        # Sample 200 items with fixed seed for reproducibility (or all of them at
        # the original-resolution baseline - see sample_indices_or_all)
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)

        # Select the sampled items
        self.samples = full_dataset.select(indices)
        self.field = field
        self.position = position
        self.scale = scale
        self.interpolation = interpolation

        if self.scale:
            if self.scale in [2,4,8]:
                self.upscaler = ImageProcessor_upscaling(scale = self.scale)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale}.")
            else: 
                self.scale_new = min(s for s in [2,4,8] if s >= self.scale) 
                self.upscaler = ImageProcessor_upscaling(scale = self.scale_new)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale_new}.")
                logger.info(f"Image will be downsized to {self.scale} to match the requirement.")

        logger.info(f"Loaded {len(self.samples)} samples from MathVerse dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.scale:
            img_orig = sample['image'].convert('RGB')
            img = self.upscaler.upscale(img_orig)
            
            if self.scale not in [2,4,8]:
                w, h = img_orig.size
                target_size = (
                    int(round(w * self.scale)),
                    int(round(h * self.scale))
                ) 
                img = img.resize(target_size, resample=Image.LANCZOS)
        elif self.interpolation: 
            img_orig = sample['image'].convert('RGB')
            target_size = (
                int(round(img_orig.size[0] * self.interpolation)),
                int(round(img_orig.size[1] * self.interpolation))
            )
            img = img_orig.resize(target_size, resample=Image.BICUBIC)
        else:
            img = sample['image']
    
        return {
            "id": sample["sample_index"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["question"],
            "answer": sample["answer"]
        }

class MMStar(Dataset):
    """Dataset class for MMStar benchmark."""
    
    def __init__(self, dataset_name: str = "Lin-Chen/MMStar", field = None, scale = None, interpolation = None, position: str = 'center', seed:int = 42, sample_size:int = 200):
        full_dataset = load_dataset(dataset_name)['val']

        # Sample 200 items with fixed seed for reproducibility (or all of them at
        # the original-resolution baseline - see sample_indices_or_all)
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)

        # Select the sampled items
        self.samples = full_dataset.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position

        if self.scale:
            if self.scale in [2,4,8]:
                self.upscaler = ImageProcessor_upscaling(scale = self.scale)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale}.")
            else: 
                self.scale_new = min(s for s in [2,4,8] if s >= self.scale) 
                self.upscaler = ImageProcessor_upscaling(scale = self.scale_new)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale_new}.")
                logger.info(f"Image will be downsized to {self.scale} to match the requirement.")

        logger.info(f"Loaded {len(self.samples)} samples from MMStar dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.scale:
            img_orig = sample['image'].convert('RGB')
            img = self.upscaler.upscale(img_orig)
            
            if self.scale not in [2,4,8]:
                w, h = img_orig.size
                target_size = (
                    int(round(w * self.scale)),
                    int(round(h * self.scale))
                ) 
                img = img_orig.resize(target_size, resample=Image.LANCZOS)

        elif self.interpolation: 
            img_orig = sample['image'].convert('RGB')
            target_size = (
                int(round(img_orig.size[0] * self.interpolation)),
                int(round(img_orig.size[1] * self.interpolation))
            )
            img = img_orig.resize(target_size, resample=Image.BICUBIC)
        else:
            img = sample['image']
    
        return {
            "id": sample["index"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["question"],
            "answer": sample["answer"]
        }

class CharXiv(Dataset):
    """Dataset class for CharXiv benchmark."""
    
    def __init__(self, dataset_name: str = "princeton-nlp/CharXiv", field = None, scale = None, interpolation = None, position: str = 'center', seed:int = 42, sample_size:int = 200):
        full_dataset = load_dataset(dataset_name)['validation']

        # Sample 200 items with fixed seed for reproducibility (or all of them at
        # the original-resolution baseline - see sample_indices_or_all)
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)

        # Select the sampled items
        self.samples = full_dataset.select(indices)
        self.position = position

        self.field = field
        self.scale = scale
        self.interpolation = interpolation

        if self.scale:
            if self.scale in [2,4,8]:
                self.upscaler = ImageProcessor_upscaling(scale = self.scale)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale}.")
            else: 
                self.scale_new = min(s for s in [2,4,8] if s >= self.scale) 
                self.upscaler = ImageProcessor_upscaling(scale = self.scale_new)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale_new}.")
                logger.info(f"Image will be downsized to {self.scale} to match the requirement.")
                
        logger.info(f"Loaded {len(self.samples)} samples from CharXiv dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.scale:
            img_orig = sample['image'].convert('RGB')
            img = self.upscaler.upscale(img_orig)
            
            if self.scale not in [2,4,8]:
                w, h = img_orig.size
                target_size = (
                    int(round(w * self.scale)),
                    int(round(h * self.scale))
                ) 
                img = img.resize(target_size, resample=Image.LANCZOS)
        elif self.interpolation: 
            img_orig = sample['image'].convert('RGB')
            target_size = (
                int(round(img_orig.size[0] * self.interpolation)),
                int(round(img_orig.size[1] * self.interpolation))
            )
            img = img_orig.resize(target_size, resample=Image.BICUBIC)
        else:
            img = sample['image']
    
        return {
            "id": sample["original_figure_path"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["reasoning_q"],
            "answer": sample["reasoning_a"]
        }

class VQAv2Dataset(Dataset):
    """Dataset class for VQAv2 benchmark.

    Uses lmms-lab/VQAv2 (parquet-backed) instead of HuggingFaceM4/VQAv2 - the latter
    only ships a legacy Python loading script, which datasets>=4.x no longer supports.
    """

    def __init__(self, dataset_name: str = "lmms-lab/VQAv2", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        full_dataset = load_dataset(dataset_name, split='validation')
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)
        self.samples = full_dataset.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position
        self.upscaler = make_upscaler(scale) if scale else None

        logger.info(f"Loaded {len(self.samples)} samples from VQAv2 dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = apply_image_transform(sample["image"], self.scale, self.interpolation, self.position, self.upscaler)
        return {
            "id": sample["question_id"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["question"],
            "answer": sample["multiple_choice_answer"],
        }


class GQADataset(Dataset):
    """Dataset class for GQA benchmark.

    Images and questions live in separate configs of the same repo and have to be
    joined on imageId.
    """

    def __init__(self, dataset_name: str = "lmms-lab/GQA", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        instructions = load_dataset(dataset_name, 'testdev_balanced_instructions', split='testdev')
        images = load_dataset(dataset_name, 'testdev_balanced_images', split='testdev')
        self.image_by_id = {row["id"]: row["image"] for row in images}
        indices = sample_indices_or_all(len(instructions), field, scale, interpolation, sample_size, seed)
        self.samples = instructions.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position
        self.upscaler = make_upscaler(scale) if scale else None

        logger.info(f"Loaded {len(self.samples)} samples from GQA dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        raw_image = self.image_by_id[sample["imageId"]]
        img = apply_image_transform(raw_image, self.scale, self.interpolation, self.position, self.upscaler)
        return {
            "id": sample["id"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["question"],
            "answer": sample["answer"],
        }


class CVBenchDataset(Dataset):
    """Dataset class for CV-Bench benchmark (2D subset only)."""

    def __init__(self, dataset_name: str = "nyu-visionx/CV-Bench", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        full_dataset = load_dataset(dataset_name, data_files={'test': 'test_2d.parquet'}, split='test')
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)
        self.samples = full_dataset.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position
        self.upscaler = make_upscaler(scale) if scale else None

        logger.info(f"Loaded {len(self.samples)} samples from CV-Bench (2D) dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = apply_image_transform(sample["image"], self.scale, self.interpolation, self.position, self.upscaler)
        return {
            "id": sample["idx"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["prompt"],
            "answer": sample["answer"],
        }


class TextVQADataset(Dataset):
    """Dataset class for TextVQA benchmark."""

    def __init__(self, dataset_name: str = "lmms-lab/TextVQA", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        full_dataset = load_dataset(dataset_name, split='validation')
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)
        self.samples = full_dataset.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position
        self.upscaler = make_upscaler(scale) if scale else None

        logger.info(f"Loaded {len(self.samples)} samples from TextVQA dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # answers is a list of ~10 annotator strings; take the majority vote
        answer = Counter(sample["answers"]).most_common(1)[0][0]
        img = apply_image_transform(sample["image"], self.scale, self.interpolation, self.position, self.upscaler)
        return {
            "id": sample["question_id"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["question"],
            "answer": answer,
        }


class ChartQADataset(Dataset):
    """Dataset class for ChartQA benchmark."""

    def __init__(self, dataset_name: str = "lmms-lab/ChartQA", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        full_dataset = load_dataset(dataset_name, split='test')
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)
        self.samples = full_dataset.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position
        self.upscaler = make_upscaler(scale) if scale else None

        logger.info(f"Loaded {len(self.samples)} samples from ChartQA dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = apply_image_transform(sample["image"], self.scale, self.interpolation, self.position, self.upscaler)
        return {
            "id": idx,
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["question"],
            "answer": sample["answer"],
        }


class DocVQADataset(Dataset):
    """Dataset class for DocVQA benchmark."""

    def __init__(self, dataset_name: str = "lmms-lab/DocVQA", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        full_dataset = load_dataset(dataset_name, 'DocVQA', split='validation')
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)
        self.samples = full_dataset.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position
        self.upscaler = make_upscaler(scale) if scale else None

        logger.info(f"Loaded {len(self.samples)} samples from DocVQA dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = apply_image_transform(sample["image"], self.scale, self.interpolation, self.position, self.upscaler)
        return {
            "id": sample["questionId"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["question"],
            "answer": sample["answers"][0],
        }


class InfographicVQADataset(Dataset):
    """Dataset class for InfographicVQA benchmark.

    lmms-lab/InfographicVQA does not exist as its own repo - InfographicVQA is a
    config inside lmms-lab/DocVQA.
    """

    def __init__(self, dataset_name: str = "lmms-lab/DocVQA", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        full_dataset = load_dataset(dataset_name, 'InfographicVQA', split='validation')
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)
        self.samples = full_dataset.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position
        self.upscaler = make_upscaler(scale) if scale else None

        logger.info(f"Loaded {len(self.samples)} samples from InfographicVQA dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = apply_image_transform(sample["image"], self.scale, self.interpolation, self.position, self.upscaler)
        return {
            "id": sample["questionId"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["question"],
            "answer": sample["answers"][0],
        }


class VStarBenchDataset(Dataset):
    """Dataset class for V*Bench benchmark.

    The image column is only a relative path string, so the repo snapshot has to be
    downloaded once and images opened from disk.
    """

    def __init__(self, dataset_name: str = "craigwu/vstar_bench", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        full_dataset = load_dataset(dataset_name)['test']
        self.local_dir = snapshot_download(dataset_name, repo_type='dataset')
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)
        self.samples = full_dataset.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position
        self.upscaler = make_upscaler(scale) if scale else None

        logger.info(f"Loaded {len(self.samples)} samples from V*Bench dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        raw_image = Image.open(os.path.join(self.local_dir, sample["image"]))
        img = apply_image_transform(raw_image, self.scale, self.interpolation, self.position, self.upscaler)
        return {
            "id": f"{sample['category']}_{sample['question_id']}",
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": sample["text"],
            "answer": sample["label"],
        }


class HRBenchDataset(Dataset):
    """Dataset class for HR-Bench benchmark (8K subset).

    The image column is a base64-encoded JPEG string rather than a decoded image
    feature, and the multiple-choice options live in separate A/B/C/D columns.
    """

    def __init__(self, dataset_name: str = "DreamMr/HR-Bench", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        full_dataset = load_dataset(dataset_name, 'hrbench_version_split', split='hrbench_8k')
        indices = sample_indices_or_all(len(full_dataset), field, scale, interpolation, sample_size, seed)
        self.samples = full_dataset.select(indices)

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position
        self.upscaler = make_upscaler(scale) if scale else None

        logger.info(f"Loaded {len(self.samples)} samples from HR-Bench (8K) dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        raw_image = Image.open(io.BytesIO(base64.b64decode(sample["image"])))
        img = apply_image_transform(raw_image, self.scale, self.interpolation, self.position, self.upscaler)
        question = (
            f"{sample['question']}\n"
            f"(A) {sample['A']}\n(B) {sample['B']}\n(C) {sample['C']}\n(D) {sample['D']}"
        )
        return {
            "id": sample["index"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": question,
            "answer": sample["answer"],
        }


class MMERealWorldLiteDataset(Dataset):
    """Dataset class for MME-RealWorld-Lite benchmark.

    Ships as a single zip (images + one JSON metadata file) rather than a
    parquet/arrow dataset, so it needs a one-time download + extraction.
    """

    def __init__(self, dataset_name: str = "yifanzhang114/MME-RealWorld-Lite", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        local_dir = snapshot_download(dataset_name, repo_type='dataset')
        self.extract_dir = os.path.join(local_dir, "_extracted")
        if not os.path.isdir(self.extract_dir):
            with zipfile.ZipFile(os.path.join(local_dir, "data.zip")) as zf:
                zf.extractall(self.extract_dir)

        with open(os.path.join(self.extract_dir, "data", "MME-RealWorld-Lite.json")) as f:
            records = json.load(f)

        indices = sample_indices_or_all(len(records), field, scale, interpolation, sample_size, seed)
        self.samples = [records[i] for i in indices]

        self.field = field
        self.scale = scale
        self.interpolation = interpolation
        self.position = position
        self.upscaler = make_upscaler(scale) if scale else None

        logger.info(f"Loaded {len(self.samples)} samples from MME-RealWorld-Lite dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        raw_image = Image.open(os.path.join(self.extract_dir, "data", "imgs", sample["Image"]))
        img = apply_image_transform(raw_image, self.scale, self.interpolation, self.position, self.upscaler)
        question = sample["Text"] + "\n" + "\n".join(sample["Answer choices"])
        return {
            "id": sample["Question_id"],
            "image": img,
            "field": self.field,
            "position": self.position,
            "question": question,
            "answer": sample["Ground truth"],
        }
