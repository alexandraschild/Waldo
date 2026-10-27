from datasets import load_dataset
import logging
from torch.utils.data import Dataset
import random
import base64
import io
import json
import os
import zipfile
from collections import Counter

from PIL import Image
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class MMVetDataset(Dataset):
    """Dataset class for MMVet benchmark."""

    def __init__(self, dataset_name: str = "whyu/mm-vet", field = None, scale = None, interpolation = None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name)['test']
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        self.field = field
        self.position = position
        logger.info(f"Loaded {len(self.samples)} samples from MMVet dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["id"],
            "image": sample["image"],
            "question": sample["question"],
            "answer": sample["answer"],
            "field": self.field,
            "position": self.position,
        }


class MMEvalDataset(Dataset):
    """Dataset class for MMEval benchmark."""

    def __init__(self, dataset_name: str = "darkyarding/MME", field = None, scale = None, interpolation = None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name)['test']
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        self.field = field
        self.position = position
        logger.info(f"Loaded {len(self.samples)} samples from MMEval dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["question_id"],
            "image": sample["image"],
            "question": sample["question"],
            "answer": sample["answer"],
            "field": self.field,
            "position": self.position,
        }


class MathVerseDataset(Dataset):
    """Dataset class for MathVerse benchmark."""

    def __init__(self, dataset_name: str = "AI4Math/MathVerse", field = None, scale = None, interpolation = None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name, 'testmini')['testmini']
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        self.field = field
        self.position = position
        logger.info(f"Loaded {len(self.samples)} samples from MathVerse dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["sample_index"],
            "image": sample["image"],
            "question": sample["question"],
            "answer": sample["answer"],
            "field": self.field,
            "position": self.position,
        }


class MMStar(Dataset):
    """Dataset class for MMStar benchmark."""

    def __init__(self, dataset_name: str = "Lin-Chen/MMStar", field = None, scale = None, interpolation = None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name)['val']
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        self.field = field
        self.position = position
        logger.info(f"Loaded {len(self.samples)} samples from MMStar dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["index"],
            "image": sample["image"],
            "question": sample["question"],
            "answer": sample["answer"],
            "field": self.field,
            "position": self.position,
        }


class CharXiv(Dataset):
    """Dataset class for CharXiv benchmark."""

    def __init__(self, dataset_name: str = "princeton-nlp/CharXiv", field = None, scale = None, interpolation = None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name)['validation']
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        self.field = field
        self.position = position
        logger.info(f"Loaded {len(self.samples)} samples from CharXiv dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["original_figure_path"],
            "image": sample["image"],
            "question": sample["reasoning_q"],
            "answer": sample["reasoning_a"],
            "field": self.field,
            "position": self.position,
        }

class VQAv2Dataset(Dataset):
    """Dataset class for VQAv2 benchmark.

    Uses lmms-lab/VQAv2 (parquet-backed) instead of HuggingFaceM4/VQAv2 - the latter
    only ships a legacy Python loading script, which datasets>=4.x no longer supports.
    """

    def __init__(self, dataset_name: str = "lmms-lab/VQAv2", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name, split='validation')
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        logger.info(f"Loaded {len(self.samples)} samples from VQAv2 dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["question_id"],
            "image": sample["image"],
            "question": sample["question"],
            "answer": sample["multiple_choice_answer"],
        }


class GQADataset(Dataset):
    """Dataset class for GQA benchmark.

    Images and questions live in separate configs of the same repo and have to be
    joined on imageId.
    """

    def __init__(self, dataset_name: str = "lmms-lab/GQA", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        instructions = load_dataset(dataset_name, 'testdev_balanced_instructions', split='testdev')
        images = load_dataset(dataset_name, 'testdev_balanced_images', split='testdev')
        self.image_by_id = {row["id"]: row["image"] for row in images}
        indices = list(range(len(instructions)))
        sampled_indices = random.sample(indices, min(sample_size, len(instructions)))
        self.samples = instructions.select(sampled_indices)
        logger.info(f"Loaded {len(self.samples)} samples from GQA dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["id"],
            "image": self.image_by_id[sample["imageId"]],
            "question": sample["question"],
            "answer": sample["answer"],
        }


class CVBenchDataset(Dataset):
    """Dataset class for CV-Bench benchmark (2D subset only)."""

    def __init__(self, dataset_name: str = "nyu-visionx/CV-Bench", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name, data_files={'test': 'test_2d.parquet'}, split='test')
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        logger.info(f"Loaded {len(self.samples)} samples from CV-Bench (2D) dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["idx"],
            "image": sample["image"],
            "question": sample["prompt"],
            "answer": sample["answer"],
        }


class TextVQADataset(Dataset):
    """Dataset class for TextVQA benchmark."""

    def __init__(self, dataset_name: str = "lmms-lab/TextVQA", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name, split='validation')
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        logger.info(f"Loaded {len(self.samples)} samples from TextVQA dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # answers is a list of ~10 annotator strings; take the majority vote
        answer = Counter(sample["answers"]).most_common(1)[0][0]
        return {
            "id": sample["question_id"],
            "image": sample["image"],
            "question": sample["question"],
            "answer": answer,
        }


class ChartQADataset(Dataset):
    """Dataset class for ChartQA benchmark."""

    def __init__(self, dataset_name: str = "lmms-lab/ChartQA", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name, split='test')
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        logger.info(f"Loaded {len(self.samples)} samples from ChartQA dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": idx,
            "image": sample["image"],
            "question": sample["question"],
            "answer": sample["answer"],
        }


class DocVQADataset(Dataset):
    """Dataset class for DocVQA benchmark."""

    def __init__(self, dataset_name: str = "lmms-lab/DocVQA", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name, 'DocVQA', split='validation')
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        logger.info(f"Loaded {len(self.samples)} samples from DocVQA dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["questionId"],
            "image": sample["image"],
            "question": sample["question"],
            "answer": sample["answers"][0],
        }


class InfographicVQADataset(Dataset):
    """Dataset class for InfographicVQA benchmark.

    lmms-lab/InfographicVQA does not exist as its own repo - InfographicVQA is a
    config inside lmms-lab/DocVQA.
    """

    def __init__(self, dataset_name: str = "lmms-lab/DocVQA", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name, 'InfographicVQA', split='validation')
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        logger.info(f"Loaded {len(self.samples)} samples from InfographicVQA dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["questionId"],
            "image": sample["image"],
            "question": sample["question"],
            "answer": sample["answers"][0],
        }


class VStarBenchDataset(Dataset):
    """Dataset class for V*Bench benchmark.

    The image column is only a relative path string, so the repo snapshot has to be
    downloaded once and images opened from disk.
    """

    def __init__(self, dataset_name: str = "craigwu/vstar_bench", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name)['test']
        self.local_dir = snapshot_download(dataset_name, repo_type='dataset')
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        logger.info(f"Loaded {len(self.samples)} samples from V*Bench dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(os.path.join(self.local_dir, sample["image"]))
        return {
            "id": f"{sample['category']}_{sample['question_id']}",
            "image": image,
            "question": sample["text"],
            "answer": sample["label"],
        }


class HRBenchDataset(Dataset):
    """Dataset class for HR-Bench benchmark (8K subset).

    The image column is a base64-encoded JPEG string rather than a decoded image
    feature, and the multiple-choice options live in separate A/B/C/D columns.
    """

    def __init__(self, dataset_name: str = "DreamMr/HR-Bench", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        full_dataset = load_dataset(dataset_name, 'hrbench_version_split', split='hrbench_8k')
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)
        logger.info(f"Loaded {len(self.samples)} samples from HR-Bench (8K) dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(io.BytesIO(base64.b64decode(sample["image"])))
        question = (
            f"{sample['question']}\n"
            f"(A) {sample['A']}\n(B) {sample['B']}\n(C) {sample['C']}\n(D) {sample['D']}"
        )
        return {
            "id": sample["index"],
            "image": image,
            "question": question,
            "answer": sample["answer"],
        }


class MMERealWorldLiteDataset(Dataset):
    """Dataset class for MME-RealWorld-Lite benchmark.

    Ships as a single zip (images + one JSON metadata file) rather than a
    parquet/arrow dataset, so it needs a one-time download + extraction.
    """

    def __init__(self, dataset_name: str = "yifanzhang114/MME-RealWorld-Lite", field=None, scale=None, interpolation=None, position: str = 'center', sample_size: int = 200, seed: int = 42):
        random.seed(seed)
        local_dir = snapshot_download(dataset_name, repo_type='dataset')
        self.extract_dir = os.path.join(local_dir, "_extracted")
        if not os.path.isdir(self.extract_dir):
            with zipfile.ZipFile(os.path.join(local_dir, "data.zip")) as zf:
                zf.extractall(self.extract_dir)

        with open(os.path.join(self.extract_dir, "data", "MME-RealWorld-Lite.json")) as f:
            records = json.load(f)

        indices = list(range(len(records)))
        sampled_indices = random.sample(indices, min(sample_size, len(records)))
        self.samples = [records[i] for i in sampled_indices]
        logger.info(f"Loaded {len(self.samples)} samples from MME-RealWorld-Lite dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(os.path.join(self.extract_dir, "data", "imgs", sample["Image"]))
        question = sample["Text"] + "\n" + "\n".join(sample["Answer choices"])
        return {
            "id": sample["Question_id"],
            "image": image,
            "question": question,
            "answer": sample["Ground truth"],
        }
