import torch
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoConfig,
    AutoProcessor,
    MllamaForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoModelForImageTextToText,
    PaliGemmaForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel,
    DynamicCache,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
import inspect
import functools
import logging
import queue
import torch.multiprocessing as mp

from .utils import build_field_images

logger = logging.getLogger(__name__)

FASTVLM_IMAGE_TOKEN_INDEX = -200


def _get_image_token_id(model) -> Optional[int]:
    """Best-effort lookup of the placeholder image token id across VLM config conventions
    (top-level vs. nested text_config, image_token_index vs. image_token_id naming)."""
    for config in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if config is None:
            continue
        for attr in ("image_token_index", "image_token_id"):
            value = getattr(config, attr, None)
            if value is not None:
                return value
    return None


def _pixel_hw_from_tensor(pixel_values) -> Optional[Tuple[int, int]]:
    """(width, height) actually fed to the vision encoder. Only valid when pixel_values is a
    plain image tensor (..., C, H, W); models that pre-flatten into patches (e.g. Qwen) need
    architecture-specific handling instead."""
    if pixel_values is None or pixel_values.dim() < 4:
        return None
    return (int(pixel_values.shape[-1]), int(pixel_values.shape[-2]))


def _vision_token_counts(input_ids: torch.Tensor, image_token_id: Optional[int]) -> List[Optional[int]]:
    """Per-sample count of the image placeholder token in the tokenized prompt. For processors
    that expand '<image>' into one token per vision patch before generate() (Llava, Qwen,
    Pixtral, InternVL, PaliGemma), this count equals the number of vision tokens the LLM
    actually attends to."""
    if image_token_id is None:
        return [None] * input_ids.shape[0]
    return [int((row == image_token_id).sum().item()) for row in input_ids]


def _image_size_fields(original_size: Tuple[int, int], resized_size: Optional[Tuple[int, int]]) -> Dict:
    resized = {"width": resized_size[0], "height": resized_size[1]} if resized_size else None
    return {
        "image_original_size_px": {"width": original_size[0], "height": original_size[1]},
        "image_resized_size_px": resized,
        "image_was_resized": (resized is not None and resized_size != original_size),
    }


def _question_token_count(tokenizer, question: str) -> int:
    """Token count of just the per-sample question text (as passed to the model - already
    includes any benchmark answer-format suffix baked in upstream by _SuffixedDataset),
    independent of chat-template scaffolding and vision placeholder tokens."""
    return len(tokenizer(question, add_special_tokens=False)["input_ids"])


def _output_token_count(tokenizer, text: str) -> int:
    """Token count of the generated answer, computed by re-tokenizing the decoded response text
    rather than slicing the (possibly batch-padded-with-early-stop) generate() output tensor."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0


def _system_prompt_token_counts(
    attention_mask: Optional[torch.Tensor],
    prompt_len: int,
    vision_token_counts: List[Optional[int]],
    question_token_counts: List[int],
    vision_tokens_in_input_ids: bool,
) -> List[Optional[int]]:
    """Per-sample count of fixed chat-template/system scaffolding tokens: whatever's left in the
    real (non-padded) prompt once the vision placeholder tokens actually inlined into input_ids
    (only when vision_tokens_in_input_ids - false for cross-attention/embed-splice models like
    Mllama and FastVLM, where input_ids carries at most one placeholder token per image rather
    than one-per-patch) and the per-sample question text are subtracted out."""
    counts = []
    for i, question_count in enumerate(question_token_counts):
        vision_count = vision_token_counts[i] if vision_tokens_in_input_ids else 0
        if vision_tokens_in_input_ids and vision_count is None:
            counts.append(None)
            continue
        real_len = int(attention_mask[i].sum().item()) if attention_mask is not None else prompt_len
        counts.append(real_len - vision_count - question_count)
    return counts


_NULL_VISION_FIELDS = {
    "num_vision_tokens": None,
    "num_system_prompt_tokens": None,
    "num_question_tokens": None,
    "num_output_tokens": None,
    "image_original_size_px": None,
    "image_resized_size_px": None,
    "image_was_resized": None,
}


def _legacy_kv_from_cache(cache) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
    """Extract Molmo's expected `((key, value), ...)` per-layer tuple format out of a modern
    transformers Cache object (e.g. DynamicCache). None if the cache is empty/uninitialized,
    matching what Molmo's own code expects to see on the first (prefill) step."""
    layers = getattr(cache, "layers", None)
    if not layers:
        return None
    return tuple((layer.keys, layer.values) for layer in layers)


def _cache_from_legacy_kv(legacy_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]) -> DynamicCache:
    """Inverse of _legacy_kv_from_cache: rebuild a DynamicCache from Molmo's legacy per-layer
    tuple output, so the surrounding transformers generate() loop keeps getting the modern Cache
    API it expects from one decoding step to the next."""
    cache = DynamicCache()
    if legacy_kv is not None:
        for i, (key, value) in enumerate(legacy_kv):
            cache.update(key, value, i)
    return cache


def _patch_molmo_remote_code(model_name: str) -> None:
    """allenai/Molmo-7B-D-0924's trust_remote_code modeling script (modeling_molmo.py) predates
    three transformers API changes it now runs against:
      1. `all_tied_weights_keys` is expected as an instance attribute, set inside
         PreTrainedModel.post_init() - Molmo's __init__ never calls post_init(), so any internal
         code that reads it directly raises AttributeError before from_pretrained returns.
      2. PreTrainedModel.tie_weights() gained `missing_keys`/`recompute_mapping` kwargs that
         transformers now always passes when calling it during weight loading; Molmo's override
         keeps the old no-argument signature, so that call raises TypeError.
      3. generate() now always passes a modern Cache object (e.g. DynamicCache, even when empty)
         as `past_key_values`, rather than `None` pre-first-step and a legacy
         `((key, value), ...)` per-layer tuple thereafter. Molmo's own forward does
         `past_key_values[0][0].size(-2)`, which isn't valid on a Cache object, and the
         to/from-legacy-cache conversion helpers transformers used to provide automatically for
         exactly this situation have since been removed.
    Patches around all three without touching Molmo's own code or downgrading transformers for
    every other worker."""
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    class_ref = getattr(config, "auto_map", {}).get("AutoModelForCausalLM")
    if class_ref is None:
        return
    molmo_cls = get_class_from_dynamic_module(class_ref, model_name)

    if "missing_keys" not in inspect.signature(molmo_cls.tie_weights).parameters:
        original_tie_weights = molmo_cls.tie_weights

        @functools.wraps(original_tie_weights)
        def _tie_weights(self, *args, **kwargs):
            return original_tie_weights(self)

        molmo_cls.tie_weights = _tie_weights

    if not getattr(molmo_cls.forward, "_waldo_cache_bridge", False):
        original_forward = molmo_cls.forward

        # transformers' generate() validates model_kwargs (attention_mask, images, etc.) by
        # inspecting self.forward's signature (falling back to it since Molmo's own
        # prepare_inputs_for_generation accepts **kwargs). functools.wraps preserves that
        # signature via __wrapped__ so inspect.signature() sees Molmo's real parameter list
        # instead of this wrapper's generic *args/**kwargs.
        @functools.wraps(original_forward)
        def _forward(self, *args, past_key_values=None, **kwargs):
            cache_obj = None
            if past_key_values is not None and not isinstance(past_key_values, tuple):
                cache_obj = past_key_values
                past_key_values = _legacy_kv_from_cache(cache_obj)

            outputs = original_forward(self, *args, past_key_values=past_key_values, **kwargs)

            if cache_obj is not None and outputs.past_key_values is not None:
                outputs.past_key_values = _cache_from_legacy_kv(outputs.past_key_values)

            return outputs

        _forward._waldo_cache_bridge = True
        molmo_cls.forward = _forward


_EXPERIMENT_IMAGE_KEYS = ("field", "resized", "bicubic")


class BaseVLMWorker:
    """Shared queue loop + field/resized/bicubic experiment orchestration.

    Subclasses implement ``load_model()`` (set ``self.model``/``self.processor``
    or equivalent) and ``process_batch(samples)`` - run one batched forward
    pass over ``samples`` (each a dict with at least "id"/"question"/"image"/
    "answer") and return one result dict per sample, in the same order. That
    part is completely model-specific (batching, prompting, decoding) and is
    untouched by the experiment logic below.

    For samples that carry a "field" value (main_field.py's runs; ``field=0``
    counts - it's the meaningful 0%-border baseline, not "no experiment"),
    ``process_batch`` is invoked three times per collated batch, once per
    image variant produced by ``build_field_images``:
      1. field   - original embedded in a larger blank canvas at the
                   requested percent,
      2. resized - plain original (no border) downscaled by the same
                   coefficient the field image experiences during model
                   preprocessing (isolates content downscaling from the
                   added-border effect),
      3. bicubic - ``resized`` scaled back up to the original resolution via
                   bicubic interpolation (same resolution as the input, but
                   with the detail lost during the downscale).
    The three per-variant results are merged into a single row per sample:
    the "field" variant's keys are kept as-is (``generated_answer``, etc.,
    for backwards compatibility with the plain single-experiment schema),
    and the "resized"/"bicubic" variants are merged in under a matching
    ``_resized``/``_bicubic`` suffix (``generated_answer_resized``, etc.).

    Samples with no "field" key set (main_scale.py / main_interpolation.py,
    which already apply their own transform inline in the dataset) skip the
    3x split entirely and go through ``process_batch`` once, unchanged.
    """

    def __init__(self, gpu_id: int, model_name: str, work_queue: mp.Queue,
                 result_queue: mp.Queue, stop_event: mp.Event, batch_size: int = 8):
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.batch_size = batch_size
        self.device = f"cuda:{gpu_id}"

    def load_model(self):
        raise NotImplementedError

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        raise NotImplementedError

    def _field_image_processor(self):
        """Image processor ``build_field_images`` should probe to figure out
        how this model resizes images. Defaults to ``self.processor``
        (an ``AutoProcessor``); overridden by workers that don't have one."""
        return self.processor

    def process_experiment_batch(self, samples: List[Dict]) -> List[Dict]:
        """Split ``samples`` into the field-experiment group (run 3x) and the
        plain group (run once), preserving input order in the output."""
        plain = [s for s in samples if s.get("field") is None]
        field_samples = [s for s in samples if s.get("field") is not None]

        results_by_id = {r["id"]: r for r in (self.process_batch(plain) if plain else [])}

        if field_samples:
            variant_batches = {name: [] for name in _EXPERIMENT_IMAGE_KEYS}
            size_infos = []
            for sample in field_samples:
                field_img, resized_img, bicubic_img, info = build_field_images(
                    self._field_image_processor(),
                    self.model_name,
                    sample["image"],
                    sample.get("field"),
                    sample.get("position") or "center",
                )
                images = {"field": field_img, "resized": resized_img, "bicubic": bicubic_img}
                for name in _EXPERIMENT_IMAGE_KEYS:
                    variant_batches[name].append({**sample, "image": images[name]})
                size_infos.append(info)

            per_variant_results = {
                name: self.process_batch(variant_batches[name]) for name in _EXPERIMENT_IMAGE_KEYS
            }

            for i, sample in enumerate(field_samples):
                merged = dict(per_variant_results["field"][i])
                for name in ("resized", "bicubic"):
                    for key, value in per_variant_results[name][i].items():
                        if key in ("id", "question", "ground_truth", "image_original_size_px"):
                            # "image_original_size_px" here is just that variant's own input
                            # image size (== image_resized_experiment_size_px for "resized",
                            # == image_original_size_px for "bicubic" by construction) - always
                            # a duplicate of a field set below by size_infos[i], under a name
                            # that would misleadingly suggest it's the true source image size.
                            continue
                        merged[f"{key}_{name}"] = value
                merged.update(size_infos[i])
                results_by_id[merged["id"]] = merged

        return [results_by_id[s["id"]] for s in samples]

    def run(self):
        """Main worker loop."""
        try:
            self.load_model()

            processed_count = 0
            done = False
            while not done and not self.stop_event.is_set():
                try:
                    item = self.work_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is None:
                    break

                batch = [item]
                while len(batch) < self.batch_size:
                    try:
                        item = self.work_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        done = True
                        break
                    batch.append(item)

                for result in self.process_experiment_batch(batch):
                    self.result_queue.put(result)

                processed_count += len(batch)
                logger.info(f"GPU {self.gpu_id}: Processed {processed_count} samples")

            logger.info(f"Worker on GPU {self.gpu_id} finished. Processed {processed_count} samples")

        except Exception as e:
            logger.error(f"Failed to initialize worker on GPU {self.gpu_id}: {str(e)}")


class LLavaWorker(BaseVLMWorker):
    """Worker class for individual GPU inference."""

    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")

        torch.cuda.set_device(self.gpu_id)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map={"": self.device},
            trust_remote_code=True,
            attn_implementation="sdpa"
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"

        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """Process a batch of samples in a single forward pass."""
        try:
            prompts = []
            images = []
            original_sizes = []
            for sample in samples:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": sample["question"]},
                        ]
                    }
                ]
                prompts.append(self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                ))
                images.append(sample["image"])
                original_sizes.append(sample["image"].size)

            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device, dtype=self.model.dtype)

            prompt_len = inputs["input_ids"].shape[1]

            image_token_id = _get_image_token_id(self.model)
            resized_hw = _pixel_hw_from_tensor(inputs.get("pixel_values"))
            vision_token_counts = _vision_token_counts(inputs["input_ids"], image_token_id)
            question_token_counts = [
                _question_token_count(self.processor.tokenizer, sample["question"]) for sample in samples
            ]
            system_prompt_token_counts = _system_prompt_token_counts(
                inputs.get("attention_mask"), prompt_len, vision_token_counts, question_token_counts,
                vision_tokens_in_input_ids=True,
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

            responses = self.processor.batch_decode(
                outputs[:, prompt_len:],
                skip_special_tokens=True
            )

            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": response.strip(),
                    "ground_truth": sample.get("answer", ""),
                    "num_vision_tokens": vision_token_counts[i],
                    "num_system_prompt_tokens": system_prompt_token_counts[i],
                    "num_question_tokens": question_token_counts[i],
                    "num_output_tokens": _output_token_count(self.processor.tokenizer, response.strip()),
                    **_image_size_fields(original_sizes[i], resized_hw),
                }
                for i, (sample, response) in enumerate(zip(samples, responses))
            ]

        except Exception as e:
            logger.error(f"Error processing batch on GPU {self.gpu_id}: {str(e)}")
            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": f"ERROR: {str(e)}",
                    "ground_truth": sample.get("answer", ""),
                    **_NULL_VISION_FIELDS,
                }
                for sample in samples
            ]


class LLamaWorker(BaseVLMWorker):
    """Worker class for individual GPU inference."""

    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")

        torch.cuda.set_device(self.gpu_id)

        # Load LLaMA 3.2 Vision model (Mllama)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map={"": self.device},
            trust_remote_code=True,
            attn_implementation="sdpa"
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"

        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """Process a batch of samples in a single forward pass."""
        try:
            prompts = []
            images = []
            original_sizes = []
            for sample in samples:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": sample["question"]},
                        ]
                    }
                ]
                prompts.append(self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                ))
                # Mllama's processor treats a flat `images` list as all belonging to one
                # batch item (expecting as many <image> tokens as images in that single
                # text), not as one image per sample - so each sample's image(s) must be
                # nested in their own sub-list, matching `text`'s batch dimension.
                images.append([sample["image"]])
                original_sizes.append(sample["image"].size)

            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device, dtype=self.model.dtype)

            prompt_len = inputs["input_ids"].shape[1]

            # Mllama attends to images via cross-attention rather than inlining one token per
            # vision patch into input_ids, so the placeholder-token-count trick used by the
            # other workers only ever yields ~1 per image here. The real vision token count
            # has to come from the tile layout instead: aspect_ratio_mask marks which of the
            # (padded) tiles per image are real, and each real tile contributes a fixed number
            # of patches (+1 cls token) determined by the vision encoder's config.
            resized_hw = None
            vision_token_counts = [None] * len(samples)
            try:
                pixel_values = inputs.get("pixel_values")
                aspect_ratio_mask = inputs.get("aspect_ratio_mask")
                vision_config = self.model.config.vision_config
                patches_per_tile = (vision_config.image_size // vision_config.patch_size) ** 2 + 1
                if pixel_values is not None:
                    resized_hw = (int(pixel_values.shape[-1]), int(pixel_values.shape[-2]))
                if aspect_ratio_mask is not None:
                    vision_token_counts = [
                        int(aspect_ratio_mask[i, 0].sum().item()) * patches_per_tile
                        for i in range(len(samples))
                    ]
            except Exception as vision_meta_err:
                logger.warning(f"Could not derive Mllama vision token count: {vision_meta_err}")

            question_token_counts = [
                _question_token_count(self.processor.tokenizer, sample["question"]) for sample in samples
            ]
            system_prompt_token_counts = _system_prompt_token_counts(
                inputs.get("attention_mask"), prompt_len, vision_token_counts, question_token_counts,
                vision_tokens_in_input_ids=False,
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

            responses = self.processor.batch_decode(
                outputs[:, prompt_len:],
                skip_special_tokens=True
            )

            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": response.strip(),
                    "ground_truth": sample.get("answer", ""),
                    "num_vision_tokens": vision_token_counts[i],
                    "num_system_prompt_tokens": system_prompt_token_counts[i],
                    "num_question_tokens": question_token_counts[i],
                    "num_output_tokens": _output_token_count(self.processor.tokenizer, response.strip()),
                    **_image_size_fields(original_sizes[i], resized_hw),
                }
                for i, (sample, response) in enumerate(zip(samples, responses))
            ]

        except Exception as e:
            logger.error(f"Error processing batch on GPU {self.gpu_id}: {str(e)}")
            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": f"ERROR: {str(e)}",
                    "ground_truth": sample.get("answer", ""),
                    **_NULL_VISION_FIELDS,
                }
                for sample in samples
            ]


class QwenWorker(BaseVLMWorker):
    """Worker class for individual GPU inference."""

    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")

        torch.cuda.set_device(self.gpu_id)

        # Load Qwen Vision Language model (Qwen2-VL / Qwen2.5-VL / Qwen3-VL)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map={"": self.device},
            attn_implementation="sdpa",
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"

        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """Process a batch of samples in a single forward pass."""
        try:
            prompts = []
            images = []
            original_sizes = []
            for sample in samples:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": sample["question"]},
                        ]
                    }
                ]
                prompts.append(self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                ))
                images.append(sample["image"])
                original_sizes.append(sample["image"].size)

            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device, dtype=self.model.dtype)

            prompt_len = inputs["input_ids"].shape[1]

            image_token_id = _get_image_token_id(self.model)
            vision_token_counts = _vision_token_counts(inputs["input_ids"], image_token_id)

            # Qwen's image processor patchifies each image into a flat (num_patches, patch_dim)
            # tensor rather than a (batch, C, H, W) image, so the resolution actually fed to the
            # vision tower has to be recovered from image_grid_thw * patch_size instead of the
            # pixel_values tensor shape.
            resized_sizes = [None] * len(samples)
            image_grid_thw = inputs.get("image_grid_thw")
            patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", None)
            if image_grid_thw is not None and patch_size is not None:
                try:
                    resized_sizes = [
                        (int(image_grid_thw[i][2]) * patch_size, int(image_grid_thw[i][1]) * patch_size)
                        for i in range(len(samples))
                    ]
                except Exception as vision_meta_err:
                    logger.warning(f"Could not derive Qwen resized image size: {vision_meta_err}")

            question_token_counts = [
                _question_token_count(self.processor.tokenizer, sample["question"]) for sample in samples
            ]
            system_prompt_token_counts = _system_prompt_token_counts(
                inputs.get("attention_mask"), prompt_len, vision_token_counts, question_token_counts,
                vision_tokens_in_input_ids=True,
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )

            responses = self.processor.batch_decode(
                outputs[:, prompt_len:],
                skip_special_tokens=True
            )

            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": response.strip(),
                    "ground_truth": sample.get("answer", ""),
                    "input_tokens": inputs["input_ids"].shape,
                    "num_vision_tokens": vision_token_counts[i],
                    "num_system_prompt_tokens": system_prompt_token_counts[i],
                    "num_question_tokens": question_token_counts[i],
                    "num_output_tokens": _output_token_count(self.processor.tokenizer, response.strip()),
                    **_image_size_fields(original_sizes[i], resized_sizes[i]),
                }
                for i, (sample, response) in enumerate(zip(samples, responses))
            ]

        except Exception as e:
            logger.error(f"Error processing batch on GPU {self.gpu_id}: {str(e)}")
            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": f"ERROR: {str(e)}",
                    "ground_truth": sample.get("answer", ""),
                    **_NULL_VISION_FIELDS,
                }
                for sample in samples
            ]


class PixtralWorker(BaseVLMWorker):
    """Worker class for individual GPU inference."""

    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")

        torch.cuda.set_device(self.gpu_id)

        # Load Pixtral Vision Language model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            device_map={"": self.device},
            dtype=torch.float16,
            attn_implementation="sdpa"
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            dtype=torch.float16
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"

        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """Process a batch of samples in a single forward pass."""
        try:
            prompts = []
            images = []
            original_sizes = []
            for sample in samples:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": sample["question"]},
                        ]
                    }
                ]
                prompts.append(self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                ))
                images.append(sample["image"])
                original_sizes.append(sample["image"].size)

            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device, dtype=self.model.dtype)

            prompt_len = inputs["input_ids"].shape[1]

            image_token_id = _get_image_token_id(self.model)
            resized_hw = _pixel_hw_from_tensor(inputs.get("pixel_values"))
            vision_token_counts = _vision_token_counts(inputs["input_ids"], image_token_id)
            question_token_counts = [
                _question_token_count(self.processor.tokenizer, sample["question"]) for sample in samples
            ]
            system_prompt_token_counts = _system_prompt_token_counts(
                inputs.get("attention_mask"), prompt_len, vision_token_counts, question_token_counts,
                vision_tokens_in_input_ids=True,
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )

            responses = self.processor.batch_decode(
                outputs[:, prompt_len:],
                skip_special_tokens=True
            )

            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": response.strip(),
                    "ground_truth": sample.get("answer", ""),
                    "num_vision_tokens": vision_token_counts[i],
                    "num_system_prompt_tokens": system_prompt_token_counts[i],
                    "num_question_tokens": question_token_counts[i],
                    "num_output_tokens": _output_token_count(self.processor.tokenizer, response.strip()),
                    **_image_size_fields(original_sizes[i], resized_hw),
                }
                for i, (sample, response) in enumerate(zip(samples, responses))
            ]

        except Exception as e:
            logger.error(f"Error processing batch on GPU {self.gpu_id}: {str(e)}")
            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": f"ERROR: {str(e)}",
                    "ground_truth": sample.get("answer", ""),
                    **_NULL_VISION_FIELDS,
                }
                for sample in samples
            ]


class InternVLWorker(BaseVLMWorker):
    """Worker class for individual GPU inference."""

    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")

        torch.cuda.set_device(self.gpu_id)

        # Native transformers support, e.g. OpenGVLab/InternVL3-*-hf checkpoints
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map={"": self.device},
            attn_implementation="sdpa",
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"

        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """Process a batch of samples in a single forward pass."""
        try:
            prompts = []
            images = []
            original_sizes = []
            for sample in samples:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": sample["question"]},
                        ]
                    }
                ]
                prompts.append(self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                ))
                images.append(sample["image"])
                original_sizes.append(sample["image"].size)

            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device, dtype=self.model.dtype)

            prompt_len = inputs["input_ids"].shape[1]

            image_token_id = _get_image_token_id(self.model)
            # InternVL dynamically tiles each image into a variable number of fixed-size crops;
            # pixel_values shape gives that fixed per-tile resolution, not a single whole-image
            # size, and the placeholder-token count already reflects however many tiles the
            # processor chose for each sample.
            resized_hw = _pixel_hw_from_tensor(inputs.get("pixel_values"))
            vision_token_counts = _vision_token_counts(inputs["input_ids"], image_token_id)
            question_token_counts = [
                _question_token_count(self.processor.tokenizer, sample["question"]) for sample in samples
            ]
            system_prompt_token_counts = _system_prompt_token_counts(
                inputs.get("attention_mask"), prompt_len, vision_token_counts, question_token_counts,
                vision_tokens_in_input_ids=True,
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )

            responses = self.processor.batch_decode(
                outputs[:, prompt_len:],
                skip_special_tokens=True
            )

            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": response.strip(),
                    "ground_truth": sample.get("answer", ""),
                    "num_vision_tokens": vision_token_counts[i],
                    "num_system_prompt_tokens": system_prompt_token_counts[i],
                    "num_question_tokens": question_token_counts[i],
                    "num_output_tokens": _output_token_count(self.processor.tokenizer, response.strip()),
                    **_image_size_fields(original_sizes[i], resized_hw),
                }
                for i, (sample, response) in enumerate(zip(samples, responses))
            ]

        except Exception as e:
            logger.error(f"Error processing batch on GPU {self.gpu_id}: {str(e)}")
            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": f"ERROR: {str(e)}",
                    "ground_truth": sample.get("answer", ""),
                    **_NULL_VISION_FIELDS,
                }
                for sample in samples
            ]


class PaliGemmaWorker(BaseVLMWorker):
    """Worker class for individual GPU inference.

    Unlike the other workers, PaliGemma has no chat template - the processor takes a
    plain prompt string per sample and inserts the image tokens itself.
    """

    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")

        torch.cuda.set_device(self.gpu_id)

        # PaliGemma2 checkpoints are gated - requires HF_TOKEN with the license accepted
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map={"": self.device},
            attn_implementation="sdpa",
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"

        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """Process a batch of samples in a single forward pass."""
        try:
            prompts = [sample["question"] for sample in samples]
            images = [sample["image"] for sample in samples]
            original_sizes = [img.size for img in images]

            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device, dtype=self.model.dtype)

            prompt_len = inputs["input_ids"].shape[1]

            image_token_id = _get_image_token_id(self.model)
            resized_hw = _pixel_hw_from_tensor(inputs.get("pixel_values"))
            vision_token_counts = _vision_token_counts(inputs["input_ids"], image_token_id)
            question_token_counts = [
                _question_token_count(self.processor.tokenizer, sample["question"]) for sample in samples
            ]
            system_prompt_token_counts = _system_prompt_token_counts(
                inputs.get("attention_mask"), prompt_len, vision_token_counts, question_token_counts,
                vision_tokens_in_input_ids=True,
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )

            responses = self.processor.batch_decode(
                outputs[:, prompt_len:],
                skip_special_tokens=True
            )

            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": response.strip(),
                    "ground_truth": sample.get("answer", ""),
                    "num_vision_tokens": vision_token_counts[i],
                    "num_system_prompt_tokens": system_prompt_token_counts[i],
                    "num_question_tokens": question_token_counts[i],
                    "num_output_tokens": _output_token_count(self.processor.tokenizer, response.strip()),
                    **_image_size_fields(original_sizes[i], resized_hw),
                }
                for i, (sample, response) in enumerate(zip(samples, responses))
            ]

        except Exception as e:
            logger.error(f"Error processing batch on GPU {self.gpu_id}: {str(e)}")
            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": f"ERROR: {str(e)}",
                    "ground_truth": sample.get("answer", ""),
                    **_NULL_VISION_FIELDS,
                }
                for sample in samples
            ]


class FastVLMWorker(BaseVLMWorker):
    """Worker class for individual GPU inference.

    Apple FastVLM uses trust_remote_code custom modeling (LlavaQwen2ForCausalLM), not
    a standard AutoProcessor. The image placeholder has to be spliced into input_ids
    manually as a literal token id (FASTVLM_IMAGE_TOKEN_INDEX), and images go through
    the model's own vision tower image processor rather than AutoProcessor. The model's
    overridden generate() converts everything to inputs_embeds internally, so - unlike
    the other workers - its output contains ONLY the newly generated tokens, with no
    prompt prefix to slice off.
    """

    def _field_image_processor(self):
        return self.model.get_vision_tower().image_processor

    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")

        torch.cuda.set_device(self.gpu_id)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map={"": self.device},
            trust_remote_code=True,
        )

        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """Process a batch of samples in a single forward pass."""
        try:
            input_id_rows = []
            images = []
            original_sizes = []
            for sample in samples:
                messages = [{"role": "user", "content": f"<image>\n{sample['question']}"}]
                rendered = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                pre, post = rendered.split("<image>", 1)
                pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids[0]
                post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids[0]
                image_tok = torch.tensor([FASTVLM_IMAGE_TOKEN_INDEX], dtype=pre_ids.dtype)
                input_id_rows.append(torch.cat([pre_ids, image_tok, post_ids], dim=0))
                original_sizes.append(sample["image"].size)
                images.append(sample["image"].convert("RGB"))

            pad_id = self.tokenizer.pad_token_id
            max_len = max(row.shape[0] for row in input_id_rows)
            input_ids = torch.full((len(input_id_rows), max_len), pad_id, dtype=input_id_rows[0].dtype)
            attention_mask = torch.zeros((len(input_id_rows), max_len), dtype=torch.long)
            for i, row in enumerate(input_id_rows):
                input_ids[i, :row.shape[0]] = row
                attention_mask[i, :row.shape[0]] = 1
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            vision_tower = self.model.get_vision_tower()
            pixel_values = vision_tower.image_processor(images=images, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(self.device, dtype=self.model.dtype)

            resized_hw = _pixel_hw_from_tensor(pixel_values)

            # FastVLM splices a single placeholder token into input_ids and expands it into
            # per-patch embeddings inside generate() itself, so the vision token count isn't
            # visible in input_ids like it is for the AutoProcessor-based workers. The only way
            # to read it off is to run the vision tower once more, standalone, on the same
            # pixel_values and check its output sequence length.
            vision_token_count = None
            try:
                with torch.inference_mode():
                    image_features = vision_tower(pixel_values)
                if hasattr(image_features, "last_hidden_state"):
                    image_features = image_features.last_hidden_state
                vision_token_count = int(image_features.shape[1])
            except Exception as vision_meta_err:
                logger.warning(f"Could not derive FastVLM vision token count: {vision_meta_err}")
            vision_token_counts = [vision_token_count] * len(samples)
            question_token_counts = [
                _question_token_count(self.tokenizer, sample["question"]) for sample in samples
            ]
            system_prompt_token_counts = _system_prompt_token_counts(
                attention_mask, max_len, vision_token_counts, question_token_counts,
                vision_tokens_in_input_ids=False,
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=pixel_values,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=pad_id,
                )

            # generate() here returns inputs_embeds-derived output, i.e. only the new
            # tokens - no prompt prefix to slice off, unlike the other workers.
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": response.strip(),
                    "ground_truth": sample.get("answer", ""),
                    "num_vision_tokens": vision_token_counts[i],
                    "num_system_prompt_tokens": system_prompt_token_counts[i],
                    "num_question_tokens": question_token_counts[i],
                    "num_output_tokens": _output_token_count(self.tokenizer, response.strip()),
                    **_image_size_fields(original_sizes[i], resized_hw),
                }
                for i, (sample, response) in enumerate(zip(samples, responses))
            ]

        except Exception as e:
            logger.error(f"Error processing batch on GPU {self.gpu_id}: {str(e)}")
            return [
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": f"ERROR: {str(e)}",
                    "ground_truth": sample.get("answer", ""),
                    **_NULL_VISION_FIELDS,
                }
                for sample in samples
            ]


class MolmoWorker(BaseVLMWorker):
    """Worker class for individual GPU inference.

    Molmo's processor.process() only supports one sample at a time (a variable
    number of image "crops" per image and no batching/collation helper is shipped in
    the repo), so unlike the other workers this one processes each sample in the
    batch individually rather than in a single collated forward pass.
    """

    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")

        torch.cuda.set_device(self.gpu_id)

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        _patch_molmo_remote_code(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
        )

        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """Process samples one at a time - see class docstring."""
        results = []
        for sample in samples:
            try:
                original_size = sample["image"].size

                inputs = self.processor.process(
                    images=[sample["image"].convert("RGB")],
                    text=sample["question"],
                )
                inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}

                # Molmo splits each image into a variable number of ~336px crops, so there is
                # no single "resized to WxH" answer the way there is for single-crop models -
                # leave that field null rather than guess. image_input_idx does reliably tell
                # us, per official Molmo processor semantics, which positions in input_ids are
                # real image-patch tokens (>=0) vs. padding (-1), so vision token count is
                # still derivable from it.
                vision_token_count = None
                try:
                    image_input_idx = inputs.get("image_input_idx")
                    if image_input_idx is not None:
                        vision_token_count = int((image_input_idx >= 0).sum().item())
                except Exception as vision_meta_err:
                    logger.warning(f"Could not derive Molmo vision token count: {vision_meta_err}")

                question_token_count = _question_token_count(self.processor.tokenizer, sample["question"])
                system_prompt_token_count = _system_prompt_token_counts(
                    None, inputs["input_ids"].size(1), [vision_token_count], [question_token_count],
                    vision_tokens_in_input_ids=True,
                )[0]

                with torch.inference_mode():
                    output = self.model.generate_from_batch(
                        inputs,
                        GenerationConfig(
                            max_new_tokens=512,
                            do_sample=False,
                            stop_strings="<|endoftext|>",
                            # This transformers version defaults GenerationConfig.use_cache to
                            # None (resolved later inside generate()), but Molmo's remote code
                            # (modeling_molmo.py) does `assert generation_config.use_cache` up
                            # front in generate_from_batch(), before that resolution happens -
                            # so it must be set explicitly here.
                            use_cache=True,
                        ),
                        tokenizer=self.processor.tokenizer,
                    )

                generated_tokens = output[0, inputs["input_ids"].size(1):]
                response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                results.append({
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": response.strip(),
                    "ground_truth": sample.get("answer", ""),
                    "num_vision_tokens": vision_token_count,
                    "num_system_prompt_tokens": system_prompt_token_count,
                    "num_question_tokens": question_token_count,
                    "num_output_tokens": _output_token_count(self.processor.tokenizer, response.strip()),
                    **_image_size_fields(original_size, None),
                })

            except Exception as e:
                logger.error(
                    f"Error processing sample {sample['id']} on GPU {self.gpu_id}: "
                    f"{type(e).__name__}: {e}",
                    exc_info=True,
                )
                results.append({
                    "id": sample["id"],
                    "question": sample["question"],
                    "generated_answer": f"ERROR: {type(e).__name__}: {e}",
                    "ground_truth": sample.get("answer", ""),
                    **_NULL_VISION_FIELDS,
                })

        return results
