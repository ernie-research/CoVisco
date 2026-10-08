"""collator.py - Multimodal collator.

Responsibilities:
  1) Tokenize messages processed by mm_plugin
  2) Adjust the <|image_pad|> count according to the dynamic strategy
  3) Assemble image/video pixel_values into batch tensors
  4) Build labels (set everything except assistant tokens to -100)
"""
from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from .plugin import CoViscoPlugin


# Shared image transform (aligned with the LLaVA-OneVision-2 processor: Resize+CenterCrop+Normalize)
IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMG_STD = (0.26862954, 0.26130258, 0.27577711)


def build_image_transform(image_size: int = 224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMG_MEAN), std=list(IMG_STD)),
    ])


def build_native_image_transform():
    """Transform for native resolution: no Resize/CenterCrop, only ToTensor + Normalize.

    The resize is computed externally by compute_native_size from each image's original dimensions.
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMG_MEAN), std=list(IMG_STD)),
    ])


def compute_native_size(
    width: int,
    height: int,
    patch_size: int,
    min_patches: int = 0,
    max_patches: int = 0,
) -> tuple:
    """Adjust (width, height) to the nearest dimensions divisible by patch_size.

    Round each dimension to the nearest patch count (the nearest divisible size), then
    scale proportionally as needed to constrain the total patch count to
    [min_patches, max_patches]. A value of 0 for min_patches / max_patches disables
    that bound.

    Returns:
        (new_h, new_w), both integer multiples of patch_size and at least patch_size.
    """
    h = max(1, int(round(height / patch_size)))
    w = max(1, int(round(width / patch_size)))

    if max_patches > 0 and h * w > max_patches:
        scale = math.sqrt(max_patches / (h * w))
        h = max(1, int(math.floor(h * scale)))
        w = max(1, int(math.floor(w * scale)))
        # With extreme aspect ratios, one side may be clamped to 1. Proportional scaling
        # can still exceed the budget, so shorten the longer side step by step.
        while h * w > max_patches and max(h, w) > 1:
            if h >= w:
                h -= 1
            else:
                w -= 1
    if min_patches > 0 and h * w < min_patches:
        scale = math.sqrt(min_patches / (h * w))
        h = max(1, int(math.ceil(h * scale)))
        w = max(1, int(math.ceil(w * scale)))

    return h * patch_size, w * patch_size


def _transform_frames_batch(frames: List[Image.Image], transform) -> torch.Tensor:
    """Transform frames in parallel and stack them into (T, 3, H, W).

    Inside a multiprocessing DataLoader worker, a thread pool avoids GIL issues because
    torchvision's PIL resize/crop releases the GIL. It is about 2-4x faster than a serial
    loop when T is large.
    """
    from concurrent.futures import ThreadPoolExecutor
    def _t(f):
        return transform(f.convert("RGB"))
    # For few frames, thread overhead outweighs the benefit; run serially.
    if len(frames) <= 8:
        return torch.stack([_t(f) for f in frames])
    with ThreadPoolExecutor(max_workers=min(4, len(frames))) as pool:
        tensors = list(pool.map(_t, frames))
    return torch.stack(tensors)


@dataclass
class CollatorConfig:
    image_size: int = 224          # Default resolution (shared by images/videos, kept for compatibility)
    image_size_image: int = 0      # Image-specific resolution; 0 reuses image_size
    image_size_video: int = 0      # Video-frame resolution; 0 reuses image_size
    patch_size: int = 14           # ViT patch size, used to compute patches per frame
    segment_t_size: int = 32
    num_query_per_seg: int = 100
    max_seq_length: int = 8192
    ignore_index: int = -100
    # Uniform-sampling path parameters (video only)
    uniform_sample_n: int = 32          # Number of frames after uniform downsampling
    uniform_segment_t_size: int = 8     # segment_t_size for the uniform path
    uniform_train_prob: float = 0.5     # Probability of selecting the uniform-sampling path
    frame_cat_prob: float = 0.0          # Frame-concatenation augmentation probability (uniform path during training only)
    # Actual tokens sent to the ViT per segment in visidx mode (determined during dataset creation).
    # When nonzero, override the static segment_t_size * patches_per_frame estimate for pad counts.
    # Images are unaffected (image always equals patches_per_frame).
    vit_tokens_per_seg_video: int = 0   # 0 = estimate with segment_t_size * ppf (legacy behavior)
    # Dynamic resolution: candidate resolutions, images only. Empty or <=1 item falls back to fixed image_size_image.
    # A resolution is sampled per batch; all images in the batch use the same resolution.
    candidate_resolutions: Optional[List[int]] = None
    # Native resolution: images only (videos always use a fixed resolution). Each image is resized to
    # the nearest dimensions divisible by patch_size while preserving aspect ratio; patch count is constrained to
    # [native_min_patches, native_max_patches].
    # Different image sizes cannot be stacked, so enable this only when the image batch has exactly one sample;
    # otherwise fall back to candidate_resolutions / the fixed-resolution path for compatibility.
    native_resolution: bool = False
    native_min_patches: int = 256    # 256 = 224x224 / 14
    native_max_patches: int = 1296   # 1296 = 504x504 / 14

    def patches_per_frame(self, modality: str = "image") -> int:
        """Number of patches in one image frame = (image_size / patch_size)^2."""
        sz = (self.image_size_image if modality == "image" else self.image_size_video) or self.image_size
        return (sz // self.patch_size) ** 2

    def vit_patches_per_seg(self, modality: str = "image", uniform_mode: bool = False) -> int:
        """Actual tokens sent to the ViT per segment, used to compute pad counts.

        For the regular video path (visidx), return vit_tokens_per_seg_video directly when
        configured (>0), avoiding an oversized pad count from segment_t_size * patches_per_frame (8192).
        """
        ppf = self.patches_per_frame(modality)
        if modality == "image":
            return ppf
        if uniform_mode:
            return self.uniform_segment_t_size * ppf
        # visidx path: prefer the explicitly configured actual token count
        if self.vit_tokens_per_seg_video > 0:
            return self.vit_tokens_per_seg_video
        return self.segment_t_size * ppf


class CoViscoCollator:
    """Multimodal training collator.

    Format of each sample after mm_plugin:

        "images": List[PIL.Image] or None
        "videos": List[List[PIL.Image]] or None
        "token_plan": TokenPlan (provided by the sampler)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        plugin: CoViscoPlugin,
        image_pad_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
        config: CollatorConfig,
        dynamic_token_config=None,
    ):
        self.tokenizer = tokenizer
        self.plugin = plugin
        self.image_pad_token_id = image_pad_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.config = config
        self.dynamic_token_config = dynamic_token_config

        # Build separate transforms for images/videos so each uses the correct resolution.
        # image_size_image/video of 0 means "reuse the shared image_size".
        sz_img = config.image_size_image if config.image_size_image > 0 else config.image_size
        sz_vid = config.image_size_video if config.image_size_video > 0 else config.image_size
        self.video_transform = build_image_transform(sz_vid)

        # Dynamic resolution: prebuild an image transform for every candidate resolution.
        # If candidate_resolutions is unset, build one for fixed sz_img (compatibility).
        self._img_transforms: Dict[int, Any] = {}
        self._dynamic_resolution = bool(config.candidate_resolutions and len(config.candidate_resolutions) > 1)
        if self._dynamic_resolution:
            for sz in config.candidate_resolutions:
                self._img_transforms[sz] = build_image_transform(sz)
        else:
            self._img_transforms[sz_img] = build_image_transform(sz_img)
        # Keep compatibility with code that directly references self.image_transform. With dynamic
        # resolution, sz_img may not be in candidate_resolutions (for example, --image-size-image
        # 336 with candidates 224/448/504), so add it on demand to avoid a construction-time KeyError.
        if sz_img not in self._img_transforms:
            self._img_transforms[sz_img] = build_image_transform(sz_img)
        self.image_transform = self._img_transforms[sz_img]

        # Native-resolution transform (ToTensor+Normalize only; resize each image separately)
        self.native_transform = build_native_image_transform() if config.native_resolution else None

    def _apply_chat_template(self, messages: List[dict]) -> str:
        """Use tokenizer.apply_chat_template to turn messages into a string.

        Pass enable_thinking=False; whether it takes effect depends on the chat template.
        Locate labels from the fully rendered result rather than assuming this option changes
        the template output.
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        return text

    def _adjust_image_pad_count(self, text: str, plan: "TokenPlan", modality: str = "image",
                                uniform_mode: bool = False, patches_per_frame: int = 0) -> str:
        """Adjust the <|image_pad|> count in each vision block of text according to the dynamic plan.

        The plugin inserts num_query_per_seg pads per segment (the query_only default). Replace
        the count in each vision block with the value required by the plan. Image and video pad
        counts differ for vit_only, so modality is required. query_and_vit also needs uniform_mode
        to use the correct segment patch count.
        patches_per_frame: Patches per frame after dynamic-resolution sampling; 0 uses the config default.
        """
        vs = self.plugin.vision_start_token
        ve = self.plugin.vision_end_token
        pad = self.plugin.image_token
        new_pad_count = self._expected_pad_count(
            plan, modality=modality, uniform_mode=uniform_mode, patches_per_frame=patches_per_frame
        )
        new_block = vs + pad * new_pad_count + ve

        # Use string splitting instead of regex so special characters (|) are not
        # misinterpreted as alternation operators.
        parts = text.split(vs)
        # parts[0] is the content before vs; each parts[1:] contains a vision block ending
        # at ve followed by the text after ve.
        result = [parts[0]]
        n_visual_blocks = len(parts) - 1
        if n_visual_blocks > 1:
            # When a sample contains multiple vision blocks, apply the same pad count to all
            # blocks (assuming a single modality). If mixed-modality placeholders are detected
            # in a future extension, handle them separately; for now use a defensive assertion.
            n_image_ph = text.count(self.plugin.vision_start_token)
            if n_image_ph != n_visual_blocks:
                raise ValueError(
                    f"_adjust_image_pad_count: vision block count mismatch "
                    f"(split={n_visual_blocks}, vision_start count={n_image_ph}). "
                    "Mixed-modality samples (image + video in one turn) are not supported."
                )
        for part in parts[1:]:
            # part has the form: "<|image_pad|>...<|vision_end|>text after the block"
            ve_idx = part.find(ve)
            if ve_idx == -1:
                # Malformed block: no vision_end was found, so preserve it unchanged
                result.append(vs + part)
            else:
                # Replace the entire vision block with the new block
                after = part[ve_idx + len(ve):]
                result.append(new_block + after)
        return "".join(result)

    def _expected_pad_count(self, plan: "TokenPlan", modality: str = "image",
                            uniform_mode: bool = False, patches_per_frame: int = 0) -> int:
        """Number of pads to insert for each <image>/<video> placeholder.

        For query_and_vit, pad count = (Q + K) * S, where
          K = round(P * vit_ratio), and P is the actual ViT patches per segment (computed from config).
        This keeps the pad count written by the collator exactly equal to the vision_embeds token
        count produced by forward.

        patches_per_frame: Patches per frame after dynamic-resolution sampling; 0 uses the config default.
        """
        if plan.num_segments == 0:
            return 0
        if plan.strategy == "vit_only":
            # Dynamic resolution: for images, use the actual patches_per_frame (>0) instead of
            # the configured fixed value, so vit_only pad count equals the ViT output token count.
            if patches_per_frame > 0 and modality == "image":
                return patches_per_frame * plan.num_segments
            vit_k = plan.vit_tokens_for_modality(modality)
            return vit_k * plan.num_segments
        if plan.strategy == "query_and_vit":
            # Dynamic resolution: use the passed patches_per_frame (>0) instead of the config default.
            if patches_per_frame > 0:
                P = patches_per_frame  # One image segment equals one frame's patch count
            else:
                P = self.config.vit_patches_per_seg(modality, uniform_mode=uniform_mode)
            K = plan.k_from_vit_tokens(P)
            return (plan.num_query_per_seg + K) * plan.num_segments
        # query_only
        return plan.total_tokens_per_seg(modality) * plan.num_segments

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Each feature = {"messages": [...], "images": [...], "videos": [...], "visidx": np.ndarray|None, "token_plan": TokenPlan}."""
        # -------------------------------------------------------------------
        # Batch-level sampling decision: use the uniform-sampling path with uniform_train_prob.
        # Requirements:
        #   1. The batch must contain videos.
        #   2. The strategy must be query_and_vit (vit_only sends all vit tokens directly to the LLM;
        #      the uniform path outputs 2048 tokens/segment while placeholders are fixed at 1024,
        #      causing a mismatch; uniform is pointless and wasteful for query_only because vit
        #      tokens do not reach the LLM).
        # -------------------------------------------------------------------
        has_video_feat = any(feat.get("videos") for feat in features)
        has_image_feat = any(feat.get("images") for feat in features)
        uniform_mode = False  # False by default; overridden below when conditions are met

        all_input_ids: List[List[int]] = []
        all_labels: List[List[int]] = []
        pixel_values_list: List[Optional[torch.Tensor]] = []   # Video (B, 3, T, H, W) or image (B, 3, H, W)
        modality_list: List[str] = []
        token_plans: List[Any] = []
        visidx_list: List[Optional[torch.Tensor]] = []

        # Dynamic resolution: sample once per batch (image-only batches; videos keep fixed resolution).
        # Use np.random instead of random to synchronize with the DataLoader generator. The PyTorch
        # DataLoader generator controls only torch tensor operations; Python random needs separate handling.
        chosen_img_size = 0
        batch_ppf = 0
        
        # Check whether the batch contains OCR samples (OCR is considered for images only)
        batch_is_ocr = False
        if has_image_feat and not has_video_feat:
            for feat in features:
                if feat.get("is_ocr"):
                    batch_is_ocr = True
                    break
        
        # -------------------------------------------------------------------
        # Native resolution (images only): images with different sizes cannot be stacked, so
        # enable it only when the image batch has exactly one sample; otherwise fall back to
        # dynamic/fixed resolution for compatibility.
        # -------------------------------------------------------------------
        image_feats = [f for f in features if f.get("images")]
        native_mode = (
            self.config.native_resolution
            and not has_video_feat
            and len(image_feats) == 1
            and len(image_feats[0]["images"]) > 0
        )
        native_hw = None
        if native_mode:
            img0 = image_feats[0]["images"][0]
            native_hw = compute_native_size(
                width=img0.width,
                height=img0.height,
                patch_size=self.config.patch_size,
                min_patches=self.config.native_min_patches,
                max_patches=self.config.native_max_patches,
            )
            batch_ppf = (native_hw[0] // self.config.patch_size) * (native_hw[1] // self.config.patch_size)

        if has_image_feat and not has_video_feat and self._dynamic_resolution and not native_mode:
            if batch_is_ocr:
                # Use the highest resolution for OCR samples to improve text recognition
                chosen_img_size = max(self.config.candidate_resolutions)
            else:
                # Choose a resolution randomly for non-OCR samples.
                # Synchronize indirectly through torch.get_rng_state(): the DataLoader receives a
                # fixed-seed generator, so use torch.randint instead of random.choice to ensure
                # collator workers on all ranks choose the same resolution.
                idx = torch.randint(0, len(self.config.candidate_resolutions), (1,)).item()
                chosen_img_size = self.config.candidate_resolutions[idx]
            batch_ppf = (chosen_img_size // self.config.patch_size) ** 2

        # If the collator has dynamic_token_config, sample one plan for the whole batch
        # (ignore the plan attached to each sample).
        batch_plan = None
        if self.dynamic_token_config is not None:
            # Read num_segments / modality from the same visual sample to avoid a mismatch
            # where first_num_segments comes from an image while first_modality comes from a video.
            first_num_segments = 1
            first_modality = "image"
            for feat in features:
                if feat.get("videos"):
                    first_modality = "video"
                    plan_in = feat.get("token_plan")
                    if plan_in is not None and hasattr(plan_in, "num_segments"):
                        first_num_segments = plan_in.num_segments
                    break
                if feat.get("images"):
                    first_modality = "image"
                    plan_in = feat.get("token_plan")
                    if plan_in is not None and hasattr(plan_in, "num_segments"):
                        first_num_segments = plan_in.num_segments
                    break
            
            # batch_is_ocr was already computed above during dynamic-resolution selection; reuse it here
            
            batch_plan = self.dynamic_token_config.sample(
                num_segments=first_num_segments,
                modality=first_modality,
                is_ocr=batch_is_ocr,
            )

        # Uniform mode is triggered only for video batches using query_and_vit
        if has_video_feat and batch_plan is not None and batch_plan.strategy == "query_and_vit":
            uniform_mode = random.random() < self.config.uniform_train_prob

        for feat in features:
            messages = feat["messages"]
            # Skip samples with empty messages (cannot participate in training)
            if not messages:
                continue
            modality_feat = "video" if feat.get("videos") else ("image" if feat.get("images") else "text")
            plan = batch_plan if batch_plan is not None else feat.get("token_plan")
            if plan is not None and hasattr(plan, "num_query_per_seg"):
                plan.num_query_per_seg = self.config.num_query_per_seg

            # 1) Render the chat-template string
            text = self._apply_chat_template(messages)

            # 1.5) A text sample still contains a vision block: upstream visual data is missing
            # (for example, mp4 decoding produced no frames), but the plugin expanded
            # <image>/<video> into an <|image_pad|> block. Without pixel_values for the ViT,
            # these pad tokens would be treated as regular tokens for loss; if the whole batch
            # were like this, forward would train on pad tokens through the text-only branch.
            # Discard the sample.
            if modality_feat == "text" and self.plugin.vision_start_token in text:
                warnings.warn(
                    "[collator] sample discarded: no images/videos but messages contain a "
                    "vision block (<|vision_start|>...), usually because upstream visual "
                    "data failed to decode.",
                    stacklevel=2,
                )
                continue

            # 2) Adjust the pad count from the plan (for vit_only, compute it from each sample's modality)
            if plan is not None:
                text = self._adjust_image_pad_count(
                    text, plan, modality=modality_feat,
                    uniform_mode=uniform_mode, patches_per_frame=batch_ppf
                )

            # 3) Tokenize (add_special_tokens=False; the chat template already added them)
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True,
                                 max_length=self.config.max_seq_length)
            input_ids = enc.input_ids[0]

            # 3.5) Truncation check: max_length truncates from the end. When the vision block
            # follows a long prompt, some <|image_pad|> tokens may be cut, making the pad count
            # smaller than the ViT output. masked_scatter in forward would raise ValueError and
            # stop training, so detect and discard the sample here.
            if modality_feat in ("image", "video"):
                n_blocks = text.count(self.plugin.vision_start_token)
                # If a visual sample has no <image>/<video> placeholder in messages, the plugin
                # cannot expand a vision block and the pad count is 0. expected_pads below is also
                # 0, so validation would pass while pixel_values still reaches the ViT and creates
                # visual tokens; forward would then raise ValueError in masked_scatter. Discard early.
                if n_blocks == 0:
                    warnings.warn(
                        f"[collator] sample discarded: modality={modality_feat} but messages have no "
                        "<image>/<video> placeholder, so no vision block after expansion "
                        "(<|image_pad|> count is 0). Check whether json.messages in this "
                        "shard is missing the placeholder.",
                        stacklevel=2,
                    )
                    continue
            if plan is not None and modality_feat in ("image", "video"):
                expected_pads = n_blocks * self._expected_pad_count(
                    plan, modality=modality_feat,
                    uniform_mode=uniform_mode, patches_per_frame=batch_ppf,
                )
                got_pads = int((input_ids == self.image_pad_token_id).sum())
                if got_pads != expected_pads:
                    warnings.warn(
                        f"[collator] sample discarded: <|image_pad|> count {got_pads} != expected {expected_pads}"
                        f" (seq_len={input_ids.shape[0]}, max_seq_length={self.config.max_seq_length}). "
                        "Usually caused by the vision block following a long prompt and being truncated "
                        "by max_seq_length; increase max_seq_length or reduce the vision token count.",
                        stacklevel=2,
                    )
                    continue

            token_plans.append(plan)

            # 4) Mask non-assistant tokens (the standard chat-template approach).
            # Simplify by masking the prompt (the simplest assistant-only loss at collator time).
            labels = self._build_labels(
                input_ids, messages, plan=plan, modality=modality_feat,
                uniform_mode=uniform_mode, patches_per_frame=batch_ppf
            )

            all_input_ids.append(input_ids.tolist())
            all_labels.append(labels.tolist())

            # 5) Process pixel_values
            if feat.get("videos") is not None and len(feat["videos"]) > 0:
                # Use the first video (each sample contains only one video)
                frames = feat["videos"][0]  # List[PIL.Image]
                pv = _transform_frames_batch(frames, self.video_transform)  # (T, 3, H, W)
                pv = pv.permute(1, 0, 2, 3)  # (3, T, H, W)
                pixel_values_list.append(pv)
                modality_list.append("video")
                # Uniform mode does not use visidx (the encoder performs uniform downsampling internally)
                if uniform_mode:
                    visidx_list.append(None)
                else:
                    # visidx: (n_vit_candidate,) int32 numpy -> LongTensor
                    raw_visidx = feat.get("visidx", None)
                    if raw_visidx is not None:
                        import numpy as np
                        visidx_list.append(torch.from_numpy(np.asarray(raw_visidx)).long())
                    else:
                        visidx_list.append(None)
            elif feat.get("images") is not None and len(feat["images"]) > 0:
                img = feat["images"][0].convert("RGB")
                if native_mode:
                    # Native resolution: resize to the nearest dimensions divisible by patch_size while preserving aspect ratio
                    img = img.resize((native_hw[1], native_hw[0]), Image.BICUBIC)
                    pv = self.native_transform(img)  # (3, H, W)
                else:
                    img_tf = self._img_transforms[chosen_img_size] if chosen_img_size else self.image_transform
                    pv = img_tf(img)  # (3, H, W)
                pixel_values_list.append(pv)
                modality_list.append("image")
                visidx_list.append(None)
            else:
                # Text-only
                pixel_values_list.append(None)
                modality_list.append("text")
                visidx_list.append(None)

        # 6) Pad into a batch.
        # If every sample was skipped (all_input_ids is empty), return a 1x2 text-only dummy.
        # A zero-row tensor is invalid: Qwen3 attention would fail at q_proj(...).view(0,0,-1,128)
        # because reshaping zero elements to a shape containing -1 is ambiguous. The dummy labels
        # are all ignore_index, so the text-only forward produces NaN loss; the training loop skips
        # the micro-batch as a non-finite loss, keeping the call sequence consistent across ranks.
        if not all_input_ids:
            dummy_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            return {
                "input_ids": torch.full((1, 2), dummy_id, dtype=torch.long),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
                "labels": torch.full((1, 2), self.config.ignore_index, dtype=torch.long),
                "pixel_values": None,
                "modality": "text",
                "visidx": None,
                "uniform_sample_frames": False,
                "uniform_sample_n": self.config.uniform_sample_n,
                "uniform_segment_t_size": self.config.uniform_segment_t_size,
                "frame_cat_prob": 0.0,
            }

        max_len = max(len(x) for x in all_input_ids)
        padded_ids, padded_labels, attn_mask = [], [], []
        for ids, lbl in zip(all_input_ids, all_labels):
            pad = max_len - len(ids)
            padded_ids.append(ids + [self.tokenizer.pad_token_id] * pad)
            padded_labels.append(lbl + [self.config.ignore_index] * pad)
            attn_mask.append([1] * len(ids) + [0] * pad)
        input_ids = torch.tensor(padded_ids, dtype=torch.long)
        labels = torch.tensor(padded_labels, dtype=torch.long)
        attention_mask = torch.tensor(attn_mask, dtype=torch.long)

        # 7) Combine pixel_values into a batch (separate video/image handling).
        # A batch must contain one modality only: all video, all image, or all text.
        # Raise immediately for mixed modalities instead of silently dropping one pixel_values group.
        video_indices = [i for i, m in enumerate(modality_list) if m == "video"]
        image_indices = [i for i, m in enumerate(modality_list) if m == "image"]
        if video_indices and image_indices:
            raise ValueError(
                f"Batch contains both image samples ({len(image_indices)}) and video samples "
                f"({len(video_indices)}). Group by modality in the dataset so each batch "
                "contains a single modality."
            )
        videos = [pixel_values_list[i] for i in video_indices]
        images = [pixel_values_list[i] for i in image_indices]

        # Filter text-only samples when the batch contains video/image samples; text-only samples
        # cannot participate in forward because their pixel_values is None and cannot be stacked with visual samples.
        if video_indices:
            # Keep input_ids/labels/attention_mask for video samples
            input_ids = input_ids[video_indices]
            labels = labels[video_indices]
            attention_mask = attention_mask[video_indices]
        elif image_indices:
            # Keep input_ids/labels/attention_mask for image samples
            input_ids = input_ids[image_indices]
            labels = labels[image_indices]
            attention_mask = attention_mask[image_indices]
        # else: text-only batch, preserve as-is

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if videos:
            out["pixel_values"] = torch.stack(videos)  # (B, 3, T, H, W)
            out["modality"] = "video"
            # visidx: stack into (B, L) when every video has visidx; otherwise set None
            vid_visidx = [visidx_list[i] for i in video_indices]
            if all(v is not None for v in vid_visidx):
                out["visidx"] = torch.stack(vid_visidx)  # (B, L)
            else:
                out["visidx"] = None
            # Uniform-sampling mode flag and parameters
            out["uniform_sample_frames"] = uniform_mode
            out["uniform_sample_n"] = self.config.uniform_sample_n
            out["uniform_segment_t_size"] = self.config.uniform_segment_t_size
            out["frame_cat_prob"] = self.config.frame_cat_prob
        elif images:
            out["pixel_values"] = torch.stack(images)  # (B, 3, H, W)
            out["modality"] = "image"
            out["visidx"] = None
            out["uniform_sample_frames"] = False
            out["uniform_sample_n"] = self.config.uniform_sample_n
            out["uniform_segment_t_size"] = self.config.uniform_segment_t_size
            out["frame_cat_prob"] = 0.0
        else:
            out["pixel_values"] = None
            out["modality"] = "text"
            out["visidx"] = None
            out["uniform_sample_frames"] = False
            out["uniform_sample_n"] = self.config.uniform_sample_n
            out["uniform_segment_t_size"] = self.config.uniform_segment_t_size
            out["frame_cat_prob"] = 0.0

        non_null_plans = [p for p in token_plans if p is not None]
        if non_null_plans:
            first_plan = non_null_plans[0]
            if any(p != first_plan for p in non_null_plans):
                raise ValueError("token_plan is inconsistent within a batch; group by token_plan or use micro-batch=1.")
            out["token_plan"] = first_plan
        return out

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        messages: List[dict],
        plan=None,
        modality: str = "image",
        uniform_mode: bool = False,
        patches_per_frame: int = 0,
    ) -> torch.Tensor:
        """Set everything except assistant tokens to -100.

        Scan messages in role order without assuming strict user-assistant alternation, so
        system roles and irregular multi-turn structures remain supported. For a fast tokenizer,
        map labels from character spans in the complete chat-template rendering to avoid boundary
        drift caused by differences in how models render history or the final turn. Retain the old
        prefix/full alignment logic as a fallback for slow tokenizers or failed localization.

        plan / modality / uniform_mode / patches_per_frame: when the collator applies
        _adjust_image_pad_count to text, perform the same replacement in both span localization
        and the fallback path.
        """
        import warnings
        labels = torch.full_like(input_ids, self.config.ignore_index)
        seq_len = input_ids.shape[0]

        # Defensive check: return fully masked labels immediately when messages is empty
        if not messages:
            return labels

        # Fast-tokenizer path: locate each assistant content in the complete rendering that
        # exactly matches input_ids. This avoids prefix/full length drift when a template injects
        # think/BOS only for the final turn. Keep the legacy logic below as a slow-tokenizer fallback.
        if getattr(self.tokenizer, "is_fast", False):
            full_text = self._apply_chat_template(messages)
            if plan is not None:
                full_text = self._adjust_image_pad_count(
                    full_text, plan, modality=modality,
                    uniform_mode=uniform_mode, patches_per_frame=patches_per_frame
                )

            can_use_span_path = True
            if can_use_span_path:
                encoded = self.tokenizer(
                    full_text, add_special_tokens=False,
                    return_offsets_mapping=True, truncation=False,
                )
                rendered_ids = encoded["input_ids"]
                offsets = encoded["offset_mapping"]
                compare_len = min(seq_len, len(rendered_ids))
                if input_ids[:compare_len].tolist() != rendered_ids[:compare_len]:
                    can_use_span_path = False
                    warnings.warn(
                        "[collator] full-text tokenization does not match input_ids; "
                        "falling back to legacy label alignment.",
                        stacklevel=2,
                    )

            if can_use_span_path:
                special_ids = set(self.tokenizer.all_special_ids)
                special_ids.difference_update({
                    self.image_pad_token_id,
                    self.vision_start_token_id,
                    self.vision_end_token_id,
                })
                span_failed = False
                for i, msg in enumerate(messages):
                    if msg.get("role") != "assistant":
                        continue
                    content = msg.get("content")
                    if not isinstance(content, str):
                        span_failed = True
                        break
                    if content == "":
                        warnings.warn(
                            f"[collator] assistant turn {i} has empty content; "
                            "skipping its loss labels.",
                            stacklevel=2,
                        )
                        continue

                    # Make the sentinel's boundary characters differ from the real content so
                    # common-prefix/suffix scanning does not consume content boundary characters (for example, "__foo").
                    marker_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                    marker_start_char = next(
                        (char for char in marker_alphabet if not content.startswith(char)),
                        None,
                    )
                    marker_end_char = next(
                        (char for char in reversed(marker_alphabet) if not content.endswith(char)),
                        None,
                    )
                    if marker_start_char is None or marker_end_char is None:
                        span_failed = True
                        break
                    marker_start = f"{marker_start_char}__OV_ASSISTANT_SPAN_START_{i}_7f3c__"
                    marker_end = f"__OV_ASSISTANT_SPAN_END_{i}_7f3c__{marker_end_char}"
                    if marker_start in full_text or marker_end in full_text:
                        span_failed = True
                        break

                    # Preserve the original content and add markers only on its sides, avoiding
                    # changes to how the template interprets <think>, newlines, or other structured content.
                    probe_messages = [dict(item) for item in messages]
                    probe_messages[i]["content"] = (
                        marker_start + content + marker_end
                    )
                    probe_text = self._apply_chat_template(probe_messages)
                    if plan is not None:
                        probe_text = self._adjust_image_pad_count(
                            probe_text, plan, modality=modality,
                            uniform_mode=uniform_mode, patches_per_frame=patches_per_frame
                        )

                    prefix_len = 0
                    common_len = min(len(full_text), len(probe_text))
                    while prefix_len < common_len and full_text[prefix_len] == probe_text[prefix_len]:
                        prefix_len += 1
                    suffix_len = 0
                    while (
                        suffix_len < common_len - prefix_len
                        and full_text[len(full_text) - 1 - suffix_len]
                        == probe_text[len(probe_text) - 1 - suffix_len]
                    ):
                        suffix_len += 1
                    char_start = prefix_len
                    char_end = len(full_text) - suffix_len
                    if char_end <= char_start:
                        span_failed = True
                        break

                    # Find tokens in offset_mapping whose spans intersect the assistant content range.
                    overlap = [
                        k for k, (tok_start, tok_end) in enumerate(offsets)
                        if tok_end > char_start and tok_start < char_end
                    ]
                    if not overlap:
                        span_failed = True
                        break
                    start = overlap[0]
                    end = overlap[-1] + 1

                    # Preserve the end special token immediately following content (such as <|im_end|>)
                    # and one separator newline, matching the previous behavior; never cross into
                    # the next turn's start token.
                    while end < len(rendered_ids) and rendered_ids[end] in special_ids:
                        end += 1
                    if end < len(rendered_ids):
                        next_piece = self.tokenizer.decode(
                            [rendered_ids[end]], skip_special_tokens=False
                        )
                        if next_piece.strip() == "":
                            end += 1

                    if start >= seq_len:
                        warnings.warn(
                            f"[collator] assistant turn {i} starts beyond seq_len={seq_len}; "
                            "this turn contributes zero loss after truncation.",
                            stacklevel=2,
                        )
                        continue
                    labels[start:min(end, seq_len)] = input_ids[start:min(end, seq_len)]

                if not span_failed:
                    # Vision tokens are never language-model prediction targets.
                    vision_token_ids = {
                        self.image_pad_token_id,
                        self.vision_start_token_id,
                        self.vision_end_token_id,
                    }
                    for vid in vision_token_ids:
                        labels[input_ids == vid] = self.config.ignore_index
                    return labels
                # The fast path may already have written earlier assistant turns; clear them
                # before fallback to avoid combining boundary results from both approaches.
                labels.fill_(self.config.ignore_index)
                warnings.warn(
                    "[collator] failed to locate an assistant span from the full chat "
                    "template; falling back to legacy label alignment.",
                    stacklevel=2,
                )

        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            # Some chat templates cannot render an empty prefix; without reliable user context
            # for the first assistant turn, skip that turn to avoid a DataLoader crash.
            if i == 0:
                warnings.warn(
                    "[collator] first message is assistant; skipping this turn "
                    "because no user context exists for label alignment.",
                    stacklevel=2,
                )
                continue

            # Start of the assistant turn: the first i messages + add_generation_prompt=True.
            # Disable thinking to match _apply_chat_template (text generation during training).
            prefix = self.tokenizer.apply_chat_template(
                messages[:i], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            # End of the assistant turn: the first i+1 messages
            full = self.tokenizer.apply_chat_template(
                messages[:i + 1], tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )

            # Match the pad adjustment applied to text in __call__; otherwise prefix_tok length
            # shifts with the visual token count, producing incorrect label boundaries (leakage or missing labels).
            if plan is not None:
                prefix = self._adjust_image_pad_count(
                    prefix, plan, modality=modality,
                    uniform_mode=uniform_mode, patches_per_frame=patches_per_frame
                )
                full = self._adjust_image_pad_count(
                    full, plan, modality=modality,
                    uniform_mode=uniform_mode, patches_per_frame=patches_per_frame
                )

            prefix_tok = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids[0]
            full_tok = self.tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids[0]

            prefix_len = len(prefix_tok)
            start = prefix_len
            if prefix_len <= seq_len:
                cmp_len = min(prefix_len, seq_len)
                if not torch.equal(prefix_tok[:cmp_len], input_ids[:cmp_len]):
                    found = False
                    search_lo = max(0, prefix_len - 5)
                    for k in range(min(prefix_len, seq_len), search_lo - 1, -1):
                        if torch.equal(prefix_tok[:k], input_ids[:k]):
                            start = k
                            found = True
                            break
                    if not found:
                        warnings.warn(
                            f"[collator] assistant turn {i}: BPE boundary mismatch, "
                            f"cannot reliably determine label start near prefix_len={prefix_len}. "
                            "This turn will be skipped to avoid label leakage.",
                            stacklevel=2,
                        )
                        continue

            raw_end = min(len(full_tok), seq_len)
            end = raw_end
            if raw_end > 0 and raw_end <= seq_len:
                check_len = min(2, raw_end, len(full_tok))
                if check_len > 0 and not torch.equal(
                    full_tok[raw_end - check_len: raw_end],
                    input_ids[raw_end - check_len: raw_end],
                ):
                    search_end_hi = min(raw_end + 2, seq_len, len(full_tok))
                    search_end_lo = max(start + 1, raw_end - 2)
                    for cand_end in range(search_end_hi, search_end_lo - 1, -1):
                        chk = min(2, cand_end, len(full_tok))
                        if chk > 0 and torch.equal(
                            full_tok[cand_end - chk: cand_end],
                            input_ids[cand_end - chk: cand_end],
                        ):
                            end = cand_end
                            break

            if start >= seq_len:
                warnings.warn(
                    f"[collator] assistant turn {i} start={start} >= seq_len={seq_len} "
                    f"(max_seq_length={self.config.max_seq_length}); "
                    "this sample contributes zero loss. Consider increasing max_seq_length.",
                    stacklevel=2,
                )
                continue
            if end <= start:
                continue
            labels[start:end] = input_ids[start:end]

        # Mask visual special tokens (<|image_pad|> / <|vision_start|> / <|vision_end|>) so
        # they are not treated as prediction targets if they happen to fall in an assistant span.
        vision_token_ids = {self.image_pad_token_id, self.vision_start_token_id, self.vision_end_token_id}
        for vid in vision_token_ids:
            labels[input_ids == vid] = self.config.ignore_index

        return labels
