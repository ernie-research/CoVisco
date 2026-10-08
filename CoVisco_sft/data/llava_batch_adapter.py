"""LLaVA-OneVision-2 batch adapter.

Data loading, packing, and the chat template remain as close as possible to
LLaVA-OneVision-2; this module adapts field names and the dynamic query/vit token plan
at the model boundary.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .dynamic_strategy import DynamicTokenConfig, TokenPlan


LLAVA_IMAGE_TOKEN_ID = 151655
LLAVA_VIDEO_TOKEN_ID = 151656


class LlavaOneVisionBatchAdapter:
    """Convert an LLaVA-OneVision-2 dataloader batch to this model's forward batch."""

    def __init__(
        self,
        dynamic_token_config: DynamicTokenConfig,
        num_query_per_seg: int,
        image_token_id: int = LLAVA_IMAGE_TOKEN_ID,
        video_token_id: int = LLAVA_VIDEO_TOKEN_ID,
        model_image_pad_token_id: Optional[int] = None,
        vit_patches_per_seg_image: int = 256,   # (image_size // patch_size)^2, patches per image segment
        vit_patches_per_seg_video: int = 256,   # segment_t_size * (image_size // patch_size)^2, patches per video segment (non-uniform mode)
        vit_patches_per_seg_video_uniform: Optional[int] = None,  # Patches per video segment in uniform_sample_frames mode
    ):
        self.dynamic_token_config = dynamic_token_config
        self.num_query_per_seg = num_query_per_seg
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.model_image_pad_token_id = model_image_pad_token_id or image_token_id
        self.vit_patches_per_seg_image = vit_patches_per_seg_image
        self.vit_patches_per_seg_video = vit_patches_per_seg_video
        # Uniform mode uses a different segment_t_size; fall back to the regular video value if unset
        self.vit_patches_per_seg_video_uniform = (
            vit_patches_per_seg_video_uniform
            if vit_patches_per_seg_video_uniform is not None
            else vit_patches_per_seg_video
        )

    def _num_segments_from_batch(self, batch: Dict[str, Any], modality: str) -> int:
        grid_key = "video_grid_thw" if modality == "video" else "image_grid_thw"
        grid = batch.get(grid_key)
        if torch.is_tensor(grid) and grid.numel() > 0:
            return int(grid.shape[0])
        return 1

    def _sample_plan(self, batch: Dict[str, Any], modality: str) -> Optional[TokenPlan]:
        if modality == "text":
            return None
        plan = self.dynamic_token_config.sample(
            num_segments=self._num_segments_from_batch(batch, modality),
            modality=modality,
        )
        plan.num_query_per_seg = self.num_query_per_seg
        return plan

    def _attention_mask(self, batch: Dict[str, Any], input_ids: torch.Tensor) -> torch.Tensor:
        attn_mask = batch.get("attention_mask", batch.get("attn_mask"))
        if torch.is_tensor(attn_mask):
            # Hugging Face convention: 1 / True participates in attention, while
            # 0 / False is ignored. A direct long conversion is sufficient; do not invert it.
            return attn_mask.long()
        return torch.ones_like(input_ids, dtype=torch.long)

    def _modality(self, batch: Dict[str, Any], input_ids: torch.Tensor) -> str:
        if torch.is_tensor(input_ids):
            if torch.any(input_ids == self.video_token_id).item():
                return "video"
            if torch.any(input_ids == self.image_token_id).item():
                return "image"
        if "pixel_values_videos" in batch or "videos" in batch or "video_grid_thw" in batch:
            return "video"
        if "imgs" in batch or "images" in batch or "pixel_values" in batch or "image_grid_thw" in batch:
            return "image"
        return "text"

    def _vision_token_id(self, modality: str) -> int:
        return self.video_token_id if modality == "video" else self.image_token_id

    @staticmethod
    def _contiguous_runs(positions: torch.Tensor) -> List[Tuple[int, int]]:
        """Group consecutive integers in positions into [(start, end), ...] half-open intervals."""
        if positions.numel() == 0:
            return []
        runs: List[Tuple[int, int]] = []
        start = int(positions[0].item())
        prev = start
        for p in positions[1:].tolist():
            p = int(p)
            if p != prev + 1:
                runs.append((start, prev + 1))
                start = p
            prev = p
        runs.append((start, prev + 1))
        return runs

    def _target_counts_for_runs(
        self,
        num_runs: int,
        token_plan: TokenPlan,
        modality: str,
        P: int,
    ) -> List[int]:
        """Compute the target token count for each visual slot run.
        Supports two layouts:
          - num_runs == 1: one large placeholder block, total = plan.total_tokens()
          - num_runs == plan.num_segments: one placeholder per segment, each gets total_tokens_per_seg()
        """
        per_seg = token_plan.total_tokens_per_seg(
            modality=modality,
            num_vit_per_seg=P if token_plan.strategy == "query_and_vit" else None,
        )
        if num_runs == 1:
            return [per_seg * token_plan.num_segments]
        if num_runs == token_plan.num_segments:
            return [per_seg] * num_runs
        raise ValueError(
            f"Cannot align {num_runs} visual slot runs with token_plan.num_segments="
            f"{token_plan.num_segments}. Use one visual placeholder per sample or "
            "one placeholder per segment."
        )

    def _resolve_vit_patches_per_seg(self, modality: str, uniform_mode: bool = False) -> int:
        """Return the correct number of vit patches per segment for the modality and sampling mode."""
        if modality != "video":
            return self.vit_patches_per_seg_image
        return self.vit_patches_per_seg_video_uniform if uniform_mode else self.vit_patches_per_seg_video

    def _resize_vision_slots(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        modality: str,
        token_plan: Optional[TokenPlan],
        uniform_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_plan is None or modality == "text":
            return input_ids, labels, attention_mask

        source_token_id = self._vision_token_id(modality)
        P = self._resolve_vit_patches_per_seg(modality, uniform_mode=uniform_mode)

        resized_ids: List[torch.Tensor] = []
        resized_labels: List[torch.Tensor] = []
        resized_masks: List[torch.Tensor] = []
        max_len = 0

        for sample_ids, sample_labels, sample_mask in zip(input_ids, labels, attention_mask):
            positions = (sample_ids == source_token_id).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                resized_ids.append(sample_ids)
                resized_labels.append(sample_labels)
                resized_masks.append(sample_mask)
                max_len = max(max_len, sample_ids.shape[0])
                continue

            runs = self._contiguous_runs(positions)
            target_counts = self._target_counts_for_runs(len(runs), token_plan, modality, P)

            id_parts: List[torch.Tensor] = []
            lbl_parts: List[torch.Tensor] = []
            mask_parts: List[torch.Tensor] = []
            cursor = 0
            for (start, end), target_count in zip(runs, target_counts):
                id_parts.append(sample_ids[cursor:start])
                lbl_parts.append(sample_labels[cursor:start])
                mask_parts.append(sample_mask[cursor:start])
                id_parts.append(torch.full(
                    (target_count,), self.model_image_pad_token_id,
                    dtype=sample_ids.dtype, device=sample_ids.device,
                ))
                lbl_parts.append(torch.full(
                    (target_count,), -100,
                    dtype=sample_labels.dtype, device=sample_labels.device,
                ))
                mask_parts.append(torch.ones(
                    (target_count,), dtype=sample_mask.dtype, device=sample_mask.device,
                ))
                cursor = end
            id_parts.append(sample_ids[cursor:])
            lbl_parts.append(sample_labels[cursor:])
            mask_parts.append(sample_mask[cursor:])

            ids = torch.cat(id_parts)
            lbl = torch.cat(lbl_parts)
            mask = torch.cat(mask_parts)
            resized_ids.append(ids)
            resized_labels.append(lbl)
            resized_masks.append(mask)
            max_len = max(max_len, ids.shape[0])

        return (
            self._pad_1d(resized_ids, max_len, pad_value=0),
            self._pad_1d(resized_labels, max_len, pad_value=-100),
            self._pad_1d(resized_masks, max_len, pad_value=0),
        )

    def _pad_1d(self, values: List[torch.Tensor], max_len: int, pad_value: int) -> torch.Tensor:
        padded = []
        for value in values:
            pad_len = max_len - value.shape[0]
            if pad_len > 0:
                pad = torch.full((pad_len,), pad_value, dtype=value.dtype, device=value.device)
                value = torch.cat([value, pad])
            padded.append(value)
        return torch.stack(padded)

    def adapt(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = batch.get("input_ids", batch.get("tokens"))
        if input_ids is None:
            raise KeyError("LLaVA batch must contain `tokens` or `input_ids`.")
        labels = batch.get("labels")
        if labels is None:
            raise KeyError("LLaVA batch must contain `labels`.")

        input_ids = input_ids.long()
        labels = labels.long()
        modality = self._modality(batch, input_ids)
        attention_mask = self._attention_mask(batch, input_ids)
        token_plan = self._sample_plan(batch, modality)
        uniform_mode = bool(batch.get("uniform_sample_frames", False))
        input_ids, labels, attention_mask = self._resize_vision_slots(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            modality=modality,
            token_plan=token_plan,
            uniform_mode=uniform_mode,
        )

        if modality == "video":
            pixel_values = batch.get("pixel_values_videos", batch.get("videos"))
        elif modality == "image":
            pixel_values = batch.get("imgs", batch.get("images", batch.get("pixel_values")))
        else:
            pixel_values = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "modality": modality,
            "token_plan": token_plan,
            "image_grid_thw": batch.get("image_grid_thw"),
            "video_grid_thw": batch.get("video_grid_thw"),
            "patch_positions": batch.get("patch_positions"),
            # Pass video-sampling parameters through to forward() so the ViT encoder uses the correct mode
            "visidx": batch.get("visidx"),
            "uniform_sample_frames": uniform_mode,
            "uniform_sample_n": batch.get("uniform_sample_n", 32),
            "uniform_segment_t_size": batch.get("uniform_segment_t_size", 8),
        }


def is_llava_onevision_batch(batch: Dict[str, Any]) -> bool:
    """Return whether the batch comes from an LLaVA-OneVision-2/Energon-style dataloader."""
    return "tokens" in batch or "imgs" in batch or "pixel_values_videos" in batch
