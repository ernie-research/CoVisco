"""plugin.py - Simplified CoVisco mm_plugin.

Does not depend on the Qwen2-VL image_processor; the dynamic strategy determines
the token count, and image_grid_thw uses a placeholder value while retaining the field
for future use.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

from PIL import Image

from .dynamic_strategy import TokenPlan


class CoViscoPlugin:
    """Simplified mm_plugin.

    Keep the public interface as close as possible to LLaVA-OneVision-2's
    Qwen2VLPlugin:
      - image_token / video_token placeholders
    """

    PLACEHOLDER_IMAGE = "<image>"
    PLACEHOLDER_VIDEO = "<video>"

    def __init__(
        self,
        image_token: str = "<|image_pad|>",
        video_token: str = "<|image_pad|>",   # Shared token
        vision_start_token: str = "<|vision_start|>",
        vision_end_token: str = "<|vision_end|>",
        sms: int = 1,
        num_query_per_seg: int = 100,
    ):
        self.image_token = image_token
        self.video_token = video_token
        self.vision_start_token = vision_start_token
        self.vision_end_token = vision_end_token
        self.sms = sms
        self.num_query_per_seg = num_query_per_seg

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _count_placeholders(messages: List[dict], placeholder: str) -> int:
        return sum(m["content"].count(placeholder) for m in messages)

    def _wrap_visual_block(self, n_tokens: int) -> str:
        """Generate a visual token sequence (vision_start + n * image_pad + vision_end)."""
        return f"{self.vision_start_token}{self.image_token * n_tokens}{self.vision_end_token}"

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------
    def process_messages(
        self,
        messages: List[dict],
        images: Optional[List[Image.Image]] = None,
        videos: Optional[List[List[Image.Image]]] = None,
        processor=None,
    ) -> Tuple[List[dict], dict]:
        """
        Args:
            messages: [{"role": ..., "content": "..."}] with <image> / <video> placeholders
            images: List of images (PIL.Image)
            videos: List of videos, each represented as a list of frames (List[PIL.Image])
            processor: Reserved for future use

        Returns:
            messages: Messages with <image>/<video> replaced by vision blocks
            mm_inputs: Dict containing images/videos (raw PIL lists, processed by the collator)
        """
        images = images or []
        videos = videos or []

        n_img_placeholders = self._count_placeholders(messages, self.PLACEHOLDER_IMAGE)
        n_vid_placeholders = self._count_placeholders(messages, self.PLACEHOLDER_VIDEO)

        # Replace <image>
        n_used_img = 0
        for msg in messages:
            content = msg["content"]
            while self.PLACEHOLDER_IMAGE in content:
                if n_used_img >= len(images):
                    raise ValueError(
                        f"<image> placeholders ({n_used_img + 1}) exceed provided images ({len(images)})"
                    )
                # Image: query_only defaults to one segment of num_query_per_seg tokens per image
                n_tokens = self.num_query_per_seg
                content = content.replace(
                    self.PLACEHOLDER_IMAGE, self._wrap_visual_block(n_tokens), 1
                )
                n_used_img += 1
            msg["content"] = content

        # Replace <video>
        n_used_vid = 0
        for msg in messages:
            content = msg["content"]
            while self.PLACEHOLDER_VIDEO in content:
                if n_used_vid >= len(videos):
                    raise ValueError(
                        f"<video> placeholders ({n_used_vid + 1}) exceed provided videos ({len(videos)})"
                    )
                # Video: query_only defaults to num_query_per_seg * num_segments.
                # num_segments depends on segment_t_size and the actual frame count; the
                # collator computes it later. Insert num_query_per_seg placeholder tokens
                # here so tokenization succeeds; the collator adjusts them using the strategy.
                n_tokens = self.num_query_per_seg
                content = content.replace(
                    self.PLACEHOLDER_VIDEO, self._wrap_visual_block(n_tokens), 1
                )
                n_used_vid += 1
            msg["content"] = content

        mm_inputs = {
            "images": images,           # List[PIL.Image]
            "videos": videos,           # List[List[PIL.Image]]
            "n_images": len(images),
            "n_videos": len(videos),
            "image_grid_thw": None,     # Not required in the initial version; retained for compatibility
            "video_grid_thw": None,
        }
        return messages, mm_inputs

    # ------------------------------------------------------------------
    # Utilities for the collator
    # ------------------------------------------------------------------
    def compute_video_segments(self, n_frames: int, segment_t_size: int) -> int:
        """One segment contains segment_t_size frames.

        Fewer frames still count as one segment; extra incomplete frames are rounded down.
        """
        if n_frames < segment_t_size:
            return 1
        return n_frames // segment_t_size
