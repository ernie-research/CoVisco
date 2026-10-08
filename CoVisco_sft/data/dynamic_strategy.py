"""dynamic_strategy.py - Dynamic token composition strategies during training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class TokenPlan:
    """Concrete token composition selected for one step.

    query_and_vit: use vit_ratio (retention ratio, 0-1); forward computes top-K at
      runtime from the actual vit_tokens.shape.
    vit_only: use vit_per_seg_image / vit_per_seg_video (absolute token counts).
    query_only: vit_ratio=0 and vit_per_seg=0.
    """
    strategy: str           # "query_only" | "vit_only" | "query_and_vit"
    vit_per_seg: int        # Vit tokens per segment for vit_only (0 / unused for query_and_vit)
    arrangement: str        # "interleave" | "concat"
    num_segments: int       # Number of video segments (1 for images)
    num_query_per_seg: int = 100
    vit_per_seg_image: Optional[int] = None   # Image vit tokens per segment for vit_only
    vit_per_seg_video: Optional[int] = None   # Video vit tokens per segment for vit_only
    vit_ratio: float = 0.0  # Vit-token retention ratio for query_and_vit (0-1)

    def vit_tokens_for_modality(self, modality: str) -> int:
        """Return vit tokens per segment by modality for vit_only; otherwise return vit_per_seg."""
        if self.strategy != "vit_only":
            return self.vit_per_seg
        if modality == "video" and self.vit_per_seg_video is not None:
            return self.vit_per_seg_video
        if self.vit_per_seg_image is not None:
            return self.vit_per_seg_image
        return self.vit_per_seg

    def k_from_vit_tokens(self, num_vit_tokens_per_seg: int) -> int:
        """Compute top-K for query_and_vit from the actual vit tokens per segment and vit_ratio.
        Retain at least one token.
        """
        return max(1, round(num_vit_tokens_per_seg * self.vit_ratio))

    def total_tokens_per_seg(self, modality: str = "image", num_vit_per_seg: Optional[int] = None) -> int:
        """Number of tokens sent to the LLM per segment.

        query_and_vit requires num_vit_per_seg (the actual vit tokens per segment) to
        infer K. If num_vit_per_seg is None, return query_per_seg as a lower bound for
        approximate estimates.
        """
        if self.strategy == "query_only":
            return self.num_query_per_seg
        if self.strategy == "vit_only":
            return self.vit_tokens_for_modality(modality)
        # query_and_vit
        if num_vit_per_seg is not None:
            return self.num_query_per_seg + self.k_from_vit_tokens(num_vit_per_seg)
        return self.num_query_per_seg  # Lower-bound estimate only

    def total_tokens(self, num_query_per_seg: int, modality: str = "image", num_vit_per_seg: Optional[int] = None) -> int:
        if self.strategy == "query_only":
            return num_query_per_seg * self.num_segments
        if self.strategy == "vit_only":
            return self.vit_tokens_for_modality(modality) * self.num_segments
        # query_and_vit
        if num_vit_per_seg is not None:
            return (num_query_per_seg + self.k_from_vit_tokens(num_vit_per_seg)) * self.num_segments
        return num_query_per_seg * self.num_segments  # Lower-bound estimate only


class DynamicTokenConfig:
    """Controls dynamic token composition during training.

    query_and_vit uses vit_token_ratios to control retention (0-1).
    vit_only supports independent vit_token_counts for images and videos:
      - vit_token_counts_image: candidate vit-token counts per image segment
      - vit_token_counts_video: candidate vit-token counts per video segment (same as image when None)

    Strategy sampling probabilities (p_query_only / p_vit_only / p_query_and_vit) are
    shared by images and videos by default. probs_image / probs_video can override them
    per modality; when both are None, behavior is unchanged. The query_and_vit
    vit_token_ratios can likewise be overridden with vit_token_ratios_image /
    vit_token_ratios_video.
    """

    def __init__(
        self,
        enabled: bool = True,
        p_query_only: float = 0.2,
        p_vit_only: float = 0.2,
        p_query_and_vit: float = 0.6,
        vit_token_ratios: tuple = (0.25, 0.5, 1.0),
        arrangement: str = "interleave",
        vit_token_counts_image: Optional[tuple] = None,
        vit_token_counts_video: Optional[tuple] = None,
        probs_image: Optional[Any] = None,
        probs_video: Optional[Any] = None,
        vit_token_ratios_image: Optional[tuple] = None,
        vit_token_ratios_video: Optional[tuple] = None,
    ):
        if abs(p_query_only + p_vit_only + p_query_and_vit - 1.0) > 1e-6:
            raise ValueError("p_query_only + p_vit_only + p_query_and_vit must sum to 1")
        self.enabled = enabled
        self.probs = [p_query_only, p_vit_only, p_query_and_vit]
        self.strategies = ["query_only", "vit_only", "query_and_vit"]
        # Override strategy probabilities by modality; fall back to shared self.probs when unset
        self.probs_image = self._resolve_probs(probs_image, self.probs, "image")
        self.probs_video = self._resolve_probs(probs_video, self.probs, "video")
        self.vit_token_ratios = list(vit_token_ratios)
        # Override query_and_vit retention-ratio candidates by modality; fall back to shared vit_token_ratios
        self.vit_token_ratios_image = self._resolve_ratios(
            vit_token_ratios_image, self.vit_token_ratios, "image")
        self.vit_token_ratios_video = self._resolve_ratios(
            vit_token_ratios_video, self.vit_token_ratios, "video")
        self.arrangement = arrangement
        # For vit_only: separate absolute token-count candidates for images and videos
        _default_counts = [256]  # Preserve the meaning of the legacy field
        self.vit_token_counts_image: list = list(vit_token_counts_image) if vit_token_counts_image else _default_counts
        self.vit_token_counts_video: list = list(vit_token_counts_video) if vit_token_counts_video else _default_counts

    @staticmethod
    def _resolve_ratios(spec, default: list, modality: str) -> list:
        """Validate and return vit_token_ratios candidates for a modality."""
        if spec is None:
            return list(default)
        ratios = [float(x) for x in spec]
        if not ratios:
            raise ValueError(f"vit_token_ratios_{modality} must not be an empty list")
        if any(not (0.0 < r <= 1.0) for r in ratios):
            raise ValueError(
                f"each entry of vit_token_ratios_{modality} must be in (0, 1], got {ratios}"
            )
        return ratios

    def _resolve_probs(self, spec, default: list, modality: str) -> list:
        """Parse a strategy-name dict or length-three sequence into self.strategies order."""
        if spec is None:
            return list(default)
        if isinstance(spec, dict):
            unknown = set(spec) - set(self.strategies)
            if unknown:
                raise ValueError(
                    f"p_{modality} contains unknown strategy names {sorted(unknown)}; available: {self.strategies}"
                )
            probs = [float(spec.get(s, 0.0)) for s in self.strategies]
        else:
            probs = [float(x) for x in spec]
            if len(probs) != 3:
                raise ValueError(
                    f"p_{modality} as a sequence must have length 3 (order {self.strategies}), got {probs}"
                )
        if abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError(f"p_{modality} probabilities must sum to 1, got {probs} (sum={sum(probs)})")
        return probs

    def probs_for_modality(self, modality: str) -> list:
        return self.probs_video if modality == "video" else self.probs_image

    def vit_token_ratios_for_modality(self, modality: str) -> list:
        return self.vit_token_ratios_video if modality == "video" else self.vit_token_ratios_image

    def sample(
        self,
        num_segments: int = 1,
        modality: str = "image",
        is_ocr: bool = False,
        generator: Optional[np.random.Generator] = None,
    ) -> TokenPlan:
        """Call once per step to return a plan.

        modality: "image" | "video"; selects token_counts for vit_only and the
                  modality-specific strategy probabilities (probs_image / probs_video)
                  and query_and_vit ratio candidates (vit_token_ratios_image / _video).
        is_ocr: Whether this is an OCR sample. OCR images use query_and_vit with
                vit_ratio=1.0; videos ignore this parameter.
        """
        probs = self.probs_for_modality(modality)
        ratios = self.vit_token_ratios_for_modality(modality)
        # OCR image samples: use vit_only (all ViT patch tokens, without token_selector)
        if is_ocr and modality == "image":
            return TokenPlan(
                strategy="vit_only",
                vit_per_seg=256,  # All patches
                arrangement=self.arrangement,
                num_segments=num_segments,
            )

        if not self.enabled:
            ratio = float(ratios[0])
            return TokenPlan(
                strategy="query_and_vit",
                vit_per_seg=0,
                arrangement=self.arrangement,
                num_segments=num_segments,
                vit_ratio=ratio,
            )

        strategy = (
            generator.choice(self.strategies, p=probs)
            if generator is not None
            else np.random.choice(self.strategies, p=probs)
        )

        if strategy == "query_only":
            return TokenPlan(
                strategy=strategy,
                vit_per_seg=0,
                arrangement=self.arrangement,
                num_segments=num_segments,
                vit_ratio=0.0,
            )

        if strategy == "vit_only":
            K_image = int(self.vit_token_counts_image[0])
            K_video = int(self.vit_token_counts_video[0])
            K = K_video if modality == "video" else K_image
            return TokenPlan(
                strategy=strategy,
                vit_per_seg=K,
                arrangement=self.arrangement,
                num_segments=num_segments,
                vit_per_seg_image=K_image,
                vit_per_seg_video=K_video,
                vit_ratio=0.0,
            )

        # query_and_vit: sample a ratio from this modality's candidates
        ratio = float(
            generator.choice(ratios) if generator is not None
            else np.random.choice(ratios)
        )
        return TokenPlan(
            strategy=strategy,
            vit_per_seg=0,
            arrangement=self.arrangement,
            num_segments=num_segments,
            vit_ratio=ratio,
        )
