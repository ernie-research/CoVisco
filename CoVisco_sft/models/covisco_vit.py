"""CoVisco ViT (SFT/mid-training variant).

Reuses the CoViscoEncoderModel implementation from the covisco repository,
including all transformer blocks, 4D RoPE, and query tokens. Changes:
  1) No pooling head
  2) Split query/vit tokens at the end of forward
  3) Support the uniform_sample_frames path (50/50 uniform-frame sampling during SFT)
  4) num_query_per_seg must be a perfect square (the query RoPE grid uses a square approximation)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from ._covisco_encoder_src import CoViscoEncoderConfig, CoViscoEncoderModel


class CoViscoViT(nn.Module):
    """CoVisco ViT returning query_tokens and vit_tokens."""

    def __init__(self, config: CoViscoEncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = CoViscoEncoderModel(config)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def num_query_per_seg(self) -> int:
        return self.config.num_query_per_seg

    def forward(
        self,
        pixel_values: torch.Tensor,           # (B, 3, T, H, W) or (B, 3, H, W)
        visidx: Optional[torch.Tensor] = None, # (B, L_vit_candidate) precomputed token indices
        modality: str = "image",               # "image" | "video"
        segment_offset: int = 0,
        uniform_sample_frames: bool = False,   # Uniform-sampling path (aligned with pretraining)
        uniform_sample_n: int = 32,
        uniform_segment_t_size: int = 8,
        frame_cat_prob: float = 0.0,           # Frame-concatenation augmentation probability (uniform path during training only)
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[BaseModelOutput]]:
        """
        Returns:
            query_tokens: (B, S, num_query_per_seg, D)        # Always num_query_per_seg
            vit_tokens:   (B, S, N_vit_per_seg, D)            # All vit tokens in the segment
            encoder_outputs: Optional transformers BaseModelOutput
        """
        out = self.encoder(
            pixel_values=pixel_values,
            visible_indices=visidx,
            segment_offset=segment_offset,
            uniform_sample_frames=uniform_sample_frames,
            uniform_sample_n=uniform_sample_n,
            uniform_segment_t_size=uniform_segment_t_size,
            frame_cat_prob=frame_cat_prob,
            return_dict=return_dict,
        )
        if return_dict:
            return out.query_tokens, out.vit_tokens, None
        # Tuple form: (query_tokens, vit_tokens, hidden_states, attentions)
        return out[0], out[1], None

    def load_covisco_pretrained(self, state_dict: dict, strict: bool = False) -> tuple:
        """
        Load from covisco pretrained weights. Missing head/layernorm_post keys are allowed.
        Keys in state_dict should already use the encoder.* prefix (with the outer
        covisco_encoder. prefix removed).
        """
        return self.encoder.load_state_dict(state_dict, strict=strict)
