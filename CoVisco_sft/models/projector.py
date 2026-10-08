"""Projector modules (configurable ``sms``).

TwoLayerMLPProjector maps vision tokens to the LLM hidden size.
With sms=1 it performs a direct 1:1 mapping (no spatial merge).
The entry point for sms=2/3 is reserved.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TwoLayerMLPProjector(nn.Module):
    """Two-layer MLP projector.

    For sms > 1, tokens must be reorganized into a 2D grid for spatial merging.
    """

    def __init__(self, vision_hidden: int, llm_hidden: int, sms: int = 1, ffn_mult: int = 2):
        super().__init__()
        self.sms = sms
        input_dim = vision_hidden * sms * sms
        self.fc1 = nn.Linear(input_dim, ffn_mult * llm_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_mult * llm_hidden, llm_hidden)

    def forward(self, x: torch.Tensor, spatial_grid: Optional[tuple] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, vision_hidden) or (B, vision_hidden), flattened vision tokens
            spatial_grid: (h, w) grid size required for reshape when sms > 1

        Returns:
            (B, T, llm_hidden) or (B, llm_hidden)
        """
        if self.sms == 1:
            return self.fc2(self.act(self.fc1(x)))
        # Spatial merging for sms > 1 is not implemented in this initial version.
        raise NotImplementedError(
            f"The spatial-merge path for TwoLayerMLPProjector.sms={self.sms} is not yet implemented. "
            "Tokens must be reshaped into (B, h, w, vision_hidden) and each sms x sms block flattened."
        )
