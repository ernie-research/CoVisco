"""Token selector for CoVisco ViT.

Reuses `LearnableTokenSelector` from the onevision repository. It is a
query-guided vit token selector with:
  - A lightweight two-layer transformer with vit self-attention, vit-query cross-attention, and FFN
  - Differentiable top-K selection (DynamicViT/EViT-style score gating)
  - Gumbel-Top-K stochastic relaxation

This module is an nn.Module with trainable parameters that are fine-tuned during SFT.
"""
from ._token_selector_src import (
    LearnableTokenSelector,
    TokenSelectorLayer,
    _straight_through_topk,
    _mmr_topk,
    _perturb_logits,
)

__all__ = [
    "LearnableTokenSelector",
    "TokenSelectorLayer",
    "_straight_through_topk",
    "_mmr_topk",
    "_perturb_logits",
]
