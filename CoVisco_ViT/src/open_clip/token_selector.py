"""
Token Selector Module for CoVisco SigLIP

This file is a home for different *token selector* designs. A token
selector consumes the per-segment query tokens (the summary tokens
produced by OneVision) together with the corresponding fine-grained ViT
tokens, and returns a small subset of the fine-grained tokens that best
complement the query tokens.

Why a separate module?
    * The query tokens already summarize each segment and can be used on
      their own for most downstream heads.
    * Sometimes downstream heads need finer-grained evidence, but the
      raw ViT tokens are too numerous (e.g. up to ~4k tokens per video
      segment) to keep all of them. We therefore want to keep only the
      top-K tokens that are most *informative for the queries*.

Currently implemented selectors:
    * ``LearnableTokenSelector``
        A lightweight (default 2-layer) transformer that lets every ViT
        token interact with the segment's query tokens, scores every ViT
        token via sigmoid, then picks the top-K. Because top-K is not
        differentiable, gradient flow is preserved via either the
        straight-through score-gating trick (DynamicViT-style) or the
        Gumbel-Sigmoid relaxation.

Conventions
    * Tokens carry an explicit segment dimension throughout, matching
      the layout of ``covisco_model.py``:
          query_tokens: (B, S, Q, D)
          vit_tokens:   (B, S, P, D)
      Selection is performed independently per segment.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable top-K helpers
# ---------------------------------------------------------------------------

def _straight_through_topk(
    scores: torch.Tensor,
    features: torch.Tensor,
    k: int,
    pad_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable top-K via score-gated straight-through.

    Forward: pick the indices of the top-K scores and gather features.
    Backward: each selected feature is multiplied by its (continuous)
    sigmoid score, so gradients flow back through the scoring network
    even though ``topk`` itself is non-differentiable. This is the
    mechanism used by DynamicViT / EViT.

    Args:
        scores:   (BS, P) sigmoid scores in [0, 1].
        features: (BS, P, D) source features (typically the contextualized
            ViT tokens produced by the selector's transformer).
        k:        Number of tokens to keep per row. Caller is responsible
            for ensuring ``k <= P``.
        pad_mask: (BS, P) bool, True where a token is padding/invalid.

    Returns:
        selected_features: (BS, K, D) score-gated selected features.
        topk_scores:       (BS, K)    continuous scores of selected tokens.
        topk_indices:      (BS, K)    indices into the original P axis.
    """
    if pad_mask is not None:
        # Padding tokens must never be selected.
        scores = scores.masked_fill(pad_mask, float("-inf"))

    topk_scores, topk_indices = scores.topk(k, dim=-1)            # (BS, K)
    gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, features.size(-1))
    selected = torch.gather(features, dim=1, index=gather_idx)    # (BS, K, D)

    # Score-gate so gradients flow back through the scoring network.
    selected = selected * topk_scores.unsqueeze(-1)

    return selected, topk_scores, topk_indices


def _perturb_logits(
    logits: torch.Tensor,
    score_activation: str,
    temperature: float,
) -> torch.Tensor:
    """Add the appropriate Gumbel noise for a stochastic relaxation of top-K.

    * ``"sigmoid"`` -> logistic noise (= Gumbel(0,1) - Gumbel(0,1)). This
      is the Gumbel-Sigmoid trick and treats each token as an
      independent {keep, drop} Bernoulli decision.
    * ``"softmax"`` -> standard Gumbel(0,1) noise. Adding it to logits
      and taking the top-K of the result is the Gumbel-Top-K (a.k.a.
      Plackett-Luce) trick, which samples K items without replacement
      from a categorical distribution.

    Returned logits should be re-activated (sigmoid or softmax) and then
    passed to ``_straight_through_topk`` for the actual hard selection.
    """
    if score_activation == "sigmoid":
        u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        noise = torch.log(u) - torch.log1p(-u)
    else:  # softmax
        u = torch.rand_like(logits).clamp_(min=1e-6)
        noise = -torch.log(-torch.log(u))
    return (logits + noise) / temperature


# ---------------------------------------------------------------------------
# Scoring transformer block
# ---------------------------------------------------------------------------

class TokenSelectorLayer(nn.Module):
    """One block of the scoring transformer.

    Each block runs (on the ViT tokens of a single segment):
        ViT  --self-attention->  ViT      (only if ``use_self_attn``)
        ViT  --cross-attn-->     Query    (queries see all query tokens)
        FFN
    The output is a contextualized ViT representation used for scoring.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        use_self_attn: bool = True,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.use_self_attn = use_self_attn

        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True,
            )
            self.norm_self = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        vit: torch.Tensor,
        query: torch.Tensor,
        vit_key_padding_mask: Optional[torch.Tensor] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vit:   (BS, P, D) per-segment ViT tokens (the queries here).
            query: (BS, Q, D) per-segment summary tokens (the K/V here).
            vit_key_padding_mask:   (BS, P) bool, True at padding positions.
            query_key_padding_mask: (BS, Q) bool, True at padding positions.

        Returns:
            (BS, P, D) contextualized ViT tokens.
        """
        # 1) Self-attention among ViT tokens of the same segment.
        if self.use_self_attn:
            attn_out, _ = self.self_attn(
                vit, vit, vit,
                key_padding_mask=vit_key_padding_mask,
                need_weights=False,
            )
            vit = self.norm_self(vit + self.dropout(attn_out))

        # 2) Cross-attention from ViT tokens to summary query tokens.
        cross_out, _ = self.cross_attn(
            vit, query, query,
            key_padding_mask=query_key_padding_mask,
            need_weights=False,
        )
        vit = self.norm_cross(vit + self.dropout(cross_out))

        # 3) Feed-forward.
        ffn_out = self.ffn(vit)
        vit = self.norm_ffn(vit + self.dropout(ffn_out))
        return vit


# ---------------------------------------------------------------------------
# Learnable token selector
# ---------------------------------------------------------------------------

class LearnableTokenSelector(nn.Module):
    """Pick the top-K fine-grained ViT tokens that best complement the
    query (summary) tokens of each segment.

    Architecture
        * A lightweight (default 2 layers) transformer where every ViT
          token can attend to (a) other ViT tokens of the same segment
          via self-attention, and (b) all query tokens of the same
          segment via cross-attention. The contextualized ViT tokens are
          projected to a single scalar logit per token.
        * ``sigmoid(logit)`` gives the keep probability ``p_i`` for each
          fine-grained token.

    End-to-end learnability
        Top-K is not differentiable. We provide two interchangeable
        differentiable selectors, picked via ``selection_mode``:

        * ``"straight_through"`` (default)
            Forward: take the hard top-K indices.
            Backward: selected features are multiplied by their (sigmoid
            or softmax) scores so gradients flow back through the
            scoring network (DynamicViT / EViT style score-gating).

        * ``"gumbel"``
            Adds the appropriate Gumbel noise to the logits during
            training (logistic noise for sigmoid mode, Gumbel(0,1) noise
            for softmax mode = Gumbel-Top-K / Plackett-Luce trick) and
            then performs the same straight-through selection. Disabled
            at evaluation (we fall back to ``"straight_through"``
            automatically). Useful for stochastic exploration early in
            training.

    Choice of score activation (``score_activation``)
        Top-K ranking is invariant under any monotone activation, so
        sigmoid and softmax pick the same indices. The difference shows
        up in the score-gating step: the scalar that multiplies each
        selected feature determines the forward signal magnitude and the
        gradient flow back into the scorer.

        * ``"sigmoid"`` (default)
            Each token gets an independent keep-probability in [0, 1];
            scores have an absolute meaning ("is this token useful?").
            Gating values stay in a healthy range and natural auxiliary
            losses (e.g. ``scores.mean()`` for sparsity, or budget
            constraints on ``scores.sum(-1)``) are available.

        * ``"softmax"``
            Scores compete and sum to 1 over the P tokens of a segment.
            Picks the same top-K, but multiplies kept features by a
            potentially very small number (~1/P), which can shrink the
            forward signal when P is large (videos). Useful if you want
            a calibrated importance distribution or a strong sparsity
            prior; consider scaling the downstream consumer's LayerNorm
            accordingly.

    Inputs / Outputs
        Inputs:
            query_tokens: (B, S, Q, D) per-segment summary tokens.
            vit_tokens:   (B, S, P, D) per-segment fine-grained ViT tokens.
            top_k:        int, number of ViT tokens to keep per segment.

        Returns a dict with:
            selected_tokens: (B, S, K, D) the selected (score-gated)
                tokens. ``K = min(top_k, P)``.
            indices:         (B, S, K)    indices into the original P axis
                (so the caller can recover position info if needed).
            scores:          (B, S, P)    per-token sigmoid keep
                probability over the full original token set.
            logits:          (B, S, P)    raw score-head logits (useful
                if you want to add an auxiliary regularizer such as a
                sparsity loss on the keep probabilities).
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        use_self_attn: bool = True,
        selection_mode: str = "straight_through",
        score_activation: str = "sigmoid",
        gumbel_temperature: float = 1.0,
    ):
        super().__init__()
        if selection_mode not in ("straight_through", "gumbel"):
            raise ValueError(
                f"selection_mode must be 'straight_through' or 'gumbel', "
                f"got {selection_mode!r}"
            )
        if score_activation not in ("sigmoid", "softmax"):
            raise ValueError(
                f"score_activation must be 'sigmoid' or 'softmax', "
                f"got {score_activation!r}"
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.hidden_size = hidden_size
        self.selection_mode = selection_mode
        self.score_activation = score_activation
        self.gumbel_temperature = gumbel_temperature

        self.layers = nn.ModuleList([
            TokenSelectorLayer(
                d_model=hidden_size,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                use_self_attn=use_self_attn,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.score_head = nn.Linear(hidden_size, 1)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.score_head.weight)
        # Zero bias -> initial keep-prob ~ 0.5, which gives a balanced
        # starting point for the score-gating gradient.
        nn.init.zeros_(self.score_head.bias)

    @staticmethod
    def _clamp_topk(top_k: int, num_tokens: int) -> int:
        if top_k is None or top_k >= num_tokens:
            return num_tokens
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        return top_k

    def forward(
        self,
        query_tokens: torch.Tensor,
        vit_tokens: torch.Tensor,
        top_k: int,
        vit_padding_mask: Optional[torch.Tensor] = None,
        query_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if query_tokens.dim() != 4:
            raise ValueError(
                f"query_tokens must be (B, S, Q, D), got {tuple(query_tokens.shape)}"
            )
        if vit_tokens.dim() != 4:
            raise ValueError(
                f"vit_tokens must be (B, S, P, D), got {tuple(vit_tokens.shape)}"
            )

        b, s, q, d = query_tokens.shape
        b_v, s_v, p, d_v = vit_tokens.shape
        if not (b == b_v and s == s_v and d == d_v):
            raise ValueError(
                f"shape mismatch between query_tokens={tuple(query_tokens.shape)} "
                f"and vit_tokens={tuple(vit_tokens.shape)}"
            )
        if d != self.hidden_size:
            raise ValueError(
                f"hidden dim mismatch: tokens have D={d} but selector was built "
                f"with hidden_size={self.hidden_size}"
            )

        k = self._clamp_topk(top_k, p)
        bs = b * s

        # Flatten (B, S) -> BS so each segment is processed independently.
        vit_flat = vit_tokens.reshape(bs, p, d)
        query_flat = query_tokens.reshape(bs, q, d)

        vit_pad_flat = (
            vit_padding_mask.reshape(bs, p) if vit_padding_mask is not None else None
        )
        query_pad_flat = (
            query_padding_mask.reshape(bs, q) if query_padding_mask is not None else None
        )

        # Run the scoring transformer.
        x = vit_flat
        for layer in self.layers:
            x = layer(
                x, query_flat,
                vit_key_padding_mask=vit_pad_flat,
                query_key_padding_mask=query_pad_flat,
            )
        x = self.norm(x)

        # Per-token logit.
        logits = self.score_head(x).squeeze(-1)              # (BS, P)
        if vit_pad_flat is not None:
            # Make sure padded positions get -inf so they cannot be in top-K
            # and have probability 0 in the score map we expose.
            logits = logits.masked_fill(vit_pad_flat, float("-inf"))

        # Optionally perturb the logits with the right Gumbel noise so
        # top-K becomes a stochastic relaxation during training.
        if self.selection_mode == "gumbel" and self.training:
            logits_for_topk = _perturb_logits(
                logits, self.score_activation, self.gumbel_temperature,
            )
        else:
            logits_for_topk = logits

        # Activate to obtain the score that will gate the gradient flow.
        if self.score_activation == "sigmoid":
            scores_for_topk = torch.sigmoid(logits_for_topk)
        else:  # softmax — competes over the P tokens of each segment
            scores_for_topk = torch.softmax(logits_for_topk, dim=-1)

        # Score-gated hard top-K with STE for gradient flow.
        selected, topk_scores, topk_idx = _straight_through_topk(
            scores=scores_for_topk,
            features=vit_flat,
            k=k,
            pad_mask=vit_pad_flat,
        )

        # Expose the full deterministic (un-perturbed) score map so callers
        # can add auxiliary regularizers (e.g. sparsity loss).
        if self.score_activation == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = torch.softmax(logits, dim=-1)

        return {
            "selected_tokens": selected.view(b, s, k, d),
            "indices": topk_idx.view(b, s, k),
            "scores": scores.view(b, s, p),
            "logits": logits.view(b, s, p),
        }
