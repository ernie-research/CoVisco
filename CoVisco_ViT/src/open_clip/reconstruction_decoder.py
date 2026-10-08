"""
Reconstruction Decoder for CoVisco SigLIP Training

This module implements a lightweight transformer decoder that reconstructs
image tokens from OneVision's query tokens. The decoder uses learnable query
tokens that attend to the encoded query tokens (as key/value) to reconstruct
the original ViT patch tokens.
"""

from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn import functional as F


class TransformerDecoderLayer(nn.Module):
    """
    A single transformer decoder layer with self-attention and cross-attention.

    This is a simplified version of the decoder layer, designed for reconstruction.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor of shape (batch_size, tgt_len, d_model)
            memory: Key/Value tensor from encoder of shape (batch_size, src_len, d_model)
            src_mask: Optional mask for source (key/value) tokens
            tgt_mask: Optional mask for target (query) tokens

        Returns:
            Output tensor of shape (batch_size, tgt_len, d_model)
        """
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention (if memory is provided)
        if memory is not None:
            cross_attn_output, _ = self.cross_attn(x, memory, memory, attn_mask=src_mask)
            x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


class ReconstructionDecoder(nn.Module):
    """
    Lightweight Transformer Decoder for reconstructing image tokens from query tokens.

    Mechanism:
    - Learnable query tokens serve as the initial queries
    - OneVision's output query tokens serve as Key/Value
    - The decoder reconstructs the original ViT patch tokens

    Two independent sets of reconstruct_queries are maintained:
    - reconstruct_queries_image: for image inputs (num_seg == 1)
    - reconstruct_queries_video: for video inputs (num_seg > 1)

    This handles the case where image and video have different resolutions and
    thus different numbers of ViT tokens to reconstruct.
    """

    def __init__(
        self,
        hidden_size: int,
        num_image_query_tokens: int = 256,
        num_video_query_tokens: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.,
        layer_norm_eps: float = 1e-5,
    ):
        """
        Args:
            hidden_size: Dimension of the hidden state
            num_image_query_tokens: Number of learnable query tokens for image reconstruction
            num_video_query_tokens: Number of learnable query tokens for video frame reconstruction
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feed-forward network (default: 4 * hidden_size)
            dropout: Dropout rate
            layer_norm_eps: Epsilon for layer normalization
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_image_query_tokens = num_image_query_tokens
        self.num_video_query_tokens = num_video_query_tokens

        # Two independent sets of learnable query tokens for image vs video reconstruction
        self.reconstruct_queries_image = nn.Parameter(
            torch.randn(1, num_image_query_tokens, hidden_size) * 0.02
        )
        self.reconstruct_queries_video = nn.Parameter(
            torch.randn(1, num_video_query_tokens, hidden_size) * 0.02
        )

        # Decoder layers
        if dim_feedforward is None:
            dim_feedforward = hidden_size * 4

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters."""
        # Initialize query tokens
        nn.init.normal_(self.reconstruct_queries_image, std=0.02)
        nn.init.normal_(self.reconstruct_queries_video, std=0.02)

        # Initialize output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        query_tokens: torch.Tensor,
        return_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the reconstruction decoder.

        Args:
            query_tokens: OneVision encoder's query tokens (B, num_seg, N_q_enc, D)
            return_attentions: Whether to return attention weights

        Returns:
            reconstructed: Reconstructed tokens
                - image (num_seg==1): (B, 1, num_image_query_tokens, D)
                - video (num_seg>1):  (B, num_seg-1, num_video_query_tokens, D)
        """
        assert query_tokens.dim() == 4, f"query tokens shape should be (batch_size, num_seg, len_query, dim), but got {query_tokens.shape}"
        batch_size, num_seg, len_query, dim = query_tokens.shape

        if num_seg > 1:
            # Video: next-frame prediction, drop last segment (no supervision target)
            query_tokens = query_tokens[:, :-1, :, :]  # (B, num_seg-1, N, D)
            effective_batch = batch_size * (num_seg - 1)
            query_tokens = query_tokens.reshape(effective_batch, len_query, dim)
            queries = self.reconstruct_queries_video.expand(effective_batch, -1, -1)
        else:
            # Image: num_seg == 1
            effective_batch = batch_size
            query_tokens = query_tokens.reshape(effective_batch, len_query, dim)
            queries = self.reconstruct_queries_image.expand(effective_batch, -1, -1)
        # Pass through decoder layers
        attentions = []
        for layer in self.layers:
            queries = layer(queries, memory=query_tokens)

            if return_attentions:
                # Capture attention weights (for visualization/debugging)
                with torch.no_grad():
                    _, attn_weights = layer.cross_attn(queries, query_tokens, query_tokens)
                    attentions.append(attn_weights)

        # Project to hidden size
        reconstructed = self.output_proj(queries)  # (effective_batch, N_q, D)
        if num_seg > 1:
            reconstructed = reconstructed.reshape(batch_size, num_seg - 1, -1, dim)  # (B, num_seg-1, N_q_video, D)
        else:
            reconstructed = reconstructed.view(batch_size, 1, -1, dim)  # (B, 1, N_q_image, D)
        if return_attentions:
            return reconstructed, attentions
        return reconstructed
