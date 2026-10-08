"""
CoVisco SigLIP Model

This module implements a CoVisco model with multiple projection heads for
SigLIP training using pre-extracted embeddings as supervision signals.

The model includes:
1. CoVisco encoder (trainable)
2. Multiple projection heads (to_image, to_image_caption, to_video_caption)
3. Independent logit_scale and logit_bias for each contrast pair
4. Optional reconstruction decoder
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .covisco_vit import CoViscoEncoderModel, CoViscoEncoderConfig
from .reconstruction_decoder import ReconstructionDecoder


class ContrastiveHead(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.layer_norm(x))


class CoViscoModel(nn.Module):
    """
    CoVisco model with multiple projection heads for SigLIP training.

    Each contrast pair (image-caption, image-image, video-caption) uses independent
    logit_scale and logit_bias parameters, allowing different modalities to have
    different temperature and bias values.

    Args:
        covisco_config: CoViscoEncoderConfig for the vision encoder
        image_embed_dim: Dimension of pre-extracted image embeddings
        image_caption_embed_dim: Dimension of pre-extracted image caption text embeddings
        video_caption_embed_dim: Dimension of pre-extracted video caption text embeddings
        use_reconstruction: Whether to use reconstruction loss
        decoder_layers: Number of layers in reconstruction decoder
        decoder_num_image_queries: Number of query tokens for image reconstruction
        decoder_num_video_queries: Number of query tokens for video frame reconstruction
        init_logit_scale_image_caption: Initial logit scale for image-caption contrast
        init_logit_scale_image_image: Initial logit scale for image-image contrast
        init_logit_scale_video_caption: Initial logit scale for video-caption contrast
        init_logit_bias_image_caption: Initial logit bias for image-caption contrast
        init_logit_bias_image_image: Initial logit bias for image-image contrast
        init_logit_bias_video_caption: Initial logit bias for video-caption contrast
    """

    def __init__(
        self,
        covisco_config: CoViscoEncoderConfig,
        image_embed_dim: int = 1024,
        image_caption_embed_dim: int = 1024,
        video_caption_embed_dim: int = 1024,
        use_reconstruction: bool = False,
        decoder_layers: int = 6,
        decoder_num_image_queries: int = 256,
        decoder_num_video_queries: int = 256,
        decoder_num_heads: int = 8,
        # Logit scale initialization
        # Note: SigLIP uses logit_scale directly as temperature (not log of temperature like CLIP)
        # CLIP:    logits = exp(logit_scale) * features @ features.T, init = ln(1/0.07)
        # SigLIP:  logits = logit_scale * features @ features.T, init = 10.0 (direct temperature)
        init_logit_scale_image_caption: float = 10.0,
        init_logit_scale_image_image: float = 10.0,
        init_logit_scale_video_caption: float = 10.0,
        # Logit bias initialization
        init_logit_bias_image_caption: float = -10.0,
        init_logit_bias_image_image: float = -10.0,
        init_logit_bias_video_caption: float = -10.0,
    ):
        super().__init__()

        self.hidden_size = covisco_config.hidden_size
        self.num_query_per_seg = covisco_config.num_query_per_seg

        # 1. CoVisco Encoder (trainable)
        self.encoder = CoViscoEncoderModel(covisco_config)
        # Create a visual attribute for compatibility with factory code
        self.visual = self.encoder

        # 2. Multiple projection heads
        self.proj_to_image = ContrastiveHead(covisco_config.hidden_size, image_embed_dim)
        self.proj_to_image_caption = ContrastiveHead(covisco_config.hidden_size, image_caption_embed_dim)
        self.proj_to_video_caption = ContrastiveHead(covisco_config.hidden_size, video_caption_embed_dim)
        
        
        # self.proj_to_video_caption = nn.Parameter(
        #     torch.randn(covisco_config.hidden_size, text_embed_dim)
        #     / covisco_config.hidden_size ** 0.5
        # )

        # 3. Independent logit_scale for each contrast pair
        self.logit_scale_image_caption = nn.Parameter(
            torch.ones(1) * init_logit_scale_image_caption
        )
        self.logit_scale_image_image = nn.Parameter(
            torch.ones(1) * init_logit_scale_image_image
        )
        self.logit_scale_video_caption = nn.Parameter(
            torch.ones(1) * init_logit_scale_video_caption
        )

        # 4. Independent logit_bias for each contrast pair
        self.logit_bias_image_caption = nn.Parameter(
            torch.ones(1) * init_logit_bias_image_caption
        )
        self.logit_bias_image_image = nn.Parameter(
            torch.ones(1) * init_logit_bias_image_image
        )
        self.logit_bias_video_caption = nn.Parameter(
            torch.ones(1) * init_logit_bias_video_caption
        )

        # 5. Optional reconstruction decoder
        self.use_reconstruction = use_reconstruction
        if use_reconstruction:
            self.reconstruction_decoder = ReconstructionDecoder(
                hidden_size=covisco_config.hidden_size,
                num_image_query_tokens=decoder_num_image_queries,
                num_video_query_tokens=decoder_num_video_queries,
                num_layers=decoder_layers,
                num_heads=decoder_num_heads,
            )

    def set_grad_checkpointing(self, enable: bool = True):
        """Toggle activation checkpointing in the vision encoder (main_covisco --grad-checkpointing)."""
        self.encoder.set_grad_checkpointing(enable)

    def forward(
        self,
        pixel_values: torch.Tensor,
        visible_indices: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        return_attentions: bool = False,
        run_decoder: bool = True,
        modality: Optional[str] = None,
        segment_offset: Optional[int] = 0,
        uniform_sample_frames: Optional[bool] = False,
        frame_concat: Optional[bool] = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the CoVisco SigLIP model.

        Args:
            pixel_values: Input images/videos of shape (B, C, H, W) or (B, T, C, H, W)
            return_intermediates: Whether to return intermediate outputs (query_tokens, vit_tokens)
            return_attentions: Whether to return attention weights
            run_decoder: Whether to run the reconstruction decoder. Set to False during
                gradient accumulation cache phase to save computation.
            modality: Controls which projection heads to compute.
                "image" -> to_image, to_image_caption, to_image_video_caption
                "video" -> to_video_caption only
                None    -> all projections (backward compatible)
            uniform_sample_frames: Whether to use uniform frame sampling for video inputs.

        Returns:
            Dictionary containing model outputs.
        """
        # Get encoder outputs with return_dict=True to access all outputs
        encoder_outputs = self.encoder(
            pixel_values,
            visible_indices=visible_indices,
            output_attentions=return_attentions,
            output_hidden_states=return_intermediates,
            return_dict=True,
            segment_offset=segment_offset,
            uniform_sample_frames=uniform_sample_frames,
            frame_concat=frame_concat,
        )

        # Extract pooled output and sequence output
        pooled_output = encoder_outputs.pooler_output  # (B, D) or (B, S*Q, D) -> pooled to (B, D)
        sequence_output = encoder_outputs.last_hidden_state  # (B, S, N, D)

        # Reshape sequence_output for easier processing
        batch_size, num_segments, num_tokens, hidden_size = sequence_output.shape

        # Extract query tokens (first num_query_per_seg tokens in each segment)
        query_tokens = sequence_output[:, :, :self.num_query_per_seg, :]  # (B, S, Q, D)
        # query_tokens_flat = query_tokens.reshape(batch_size, -1, hidden_size)  # (B, S*Q, D)

        # Extract ViT patch tokens (remaining tokens)
        vit_tokens = sequence_output[:, :, self.num_query_per_seg:, :]  # (B, S, N-Q, D)
        # vit_tokens_flat = vit_tokens.reshape(batch_size, -1, hidden_size)  # (B, S*(N-Q), D)

        # Ensure pooled_output is (B, D) - handle case where it might be (B, S*Q, D)
        # if pooled_output.dim() == 3:
            # Global average pooling if we have segment-level outputs
            # pooled_output = pooled_output.mean(dim=1)

        # Apply projections selectively based on modality
        outputs = {'pooled_output': pooled_output}

        if modality is None or modality == "image":
            outputs['to_image'] = F.normalize(self.proj_to_image(pooled_output), dim=-1)
            outputs['to_image_caption'] = F.normalize(self.proj_to_image_caption(pooled_output), dim=-1)
            # Images whose captions were encoded by the *video* text encoder share the
            # video-caption head: same projection weights, separate output key so the
            # loss can report it independently from real video batches.
            outputs['to_image_video_caption'] = F.normalize(self.proj_to_video_caption(pooled_output), dim=-1)
        if modality is None or modality == "video":
            outputs['to_video_caption'] = F.normalize(self.proj_to_video_caption(pooled_output), dim=-1)

        # Optionally return intermediate outputs for reconstruction loss
        if return_intermediates:
            outputs['query_tokens'] = query_tokens
            outputs['vit_tokens'] = vit_tokens.detach()

        # Optionally run reconstruction decoder
        if self.use_reconstruction and run_decoder:
            reconstructed = self.reconstruction_decoder(
                query_tokens,
                return_attentions=False,
            )
            outputs['reconstructed'] = reconstructed

            # if return_attentions and isinstance(reconstructed, tuple):
                # outputs['reconstruction_attentions'] = reconstructed[1]

        if return_attentions:
            outputs['attentions'] = encoder_outputs.attentions

        return outputs

    def get_logit_scales_and_biases(self, modality: str = "image") -> Dict[str, torch.Tensor]:
        """
        Get logit_scale and logit_bias for the specified modality.

        Args:
            modality: Current modality ("image" or "video")

        Returns:
            Dictionary with logit_scales and logit_biases for this modality
        """
        logit_scales = {}
        logit_biases = {}

        if modality == "image":
            logit_scales['image_caption'] = self.logit_scale_image_caption
            logit_scales['image_image'] = self.logit_scale_image_image
            logit_biases['image_caption'] = self.logit_bias_image_caption
            logit_biases['image_image'] = self.logit_bias_image_image
            # Shared with real video batches: the image-with-video-caption pair lives in
            # the same embedding space, so it reuses the video-caption temperature/bias.
            logit_scales['video_caption'] = self.logit_scale_video_caption
            logit_biases['video_caption'] = self.logit_bias_video_caption
        elif modality == "video":
            logit_scales['video_caption'] = self.logit_scale_video_caption
            logit_biases['video_caption'] = self.logit_bias_video_caption

        return logit_scales, logit_biases
