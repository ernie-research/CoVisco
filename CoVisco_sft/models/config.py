"""Model configuration dataclasses.

Centralized configuration for the ViT, projector, LLM, and token strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ViTConfig:
    name: str = "CoVisco-L-14"
    pretrained_path: str = ""  # Local .pt/.safetensors file, HF release dir, or Hub repo id (e.g. ernie-research/CoVisco-L-14)
    hidden_size: int = 1024
    num_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    num_query_per_seg: int = 100  # Must be a perfect square
    segment_t_size: int = 32
    patch_size: int = 14
    image_size: int = 224        # Image input resolution (default, kept for compatibility)
    image_size_image: int = 0    # Image-specific resolution; 0 reuses image_size
    image_size_video: int = 0    # Video-frame resolution; 0 reuses image_size
    num_channels: int = 3
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02


@dataclass
class ProjectorConfig:
    type: str = "two_layer_mlp"
    sms: int = 1
    ffn_mult: int = 2


@dataclass
class LLMConfig:
    name: str = "Qwen3-4B"
    path: str = ""  # Hugging Face model path
    hidden_size: int = 2560
    num_layers: int = 36
    num_attention_heads: int = 32
    image_pad_token_id: int = 151655   # <|image_pad|>
    vision_start_token_id: int = 151652  # <|vision_start|>
    vision_end_token_id: int = 151653    # <|vision_end|>
    freeze: bool = True  # Freeze the LLM by default (can be enabled during SFT)


@dataclass
class TokenStrategyConfig:
    """Dynamic token composition strategy."""
    enabled: bool = True
    p_query_only: float = 0.2
    p_vit_only: float = 0.2
    p_query_and_vit: float = 0.6
    # Optional per-modality strategy sampling probabilities. If None, the modality
    # reuses the shared p_* values above.
    # YAML form (the three values must sum to 1):
    #   p_image: {query_only: 0.3, vit_only: 0.3, query_and_vit: 0.4}
    #   p_video: {query_only: 0.1, vit_only: 0.3, query_and_vit: 0.6}
    p_image: Optional[Dict[str, float]] = None
    p_video: Optional[Dict[str, float]] = None
    vit_token_ratios: Tuple[float, ...] = (0.25, 0.5, 1.0)  # Candidate vit-token retention ratios for query_and_vit
    # Optional per-modality query_and_vit ratio candidates. If None, reuse the shared vit_token_ratios.
    vit_token_ratios_image: Optional[Tuple[float, ...]] = None
    vit_token_ratios_video: Optional[Tuple[float, ...]] = None
    arrangement: str = "interleave"
    vit_token_counts_image: Optional[Tuple[int, ...]] = None  # vit tokens per image segment for vit_only
    vit_token_counts_video: Optional[Tuple[int, ...]] = None  # vit tokens per video segment for vit_only


@dataclass
class TokenSelectorConfig:
    """Query-guided vit token selector."""
    method: str = "straight_through"  # "straight_through" | "gumbel"
    score_activation: str = "sigmoid"  # "sigmoid" | "softmax"
    num_layers: int = 2
    num_heads: int = 8
    gumbel_temperature: float = 1.0
    mmr_lambda: float = 0.0  # 0.0 = disabled (top-K); >0 enables MMR at inference
    # Soft bound on the score logits: logit = s * tanh(logit / s). Keeps sigmoid
    # out of its saturated (zero-gradient, bf16-tie) regime. 0 = unbounded.
    logit_scale: float = 4.0


def build_model_config(config_dict: Dict[str, Any]) -> "ModelConfig":
    """Recursively construct ModelConfig from a YAML dictionary."""
    config_dict = dict(config_dict or {})
    return ModelConfig(
        name=config_dict.get("name", "covisco-qwen3-4b"),
        vit=ViTConfig(**config_dict.get("vit", {})),
        projector=ProjectorConfig(**config_dict.get("projector", {})),
        llm=LLMConfig(**config_dict.get("llm", {})),
        token_strategy=TokenStrategyConfig(**config_dict.get("token_strategy", {})),
        token_selector=TokenSelectorConfig(**config_dict.get("token_selector", {})),
        sparsity_lambda=float(config_dict.get("sparsity_lambda", 0.0)),
        first_frame_max_ratio=float(config_dict.get("first_frame_max_ratio", 0.0)),
    )


@dataclass
class ModelConfig:
    name: str = "covisco-qwen3-4b"
    vit: ViTConfig = field(default_factory=ViTConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    token_strategy: TokenStrategyConfig = field(default_factory=TokenStrategyConfig)
    token_selector: TokenSelectorConfig = field(default_factory=TokenSelectorConfig)
    sparsity_lambda: float = 0.0  # Token-selector sparsity regularization coefficient (0 disables it)
    first_frame_max_ratio: float = 0.0  # First-frame token ratio cap during inference (0.0 = unlimited, video only)
