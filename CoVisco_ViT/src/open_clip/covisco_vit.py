from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# from .configuration_covisco_encoder import CoViscoEncoderConfig

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from random import randint

logger = logging.get_logger(__name__)


def uniform_sample_frames(pixel_values, num_frames=16):
    """
    Uniformly sample num_frames from the T dimension of pixel_values.

    Args:
        pixel_values: tensor of shape (B, C, T, H, W)
        num_frames: number of frames to sample (default 16)

    Returns:
        tensor of shape (B, C, num_frames, H, W)
    """
    T = pixel_values.shape[2]
    if T <= num_frames:
        return pixel_values
    indices = torch.linspace(0, T - 1, num_frames, dtype=torch.long, device=pixel_values.device)
    return pixel_values[:, :, indices, :, :]
def uniform_sample_frames_and_concat(pixel_values, num_frames=16, concat_mode='horizontal'):
    """
    Uniformly sample num_frames from the T dimension of pixel_values,
    then concatenate every 4 frames into one image.
    
    Args:
        pixel_values: tensor of shape (B, C, T, H, W)
        num_frames: number of frames to sample (default 16)
        concat_mode: 'horizontal', 'vertical', or 'grid'
                     - 'horizontal': each 4 frames concatenated horizontally -> (B, C, 4, H, W*4)
                     - 'vertical': each 4 frames concatenated vertically -> (B, C, 4, H*4, W)
                     - 'grid': each 4 frames in 2x2 grid -> (B, C, 4, H*2, W*2)
    
    Returns:
        tensor of shape (B, C, 4, H', W') where H' and W' depend on concat_mode
    """
    # Step 1: Uniform sample frames
    T = pixel_values.shape[2]
    if T <= num_frames:
        sampled = pixel_values
    else:
        indices = torch.linspace(0, T - 1, num_frames, dtype=torch.long, device=pixel_values.device)
        sampled = pixel_values[:, :, indices, :, :]  # (B, C, num_frames, H, W)
    
    # Step 2: Reshape to (B, C, 4, 4, H, W) - 4 groups, each with 4 frames
    B, C, _, H, W = sampled.shape
    sampled = sampled.reshape(B, C, 4, 4, H, W)
    
    # Step 3: Concatenate each group of 4 frames
    if concat_mode == 'horizontal':
        # Concatenate along width dimension
        result = sampled.permute(0, 1, 2, 4, 3, 5).reshape(B, C, 4, H, W * 4)
    elif concat_mode == 'vertical':
        # Concatenate along height dimension
        result = sampled.permute(0, 1, 2, 3, 4, 5).reshape(B, C, 4, H * 4, W)
    elif concat_mode == 'grid':
        # Arrange in 2x2 grid: frames 0,1 in top row; frames 2,3 in bottom row
        # sampled shape: (B, C, 4, 4, H, W)
        # Rearrange to (B, C, 4, 2, 2, H, W)
        grid = sampled.reshape(B, C, 4, 2, 2, H, W)
        # Concatenate horizontally first: (B, C, 4, 2, H, W*2)
        grid = torch.cat([grid[:, :, :, 0, 0], grid[:, :, :, 0, 1]], dim=-1)  # top row
        grid_bottom = torch.cat([grid[:, :, :, 1, 0], grid[:, :, :, 1, 1]], dim=-1)  # bottom row
        # Actually let me redo this more clearly
        top_row = torch.cat([grid[:, :, :, 0, 0], grid[:, :, :, 0, 1]], dim=-1)  # (B, C, 4, H, W*2)
        bottom_row = torch.cat([grid[:, :, :, 1, 0], grid[:, :, :, 1, 1]], dim=-1)  # (B, C, 4, H, W*2)
        result = torch.cat([top_row, bottom_row], dim=-2)  # (B, C, 4, H*2, W*2)
    else:
        raise ValueError(f"Invalid concat_mode: {concat_mode}. Choose from 'horizontal', 'vertical', 'grid'.")
    
    return result

class CoViscoEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CoViscoEncoderModel`]. It is used to instantiate a
    CoVisco Encoder model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CoVisco Encoder architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        layer_norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The type of layer normalization to use. Supported values: `"layer_norm"`, `"rms_norm"`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        use_head (`bool`, *optional*, defaults to `True`):
            Whether to use the pooling head.

    Example:

    ```python
    >>> from configuration_covisco_encoder import CoViscoEncoderConfig
    >>> from modeling_covisco_encoder import CoViscoEncoderModel

    >>> # Initializing a CoViscoEncoder configuration
    >>> configuration = CoViscoEncoderConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = CoViscoEncoderModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "covisco_encoder"

    def __init__(
        self,
        output_dim=1024,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=448,
        patch_size=14,
        num_query_per_seg=100,
        segment_t_size=32,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        layer_norm_type="layer_norm",
        attention_dropout=0.0,
        initializer_range=0.02,
        rope_theta=10000.0,
        # rope_temporal_size=64,
        use_head=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dim=output_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_query_per_seg = num_query_per_seg
        self.segment_t_size = segment_t_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_type = layer_norm_type
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        # self.rope_temporal_size = rope_temporal_size  # None=use actual frames, int=fixed size (legacy: 64)
        self.use_head = use_head
        # Note: output_attentions, output_hidden_states, and use_return_dict are properties
        # inherited from PretrainedConfig and cannot be set directly
        # self.output_attentions=False
        # self.output_hidden_states=False
        # self.use_return_dict=True




try:
    from flash_attn import flash_attn_func

    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Model Docstrings
# ---------------------------------------------------------------------------

COVISCO_ENCODER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CoViscoEncoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

COVISCO_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch_size, num_channels, num_frames, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`].
        visible_indices (`torch.Tensor`, *optional*):
            Indices of visible patches for masking. Used in MAE-style pretraining or inference.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# ---------------------------------------------------------------------------
# Helper Functions & Layers
# ---------------------------------------------------------------------------


def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


def rotate_half(x):
    """
    Interleaved rotation to match Source model's implementation.
    (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D)
    # freqs: (B, L, D)

    # We need to broadcast freqs to match heads
    # (B, L, D) -> (B, 1, L, D)

    # !!! CRITICAL FIX: Cast cos/sin to q.dtype (bf16/fp16) immediately
    # freqs are typically float32, so cos() returns float32.
    # Without this cast, (q * cos) upcasts q to float32, causing FlashAttention to fail.
    cos = freqs.cos().unsqueeze(1).to(q.dtype)
    sin = freqs.sin().unsqueeze(1).to(q.dtype)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SegmentVideoRotaryEmbeddingSplit2455(nn.Module):
    """
    4D (Seg, T, H, W) Rotary frequency constructor with 2:4:5:5 split.
    Adds segment dimension to the existing VideoRotaryEmbeddingSplit466.
    """

    def __init__(self, config: CoViscoEncoderConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must also be divisible by 16 to split into 2:4:5:5."

        self.head_dim = head_dim
        self.half = half

        unit = half // 16
        self.s_size = 2 * unit
        self.t_size = 4 * unit
        self.h_size = 5 * unit
        self.w_size = 5 * unit

        self.register_buffer(
            "inv_freq_s",
            1.0 / (base ** (torch.arange(self.s_size, dtype=torch.float32) / self.s_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def forward(self, s: int, t: int, h: int, w: int, device=None):
        if device is None:
            device = self.inv_freq_s.device

        inv_s = self.inv_freq_s.to(device=device)
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        fs = torch.outer(torch.arange(s, device=device, dtype=torch.float32), inv_s)
        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        s_ids = torch.arange(s, device=device).repeat_interleave(t * h * w)
        t_ids = torch.arange(t, device=device).repeat_interleave(h * w).repeat(s)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t * s)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t * s)

        freqs = torch.cat([fs[s_ids], ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit patch positions.

        Args:
            patch_positions: [batch_size, seq_len, 4] tensor with [s, t, h, w] positions for each patch

        Returns:
            freqs: [batch_size, seq_len, half] tensor of position frequencies
        """
        device = patch_positions.device
        inv_s = self.inv_freq_s.to(device=device)
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        s_pos = patch_positions[..., 0].float()  # [batch_size, seq_len]
        t_pos = patch_positions[..., 1].float()  # [batch_size, seq_len]
        h_pos = patch_positions[..., 2].float()  # [batch_size, seq_len]
        w_pos = patch_positions[..., 3].float()  # [batch_size, seq_len]

        # Use einsum for batched outer product: [batch_size, seq_len] x [dim] -> [batch_size, seq_len, dim]
        fs = torch.einsum("bs,d->bsd", s_pos, inv_s)
        ft = torch.einsum("bs,d->bsd", t_pos, inv_t)
        fh = torch.einsum("bs,d->bsd", h_pos, inv_h)
        fw = torch.einsum("bs,d->bsd", w_pos, inv_w)

        return torch.cat([fs, ft, fh, fw], dim=-1)


class VideoRotaryEmbeddingSplit466(nn.Module):
    """
    3D (T,H,W) Rotary frequency constructor with 4:6:6 split.
    """

    def __init__(self, config: CoViscoEncoderConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must also be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.half = half

        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def forward(self, t: int, h: int, w: int, device=None):
        if device is None:
            device = self.inv_freq_t.device

        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit patch positions.

        Args:
            patch_positions: [batch_size, seq_len, 3] tensor with [t, h, w] positions for each patch

        Returns:
            freqs: [batch_size, seq_len, half] tensor of position frequencies
        """
        device = patch_positions.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        t_pos = patch_positions[..., 0].float()  # [batch_size, seq_len]
        h_pos = patch_positions[..., 1].float()  # [batch_size, seq_len]
        w_pos = patch_positions[..., 2].float()  # [batch_size, seq_len]

        # Use einsum for batched outer product: [batch_size, seq_len] x [dim] -> [batch_size, seq_len, dim]
        ft = torch.einsum("bs,d->bsd", t_pos, inv_t)
        fh = torch.einsum("bs,d->bsd", h_pos, inv_h)
        fw = torch.einsum("bs,d->bsd", w_pos, inv_w)

        return torch.cat([ft, fh, fw], dim=-1)


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-Head Attention Pooling with a learned probe (PMA-style).
    """

    def __init__(self, config: CoViscoEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        attn_output, _ = self.attention(probe, hidden_states, hidden_states)

        residual = attn_output
        attn_output = self.norm(attn_output)
        attn_output = residual + self.mlp(attn_output)

        return attn_output[:, 0]


# ---------------------------------------------------------------------------
# Modeling Components
# ---------------------------------------------------------------------------


def extract_patches(x, p):
    # x: [B, S, T, C, H, W]
    B, S, T, C, H, W = x.shape
    # Decompose H and W into h*w patches of size p x p
    h = H // p
    w = W // p
    patches = x.unfold(4, p, p).unfold(5, p, p)  # [B, S, T,C, h, w, p, p]
    patches = patches.reshape(B, S, T, C, h * w, p, p)  # [B, S, T,C, h*w, p, p]
    patches=patches.permute(0,1,2,4,3,5,6) #[B, S, T, h*w,C, p, p]
    return patches


class CoViscoEncoderEmbeddings(nn.Module):
    def __init__(self, config: CoViscoEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # Handle 4D (B, C, H, W) or 5D (B, C, T, H, W) inputs
        # if pixel_values.dim() == 4:
            # pixel_values = pixel_values.unsqueeze(2)  # (B, C, 1, H, W)

        # batch_size, channels, t_frames, height, width = pixel_values.shape

        # Merge time into batch for Conv2d
        # x_2d = pixel_values.permute(0, 2, 1, 3, 4).reshape(batch_size * t_frames, channels, height, width)

        # Patch Embed
        embeddings = self.patch_embedding(pixel_values)  # (B*T, C, Hp, Wp)
        # embeddings = embeddings.flatten(2).transpose(1, 2)  # (B*T, L_frame, C)

        # Flatten all patches
        # total_patches = t_frames * (height // self.patch_size) * (width // self.patch_size)
        # embeddings = embeddings.reshape(batch_size, total_patches, self.embed_dim)

        return embeddings


class CoViscoEncoderAttention(nn.Module):
    """Multi-headed attention with RoPE support"""

    def __init__(self, config: CoViscoEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        # hidden_states: torch.Tensor,
        query_states:torch.Tensor,
        key_states:torch.Tensor,
        value_states:torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb_q: Optional[torch.Tensor] = None,
        rotary_pos_emb_k: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # context_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # return_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = query_states.size()

        # query_states = self.q_proj(q)
        # key_states = self.k_proj(k_pre)
        # value_states = self.v_proj(v_pre)

        # (B, L, H, D) -> Transpose to (B, H, L, D)
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size,-1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if rotary_pos_emb_q is not None:
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, rotary_pos_emb_q)
            _, key_states = apply_rotary_pos_emb( key_states,key_states, rotary_pos_emb_k)
        # if context_kv_cache is not None:
            # key_states = torch.cat((context_kv_cache[0].transpose(1, 2), key_states), dim=2)
            # value_states = torch.cat((context_kv_cache[1].transpose(1, 2), value_states), dim=2)

        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, q_len):
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights + attention_mask

        # FIX: Remove dtype=torch.float32 to stay in original dtype (bf16/fp16)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        # if return_kv_cache:
            # cached_kv = (key_states.transpose(1, 2), value_states.transpose(1, 2))  # Align with Flash Attention
            # return attn_output, cached_kv
        return attn_output, attn_weights if output_attentions else None


class CoViscoEncoderFlashAttention2(nn.Module):
    """
    Multi-headed attention with RoPE support using Flash Attention 2.
    This module implements the same attention mechanism as CoViscoEncoderAttention but uses
    Flash Attention for improved performance and memory efficiency.
    """

    def __init__(self, config: CoViscoEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        query_states:torch.Tensor,
        key_states:torch.Tensor,
        value_states:torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb_q: Optional[torch.Tensor] = None,
        rotary_pos_emb_k: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # context_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # return_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using Flash Attention 2.
        """
        batch_size, q_len, _ = query_states.size()

        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)

        # Flash Attention requires (B, L, H, D) format
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states =  key_states.view(batch_size, -1, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim)

        # Apply RoPE if provided
        if rotary_pos_emb_q is not None:
            # Transpose for RoPE application: (B, L, H, D) -> (B, H, L, D)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            # NOTE: apply_rotary_pos_emb now ensures NO float32 cast happens
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, rotary_pos_emb_q)
            _, key_states = apply_rotary_pos_emb( key_states,key_states, rotary_pos_emb_k)
            # Transpose back: (B, H, L, D) -> (B, L, H, D)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
        # if context_kv_cache is not None:
            # key_states = torch.cat((context_kv_cache[0], key_states), dim=1)
            # value_states = torch.cat((context_kv_cache[1], value_states), dim=1)
        # Flash Attention forward pass
        if not _flash_attn_available:
            raise ImportError("flash_attn is not installed. Please install it to use CoViscoEncoderFlashAttention2.")

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )

        # Reshape to (B, L, embed_dim)
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        # No extra casting here.
        attn_output = self.out_proj(attn_output)
        # if return_kv_cache:
            # return attn_output, (key_states, value_states)
        return attn_output, None


COVISCO_ENCODER_ATTENTION_CLASSES = {
    "eager": CoViscoEncoderAttention,
    "flash_attention_2": CoViscoEncoderFlashAttention2,
}


class CoViscoEncoderEncoderLayer(nn.Module):
    def __init__(self, config: CoViscoEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_query_per_seg=config.num_query_per_seg
        # Get attention implementation from config, default to "flash_attention_2"
        attn_implementation = getattr(config, "_attn_implementation", "flash_attention_2")
        if attn_implementation not in COVISCO_ENCODER_ATTENTION_CLASSES:
            # Fallback to eager if flash_attention_2 is not available
            if not _flash_attn_available and attn_implementation == "flash_attention_2":
                attn_implementation = "eager"
            else:
                raise ValueError(
                    f"Unknown attention implementation: {attn_implementation}. "
                    f"Available implementations: {list(COVISCO_ENCODER_ATTENTION_CLASSES.keys())}"
                )
        self.self_attn = COVISCO_ENCODER_ATTENTION_CLASSES[attn_implementation](config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)
    def block2(self,hidden_states,rotary_pos_emb,attention_mask,output_attentions):
            b,s,n=hidden_states.shape[0],hidden_states.shape[1],hidden_states.shape[2]
            query_states=hidden_states.reshape(b*s,n,-1)# B*S N D
            query_states=self.self_attn.q_proj(query_states)
            rotary_pos_emb_q=rotary_pos_emb.reshape(b*s,n,-1)

            sq=hidden_states[:,:,:self.num_query_per_seg,:] #B S Q D
            sq_rope=rotary_pos_emb[:,:,:self.num_query_per_seg,:]# B S Q D
            sq_rope=sq_rope.unsqueeze(1).expand(-1,s,-1,-1,-1).reshape(b*s, s*self.num_query_per_seg,-1)#B*S S*Q D
            sq_states_k=self.self_attn.k_proj(sq).unsqueeze(1).expand(-1,s,-1,-1,-1).reshape(b*s, s*self.num_query_per_seg,-1) #B*S S*Q D
            sq_states_v=self.self_attn.v_proj(sq).unsqueeze(1).expand(-1,s,-1,-1,-1).reshape(b*s, s*self.num_query_per_seg,-1)#B*S S*Q D
        
            key_states=hidden_states[:,:,self.num_query_per_seg:,:]
            key_states=key_states.view(b*s,n-self.num_query_per_seg,-1)
            key_states=self.self_attn.k_proj(key_states)
            value_states=hidden_states[:,:,self.num_query_per_seg:,:]
            value_states=value_states.view(b*s,n-self.num_query_per_seg,-1)
            value_states=self.self_attn.v_proj(value_states)
            rotary_pos_emb_k=rotary_pos_emb[:,:,self.num_query_per_seg:,:]
            rotary_pos_emb_k=rotary_pos_emb_k.view(b*s,n-self.num_query_per_seg,-1)
        
            rotary_pos_emb_k=torch.cat((sq_rope,rotary_pos_emb_k),1)
            key_states=torch.cat((sq_states_k,key_states),1)
            value_states=torch.cat((sq_states_v,value_states),1)

            hidden_states, attn_weights= self.self_attn(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                rotary_pos_emb_q=rotary_pos_emb_q,
                rotary_pos_emb_k=rotary_pos_emb_k,
                output_attentions=output_attentions,
            )
            hidden_states=hidden_states.view(b,s,n,-1)
            return hidden_states, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,# B Seg n D
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,# B Seg n D
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        b,s,n=hidden_states.shape[0],hidden_states.shape[1],hidden_states.shape[2]
        hidden_states = self.layer_norm1(hidden_states)

        if layer_idx%2==0:
            hidden_states = hidden_states.view(b*s,n,-1)
            rotary_pos_emb=rotary_pos_emb.view(b*s,n,-1) 
            hidden_states, attn_weights = self.self_attn(
                query_states=self.self_attn.q_proj(hidden_states),
                key_states=self.self_attn.k_proj(hidden_states),
                value_states=self.self_attn.v_proj(hidden_states),
                attention_mask=attention_mask,
                rotary_pos_emb_q=rotary_pos_emb,
                rotary_pos_emb_k=rotary_pos_emb,
                output_attentions=output_attentions,
            )
            hidden_states = hidden_states.view(b,s,n,-1)

        else:
            hidden_states, attn_weights = self.block2(hidden_states,rotary_pos_emb,attention_mask,output_attentions)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attn_weights) if output_attentions else (hidden_states,)
        return outputs


class CoViscoEncoderEncoder(nn.Module):
    def __init__(self, config: CoViscoEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CoViscoEncoderEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.grad_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor, # B S N D
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for idx,layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.grad_checkpointing and self.training and not output_attentions:
                # Recompute layer activations during backward to trade compute for memory.
                # use_reentrant=False supports kwargs and preserves RNG state, so the
                # attention dropout mask (if attention_dropout > 0) is identical across
                # the two passes. Layers returning attention weights force the plain path.
                layer_outputs = checkpoint(
                    layer,
                    hidden_states,
                    layer_idx=idx,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    output_attentions=output_attentions,
                    layer_idx=idx
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# ---------------------------------------------------------------------------
# Main Models
# ---------------------------------------------------------------------------


@add_start_docstrings(
    "The bare CoVisco Encoder Model outputting raw hidden-states without any specific head on top.",
    COVISCO_ENCODER_START_DOCSTRING,
)
class CoViscoEncoderPreTrainedModel(PreTrainedModel):
    config_class = CoViscoEncoderConfig
    base_model_prefix = "covisco_encoder"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CoViscoEncoderEncoderLayer"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            # Fix: RMSNorm doesn't have bias, must check hasattr first
            module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()


@add_start_docstrings(
    "CoVisco Encoder Model with a vision transformer encoder.",
    COVISCO_ENCODER_START_DOCSTRING,
)
class CoViscoEncoderModel(CoViscoEncoderPreTrainedModel):
    def __init__(self, config: CoViscoEncoderConfig):
        super().__init__(config)
        self.config = config
        self.segment_t_size=config.segment_t_size
        self.embeddings = CoViscoEncoderEmbeddings(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = CoViscoEncoderEncoder(config)
        self.video_rope = SegmentVideoRotaryEmbeddingSplit2455(config)
        self.query_tokens=nn.Parameter(0.05*torch.randn(1,1,config.num_query_per_seg,self.config.hidden_size))
        self.num_query_per_seg=config.num_query_per_seg
        self.embed_dim=config.output_dim
        # Expose image_size for compatibility with factory code
        self.image_size = config.image_size
        # self.proj = nn.Parameter(torch.randn(config.hidden_size, self.embed_dim) / config.hidden_size ** 0.5)

        if config.use_head:
            self.layernorm_post = get_norm_layer(config)
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        else:
            self.layernorm_post = None
            self.head = None

        self.post_init()

    def set_grad_checkpointing(self, enable: bool = True):
        """Toggle per-layer activation checkpointing in the encoder stack."""
        self.encoder.grad_checkpointing = enable

    def offset_positions(self, visible_indices, num_token_per_frame,seg_offset=0,reset_segment_t_size=None,is_image=False):
       # visible_indices: [B, L], global token indices
       # Determine which segment each token belongs to
       if reset_segment_t_size is None:
          segment_t_size=self.segment_t_size
       else:
          segment_t_size = reset_segment_t_size
       offset = segment_t_size * num_token_per_frame  # Total tokens per segment  
       segment_ids = visible_indices // offset  # [B, L]，0, 1, 2, ...

       # Image data (one frame/segment): randomly shift within the segment during training
       # so the model does not over-focus on frame 0
       if is_image and self.training:
          rand_shift = torch.randint(0, segment_t_size-1, (1,), device=visible_indices.device).item()
       else:
          rand_shift = 0
    
       # Compute the offset: (segment_id + 1 + rand_shift) * num_token_per_frame
       adjustment = (segment_ids + 1 + rand_shift) * num_token_per_frame+seg_offset*(segment_t_size+1)*num_token_per_frame
    
       # Apply the offset
       visible_indices = visible_indices + adjustment
       return visible_indices
    def get_query_tokens_rope(self,batch_size,num_seg,device,seg_offset=0):
    #    batch_size = hidden_states.shape[0]
    #    num_seg = hidden_states.shape[1]
       num_query = self.num_query_per_seg  # 100
       # Generate query-token position indices: [B, S, N, 4] -> (s, t, h, w)
       # Assume 100 queries are uniformly distributed on a 10x10 grid
       h_vals = torch.arange(10, device=device).repeat(10)  # [100]
       w_vals = torch.arange(10, device=device).repeat_interleave(10)  # [100]
       # Expand to [B, S, N]
       segment_ids = torch.arange(num_seg, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, num_query)+seg_offset
       h_ids = h_vals.unsqueeze(0).unsqueeze(0).expand(batch_size, num_seg, -1)
       w_ids = w_vals.unsqueeze(0).unsqueeze(0).expand(batch_size, num_seg, -1)
       t_ids = torch.zeros(batch_size, num_seg, num_query, dtype=torch.long, device=device)
       query_positions = torch.stack([segment_ids, t_ids, h_ids, w_ids], dim=-1)  # [B, S, N, 4]
       query_positions=query_positions.view(batch_size,num_seg*num_query,4)
       # Generate query-token RoPE with forward_from_positions
       query_token_rope = self.video_rope.forward_from_positions(query_positions)# B L D/2
       query_token_rope=torch.cat((query_token_rope,query_token_rope),dim=-1)
       query_token_rope=query_token_rope.view(batch_size,num_seg,num_query,-1)
       return query_token_rope
    def get_segments(self,pixel_values,visible_indices=None, seg_offset=0):
        patch_size=self.config.patch_size
        batch_size=pixel_values.shape[0]
        if pixel_values.dim() == 5:#B C T H W
            # Use config.rope_temporal_size if set, otherwise use actual frame count
            num_frame=pixel_values.shape[2]
            channels=pixel_values.shape[1]
            
            # t_frames = (
                # self.config.rope_temporal_size if self.config.rope_temporal_size is not None else pixel_values.shape[2]
            # )
            height = pixel_values.shape[3]
            width = pixel_values.shape[4]

            if num_frame<self.segment_t_size:
                num_seg=1
                t_frames=num_frame
                pixel_values=pixel_values.unsqueeze(1)# B 1 C T H W
                pixel_values=pixel_values.permute(0,1,3,2,4,5)# B 1 T C H W
            else:
                assert num_frame % self.segment_t_size == 0
                num_seg=num_frame//self.segment_t_size
                t_frames=self.segment_t_size
                pixel_values=pixel_values.transpose(1,2)# B T C H W
                pixel_values=pixel_values.reshape(batch_size,num_seg,self.segment_t_size,channels,height,width)# B S T C H W
            
        else:# B C H W
            num_frame=1
            num_seg=1
            t_frames = 1
            height = pixel_values.shape[2]
            width = pixel_values.shape[3]
            pixel_values=pixel_values.unsqueeze(1).unsqueeze(1)# B 1 1 C H W
        
        h=height//patch_size
        w=width//patch_size
        total_patches=num_frame*h*w
        pixel_values=extract_patches(pixel_values,p=patch_size)#[B, S, T, h*w,C, p, p]
        pixel_values=pixel_values.reshape(batch_size,num_seg*t_frames*h*w,-1,patch_size,patch_size)

        # TODO: Select the patches to compute and obtain their positional encoding
        if visible_indices is None:
             visible_indices = (
                torch.arange(total_patches, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)# B L
            )
        else:
            assert visible_indices.shape[1]%num_seg==0
         

        visible_indices_offset=self.offset_positions(visible_indices, num_token_per_frame=h*w,seg_offset=seg_offset,is_image=(num_frame==1))# Offset to add query tokens (t=0)

         # Visible indices: B L, corresponding to contiguous token indices.
        freqs_full = self.video_rope(
                s=num_seg+seg_offset,
                t=self.segment_t_size+1,# Add query tokens as the first frame of each segment
                h=h,
                w=w,
                device=pixel_values.device,
            )
        freqs_visible = freqs_full[visible_indices_offset]# B L D/2 
        half_dim = freqs_visible.shape[-1]
        freqs_visible=freqs_visible.reshape(batch_size,num_seg,-1,half_dim)
        # Concatenate D/2 + D/2 -> D for applying rope
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)#B nums_seg L D

        
        # visible_indices=visible_indices.view(batch_size,num_seg,-1)
         # Method 1: use gather (suitable for selecting specific dimensions)
        C=pixel_values.shape[2]
        pixel_values = pixel_values.gather(
               1,  # Index along the L dimension
               visible_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, patch_size, patch_size)
               )
               # Output: [B, l, C, P, P]
        pixel_values=pixel_values.view(-1,C,patch_size,patch_size)
        pixel_values=self.embeddings(pixel_values)# [B*L, C, 1,1]
        pixel_values=pixel_values.view(batch_size,num_seg,-1,self.config.hidden_size)# Preprocessing must keep the selected token count per segment consistent and ordered by patch
        return pixel_values, freqs_visible
    def get_uniform_frame_segments(self,pixel_values,visible_indices=None, seg_offset=0):
        patch_size=self.config.patch_size
        batch_size=pixel_values.shape[0]
        segment_t_size=4
        sample_frames=16
        if pixel_values.dim() == 5:#B C T H W
            # Use config.rope_temporal_size if set, otherwise use actual frame count
            pixel_values = uniform_sample_frames(pixel_values, num_frames=sample_frames)
            num_frame=pixel_values.shape[2]
            channels=pixel_values.shape[1]
            
            # t_frames = (
                # self.config.rope_temporal_size if self.config.rope_temporal_size is not None else pixel_values.shape[2]
            # )
            height = pixel_values.shape[3]
            width = pixel_values.shape[4]

            if num_frame<segment_t_size:
                num_seg=1
                t_frames=num_frame
                pixel_values=pixel_values.unsqueeze(1)# B 1 C T H W
                pixel_values=pixel_values.permute(0,1,3,2,4,5)# B 1 T C H W
            else:
                assert num_frame % segment_t_size == 0
                num_seg=num_frame//segment_t_size
                t_frames=segment_t_size
                pixel_values=pixel_values.transpose(1,2)# B T C H W
                pixel_values=pixel_values.reshape(batch_size,num_seg,segment_t_size,channels,height,width)# B S T C H W
            
        else:# B C H W
            num_frame=1
            num_seg=1
            t_frames = 1
            height = pixel_values.shape[2]
            width = pixel_values.shape[3]
            pixel_values=pixel_values.unsqueeze(1).unsqueeze(1)# B 1 1 C H W
        
        h=height//patch_size
        w=width//patch_size
        total_patches=num_frame*h*w
        pixel_values=extract_patches(pixel_values,p=patch_size)#[B, S, T, h*w,C, p, p]
        pixel_values=pixel_values.reshape(batch_size,num_seg*t_frames*h*w,-1,patch_size,patch_size)

        # TODO: Select the patches to compute and obtain their positional encoding
        if visible_indices is None:
             visible_indices = (
                torch.arange(total_patches, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)# B L
            )
        else:
            assert visible_indices.shape[1]%num_seg==0
         

        visible_indices_offset=self.offset_positions(visible_indices, num_token_per_frame=h*w,seg_offset=seg_offset,reset_segment_t_size=segment_t_size,is_image=(num_frame==1))# Offset to add query tokens (t=0)

         # Visible indices: B L, corresponding to contiguous token indices.
        freqs_full = self.video_rope(
                s=num_seg+seg_offset,
                t=segment_t_size+1,# Add query tokens as the first frame of each segment
                h=h,
                w=w,
                device=pixel_values.device,
            )
        freqs_visible = freqs_full[visible_indices_offset]# B L D/2 
        half_dim = freqs_visible.shape[-1]
        freqs_visible=freqs_visible.reshape(batch_size,num_seg,-1,half_dim)
        # Concatenate D/2 + D/2 -> D for applying rope
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)#B nums_seg L D

        
        # visible_indices=visible_indices.view(batch_size,num_seg,-1)
         # Method 1: use gather (suitable for selecting specific dimensions)
        C=pixel_values.shape[2]
        pixel_values = pixel_values.gather(
               1,  # Index along the L dimension
               visible_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, patch_size, patch_size)
               )
               # Output: [B, l, C, P, P]
        pixel_values=pixel_values.view(-1,C,patch_size,patch_size)
        pixel_values=self.embeddings(pixel_values)# [B*L, C, 1,1]
        pixel_values=pixel_values.view(batch_size,num_seg,-1,self.config.hidden_size)# Preprocessing must keep the selected token count per segment consistent and ordered by patch
        return pixel_values, freqs_visible
    def get_uniform_frame_segments_then_concat_frames(self,pixel_values,visible_indices=None, seg_offset=0):
        patch_size=self.config.patch_size
        batch_size=pixel_values.shape[0]
        segment_t_size=1
        sample_frames=16
        if pixel_values.dim() == 5:#B C T H W
            # Use config.rope_temporal_size if set, otherwise use actual frame count
            pixel_values = uniform_sample_frames_and_concat(pixel_values, num_frames=sample_frames,concat_mode='horizontal')# B, C, 4, H', W'
            num_frame=pixel_values.shape[2]#4
            channels=pixel_values.shape[1]#3
            
            # t_frames = (
                # self.config.rope_temporal_size if self.config.rope_temporal_size is not None else pixel_values.shape[2]
            # )
            height = pixel_values.shape[3]
            width = pixel_values.shape[4]

            if num_frame<segment_t_size:
                num_seg=1
                t_frames=num_frame
                pixel_values=pixel_values.unsqueeze(1)# B 1 C T H W
                pixel_values=pixel_values.permute(0,1,3,2,4,5)# B 1 T C H W
            else:
                assert num_frame % segment_t_size == 0
                num_seg=num_frame//segment_t_size
                t_frames=segment_t_size
                pixel_values=pixel_values.transpose(1,2)# B T C H W
                pixel_values=pixel_values.reshape(batch_size,num_seg,segment_t_size,channels,height,width)# B S T C H W
            
        else:# B C H W
            num_frame=1
            num_seg=1
            t_frames = 1
            height = pixel_values.shape[2]
            width = pixel_values.shape[3]
            pixel_values=pixel_values.unsqueeze(1).unsqueeze(1)# B 1 1 C H W
        
        h=height//patch_size
        w=width//patch_size
        total_patches=num_frame*h*w
        pixel_values=extract_patches(pixel_values,p=patch_size)#[B, S, T, h*w,C, p, p]
        pixel_values=pixel_values.reshape(batch_size,num_seg*t_frames*h*w,-1,patch_size,patch_size)

        # TODO: Select the patches to compute and obtain their positional encoding
        if visible_indices is None:
             visible_indices = (
                torch.arange(total_patches, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)# B L
            )
        else:
            assert visible_indices.shape[1]%num_seg==0
         

        visible_indices_offset=self.offset_positions(visible_indices, num_token_per_frame=h*w,seg_offset=seg_offset,reset_segment_t_size=segment_t_size,is_image=(num_frame==1))# Offset to add query tokens (t=0)

         # Visible indices: B L, corresponding to contiguous token indices.
        freqs_full = self.video_rope(
                s=num_seg+seg_offset,
                t=segment_t_size+1,# Add query tokens as the first frame of each segment
                h=h,
                w=w,
                device=pixel_values.device,
            )
        freqs_visible = freqs_full[visible_indices_offset]# B L D/2 
        half_dim = freqs_visible.shape[-1]
        freqs_visible=freqs_visible.reshape(batch_size,num_seg,-1,half_dim)
        # Concatenate D/2 + D/2 -> D for applying rope
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)#B nums_seg L D

        
        # visible_indices=visible_indices.view(batch_size,num_seg,-1)
         # Method 1: use gather (suitable for selecting specific dimensions)
        C=pixel_values.shape[2]
        pixel_values = pixel_values.gather(
               1,  # Index along the L dimension
               visible_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, patch_size, patch_size)
               )
               # Output: [B, l, C, P, P]
        pixel_values=pixel_values.view(-1,C,patch_size,patch_size)
        pixel_values=self.embeddings(pixel_values)# [B*L, C, 1,1]
        pixel_values=pixel_values.view(batch_size,num_seg,-1,self.config.hidden_size)# Preprocessing must keep the selected token count per segment consistent and ordered by patch
        return pixel_values, freqs_visible
    
   
    @add_start_docstrings_to_model_forward(COVISCO_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CoViscoEncoderConfig)
    def forward(
        self,
        pixel_values: torch.Tensor,
        visible_indices: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        segment_offset:Optional[int] = 0,
        uniform_sample_frames:Optional[bool] = False,
        frame_concat:Optional[bool] = False
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoModel, AutoImageProcessor
        >>> from PIL import Image

        >>> model = AutoModel.from_pretrained("CoVisco-L-14", trust_remote_code=True)
        >>> preprocessor = AutoImageProcessor.from_pretrained("CoVisco-L-14", trust_remote_code=True)
        >>> image = Image.open("path/to/your/image.jpg")  # Replace with your image path
        >>> pixel_values = preprocessor(images=image, return_tensors="pt")["pixel_values"]
        >>> outputs = model(pixel_values)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 4. Pre-Norm & Encoder
        if uniform_sample_frames:
            if not frame_concat:
                hidden_states, freqs_visible = self.get_uniform_frame_segments(pixel_values,
                                                                   visible_indices=None,seg_offset=segment_offset)
            else:
                hidden_states, freqs_visible = self.get_uniform_frame_segments_then_concat_frames(pixel_values,
                                                                   visible_indices=None,seg_offset=segment_offset)
        else:
            hidden_states, freqs_visible = self.get_segments(pixel_values,visible_indices,segment_offset)
        if hidden_states.shape[1]==1 and hidden_states.shape[2]>1000 and self.training:
            # Randomly drop tokens, retaining 80% (dropping 20%); update hidden_states and freqs_visible together
            num_tokens = hidden_states.shape[2]  # N
            keep_num = int(num_tokens * 0.8)
            # Generate independent random keep indices for each sample
            perm = torch.rand(hidden_states.shape[0], num_tokens, device=hidden_states.device).argsort(dim=1)
            keep_indices = perm[:, :keep_num].sort(dim=1)[0]  # (B, keep_num), sorted in the original spatial order
            idx_hs = keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, hidden_states.shape[-1])  # (B, 1, keep_num, D)
            idx_rope = keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, freqs_visible.shape[-1])  # (B, 1, keep_num, D_rope)
            hidden_states = hidden_states.gather(2, idx_hs)
            freqs_visible = freqs_visible.gather(2, idx_rope)
        query_tokens=self.query_tokens.expand(hidden_states.shape[0],hidden_states.shape[1],-1,-1)# B S N D
        hidden_states=torch.cat([query_tokens,hidden_states],dim=2)
        query_token_rope=self.get_query_tokens_rope(batch_size=hidden_states.shape[0],
                                                    num_seg=hidden_states.shape[1],device=hidden_states.device,seg_offset=segment_offset)
        freqs_visible=torch.cat([query_token_rope, freqs_visible], dim=2)
        hidden_states = self.layernorm_pre(hidden_states)# B S N D

        # # fix: gather hidden_states to match freqs_visible when using sparse visible_indices
        # num_visible = visible_indices.shape[1]
        # if num_visible != total_patches:
        #     # sparse mode: select only visible patches
        #     hidden_states = hidden_states.gather(
        #         1, visible_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        #     )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_visible,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]# B S N D

        # Apply post-norm if configured
        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        # 5. Pooling Head
        pooled_output = None
        if self.head is not None:
            b=sequence_output.shape[0]
            s=sequence_output.shape[1]
            # q=sequence_output.shape[2]
            pooled_output = sequence_output[:,:,:self.num_query_per_seg,:].reshape(b,s*self.num_query_per_seg,-1)# B S*Q D
            pooled_output = self.head(pooled_output)
            # if self.proj is not None:
            #    pooled_output  = pooled_output  @ self.proj
        # return pooled_output
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
