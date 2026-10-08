"""Shared HuggingFace-release loader for the CoVisco eval scripts.

The original eval scripts (``eval_zeroshot_siglip2.py``,
``eval_mmeb_video_retrieval*.py``, ``eval_mme_v2_all*.py``) build the CoVisco
vision tower from the ``open_clip`` training package and load a raw ``.pt``
training checkpoint. This module provides a drop-in alternative that loads the
**released** vision encoder instead:

    config.json + model.safetensors + the self-contained covisco_*.py code

shipped in the Hugging Face repo (e.g. ``hf_release/CoVisco-L-14`` or the Hub
repo ``ernie-research/CoVisco-L-14``). No dependency on the ``open_clip`` package
for the vision side.

The teacher *text* encoders (SigLIP2 via open_clip, Qwen3-VL via
sentence-transformers) are NOT part of this release and are left untouched by
the ``*_hf.py`` wrappers.

``--hf_model`` accepts either:
  * a local directory containing config.json + model.safetensors + covisco_*.py
  * a Hugging Face Hub repo id (downloaded via ``snapshot_download``)
"""

import importlib
import json
import os
import sys

import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default to the Hugging Face Hub repo id for the released CoVisco-L-14 encoder.
# ``resolve_hf_model`` fetches it via snapshot_download (set HF_TOKEN for the
# private repo). Pass a local directory (e.g. ``hf_release/CoVisco-L-14``) via
# ``--hf_model`` / ``HF_MODEL`` to load from disk instead.
DEFAULT_HF_MODEL = "ernie-research/CoVisco-L-14"

# OpenAI-CLIP preprocessing constants. These match ``preprocess_cfg`` in the
# released config.json (mean/std/interpolation/resize_mode) and the transforms
# used by the original eval scripts.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def resolve_hf_model(hf_model: str) -> str:
    """Return a local directory for the CoVisco release.

    If ``hf_model`` is an existing directory it is returned as-is; otherwise it
    is treated as a Hub repo id and fetched with ``snapshot_download`` (this
    needs network access, and a token for a private repo via the standard
    ``HF_TOKEN`` env var / ``huggingface-cli login``).
    """
    if os.path.isdir(hf_model):
        return hf_model
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=hf_model, repo_type="model")


_IMPORT_CACHE = {}


def _import_covisco_modules(model_dir: str):
    """Import the self-contained covisco_vit / covisco_model from ``model_dir``.

    The released modules use top-level absolute imports (``from covisco_vit
    import ...``), so the directory must be on ``sys.path``. Cached per dir.
    """
    if model_dir in _IMPORT_CACHE:
        return _IMPORT_CACHE[model_dir]
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    covisco_vit = importlib.import_module("covisco_vit")
    covisco_model = importlib.import_module("covisco_model")
    _IMPORT_CACHE[model_dir] = (covisco_vit, covisco_model)
    return covisco_vit, covisco_model


def load_vision_model(hf_model: str, device: str = "cpu", image_size=None,
                      segment_t_size=None, video_caption_embed_dim=None):
    """Instantiate ``CoViscoModel`` from the release and load model.safetensors.

    ``image_size`` and ``segment_t_size`` override the values stored in
    config.json when given. Both are runtime-only (they do not change any weight
    shapes: patch embedding is a conv, positions are RoPE, and the abstract
    query tokens depend on ``num_query_per_seg``), so the same safetensors
    weights load cleanly under any value. The original video eval scripts build
    the config with ``segment_t_size=16`` (vs. 32 in the released config.json),
    so the wrappers pass that through to reproduce their behaviour.

    Returns ``(model, cfg, model_dir)`` where ``model`` is ``.eval().to(device)``
    and ``cfg`` is the parsed config.json.
    """
    model_dir = resolve_hf_model(hf_model)
    covisco_vit, covisco_model = _import_covisco_modules(model_dir)

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    enc_kwargs = dict(cfg["covisco_encoder_cfg"])
    if image_size is not None:
        enc_kwargs["image_size"] = image_size
    if segment_t_size is not None:
        enc_kwargs["segment_t_size"] = segment_t_size

    covisco_kwargs = dict(cfg["covisco_cfg"])
    if video_caption_embed_dim is not None:
        covisco_kwargs["video_caption_embed_dim"] = video_caption_embed_dim

    # The released CoViscoModel is a transformers.PreTrainedModel built from a
    # CoViscoConfig (which wraps these dicts), not the old keyword API
    # ``CoViscoModel(onevision_config=..., **covisco_cfg)``.
    config = covisco_model.CoViscoConfig(
        covisco_encoder_cfg=enc_kwargs,
        covisco_cfg=covisco_kwargs,
        preprocess_cfg=cfg.get("preprocess_cfg"),
    )
    model = covisco_model.CoViscoModel(config)

    from safetensors.torch import load_file

    state = load_file(os.path.join(model_dir, "model.safetensors"))
    # The release stores the encoder twice (``encoder.*`` and its alias
    # ``visual.*``); both resolve to the same module, so a clean load reports no
    # missing / unexpected keys.
    missing, unexpected = model.load_state_dict(state, strict=False)
    # ``missing`` = model params NOT present in the checkpoint, i.e. left at their
    # random init. Unlike the benign ``visual.*`` alias duplication (which only
    # produces *unexpected* keys), any missing key means part of the model is
    # uninitialized -- warn loudly and quantify so a largely-random load can't
    # hide behind a truncated line.
    if missing:
        n_model_keys = len(model.state_dict())
        print(
            f"[WARN] {len(missing)}/{n_model_keys} model params missing from "
            f"checkpoint and left at RANDOM init -- outputs will be wrong if this "
            f"is unexpected. First 10: {missing[:10]}"
        )
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected checkpoint keys ignored: {unexpected[:10]}")
    return model.eval().to(device), cfg, model_dir


def build_preprocess(image_size: int = 224, mean=_CLIP_MEAN, std=_CLIP_STD):
    """Square validation transform: shortest-edge resize + center crop.

    Mirrors open_clip's ``resize_mode='shortest'`` and the transform used by the
    original eval scripts (and config.json ``preprocess_cfg``).
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


_PATCH_SIZE = 14


def build_native_preprocess(patch_size: int = _PATCH_SIZE, max_side: int = 1400,
                            min_side=None, mean=_CLIP_MEAN, std=_CLIP_STD):
    """Aspect-ratio-preserving preprocessing at (near-)native resolution.

    Unlike ``build_preprocess`` (Resize + CenterCrop to a fixed square, which
    distorts/crops away detail -- fatal for text-heavy document pages), this
    keeps the original aspect ratio and only:
      * scales so the long side <= ``max_side`` (memory cap),
      * rounds each side to a multiple of ``patch_size`` (>= ``min_side``).

    The encoder is RoPE-only with no learned position table and derives its
    patch grid as h=H//patch_size, w=W//patch_size at runtime, so it accepts any
    such H x W (including non-square). Because a batch is a single dense tensor
    (no resolution packing), callers must encode one image at a time when
    using this.
    """
    from torchvision import transforms

    min_side = min_side or patch_size
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)

    def _round(x):
        return max(min_side, int(round(x / patch_size)) * patch_size)

    def preprocess(img):
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        nw, nh = _round(w * scale), _round(h * scale)
        img = img.resize((nw, nh), Image.BICUBIC)
        return normalize(to_tensor(img))

    return preprocess
