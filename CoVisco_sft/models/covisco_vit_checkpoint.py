"""Utilities for loading CoVisco ViT checkpoints."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _resolve_source(path_or_repo: str | Path) -> Mapping[str, torch.Tensor]:
    """Return a raw (un-remapped) state_dict from any supported source.

    Supports:
      * a local PyTorch ``.pt`` training checkpoint,
      * a local ``.safetensors`` file,
      * a local HuggingFace release directory (containing model.safetensors),
      * a HuggingFace Hub repo id (downloaded via snapshot_download).
    """
    p = Path(path_or_repo)

    # 1) Local .pt checkpoint
    if p.suffix == ".pt" and p.exists():
        return _unwrap_state_dict(torch.load(str(p), map_location="cpu"))

    # 2) Local .safetensors file
    if p.suffix == ".safetensors" and p.exists():
        from safetensors.torch import load_file

        return load_file(str(p))

    # 3) Local HF release directory, or a Hub repo id to download
    if p.is_dir():
        model_dir = str(p)
    else:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(repo_id=str(path_or_repo), repo_type="model")

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
    from safetensors.torch import load_file

    return load_file(safetensors_path)


def _unwrap_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _strip_prefix(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_vit_weights_direct(
    vision_model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict:
    """Load ViT weights by exact name matching (for original SigLIP naming).

    ``checkpoint_path`` may be a local ``.pt`` / ``.safetensors`` file, a local
    HuggingFace release directory, or a HuggingFace Hub repo id (e.g.
    ``ernie-research/CoVisco-L-14``). After removing the module./model./encoder.
    prefixes from checkpoint keys, match them exactly against
    vision_model.state_dict() keys; skip shape mismatches.
    """
    # Strip module./model. and then encoder. to align with self.vit.encoder names
    source = {}
    for k, v in _resolve_source(checkpoint_path).items():
        if not torch.is_tensor(v):
            continue
        k = _strip_prefix(k)          # Remove the module./model. prefix
        if k.startswith("encoder."):  # Remove the encoder. prefix
            k = k[len("encoder."):]
        source[k] = v
    target_sd = vision_model.state_dict()

    loaded, missing, unexpected = [], [], []

    for src_key, src_val in source.items():
        if src_key not in target_sd:
            unexpected.append(src_key)
            continue
        tgt_val = target_sd[src_key]
        if src_val.shape != tgt_val.shape:
            print(f"[ViT ckpt] shape mismatch {src_key}: ckpt {tuple(src_val.shape)} vs model {tuple(tgt_val.shape)}, skipped")
            missing.append(src_key)
            continue
        target_sd[src_key] = src_val.to(dtype=tgt_val.dtype)
        loaded.append(src_key)

    for key in target_sd:
        if key not in source:
            missing.append(key)

    vision_model.load_state_dict(target_sd, strict=False)
    return {
        "loaded": loaded,
        "missing": missing,
        "unexpected": unexpected,
        "total_src": len(source),
    }
