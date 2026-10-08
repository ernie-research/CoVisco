"""Utilities for loading CoVisco ViT checkpoints."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


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

    After removing the module./model. prefixes from checkpoint keys, match them
    exactly against vision_model.state_dict() keys; skip shape mismatches.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ViT checkpoint not found: {checkpoint_path}")

    raw = torch.load(str(checkpoint_path), map_location="cpu")
    # Strip module./model. and then encoder. to align with self.vit.encoder names
    source = {}
    for k, v in _unwrap_state_dict(raw).items():
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
