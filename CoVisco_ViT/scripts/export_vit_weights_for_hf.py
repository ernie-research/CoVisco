#!/usr/bin/env python3
"""Safely export the CoVisco ViT model weights from a training checkpoint.

"Safe" here means:
  * The checkpoint is loaded with ``torch.load(..., weights_only=True)`` so the
    restricted unpickler is used and no arbitrary code can run during load.
  * Only the model tensors are kept. Optimizer / scaler / bookkeeping entries
    (``optimizer``, ``optimizer_param_names``, ``scaler``, ``epoch``, ``name``)
    are dropped, so no training state leaks into the released artifact.
  * The output is written as ``.safetensors`` (zero-copy, no pickle), which is
    the safe format for downstream users to load.

Usage:
    python scripts/export_vit_weights_for_hf.py \
        --ckpt /path/to/covisco_vit_1765.pt \
        --out  /path/to/CoVisco-L-14.safetensors
"""
import argparse
import os

import torch
from safetensors.torch import save_file

# Non-weight entries written by the training loop (see main_covisco.py).
_NON_WEIGHT_KEYS = {"optimizer", "optimizer_param_names", "scaler", "epoch", "name"}


def extract_state_dict(ckpt: dict) -> tuple[dict, dict]:
    """Return (model_state_dict, meta) from a raw loaded checkpoint."""
    meta = {}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Preserve harmless provenance info as safetensors string metadata.
        if "epoch" in ckpt:
            meta["epoch"] = str(ckpt["epoch"])
        if "name" in ckpt:
            meta["name"] = str(ckpt["name"])
        sd = ckpt["state_dict"]
    else:
        # A bare state_dict was saved directly.
        sd = ckpt
    return sd, meta


def clean_weights(sd: dict) -> dict:
    """Keep tensors only, strip the DDP ``module.`` prefix, de-share storage."""
    cleaned = {}
    skipped = []
    for k, v in sd.items():
        if k in _NON_WEIGHT_KEYS:
            continue
        if not torch.is_tensor(v):
            skipped.append(k)
            continue
        new_k = k[len("module."):] if k.startswith("module.") else k
        # clone() guarantees contiguous, CPU, and independent storage so that
        # safetensors never trips over shared-memory tensors.
        cleaned[new_k] = v.detach().cpu().clone()
    if skipped:
        print(f"[info] skipped {len(skipped)} non-tensor entries: {skipped[:5]}"
              + (" ..." if len(skipped) > 5 else ""))
    return cleaned


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Path to the training .pt checkpoint")
    ap.add_argument("--out", required=True, help="Output .safetensors path")
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite existing file: {args.out}")

    print(f"[1/3] loading (weights_only=True): {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    raw_sd, meta = extract_state_dict(ckpt)
    weights = clean_weights(raw_sd)

    n_params = sum(t.numel() for t in weights.values())
    dtypes = sorted({str(t.dtype) for t in weights.values()})
    print(f"[2/3] kept {len(weights)} tensors | {n_params/1e6:.1f}M params | dtypes: {dtypes}")

    meta["source_checkpoint"] = os.path.basename(args.ckpt)
    meta["format"] = "pt"  # tells safetensors loaders these are torch tensors
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_file(weights, args.out, metadata=meta)
    size_gb = os.path.getsize(args.out) / 1e9
    print(f"[3/3] wrote {args.out} ({size_gb:.2f} GB) | metadata: {meta}")


if __name__ == "__main__":
    main()
