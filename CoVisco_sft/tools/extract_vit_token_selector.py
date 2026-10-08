#!/usr/bin/env python3
"""Extract only the ViT and token-selector weights from a training checkpoint.

Training checkpoints (see train/train.py) are saved as::

    {
        "step": int,
        "model_state": <LlavaCoViscoModel state_dict>,
        "optimizer_state": ...,
        "lr_scale": ...,
        "best_val_loss": ...,
        "args": ...,
    }

Inside ``model_state`` the parameters are flat with submodule prefixes such as
``vit.*``, ``token_selector.*``, ``projector.*`` and ``language_model.*``.

This script writes a new checkpoint that contains ONLY the ``vit.*`` and
``token_selector.*`` tensors. The optimizer state and every other submodule
(projector / language model) are dropped. ``step`` is always reset to 0 so
that a later resume starts from the beginning rather than the source step.

Usage::

    python tools/extract_vit_token_selector.py \
        /path/to/code/CoVisco_sft/output/sft_1.7b_vit_native_reso/step_3000.pt \
        -o output/sft_1.7b_vit_native_reso/step_3000_vit_tokensel.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

# Submodule prefixes to keep (see models/llava_covisco.py).
KEEP_PREFIXES = ("vit.", "token_selector.")


def _strip_module_prefix(state: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """Drop a leading ``module.`` added by DDP, if present."""
    out = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = value
    return out


def extract(ckpt_path: Path, out_path: Path, keep_meta: bool) -> None:
    print(f"[load] {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # The model state_dict may be nested under "model_state", or the file may
    # already be a bare state_dict.
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    state = _strip_module_prefix(state)

    kept: Dict[str, "torch.Tensor"] = {
        key: value
        for key, value in state.items()
        if key.startswith(KEEP_PREFIXES)
    }

    n_vit = sum(1 for k in kept if k.startswith("vit."))
    n_sel = sum(1 for k in kept if k.startswith("token_selector."))
    if n_vit == 0:
        raise RuntimeError("No 'vit.*' weights found in checkpoint")
    if n_sel == 0:
        raise RuntimeError("No 'token_selector.*' weights found in checkpoint")

    total_params = sum(v.numel() for v in kept.values() if hasattr(v, "numel"))
    print(f"[keep] vit tensors={n_vit} token_selector tensors={n_sel} "
          f"total_tensors={len(kept)} total_params={total_params:,}")

    src_step = ckpt.get("step") if isinstance(ckpt, dict) else None
    out: Dict[str, object] = {"model_state": kept, "step": 0}
    if keep_meta and isinstance(ckpt, dict) and "args" in ckpt:
        out["args"] = ckpt["args"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(out_path))

    src_mb = ckpt_path.stat().st_size / (1024 * 1024)
    dst_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[save] {out_path}")
    print(f"[step] {src_step} -> 0")
    print(f"[size] {src_mb:.1f} MB -> {dst_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract vit.* and token_selector.* weights from a checkpoint."
    )
    parser.add_argument("ckpt", type=Path, help="Path to the source checkpoint (.pt)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path. Default: <ckpt_stem>_vit_tokensel.pt next to source.",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Do not copy 'args' metadata into the output. 'step' is always 0.",
    )
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt
    if not ckpt_path.is_file():
        parser.error(f"checkpoint not found: {ckpt_path}")

    out_path: Path = args.output or ckpt_path.with_name(
        f"{ckpt_path.stem}_vit_tokensel.pt"
    )

    extract(ckpt_path, out_path, keep_meta=not args.no_meta)


if __name__ == "__main__":
    main()
