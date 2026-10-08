"""HuggingFace-release variant of ``eval_mmeb_video_retrieval_siglip2.py``.

Same MME-V2 video retrieval evaluation in the SigLIP2 text space (CoVisco
``to_image_caption`` branch, ``modality='image'``), but the vision tower is
loaded from the released ``config.json`` + ``model.safetensors`` (see
``covisco_hf.py``) instead of an open_clip ``.pt`` checkpoint. The SigLIP2
teacher text encoder is unchanged.

The original ``load_vision_model`` builds the encoder with ``segment_t_size=16``
and ``image_size=224``; those are reproduced here.

Usage:
    # default: load from the Hub repo ernie-research/CoVisco-L-14
    # (set HF_TOKEN for the private repo)
    python eval_mmeb_video_retrieval_siglip2_hf.py --data_root /path/to/mme_v2
    # or load from a local release dir: --hf_model hf_release/CoVisco-L-14
"""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import covisco_hf  # noqa: E402
import eval_mmeb_video_retrieval_siglip2 as base  # noqa: E402

_HF_MODEL = covisco_hf.DEFAULT_HF_MODEL


def _hf_load_vision_model(ckpt_path, video_caption_embed_dim, device):
    """Drop-in for base.load_vision_model; loads the HF release (ignores ckpt)."""
    del ckpt_path
    model, _cfg, _dir = covisco_hf.load_vision_model(
        _HF_MODEL, device=device, image_size=224, segment_t_size=16,
        video_caption_embed_dim=video_caption_embed_dim,
    )
    return model


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf_model", default=covisco_hf.DEFAULT_HF_MODEL,
                   help="Local CoVisco release dir or HF Hub repo id.")
    p.add_argument("--data_root", type=str, required=True, help="MMEB-V2 data root.")
    p.add_argument("--video_caption_embed_dim", type=int, default=4096,
                   help="Dim of the to_video_caption head (needed for loading).")
    p.add_argument("--max_frames_ret", type=int, default=64)
    p.add_argument("--video_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str,
                   default="mme_v2_vret_siglip2_hf_results.json")
    return p


def main():
    global _HF_MODEL
    args = _build_parser().parse_args()
    _HF_MODEL = args.hf_model
    base.load_vision_model = _hf_load_vision_model
    print(f"[hf] loading CoVisco vision tower from: {_HF_MODEL}")
    base.main(args)


if __name__ == "__main__":
    main()
