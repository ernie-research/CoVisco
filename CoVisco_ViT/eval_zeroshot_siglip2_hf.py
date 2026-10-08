"""HuggingFace-release variant of ``eval_zeroshot_siglip2.py``.

Identical zero-shot classification / retrieval evaluation, but the CoVisco
vision tower is loaded from the released ``config.json`` + ``model.safetensors``
(see ``covisco_hf.py``) instead of from an open_clip ``.pt`` training
checkpoint. The SigLIP2 teacher text encoder is unchanged.

Usage:
    # default: load from the Hub repo ernie-research/CoVisco-L-14
    # (set HF_TOKEN for the private repo)
    python eval_zeroshot_siglip2_hf.py --tasks classification retrieval --image_size 224
    # or load from a local release dir:
    python eval_zeroshot_siglip2_hf.py --hf_model hf_release/CoVisco-L-14 ...

All other flags (``--tasks``, ``--cls_datasets``, ``--siglip2_dir``, ...) are
inherited from the base script; ``--ckpt`` is accepted but ignored.
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import covisco_hf  # noqa: E402
import eval_zeroshot_siglip2 as base  # noqa: E402

_HF_MODEL = covisco_hf.DEFAULT_HF_MODEL


def _hf_load_vision_model(ckpt_path, video_caption_embed_dim, image_size, device):
    """Drop-in for base.load_vision_model; loads the HF release (ignores ckpt)."""
    del ckpt_path  # replaced by --hf_model
    model, _cfg, _dir = covisco_hf.load_vision_model(
        _HF_MODEL, device=device, image_size=image_size,
        video_caption_embed_dim=video_caption_embed_dim,
    )
    return model


def _hf_build_preprocess(image_size):
    return covisco_hf.build_preprocess(image_size)


def main():
    global _HF_MODEL
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--hf_model", default=covisco_hf.DEFAULT_HF_MODEL,
                     help="Local CoVisco release dir or HF Hub repo id.")
    ns, remaining = pre.parse_known_args()
    _HF_MODEL = ns.hf_model
    sys.argv = [sys.argv[0]] + remaining

    base.load_vision_model = _hf_load_vision_model
    base.build_preprocess = _hf_build_preprocess
    print(f"[hf] loading CoVisco vision tower from: {_HF_MODEL}")
    base.main()


if __name__ == "__main__":
    main()
