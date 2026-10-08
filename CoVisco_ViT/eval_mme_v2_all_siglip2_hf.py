"""HuggingFace-release variant of ``eval_mme_v2_all_siglip2.py``.

Same unified MMEB-V2 evaluator in the SigLIP2 text space (every visual task uses
the CoVisco ``to_image_caption`` projection, ``modality='image'``), but the
vision tower is loaded from the released ``config.json`` + ``model.safetensors``
(see ``covisco_hf.py``) instead of an open_clip ``.pt`` checkpoint. The SigLIP2
teacher text encoder (reused from ``eval_mme_v2_all_siglip2``) is unchanged.

Usage:
    # default: load from the Hub repo ernie-research/CoVisco-L-14
    # (set HF_TOKEN for the private repo)
    python eval_mme_v2_all_siglip2_hf.py \
        --data_root /path/to/mme_v2 --siglip2_dir /path/to/ViT-gopt-16-SigLIP2-384
    # or load from a local release dir: --hf_model hf_release/CoVisco-L-14
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import covisco_hf  # noqa: E402
import eval_mme_v2_all as evaluator  # noqa: E402
import eval_mme_v2_all_siglip2 as sig  # noqa: E402

_HF_MODEL = covisco_hf.DEFAULT_HF_MODEL


def _hf_load_model(ckpt_path, video_caption_embed_dim, device,
                   image_size=224, segment_t_size=16):
    """Drop-in for evaluator.load_model; loads the HF release (ignores ckpt)."""
    del ckpt_path
    model, _cfg, _dir = covisco_hf.load_vision_model(
        _HF_MODEL, device=device, image_size=image_size,
        segment_t_size=segment_t_size,
        video_caption_embed_dim=video_caption_embed_dim,
    )
    return model


def main():
    global _HF_MODEL
    parser = evaluator.build_parser()
    parser.description = ("MMEB-V2 evaluation using SigLIP2 text and "
                          "to_image_caption, loading the CoVisco HF release.")
    parser.add_argument("--hf_model", default=covisco_hf.DEFAULT_HF_MODEL,
                        help="Local CoVisco release dir or HF Hub repo id.")
    # SigLIP2 text-encoder args (mirror eval_mme_v2_all_siglip2.main).
    parser.add_argument("--siglip2_dir", default=None,
                        help="Local SigLIP2 dir with open_clip weights/tokenizer.")
    parser.add_argument("--siglip2_pretrained", default="webli",
                        help="open_clip pretrained tag or local weight file.")
    parser.add_argument("--siglip2_tokenizer", default=None,
                        help="Tokenizer dir/name; defaults to --siglip2_dir or HF.")
    parser.add_argument("--siglip2_text_batch_size", type=int, default=256)
    args = parser.parse_args()
    _HF_MODEL = args.hf_model

    # Swap in HF vision loading + SigLIP2 image-caption space + SigLIP2 text.
    evaluator.load_model = _hf_load_model
    evaluator._VISION_MODALITY = "image"
    evaluator._VISION_OUTPUT_KEY = "to_image_caption"
    evaluator._load_text_encoder = sig._load_siglip2_text_encoder
    print(f"[hf] loading CoVisco vision tower from: {_HF_MODEL}")

    args.device = evaluator._init_distributed(args.device)
    if evaluator._DIST_ENABLED:
        print(f"[distributed] rank {evaluator._DIST_RANK}/"
              f"{evaluator._DIST_WORLD_SIZE}, device={args.device}")
    evaluator.main(args)
    if evaluator._DIST_ENABLED:
        evaluator.dist.barrier()
        evaluator.dist.destroy_process_group()


if __name__ == "__main__":
    main()
