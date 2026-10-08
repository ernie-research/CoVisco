"""HuggingFace-release variant of ``eval_mme_v2_all.py``.

Same unified MMEB-V2 evaluator (video retrieval / classification / moment
retrieval / image_cls / visdoc) in the Qwen3-VL text space, but the CoVisco
vision tower is loaded from the released ``config.json`` + ``model.safetensors``
(see ``covisco_hf.py``) instead of an open_clip ``.pt`` checkpoint. The Qwen3-VL
text encoder is unchanged.

``--image_size`` and ``--segment_t_size`` are forwarded to the released encoder
(defaults 224 / 16, matching the base script).

Usage:
    # default: load from the Hub repo ernie-research/CoVisco-L-14
    # (set HF_TOKEN for the private repo)
    python eval_mme_v2_all_hf.py \
        --data_root /path/to/mme_v2 --text_model_path /path/to/Qwen3-VL-Embedding-8B
    # or load from a local release dir: --hf_model hf_release/CoVisco-L-14
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import covisco_hf  # noqa: E402
import eval_mme_v2_all as evaluator  # noqa: E402

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


def _add_hf_arg(parser):
    parser.add_argument("--hf_model", default=covisco_hf.DEFAULT_HF_MODEL,
                        help="Local CoVisco release dir or HF Hub repo id.")
    return parser


def main():
    global _HF_MODEL
    parser = _add_hf_arg(evaluator.build_parser())
    parser.description = "MMEB-V2 evaluation loading the CoVisco HF release."
    args = parser.parse_args()
    _HF_MODEL = args.hf_model

    evaluator.load_model = _hf_load_model
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
