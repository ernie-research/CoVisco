"""MME-V2 evaluation with the training-time SigLIP2 text space.

This is a separate entry point from ``eval_mme_v2_all.py``. It reuses the
task implementations and distributed evaluator, but changes both sides of the
image-caption space:

  text:   ViT-gopt-16-SigLIP2-384, pretrained="webli"
  vision: CoVisco output ``to_image_caption`` with ``modality="image"``

The CoVisco checkpoint must have been trained with
``image_caption_embed_dim=1536``.
"""

import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import eval_mme_v2_all as evaluator  # noqa: E402


class _GemmaTokenizer:
    """SigLIP2 tokenizer wrapper matching the training-time canonicalization."""

    def __init__(self, tokenizer, context_length=64):
        from open_clip.tokenizer import _clean_canonicalize

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.clean_fn = _clean_canonicalize

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [self.clean_fn(text) for text in texts]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.context_length,
            padding="max_length",
            truncation=True,
        )
        return encoded.input_ids


class _SigLIP2TextEncoder:
    """Expose open_clip SigLIP2 ``encode_text`` through evaluator's API."""

    def __init__(self, model, tokenizer, device, batch_size=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts, prompt=None, normalize_embeddings=True,
               show_progress_bar=False, batch_size=None, convert_to_numpy=True):
        del prompt, show_progress_bar
        batch_size = batch_size or self.batch_size
        outputs = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[start:start + batch_size]).to(self.device)
            with torch.autocast(device_type=self.device.split(":")[0]):
                embeddings = self.model.encode_text(
                    tokens, normalize=normalize_embeddings)
            outputs.append(embeddings.float().cpu())
        result = torch.cat(outputs, dim=0) if outputs else torch.empty(0, 1536)
        return result.numpy() if convert_to_numpy else result


def _load_siglip2_text_encoder(args, device):
    import open_clip
    from transformers import AutoTokenizer

    model_name = "ViT-gopt-16-SigLIP2-384"
    local_dir = args.siglip2_dir
    pretrained = args.siglip2_pretrained
    tokenizer_source = local_dir or "timm/ViT-gopt-16-SigLIP2-384"

    local_weights = None
    if local_dir and os.path.isdir(local_dir):
        for filename in ("open_clip_model.safetensors",
                         "open_clip_pytorch_model.bin"):
            candidate = os.path.join(local_dir, filename)
            if os.path.exists(candidate):
                local_weights = candidate
                break
        if not args.siglip2_tokenizer:
            tokenizer_source = local_dir

    if local_weights:
        pretrained = local_weights
    print(f"Loading SigLIP2 text encoder ({model_name}, pretrained={pretrained}) ...")
    text_model = open_clip.create_model(
        model_name, pretrained=pretrained, device=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.siglip2_tokenizer or tokenizer_source, use_fast=True)
    return _SigLIP2TextEncoder(
        text_model,
        _GemmaTokenizer(tokenizer, context_length=64),
        device,
        batch_size=args.siglip2_text_batch_size,
    )


def main():
    parser = evaluator.build_parser()
    parser.description = "MME-V2 evaluation using SigLIP2 text and to_image_caption."
    parser.add_argument(
        "--siglip2_dir",
        default=None,
        help="Local SigLIP2 directory containing open_clip weights/tokenizer.",
    )
    parser.add_argument(
        "--siglip2_pretrained",
        default="webli",
        help="open_clip pretrained tag or local weight file.",
    )
    parser.add_argument(
        "--siglip2_tokenizer",
        default=None,
        help="Tokenizer directory/name; defaults to --siglip2_dir or the HF model.",
    )
    parser.add_argument("--siglip2_text_batch_size", type=int, default=256)
    args = parser.parse_args()

    # Make every visual task use the image-caption projection trained against
    # SigLIP2 text features, including video frames and VisDoRe pages.
    evaluator._VISION_MODALITY = "image"
    evaluator._VISION_OUTPUT_KEY = "to_image_caption"
    evaluator._load_text_encoder = _load_siglip2_text_encoder

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
