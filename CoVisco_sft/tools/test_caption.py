"""test_caption.py - Test the SFT model's image captioning ability.

Usage:
  # Default config with a specific checkpoint
  python tools/test_caption.py --config configs/covisco_qwen3_1.7b.yaml \\
      --checkpoint output/sft_1.7b/step_12000.pt \\
      --image /path/to/image.jpg

  # Batch-test all images in a directory
  python tools/test_caption.py --config configs/covisco_qwen3_1.7b.yaml \\
      --checkpoint output/sft_1.7b/step_12000.pt \\
      --image-dir /path/to/images/

  # Custom prompt
  python tools/test_caption.py --config configs/covisco_qwen3_1.7b.yaml \\
      --checkpoint output/sft_1.7b/step_12000.pt \\
      --image /path/to/image.jpg \\
      --prompt "Describe this image in detail."

  # Token sampling mode: query_only / vit_only / query_and_vit (default, K=round(P*vit_ratio))
  python tools/test_caption.py --config configs/covisco_qwen3_1.7b.yaml \\
      --checkpoint output/sft_1.7b/step_12000.pt \\
      --image /path/to/image.jpg \\
      --strategy query_and_vit --vit-ratio 0.5

Run from: CoVisco_sft/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from PIL import Image
from transformers import AutoTokenizer

# Ensure imports work from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collator import build_image_transform, build_native_image_transform, compute_native_size
from models.config import build_model_config
from models.llava_covisco import LlavaCoViscoModel

# --------------------------------------------------------------------------- #
# Image preprocessing constants (aligned with the collator)
# --------------------------------------------------------------------------- #
IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMG_STD = (0.26862954, 0.26130258, 0.27577711)

# Vision special tokens (aligned with the config)
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"

DEFAULT_PROMPT = "Please describe this image in detail."


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(config_path: str, checkpoint_path: str, device: torch.device):
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    model_config = build_model_config(cfg_dict)

    print(f"[init] loading model from config: {config_path}")
    model = LlavaCoViscoModel(model_config)

    print(f"[init] loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Support multiple checkpoint formats:
    #   - {"model_state": state_dict, ...}  (format saved by train.py)
    #   - {"model": state_dict, ...}
    #   - a raw state_dict
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict) - len(missing)
    print(f"[ckpt] loaded {loaded}/{len(state_dict)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:10]}")

    # Cast everything to bfloat16 to avoid matmul errors from dtype mismatches
    # between ViT (bfloat16) and projector/LLM weights
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_config.llm.path, trust_remote_code=True)
    return model, tokenizer, model_config


# --------------------------------------------------------------------------- #
# Build the input for a single image
# --------------------------------------------------------------------------- #
def build_single_image_input(
    image: Image.Image,
    prompt: str,
    tokenizer,
    model_config,
    num_query_per_seg: int,
    device: torch.device,
    image_size: int = 224,
    strategy: str = "query_and_vit",
    vit_ratio: float = 1.0,
    native_resolution: bool = False,
    native_min_patches: int = 256,
    native_max_patches: int = 4096,
):
    """Assemble a PIL.Image + prompt into the input dict required by model forward.

    strategy: "query_only" | "vit_only" | "query_and_vit", aligned with training-time token_plan.
    vit_ratio: active only for strategy="query_and_vit", K = round(P * vit_ratio).
    native_resolution: native-resolution mode; resize with aspect ratio preserved to a
        patch_size-divisible size (patch count constrained to [native_min_patches,
        native_max_patches]), aligned with the training-time --native-resolution path.
    """
    patch_size = model_config.vit.patch_size
    image = image.convert("RGB")
    if native_resolution:
        new_h, new_w = compute_native_size(
            width=image.width,
            height=image.height,
            patch_size=patch_size,
            min_patches=native_min_patches,
            max_patches=native_max_patches,
        )
        if (new_h, new_w) != (image.height, image.width):
            image = image.resize((new_w, new_h), Image.BICUBIC)
        pixel = build_native_image_transform()(image)  # (3, H, W)
    else:
        pixel = build_image_transform(image_size)(image)  # (3, H, W)
    pixel_values = pixel.unsqueeze(0).to(device, dtype=torch.bfloat16)  # (1, 3, H, W)

    # Patches per image P is determined by the actual input size (non-square under native resolution);
    # compute the pad count from the token sampling strategy (num_segments=1, single image):
    #   - query_only:      pad_count = num_query_per_seg
    #   - vit_only:        pad_count = P
    #   - query_and_vit:   pad_count = num_query_per_seg + K，K = round(P * vit_ratio)
    H, W = pixel_values.shape[-2:]
    P = (H // patch_size) * (W // patch_size)
    print(f"[input] image size (HxW)={H}x{W}, patch_size={patch_size}, patches={P}"
          f"{' (native)' if native_resolution else ''}")

    if strategy == "query_only":
        pad_count = num_query_per_seg
    elif strategy == "vit_only":
        pad_count = P
    elif strategy == "query_and_vit":
        K = max(1, round(P * vit_ratio))
        pad_count = num_query_per_seg + K
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}; expected one of query_only/vit_only/query_and_vit")
    vision_block = VISION_START + IMAGE_PAD * pad_count + VISION_END

    messages = [
        {"role": "user", "content": f"<image>\n{prompt}"},
    ]
    # Replace the <image> placeholder with the actual vision token block
    for msg in messages:
        msg["content"] = msg["content"].replace("<image>", vision_block, 1)

    # Match the training collator: disable thinking, add_special_tokens=False
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)         # (1, seq_len)
    attention_mask = encoded["attention_mask"].to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "modality": "image",
        "token_plan": {"strategy": strategy, "vit_ratio": vit_ratio, "arrangement": "interleave"},
    }


# --------------------------------------------------------------------------- #
# Generation function
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_caption(
    model: LlavaCoViscoModel,
    tokenizer,
    model_config,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
    image_size: int = 224,
    strategy: str = "query_and_vit",
    vit_ratio: float = 1.0,
    native_resolution: bool = False,
    native_min_patches: int = 256,
    native_max_patches: int = 4096,
) -> str:
    num_query_per_seg = model_config.vit.num_query_per_seg
    inputs = build_single_image_input(
        image, prompt, tokenizer, model_config,
        num_query_per_seg=num_query_per_seg,
        device=device,
        image_size=image_size,
        strategy=strategy,
        vit_ratio=vit_ratio,
        native_resolution=native_resolution,
        native_min_patches=native_min_patches,
        native_max_patches=native_max_patches,
    )

    # Run one forward to obtain vision embeds, then use language_model.generate.
    # pixel_values must be injected into inputs_embeds inside the model before calling generate.
    # LlavaCoViscoModel does not expose a generate interface, so we do it manually:
    #   1) Call ViT + projector to get vision_embeds
    #   2) Replace the image_pad token positions with inputs_embeds
    #   3) Call language_model.generate
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    token_plan = inputs["token_plan"]

    # Step 1: ViT forward
    vit_dtype = next(model.vit.parameters()).dtype
    query_tokens, vit_tokens, _ = model.vit(
        pixel_values=pixel_values.to(vit_dtype),
        modality="image",
    )  # query_tokens: (1, S, Q, D); vit_tokens: (1, S, P, D)

    # Step 2: token_selector + arrangement (per the sampling mode given by --strategy, aligned with training token_plan)
    P = vit_tokens.shape[2]
    if strategy == "query_only":
        selected_vit = None
    elif strategy == "vit_only":
        selected_vit = vit_tokens  # All vit tokens, no selector filtering
    elif strategy == "query_and_vit":
        K = max(1, round(P * vit_ratio))
        sel_out = model.token_selector(query_tokens, vit_tokens, top_k=K)
        selected_vit = sel_out["selected_tokens"]  # (1, S, K, D)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}; expected one of query_only/vit_only/query_and_vit")
    all_vision = model._arrange_tokens(
        query_tokens=query_tokens,
        selected_vit=selected_vit,
        strategy=strategy,
        arrangement="interleave",
    )  # (1, total_tokens, D_vit)

    # Step 3: projector
    llm_dtype = model.language_model.dtype
    vision_embeds = model.projector(all_vision.to(llm_dtype))  # (1, total_tokens, D_llm)

    # Diagnostics: check that features at each stage are valid (different images should differ clearly)
    with torch.no_grad():
        print(f"[diag] pixel_values   mean={pixel_values.float().mean():.4f}  std={pixel_values.float().std():.4f}")
        print(f"[diag] query_tokens   mean={query_tokens.float().mean():.4f}  std={query_tokens.float().std():.4f}  norm={query_tokens.float().norm():.2f}")
        print(f"[diag] vit_tokens     mean={vit_tokens.float().mean():.4f}  std={vit_tokens.float().std():.4f}  norm={vit_tokens.float().norm():.2f}")
        print(f"[diag] all_vision     mean={all_vision.float().mean():.4f}  std={all_vision.float().std():.4f}  norm={all_vision.float().norm():.2f}")
        print(f"[diag] vision_embeds  mean={vision_embeds.float().mean():.4f}  std={vision_embeds.float().std():.4f}  norm={vision_embeds.float().norm():.2f}")
        has_nan = torch.isnan(vision_embeds).any().item()
        has_inf = torch.isinf(vision_embeds).any().item()
        print(f"[diag] vision_embeds  has_nan={has_nan}  has_inf={has_inf}")

    # Step 4: build inputs_embeds and fill the image_pad positions with vision_embeds
    emb_layer = model.language_model.get_input_embeddings()
    inputs_embeds = emb_layer(input_ids).clone()  # (1, seq_len, D_llm)
    pad_mask = (input_ids == model.image_pad_token_id)  # (1, seq_len)
    n_pad = pad_mask.sum().item()
    n_vision = vision_embeds.shape[1]

    # Debug: confirm the image_pad token id is correctly recognized
    image_pad_id = model.image_pad_token_id
    token_ids_in_seq = input_ids[0].tolist()
    n_image_pad_in_seq = token_ids_in_seq.count(image_pad_id)
    print(f"[debug] image_pad_token_id={image_pad_id}, "
          f"occurrences in input_ids={n_image_pad_in_seq}, "
          f"n_vision={n_vision}, input_ids.shape={input_ids.shape}")
    # Print the unique token ids in input_ids to help locate vision-token tokenization issues
    unique_ids = sorted(set(token_ids_in_seq))
    print(f"[debug] unique token ids in sequence: {unique_ids[:30]}{'...' if len(unique_ids) > 30 else ''}")

    if n_pad != n_vision:
        raise ValueError(
            f"vision tokens ({n_vision}) do not match the image_pad token count in input_ids ({n_pad})."
            f"\n  num_query_per_seg={num_query_per_seg}; check the config and the actual ViT output."
        )
    inputs_embeds[pad_mask] = vision_embeds.reshape(-1, vision_embeds.shape[-1])

    # Step 5: generate
    gen_kwargs = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    gen_out = model.language_model.generate(**gen_kwargs)
    # When inputs_embeds is passed, the sequences returned by generate contain only the newly
    # generated tokens (no prompt), because the prompt part is an embedding rather than token
    # ids and cannot be restored to ids and prepended. The previous heuristic that trimmed the
    # prompt when output_ids.shape[1] > prompt_len was unreliable: when the number of newly
    # generated tokens happened to exceed prompt_len it misjudged and cut off part of the
    # newly generated content at the beginning.
    new_ids = gen_out.sequences[0]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    return generated.strip()


# --------------------------------------------------------------------------- #
# Batch testing
# --------------------------------------------------------------------------- #
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_path: Optional[str], image_dir: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    if image_path:
        paths.append(Path(image_path))
    if image_dir:
        d = Path(image_dir)
        for ext in IMAGE_EXTS:
            paths.extend(sorted(d.glob(f"*{ext}")))
            paths.extend(sorted(d.glob(f"*{ext.upper()}")))
    return paths


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Test SFT model caption ability")
    parser.add_argument("--config", required=True, help="path to the model config YAML")
    parser.add_argument("--checkpoint", required=True, help="path to the SFT checkpoint (.pt)")
    parser.add_argument("--image", default=None, help="single image path")
    parser.add_argument("--image-dir", default=None, help="directory of images for batch testing")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Caption prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true", help="enable sampling decoding")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224, help="input image resolution (ignored with --native-resolution)")
    parser.add_argument(
        "--native-resolution", action="store_true",
        help="native-resolution input: resize with aspect ratio preserved to a patch_size-divisible size, no CenterCrop",
    )
    parser.add_argument(
        "--native-min-patches", type=int, default=256,
        help="lower bound on patch count in native-resolution mode (256 = 224x224)",
    )
    parser.add_argument(
        "--native-max-patches", type=int, default=4096,
        help="upper bound on patch count in native-resolution mode (ViT uses full attention; too large will OOM)",
    )
    parser.add_argument(
        "--strategy", default="query_and_vit",
        choices=["query_only", "vit_only", "query_and_vit"],
        help="token sampling mode: query_only (query tokens only) / vit_only (all vit tokens, no "
             "filtering) / query_and_vit (query + top-K filtered vit tokens, default, same as "
             "the main training strategy)",
    )
    parser.add_argument(
        "--vit-ratio", type=float, default=1.0,
        help="active only for --strategy=query_and_vit; K=round(P*vit_ratio) where P is the patch count per segment",
    )
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("specify at least one of --image or --image-dir")

    device = torch.device(args.device)
    model, tokenizer, model_config = load_model_and_tokenizer(
        args.config, args.checkpoint, device
    )
    print(f"[init] model loaded, device={device}")
    print(f"[init] token sampling strategy={args.strategy}, vit_ratio={args.vit_ratio}")
    if args.native_resolution:
        print(f"[init] native resolution enabled, patches in "
              f"[{args.native_min_patches}, {args.native_max_patches}]\n")
    else:
        print(f"[init] fixed image size={args.image_size}\n")

    image_paths = collect_images(args.image, args.image_dir)
    if not image_paths:
        print("[error] no image files found")
        sys.exit(1)

    for img_path in image_paths:
        print(f"{'='*60}")
        print(f"Image : {img_path}")
        print(f"Prompt: {args.prompt}")
        try:
            image = Image.open(img_path)
            caption = generate_caption(
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                image=image,
                prompt=args.prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                image_size=args.image_size,
                strategy=args.strategy,
                vit_ratio=args.vit_ratio,
                native_resolution=args.native_resolution,
                native_min_patches=args.native_min_patches,
                native_max_patches=args.native_max_patches,
            )
            print(f"Caption:\n{caption}")
        except Exception as e:
            print(f"[error] {e}")
        print()


if __name__ == "__main__":
    main()
