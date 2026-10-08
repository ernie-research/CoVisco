#!/usr/bin/env python3
"""Test the CoVisco model inference pipeline.

Verifies:
1. Model loading (ViT + Token Selector + Projector + LLM)
2. Image preprocessing
3. ViT encoding
4. Token Selector selection
5. Projector projection
6. LLM generation
7. (optional) Run inference on an evaluation dataset

Usage examples:
    # Basic test
    python scripts/test_inference.py --config configs/covisco_qwen3_4b_instruct2507.yaml

    # Test on the AI2D dataset
    python scripts/test_inference.py --config configs/covisco_qwen3_4b_instruct2507.yaml --task ai2d

    # Test on a video dataset
    python scripts/test_inference.py --config configs/covisco_qwen3_4b_instruct2507.yaml --task videomme --num_video_frames 8
"""
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# Add the project path
REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

from models.config import build_model_config
from models.llava_covisco import LlavaCoViscoModel
from transformers import AutoTokenizer


def test_inference(
    config_path: str,
    checkpoint_path: str = None,
    vit_ratio: float = 0.5,
    mmr_lambda: float = 0.0,
    segment_t_size: int = None,
):
    """Test the inference pipeline.

    Args:
        config_path: path to the model config file
        checkpoint_path: path to the trained checkpoint
        vit_ratio: fraction of vit tokens to keep under query_and_vit mode (0.0-1.0)
        mmr_lambda: MMR diversity weight (0.0 = off, >0 enables)
        segment_t_size: override config.vit.segment_t_size (None = use the config value)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] vit_ratio={vit_ratio}, mmr_lambda={mmr_lambda}, segment_t_size={segment_t_size}")

    # 1. Load config
    print(f"\n[1] Loading config from: {config_path}")
    with open(config_path) as f:
        import yaml
        config_dict = yaml.safe_load(f)
    config = build_model_config(config_dict)

    # Override segment_t_size
    if segment_t_size is not None:
        print(f"    Overriding segment_t_size: {config.vit.segment_t_size} -> {segment_t_size}")
        config.vit.segment_t_size = segment_t_size

    print(f"    ViT: {config.vit.name}")
    print(f"    LLM: {config.llm.name}")
    print(f"    segment_t_size: {config.vit.segment_t_size}")

    # 2. Build model
    print(f"\n[2] Building model...")
    model = LlavaCoViscoModel(config)
    model = model.to(device)
    model.eval()

    # Trainable parameter statistics
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total params: {n_total:,}")
    print(f"    Trainable:   {n_train:,} ({100*n_train/n_total:.2f}%)")

    # 3. Load tokenizer
    print(f"\n[3] Loading tokenizer from: {config.llm.path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm.path,
        trust_remote_code=True,
        use_fast=False,
    )

    # 4. Test text-only generation
    print(f"\n[4] Testing text-only generation...")
    prompt = "Hello, please introduce yourself."
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.language_model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
        )
    
    output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"    Prompt: {prompt}")
    print(f"    Output: {output}")

    # 5. Test image input
    print(f"\n[5] Testing image input...")
    # Create a test image (random noise)
    test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    print(f"    Test image: {test_image.size}")

    # Preprocessing
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((config.vit.image_size, config.vit.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # ViT expects input format: (B, C, T, H, W) - channel before the time dimension
    pixel_values = transform(test_image).unsqueeze(0)  # (1, 3, H, W)
    pixel_values = pixel_values.unsqueeze(2)  # (1, 3, 1, H, W) - T=1 for single image
    pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
    print(f"    Pixel values shape: {pixel_values.shape}")

    # ViT forward
    with torch.no_grad():
        query_tokens, vit_tokens, _ = model.vit(
            pixel_values=pixel_values.to(next(model.vit.parameters()).dtype),
            visidx=None,
            modality="image",
        )
    print(f"    Query tokens: {query_tokens.shape}")  # (B, S, Q, D)
    print(f"    ViT tokens:   {vit_tokens.shape}")    # (B, S, P, D)

    # Token selector (uses the command-line args vit_ratio and mmr_lambda)
    P = vit_tokens.shape[2]
    K = max(1, round(P * vit_ratio))

    # Dynamically update the token_selector's mmr_lambda
    original_mmr_lambda = model.token_selector.mmr_lambda
    model.token_selector.mmr_lambda = mmr_lambda

    with torch.no_grad():
        sel_out = model.token_selector(query_tokens, vit_tokens, top_k=K)
    selected_vit = sel_out["selected_tokens"]

    # Restore the original mmr_lambda
    model.token_selector.mmr_lambda = original_mmr_lambda

    print(f"    Token selector: P={P}, K={K}, vit_ratio={vit_ratio:.2f}, mmr_lambda={mmr_lambda:.2f}")
    print(f"    Selected vit: {selected_vit.shape}")  # (B, S, K, D)

    # Concatenate
    b, s, q, d = query_tokens.shape
    all_vision = torch.cat([query_tokens[:, 0, :, :], selected_vit[:, 0, :, :]], dim=1)
    print(f"    All vision: {all_vision.shape}")  # (B, Q+K, D)

    # Projector (convert to float32 for projector, then to LLM dtype)
    with torch.no_grad():
        projector_dtype = next(model.projector.parameters()).dtype
        vision_embeds = model.projector(all_vision.to(projector_dtype))
    print(f"    Vision embeds: {vision_embeds.shape}")  # (B, Q+K, D_llm)

    print(f"\n[SUCCESS] All components working correctly!")


def test_on_benchmark(
    config_path: str,
    checkpoint_path: str,
    task: str,
    vit_ratio: float = 0.5,
    mmr_lambda: float = 0.0,
    segment_t_size: int = None,
    num_video_frames: int = 8,
    max_new_tokens: int = 256,
    batch_size: int = 1,
    num_gpus: int = 1,
    output_path: str = None,
):
    """Test the model on a given benchmark.

    Args:
        config_path: path to the model config file
        checkpoint_path: path to the trained checkpoint
        task: evaluation task name (e.g. ai2d, chartqa, videomme, etc.)
        vit_ratio: fraction of vit tokens to keep under query_and_vit mode
        mmr_lambda: MMR diversity weight
        segment_t_size: override config.vit.segment_t_size (None = use the config value)
        num_video_frames: number of video frames
        max_new_tokens: maximum number of generated tokens
        batch_size: batch size
        num_gpus: number of GPUs
        output_path: path to save results
    """
    import subprocess
    import json
    from datetime import datetime

    # Set environment variables
    lmms_eval_dir = os.path.join(REPO_DIR, "third_party", "lmms-eval")
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise RuntimeError("set HF_HOME to the directory containing the evaluation datasets")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_DIR}:{lmms_eval_dir}"
    env["HF_HOME"] = hf_home
    env["http_proxy"] = os.environ.get("http_proxy", "")
    env["https_proxy"] = os.environ.get("https_proxy", "")
    env["no_proxy"] = os.environ.get("no_proxy", "localhost,127.0.0.1")

    # Build model arguments
    model_args = f"config_path={config_path}"
    if checkpoint_path:
        model_args += f",checkpoint_path={checkpoint_path}"
    model_args += f",conv_template=qwen_1_5,num_video_frames={num_video_frames},max_new_tokens={max_new_tokens}"
    if segment_t_size is not None:
        model_args += f",segment_t_size={segment_t_size}"

    # Build the accelerate launch command
    run_port = 12457 + hash(task) % 1000
    model_name = f"covisco_qwen3_4b_vit{vit_ratio}_mmr{mmr_lambda}"

    cmd = [
        "python", "-m", "accelerate.commands.launch",
        f"--main_process_port={run_port}",
        f"--num_processes={num_gpus}",
        "-m", "lmms_eval",
        "--model", "llava_covisco",
        "--model_args", model_args,
        "--tasks", task,
        "--batch_size", str(batch_size),
        "--log_samples",
        f"--log_samples_suffix", f"{model_name}_{datetime.now().strftime('%Y%m%d')}",
        "--output_path", output_path or f"{REPO_DIR}/eval_results/{model_name}/",
    ]

    print(f"\n{'='*60}")
    print(f"[BENCHMARK] Running evaluation on: {task}")
    print(f"{'='*60}")
    print(f"Config:       {config_path}")
    print(f"Checkpoint:   {checkpoint_path or '<none>'}")
    print(f"vit_ratio:    {vit_ratio}")
    print(f"mmr_lambda:   {mmr_lambda}")
    print(f"Num GPUs:     {num_gpus}")
    print(f"Output:       {output_path or f'{REPO_DIR}/eval_results/{model_name}/'}")
    print(f"\nCommand: {' '.join(cmd[:10])}...")
    print(f"{'='*60}\n")

    # Run the command
    result = subprocess.run(cmd, env=env, cwd=lmms_eval_dir)

    if result.returncode == 0:
        print(f"\n[SUCCESS] Evaluation completed for task: {task}")
    else:
        print(f"\n[ERROR] Evaluation failed with return code: {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the CoVisco model inference pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--vit_ratio", type=float, default=0.5,
                        help="fraction of vit tokens to keep under query_and_vit mode (0.0-1.0, default=0.5)")
    parser.add_argument("--mmr_lambda", type=float, default=0.0,
                        help="MMR diversity weight (0.0=off, >0 enables, default=0.0)")
    parser.add_argument("--segment_t_size", type=int, default=None,
                        help="override config.vit.segment_t_size (default=None, use the config value)")

    # Benchmark test arguments
    parser.add_argument("--task", type=str, default=None,
                        help="evaluation task name (e.g. ai2d, chartqa, videomme, etc.). If omitted, run only the basic test")
    parser.add_argument("--num_video_frames", type=int, default=8,
                        help="number of video frames (default=8)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="maximum number of generated tokens (default=256)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default=1)")
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="number of GPUs (default=4)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="path to save results")

    args = parser.parse_args()

    # Run the basic test first
    test_inference(args.config, args.checkpoint, args.vit_ratio, args.mmr_lambda, args.segment_t_size)

    # If a task is specified, run the benchmark test
    if args.task:
        test_on_benchmark(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            task=args.task,
            vit_ratio=args.vit_ratio,
            mmr_lambda=args.mmr_lambda,
            segment_t_size=args.segment_t_size,
            num_video_frames=args.num_video_frames,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
            output_path=args.output_path,
        )
