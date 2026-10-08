#!/usr/bin/env bash
# Visualize the token selector on cached samples.
set -euo pipefail
cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 python tools/visualize_token_selector.py \
    --ckpt "${CKPT:?set CKPT to the trained checkpoint path}" \
    --sample-cache-dir outputs/token_selector_sample_cache \
    --num-images 16 \
    --image-resolution=448 \
    --video-resolution=224 \
    --num-videos 0 \
    --top-k 128 \
    --device cuda:0 \
    --dtype bf16 \
    --video-mode uniform \
    --uniform-video-frames 64 \
    --uniform-segment-t-size 16 \
    --output-dir outputs/token_selector_viz_soft_uniform64_s16_zero_q \
    --overlay-alpha 0.25 --export-video gif --fps 4 \
    --first-frame-max-ratio 0.5 \
    --mmr --mmr-lambda 0.3
    #  --zero-query-tokens
