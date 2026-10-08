#!/bin/bash
# Test the image captioning ability of a trained checkpoint
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/covisco_qwen3_1.7b.yaml}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the trained checkpoint path}"

# This checkpoint was trained with native resolution (see NATIVE_RESOLUTION in scripts/run_sft_1.7b.sh),
# so inference also defaults to native resolution; when NATIVE_RESOLUTION=0 it falls back to a fixed IMAGE_SIZE.
IMAGE_SIZE=${IMAGE_SIZE:-448}

# ===== Image native resolution =====
# NATIVE_RESOLUTION=1: preserve aspect ratio, resize the image to the nearest size divisible
# by patch size (14), constrain total patch count to [NATIVE_MIN_PATCHES, NATIVE_MAX_PATCHES],
# and do not CenterCrop.
# Must match the value used during training, otherwise the vision token distribution/count
# will not match training.
NATIVE_RESOLUTION=${NATIVE_RESOLUTION:-1}
NATIVE_MIN_PATCHES=${NATIVE_MIN_PATCHES:-256}    # 256 = 224x224
NATIVE_MAX_PATCHES=${NATIVE_MAX_PATCHES:-4096}   # 4096 = 896x896

# Provide the input via the IMAGE (single file) or IMAGE_DIR (directory) environment variable
IMAGE=${IMAGE:-}
IMAGE_DIR=${IMAGE_DIR:-}
if [ -z "${IMAGE}" ] && [ -z "${IMAGE_DIR}" ]; then
    echo "[ERROR] set IMAGE=<path/to/image.jpg> or IMAGE_DIR=<path/to/dir>" >&2
    exit 1
fi

# token sampling mode: query_only / vit_only / query_and_vit (default, matches the main training strategy)
STRATEGY=${STRATEGY:-query_and_vit}
VIT_RATIO=${VIT_RATIO:-0.1}

ARGS=(
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --image-size "${IMAGE_SIZE}"
    --device "${DEVICE:-cuda:0}"
    --prompt "${PROMPT:-Please describe this image in detail.}"
    --strategy "${STRATEGY}"
    --vit-ratio "${VIT_RATIO}"
)

if [ "${NATIVE_RESOLUTION}" = "1" ]; then
    ARGS+=(--native-resolution
           --native-min-patches "${NATIVE_MIN_PATCHES}"
           --native-max-patches "${NATIVE_MAX_PATCHES}")
fi

if [ -n "${IMAGE_DIR}" ]; then
    ARGS+=(--image-dir "${IMAGE_DIR}")
else
    ARGS+=(--image "${IMAGE}")
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python tools/test_caption.py "${ARGS[@]}"
