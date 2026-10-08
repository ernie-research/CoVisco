#!/bin/bash
# ============================================================================
# CoVisco Long-Video Evaluation Script (256 frames/video)
# ============================================================================
# This script evaluates CoVisco models on LONG VIDEO benchmarks:
#   - MLVU            (mlvu_dev)
#   - LongVideoBench  (longvideobench_val_v)
#
# Each video is sampled at 256 frames (NUM_VIDEO_FRAMES=256).
# Video benchmarks are run ONE at a time (looped) to avoid dataset/loader
# state leaking between tasks.
#
# Usage:
#   bash scripts/eval_covisco_longvideo.sh
#   # override frames or tasks:
#   NUM_VIDEO_FRAMES=128 bash scripts/eval_covisco_longvideo.sh
#   TASKS="mlvu_dev" bash scripts/eval_covisco_longvideo.sh
# ============================================================================

set -euo pipefail

# Get the absolute path of the repo directory based on script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional HTTP(S) proxy. Leave unset for direct network access.
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

# Hugging Face Token (required for gated datasets)
# Set your HF token here or export HF_TOKEN before running
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[WARNING] HF_TOKEN not set. Gated datasets may fail."
    echo "[WARNING] Get your token from: https://huggingface.co/settings/tokens"
fi
# Pass HF_TOKEN via an environment variable; do not commit the token into the repo
export HF_TOKEN="${HF_TOKEN:-}"

# ============ LMMS-Eval Paths ============
# lmms-eval is vendored into this repo under third_party/ (see third_party/UPSTREAM.txt)
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-${REPO_DIR}/third_party/lmms-eval}"
if [ ! -d "$LMMS_EVAL_DIR" ]; then
    echo "[ERROR] lmms-eval not found: $LMMS_EVAL_DIR"
    exit 1
fi
export PYTHONPATH=${REPO_DIR}:${LMMS_EVAL_DIR}
# HF_HOME must point to a local disk (ext4/overlay), not a FUSE/network mount: the
# datasets library's filelock raises "Bad file descriptor" / "Input/output error" there.
export HF_HOME="${HF_HOME:?set HF_HOME to a local (ext4/overlay) directory for the datasets/hub cache}"
mkdir -p "$HF_HOME/datasets" "$HF_HOME/hub"
unset HF_ENDPOINT

echo "[DEBUG] REPO_DIR=${REPO_DIR}"
echo "[DEBUG] LMMS_EVAL_DIR=${LMMS_EVAL_DIR}"
echo "[DEBUG] HF_HOME=${HF_HOME}"

# ============ Configuration ============
# Model configuration
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/covisco_qwen3_4b_instruct2507_instruction_tuning.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH?set CHECKPOINT_PATH to your trained checkpoint (or set CHECKPOINT_PATH=\"\" to evaluate untrained weights on purpose)}"

# Long video tasks to evaluate (run ONE at a time via the loop below)
TASKS="${TASKS:-mlvu_dev,longvideobench_val_v}"

# Evaluation configuration
RUN_PORT="${RUN_PORT:-12457}"
MODEL_NAME="${MODEL_NAME:-covisco_qwen3_4b}"
NUM_GPUS="${NUM_GPUS:-4}"
# Long video: sample 256 frames per video
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

# ============ Validate paths ============
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ -n "$CHECKPOINT_PATH" ] && [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "[ERROR] Checkpoint file not found: $CHECKPOINT_PATH"
    echo "[ERROR] To evaluate untrained weights on purpose, explicitly set CHECKPOINT_PATH=\"\""
    exit 1
fi

# ============ Build model arguments ============
MODEL_ARGS="config_path=${CONFIG_PATH}"
if [ -n "$CHECKPOINT_PATH" ]; then
    MODEL_ARGS="${MODEL_ARGS},checkpoint_path=${CHECKPOINT_PATH}"
fi
MODEL_ARGS="${MODEL_ARGS},conv_template=qwen_1_5,num_video_frames=${NUM_VIDEO_FRAMES},max_new_tokens=${MAX_NEW_TOKENS}"

# Token selector args
# TOKEN_STRATEGY: query_only | vit_only | query_and_vit (must match the training distribution)
#   During training, image samples: vit_only 50% / query_only 20% / query_and_vit 30% (ratio 0.4~0.8),
#   OCR samples are always vit_only (full patches). So document/chart benchmarks use vit_only.
#   VIT_RATIO has no effect under vit_only.
TOKEN_STRATEGY="${TOKEN_STRATEGY:-query_only}"
VIT_RATIO="${VIT_RATIO:-1.0}"
MMR_LAMBDA="${MMR_LAMBDA:-0.3}"
MODEL_ARGS="${MODEL_ARGS},token_strategy=${TOKEN_STRATEGY},vit_ratio=${VIT_RATIO},mmr_lambda=${MMR_LAMBDA}"

# ViT segment_t_size (defaults to the config value 32; override via environment variable)
SEGMENT_T_SIZE="${SEGMENT_T_SIZE:-16}"
if [ -n "$SEGMENT_T_SIZE" ]; then
    MODEL_ARGS="${MODEL_ARGS},segment_t_size=${SEGMENT_T_SIZE}"
fi

# Image/video resolution (defaults to the config value; override via environment variable)
# IMAGE_SIZE: overrides config.vit.image_size (shared fallback for image and video)
# IMAGE_SIZE_IMAGE / IMAGE_SIZE_VIDEO: override image/video resolution separately, higher priority
IMAGE_SIZE="${IMAGE_SIZE:-}"
IMAGE_SIZE_IMAGE="${IMAGE_SIZE_IMAGE:-448}"
IMAGE_SIZE_VIDEO="${IMAGE_SIZE_VIDEO:-224}"
if [ -n "$IMAGE_SIZE" ]; then
    MODEL_ARGS="${MODEL_ARGS},image_size=${IMAGE_SIZE}"
fi
if [ -n "$IMAGE_SIZE_IMAGE" ]; then
    MODEL_ARGS="${MODEL_ARGS},image_size_image=${IMAGE_SIZE_IMAGE}"
fi
if [ -n "$IMAGE_SIZE_VIDEO" ]; then
    MODEL_ARGS="${MODEL_ARGS},image_size_video=${IMAGE_SIZE_VIDEO}"
fi

# Image native resolution mode (affects images only; videos always use fixed resolution)
# When NATIVE_RESOLUTION=1: ignore IMAGE_SIZE_IMAGE, resize each image by its original
# aspect ratio to the nearest size divisible by patch_size(14), forcing batch_size to 1
# NATIVE_MAX_PATCHES / NATIVE_MIN_PATCHES bound the total patch count (ViT attention is O(N^2))
NATIVE_RESOLUTION="${NATIVE_RESOLUTION:-1}"
NATIVE_MAX_PATCHES="${NATIVE_MAX_PATCHES:-4096}"
# Keep consistent with the training script (run_sft_4b_instruct2507.sh: NATIVE_MIN_PATCHES=256)
NATIVE_MIN_PATCHES="${NATIVE_MIN_PATCHES:-256}"
if [ "$NATIVE_RESOLUTION" = "1" ] || [ "$NATIVE_RESOLUTION" = "true" ] || [ "$NATIVE_RESOLUTION" = "True" ]; then
    MODEL_ARGS="${MODEL_ARGS},native_resolution=True,native_max_patches=${NATIVE_MAX_PATCHES},native_min_patches=${NATIVE_MIN_PATCHES}"
fi

# Model runtime dtype (default bfloat16; converts all submodules uniformly to avoid matmul dtype mismatch errors)
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
MODEL_ARGS="${MODEL_ARGS},dtype=${MODEL_DTYPE}"

# ============ Print configuration ============
echo "========================================"
echo "CoVisco Long-Video Evaluation"
echo "========================================"
echo "Config:       $CONFIG_PATH"
echo "Checkpoint:   ${CHECKPOINT_PATH:-<none>}"
echo "Tasks:        $TASKS"
echo "Model Name:   $MODEL_NAME"
echo "Num GPUs:     $NUM_GPUS"
echo "Video Frames: $NUM_VIDEO_FRAMES"
echo "token_strategy: $TOKEN_STRATEGY"
echo "vit_ratio:    $VIT_RATIO"
echo "mmr_lambda:   $MMR_LAMBDA"
echo "segment_t_size: ${SEGMENT_T_SIZE:-<config default>}"
echo "image_size:      ${IMAGE_SIZE:-<config default>}"
echo "image_size_image: ${IMAGE_SIZE_IMAGE:-<config default>}"
echo "image_size_video: ${IMAGE_SIZE_VIDEO:-<config default>}"
echo "native_resolution: ${NATIVE_RESOLUTION} (max_patches=${NATIVE_MAX_PATCHES}, min_patches=${NATIVE_MIN_PATCHES})"
echo "dtype:           $MODEL_DTYPE"
echo "========================================"

# ============ Run evaluation (one task at a time) ============
cd ${LMMS_EVAL_DIR}

# Run the comma-separated TASKS one by one to avoid state leaking between video datasets
IFS=',' read -ra TASK_ARR <<< "$TASKS"
for TASK in "${TASK_ARR[@]}"; do
    TASK="$(echo "$TASK" | xargs)"  # trim whitespace
    [ -z "$TASK" ] && continue
    echo "----------------------------------------"
    echo "[RUN] task=${TASK}  frames=${NUM_VIDEO_FRAMES}"
    echo "----------------------------------------"
    python -m accelerate.commands.launch \
        --main_process_port=$RUN_PORT \
        --num_processes=$NUM_GPUS \
        -m lmms_eval \
        --model llava_covisco \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASK" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix ${MODEL_NAME}_$(date +%Y%m%d) \
        --output_path ${REPO_DIR}/eval_results/${MODEL_NAME}/
done

echo "========================================"
echo "Evaluation completed!"
echo "Results saved to: ${REPO_DIR}/eval_results/${MODEL_NAME}/"
echo "========================================"
