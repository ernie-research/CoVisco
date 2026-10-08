#!/bin/bash
# ============================================================================
# CoVisco Video Evaluation with CODEC token filtering (visidx)
# ============================================================================
# Follows the codec-filtered evaluation flow of OneVision-Encoder/llava_next, adapted to
# this repo's model: video patch-level filtering is injected into the ViT via precomputed
# visidx (instead of mosaic + patch_positions).
#
# One-time preprocessing (run once per benchmark):
#   bash scripts/precompute_codec_visidx/preprocess_video_benchmark.sh videomme
#
# Evaluation (run video benchmarks one at a time):
#   TASKS="videomme" bash scripts/eval_covisco_codec.sh
#   TASKS="perceptiontest_val_mc" bash scripts/eval_covisco_codec.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional HTTP(S) proxy. Leave unset for direct network access.
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[WARNING] HF_TOKEN not set. Gated datasets may fail."
fi
export HF_TOKEN="${HF_TOKEN:-}"

LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-${REPO_DIR}/third_party/lmms-eval}"
if [ ! -d "$LMMS_EVAL_DIR" ]; then
    echo "[ERROR] lmms-eval not found: $LMMS_EVAL_DIR"; exit 1
fi
export PYTHONPATH=${REPO_DIR}:${LMMS_EVAL_DIR}
# HF_HOME holds the pre-downloaded videos (mp4) / hub snapshots; doc_to_visual reads from
# HF_HOME/<task>/data.
export HF_HOME="${HF_HOME:?set HF_HOME to the directory containing the pre-downloaded evaluation videos}"
# HF datasets needs a filelock when building/loading the arrow cache; that lock requires a
# local filesystem (ext4/overlay), so keep the datasets cache (tiny, annotation arrow only)
# on local disk even if the large video files live elsewhere.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:?set HF_DATASETS_CACHE to a local (ext4/overlay) cache directory}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME/hub"
# Note: do not set HF_DATASETS_OFFLINE/HF_HUB_OFFLINE. This datasets version errors out
# directly when offline instead of falling back to the cached module; when the network is
# blocked it auto-falls back to the local arrow cache (verified to load).
unset HF_ENDPOINT

# ============ Configuration ============
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/covisco_qwen3_4b_instruct2507_instruction_tuning.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH?set CHECKPOINT_PATH to your trained checkpoint (or set CHECKPOINT_PATH=\"\" to evaluate untrained weights on purpose)}"

# Video benchmark, one at a time (corresponds to the TASK used for visidx precompute)
TASKS="${TASKS:-videomme}"

RUN_PORT="${RUN_PORT:-12458}"
MODEL_NAME="${MODEL_NAME:-covisco_qwen3_4b_codec}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

# ============ codec visidx (must match the precompute parameters) ============
# Frames per segment / frame count / video resolution / patch_size directly determine the
# visidx index space; keep consistent with
# scripts/precompute_codec_visidx/preprocess_video_benchmark.sh.
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-64}"
SEGMENT_T_SIZE="${SEGMENT_T_SIZE:-16}"
IMAGE_SIZE_VIDEO="${IMAGE_SIZE_VIDEO:-224}"
# visidx asset root (<root>/<video_stem>/visidx.npy)
CODEC_VISIDX_ROOT="${CODEC_VISIDX_ROOT:?set CODEC_VISIDX_ROOT to the precomputed visidx cache root for this benchmark}"

if [ ! -d "$CODEC_VISIDX_ROOT" ]; then
    echo "[ERROR] codec visidx directory does not exist: $CODEC_VISIDX_ROOT"
    echo "[ERROR] Please run first: bash scripts/precompute_codec_visidx/preprocess_video_benchmark.sh ${TASKS}"
    exit 1
fi

# Token selector: codec pre-filtering as the candidate pool; query_and_vit then adds query tokens + learned fine selection
TOKEN_STRATEGY="${TOKEN_STRATEGY:-query_and_vit}"
VIT_RATIO="${VIT_RATIO:-0.2}"
MMR_LAMBDA="${MMR_LAMBDA:-0.3}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"

# ============ Validate paths ============
if [ ! -f "$CONFIG_PATH" ]; then echo "[ERROR] Config not found: $CONFIG_PATH"; exit 1; fi
if [ -n "$CHECKPOINT_PATH" ] && [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH"; exit 1
fi

# ============ Build model arguments ============
MODEL_ARGS="config_path=${CONFIG_PATH}"
if [ -n "$CHECKPOINT_PATH" ]; then
    MODEL_ARGS="${MODEL_ARGS},checkpoint_path=${CHECKPOINT_PATH}"
fi
MODEL_ARGS="${MODEL_ARGS},conv_template=qwen_1_5,num_video_frames=${NUM_VIDEO_FRAMES},max_new_tokens=${MAX_NEW_TOKENS}"
MODEL_ARGS="${MODEL_ARGS},token_strategy=${TOKEN_STRATEGY},vit_ratio=${VIT_RATIO},mmr_lambda=${MMR_LAMBDA}"
MODEL_ARGS="${MODEL_ARGS},segment_t_size=${SEGMENT_T_SIZE}"
MODEL_ARGS="${MODEL_ARGS},image_size_video=${IMAGE_SIZE_VIDEO}"
MODEL_ARGS="${MODEL_ARGS},codec_visidx_root=${CODEC_VISIDX_ROOT}"
MODEL_ARGS="${MODEL_ARGS},dtype=${MODEL_DTYPE}"

# ============ Print configuration ============
echo "========================================"
echo "CoVisco CODEC Video Evaluation"
echo "========================================"
echo "Config:            $CONFIG_PATH"
echo "Checkpoint:        ${CHECKPOINT_PATH:-<none>}"
echo "Tasks:             $TASKS"
echo "Num GPUs:          $NUM_GPUS"
echo "Video Frames:      $NUM_VIDEO_FRAMES"
echo "segment_t_size:    $SEGMENT_T_SIZE"
echo "image_size_video:  $IMAGE_SIZE_VIDEO"
echo "codec_visidx_root: $CODEC_VISIDX_ROOT"
echo "token_strategy:    $TOKEN_STRATEGY  (vit_ratio=$VIT_RATIO, mmr_lambda=$MMR_LAMBDA)"
echo "dtype:             $MODEL_DTYPE"
echo "========================================"

# ============ Run evaluation ============
cd ${LMMS_EVAL_DIR}

python -m accelerate.commands.launch \
    --main_process_port=$RUN_PORT \
    --num_processes=$NUM_GPUS \
    -m lmms_eval \
    --model llava_covisco \
    --model_args "$MODEL_ARGS" \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${MODEL_NAME}_$(date +%Y%m%d) \
    --output_path ${REPO_DIR}/eval_results/${MODEL_NAME}/

echo "========================================"
echo "Evaluation completed!"
echo "Results saved to: ${REPO_DIR}/eval_results/${MODEL_NAME}/"
echo "========================================"
