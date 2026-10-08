#!/bin/bash
# Token selector training: Qwen3-1.7B experimental run (multi-node DDP)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Optional HTTP(S) proxy. Leave unset for direct network access.
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

# CUDA environment (override CUDA_HOME if your toolkit lives elsewhere)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib:/usr/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:$PATH"

DATA_PATH="${DATA_PATH:?set DATA_PATH to the image training webdataset root}"
VIDEO_DATA_PATH="${VIDEO_DATA_PATH:?set VIDEO_DATA_PATH to the video training webdataset root}"
OUTPUT_DIR="${OUTPUT_DIR:-output/sft_1.7b_vit_native_reso_with_3stream_trainedViT}"

# ===== Training epochs =====
NUM_EPOCHS=${NUM_EPOCHS:-1}
# Dataset sample count: used for epoch -> step conversion (fill in actual dataset size)
NUM_SAMPLES_PER_EPOCH="${NUM_SAMPLES_PER_EPOCH:?set NUM_SAMPLES_PER_EPOCH to the training dataset size (used for epoch->step conversion)}"

# ===== Uniform frame-sampling branch =====
# A video batch has UNIFORM_TRAIN_PROB probability of taking the uniform-sampling branch;
# otherwise it takes the visidx sparse path.
# Both branches produce the same number of segments (128/32 = 32/8 = 4 segments),
# so the LLM-side token count is identical.
UNIFORM_SAMPLE_N=${UNIFORM_SAMPLE_N:-16}          # number of frames after uniform downsampling
UNIFORM_SEGMENT_T_SIZE=${UNIFORM_SEGMENT_T_SIZE:-4}  # segment_t_size for the uniform branch
UNIFORM_TRAIN_PROB=${UNIFORM_TRAIN_PROB:-0.5}     # probability of taking the uniform branch (0.0~1.0)

# ===== ViT input resolution (controlled per modality) =====
# 0 means use the image_size from the model config (default 224)
IMAGE_SIZE_IMAGE=${IMAGE_SIZE_IMAGE:-448}   # image resolution, e.g. 336
IMAGE_SIZE_VIDEO=${IMAGE_SIZE_VIDEO:-224}   # video frame resolution, e.g. 224

# ===== Image native resolution =====
# When NATIVE_RESOLUTION=1: each image is resized to the nearest size divisible by
# patch size (14) (aspect ratio preserved), with total patch count constrained to
# [NATIVE_MIN_PATCHES, NATIVE_MAX_PATCHES].
# Applies to images only; videos still use the fixed IMAGE_SIZE_VIDEO resolution.
# Since images of different sizes in the same batch cannot be stacked, native resolution
# only works when image micro-batch == 1; so when enabled it forces --micro-batch to 1
# and scales up grad-accum proportionally to keep the effective batch size.
NATIVE_RESOLUTION=${NATIVE_RESOLUTION:-1}
NATIVE_MIN_PATCHES=${NATIVE_MIN_PATCHES:-256}    # 256 = 224x224
NATIVE_MAX_PATCHES=${NATIVE_MAX_PATCHES:-4096}   # 1296 = 504x504

MICRO_BATCH=${MICRO_BATCH:-10}
GRAD_ACCUM=${GRAD_ACCUM:-4}
NATIVE_ARGS=()
if [ "${NATIVE_RESOLUTION}" = "1" ]; then
    NATIVE_ARGS=(--native-resolution
                 --native-min-patches "${NATIVE_MIN_PATCHES}"
                 --native-max-patches "${NATIVE_MAX_PATCHES}")
    GRAD_ACCUM=$((GRAD_ACCUM * MICRO_BATCH))   # keep the effective batch size unchanged
    MICRO_BATCH=1
fi

torchrun \
    --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --nnodes="${NNODES:-8}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29500}" \
    -m train.train \
    --config configs/covisco_qwen3_1.7b.yaml \
    --data-path "${DATA_PATH}" \
    --video-data-path "${VIDEO_DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --stage sft \
    --trainable-modules projector token_selector llm\
    --lr 1e-4 \
    --num-epochs "${NUM_EPOCHS}" \
    --num-samples-per-epoch "${NUM_SAMPLES_PER_EPOCH}" \
    --warmup-ratio 0.01 \
    --lr-scheduler constant\
    --grad-accum  32 \
    --micro-batch 1 \
    --video-micro-batch "${VIDEO_MICRO_BATCH:-1}" \
    --uniform-sample-n "${UNIFORM_SAMPLE_N}" \
    --uniform-segment-t-size "${UNIFORM_SEGMENT_T_SIZE}" \
    --uniform-train-prob "${UNIFORM_TRAIN_PROB}" \
    --frame-cat-prob 0. \
    --candidate-resolutions 224 336 448 504\
    --image-size-image "${IMAGE_SIZE_IMAGE}" \
    --image-size-video "${IMAGE_SIZE_VIDEO}" \
    "${NATIVE_ARGS[@]}" \
    --log-interval 10 \
    --save-interval 500 \
    --bf16 \
    # --num-train-steps 332031 
    # --ckpt /path/to/code/CoVisco_sft/output/sft_1.7b_vit_multi_reso/step_8000.pt  
    # --skip-optimizer-state
# 
# To enable val loss, append:
#   --do-valid \
#   --val-data-path /path/to/val_wds \
#   --eval-interval 500 \
#   --eval-iters 20 \
#   --eval-strategy query_and_vit \
#   --eval-vit-k 128 \
#   --save-best-on-val
