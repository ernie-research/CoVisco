#!/bin/bash
# SFT: Qwen3-4B-Instruct2507 experimental run (multi-node DDP)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Optional HTTP(S) proxy. Leave unset for direct network access.
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# Or first confirm it is a false kill:
export TORCH_NCCL_ENABLE_MONITORING=0

DATA_PATH="${DATA_PATH:?set DATA_PATH to the image training webdataset root}"
VIDEO_DATA_PATH="${VIDEO_DATA_PATH:?set VIDEO_DATA_PATH to the video training webdataset root}"
OUTPUT_DIR="${OUTPUT_DIR:-output/sft_4b_instruct2507_instruction_tuning}"

# Checkpoint to resume from. Leave empty to start from the base weights.
CKPT="${CKPT:-}"
CKPT_ARGS=()
if [ -n "${CKPT}" ]; then
    CKPT_ARGS=(--ckpt "${CKPT}")
fi

# On resume, skip image shards already trained (including the half-read one, ceil).
# Point SKIP_SHARDS at a newline-separated list of shard filenames to skip.
# Leave empty to process all shards. Videos loop over multiple rounds and are not skipped.
SKIP_SHARDS=${SKIP_SHARDS:-}
SKIP_SHARD_ARGS=()
if [ -n "${SKIP_SHARDS}" ]; then
    SKIP_SHARD_ARGS=(--skip-shards "${SKIP_SHARDS}")
fi

# ===== Training epochs =====
NUM_EPOCHS=${NUM_EPOCHS:-1}
# Dataset sample count: used for epoch -> step conversion (fill in the actual image
# sample count of DATA_PATH).
# llava_next_wds_780K = 390 shards x 2000 = 780000. The previously used 1385000 is the
# scale of VIDEO_DATA_PATH (video stats.json: total_samples=1384174), which overestimates
# num_train_steps by 1.78x; after the image data is exhausted the outer loop restarts the
# loader and reads it again.
NUM_SAMPLES_PER_EPOCH=${NUM_SAMPLES_PER_EPOCH:-1000000}

# ===== Data ordering =====
# Shards are packed by data source (samples within one tar are homogeneous in task), so
# reading sequentially makes the task composition of the global batch switch all at once
# every (shard_size*num_workers/samples_per_step) steps, giving a staircase-shaped loss.
# Interleaving K shards + a sample-level shuffle buffer removes this structure.
SHARD_INTERLEAVE=${SHARD_INTERLEAVE:-4}
SHUFFLE_BUFFER=${SHUFFLE_BUFFER:-500}

# ===== Uniform frame-sampling branch =====
# A video batch has UNIFORM_TRAIN_PROB probability of taking the uniform-sampling branch;
# otherwise it takes the visidx sparse path.
# Both branches produce the same number of segments (128/32 = 32/8 = 4 segments),
# so the LLM-side token count is identical.
VIDEO_TARGET_FRAMES=${VIDEO_TARGET_FRAMES:-64}   # frames uniformly sampled when decoding mp4 (jpg-frame datasets unaffected)
UNIFORM_SAMPLE_N=${UNIFORM_SAMPLE_N:-16}          # number of frames after uniform downsampling
UNIFORM_SEGMENT_T_SIZE=${UNIFORM_SEGMENT_T_SIZE:-4}  # segment_t_size for the uniform branch
UNIFORM_TRAIN_PROB=${UNIFORM_TRAIN_PROB:-0.5}     # probability of taking the uniform branch (0.0~1.0)

# ===== ViT input resolution (controlled per modality) =====
# 0 means use the image_size from the model config (default 224)
IMAGE_SIZE_IMAGE=${IMAGE_SIZE_IMAGE:-224}   # image resolution, e.g. 336
IMAGE_SIZE_VIDEO=${IMAGE_SIZE_VIDEO:-224}   # video frame resolution, e.g. 224

# ===== Image native resolution =====
# When NATIVE_RESOLUTION=1: each image is resized to the nearest size divisible by
# patch size (14) (aspect ratio preserved), with total patch count constrained to
# [NATIVE_MIN_PATCHES, NATIVE_MAX_PATCHES].
# Applies to images only; videos still use the fixed IMAGE_SIZE_VIDEO resolution.
# Since images of different sizes in the same batch cannot be stacked, native resolution
# only works when image micro-batch == 1.
# When enabled, grad-accum and video-micro-batch-every are recomputed so that the number
# of images and videos consumed per optimizer step matches the disabled case
# (image_mb = G*V/(V+1), video_mb = G/(V+1), with V=original MICRO_BATCH,
# G=original GRAD_ACCUM/2*(V+1)).
NATIVE_RESOLUTION=${NATIVE_RESOLUTION:-1}
NATIVE_MIN_PATCHES=${NATIVE_MIN_PATCHES:-256}    # 256 = 224x224
NATIVE_MAX_PATCHES=${NATIVE_MAX_PATCHES:-4096}  
MICRO_BATCH=${MICRO_BATCH:-1}
# GRAD_ACCUM=${GRAD_ACCUM:-16}
VIDEO_EVERY=${VIDEO_EVERY:-1}
NATIVE_ARGS=()
if [ "${NATIVE_RESOLUTION}" = "1" ]; then
    NATIVE_ARGS=(--native-resolution
                 --native-min-patches "${NATIVE_MIN_PATCHES}"
                 --native-max-patches "${NATIVE_MAX_PATCHES}")
    # VIDEO_EVERY=${MICRO_BATCH}
    # GRAD_ACCUM=$(( GRAD_ACCUM / 2 * (VIDEO_EVERY + 1) ))
    # MICRO_BATCH=1
fi
RESET_STEP=0
torchrun \
    --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29500}" \
    -m train.train \
    --config configs/covisco_qwen3_4b_instruct2507_instruction_tuning.yaml \
    --data-path "${DATA_PATH}" \
    --video-data-path "${VIDEO_DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --stage sft \
    --trainable-modules projector llm\
    --lr 5e-5 \
    --num-epochs "${NUM_EPOCHS}" \
    --num-samples-per-epoch "${NUM_SAMPLES_PER_EPOCH}" \
    --warmup-ratio 0.0 \
    --lr-scheduler constant\
    --grad-accum 64  \
    --micro-batch 1 \
    --video-micro-batch "${VIDEO_MICRO_BATCH:-1}" \
    --video-micro-batch-every "${VIDEO_EVERY}" \
    --video-target-frames "${VIDEO_TARGET_FRAMES}" \
    --uniform-sample-n "${UNIFORM_SAMPLE_N}" \
    --uniform-segment-t-size "${UNIFORM_SEGMENT_T_SIZE}" \
    --uniform-train-prob "${UNIFORM_TRAIN_PROB}" \
    --frame-cat-prob 0. \
    --image-size-image "${IMAGE_SIZE_IMAGE}" \
    --image-size-video "${IMAGE_SIZE_VIDEO}" \
    "${NATIVE_ARGS[@]}" \
    --log-interval 10 \
    --save-interval 1000 \
    --shard-interleave "${SHARD_INTERLEAVE}" \
    --shuffle-buffer "${SHUFFLE_BUFFER}" \
    --bf16 \
    --candidate-resolutions 224 336 448 504 \
    "${SKIP_SHARD_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    ${RESET_STEP:+--reset-step "${RESET_STEP}"}
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
