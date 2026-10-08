#!/bin/bash
set -euo pipefail

# =============================================================================
# CoVisco ViT (SigLIP) Training Script with WebDataset Tar Files
# =============================================================================
# --- Usage ---
# Node 0: bash ./scripts/run_covisco_train.sh 0
# Node 1: bash ./scripts/run_covisco_train.sh 1
# Node 2: bash ./scripts/run_covisco_train.sh 2
# Node 3: bash ./scripts/run_covisco_train.sh 3
# ...
# 1. Argument check
if [ -z "$1" ]; then
  echo "Error: NODE_RANK argument is missing."
  echo "Usage: bash run_covisco_train.sh <NODE_RANK>"
  exit 1
fi
RANK=$1

# 2. Environment setup
cd "$(dirname "$0")"/..
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

# CUDA memory allocator config: reduces OOM from fragmentation after eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Distributed communication config
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=$RANK

echo "====================================================="
echo "CoVisco ViT (SigLIP) Training with WebDataset Tar Files"
echo "====================================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "====================================================="

# 4. Data config - WebDataset format
# Adjust the variables below to your actual data paths
# IMAGE_DATA_PATH: image-caption / image-image contrastive data (the real image dataset, fill in yourself)
IMAGE_DATA_PATH="${IMAGE_DATA_PATH:?set IMAGE_DATA_PATH to the image training webdataset root}"
VIDEO_DATA_PATH="${VIDEO_DATA_PATH:-}"
# IMAGE_VIDCAP_DATA_PATH: image <-> video text-encoder caption contrastive data (separate dataloader)
IMAGE_VIDCAP_DATA_PATH="${IMAGE_VIDCAP_DATA_PATH:-}"

# Omit the corresponding --xxx-data-path when a path is empty to avoid building an empty shard list
IMAGE_DATA_ARGS=""
if [ -n "$IMAGE_DATA_PATH" ]; then
    IMAGE_DATA_ARGS="--image-data-path $IMAGE_DATA_PATH"
fi
VIDEO_DATA_ARGS=""
if [ -n "$VIDEO_DATA_PATH" ]; then
    VIDEO_DATA_ARGS="--video-data-path $VIDEO_DATA_PATH"
fi
IMAGE_VIDCAP_DATA_ARGS=""
if [ -n "$IMAGE_VIDCAP_DATA_PATH" ]; then
    IMAGE_VIDCAP_DATA_ARGS="--image-vidcap-data-path $IMAGE_VIDCAP_DATA_PATH"
fi

# Sample count config
# Each image shard holds about 6400 images
# Each video shard holds about 1000 videos
IMAGE_SAMPLES_PER_SHARD=${IMAGE_SAMPLES_PER_SHARD:-6400}
VIDEO_SAMPLES_PER_SHARD=${VIDEO_SAMPLES_PER_SHARD:-1000}
IMAGE_VIDCAP_SAMPLES_PER_SHARD=${IMAGE_VIDCAP_SAMPLES_PER_SHARD:-2000}

# Total sample counts (0 = compute automatically from the actual shard count; set to the true dataset size for an exact epoch->step conversion)
IMAGE_NUM_SAMPLES=${IMAGE_NUM_SAMPLES:-0}
VIDEO_NUM_SAMPLES=${VIDEO_NUM_SAMPLES:-0}
IMAGE_VIDCAP_NUM_SAMPLES=${IMAGE_VIDCAP_NUM_SAMPLES:-0}

# 5. Training parameters
MODEL_NAME="${MODEL_NAME:-CoVisco-L-14}"
IMAGE_BATCH_SIZE=${IMAGE_BATCH_SIZE:-44}
VIDEO_BATCH_SIZE=${VIDEO_BATCH_SIZE:-4}
IMAGE_VIDCAP_BATCH_SIZE=${IMAGE_VIDCAP_BATCH_SIZE:-44}
ACCUM_FREQ=${ACCUM_FREQ:-12}
IMG_ACCUM_FREQ=${IMG_ACCUM_FREQ:-46}
VID_ACCUM_FREQ=${VID_ACCUM_FREQ:-50}
IMAGE_VIDCAP_ACCUM_FREQ=${IMAGE_VIDCAP_ACCUM_FREQ:-46}
EPOCHS=${EPOCHS:-40}
LR=${LR:-3e-4}
WARMUP_STEPS=${WARMUP_STEPS:-0}
WD=${WD:-3e-5}

# Distributed SigLIP implementation (reduce is more stable than bidir)
SIGLIP_DIST_IMPL="${SIGLIP_DIST_IMPL:-reduce}"

# Loss weights
RECONSTRUCTION_WEIGHT=${RECONSTRUCTION_WEIGHT:-0.}
IMAGE_CAPTION_WEIGHT=${IMAGE_CAPTION_WEIGHT:-1.0}
IMAGE_IMAGE_WEIGHT=${IMAGE_IMAGE_WEIGHT:-1.0}
VIDEO_CAPTION_WEIGHT=${VIDEO_CAPTION_WEIGHT:-1.0}
IMAGE_VIDEO_CAPTION_WEIGHT=${IMAGE_VIDEO_CAPTION_WEIGHT:-1.0}

# Reconstruction decoder config
USE_RECONSTRUCTION=${USE_RECONSTRUCTION:-false}
DECODER_LAYERS=${DECODER_LAYERS:-8}
DECODER_NUM_QUERIES=${DECODER_NUM_QUERIES:-256}
DECODER_NUM_HEADS=${DECODER_NUM_HEADS:-8}

# Logit scale initial values
INIT_SCALE_IMAGE_CAPTION=${INIT_SCALE_IMAGE_CAPTION:-10}
INIT_SCALE_IMAGE_IMAGE=${INIT_SCALE_IMAGE_IMAGE:-10}
INIT_SCALE_VIDEO_CAPTION=${INIT_SCALE_VIDEO_CAPTION:-10}

# Logit bias initial values
INIT_BIAS_IMAGE_CAPTION=${INIT_BIAS_IMAGE_CAPTION:--10.0}
INIT_BIAS_IMAGE_IMAGE=${INIT_BIAS_IMAGE_IMAGE:--10.0}
INIT_BIAS_VIDEO_CAPTION=${INIT_BIAS_VIDEO_CAPTION:--10.0}

# Embedding dimensions
IMAGE_EMBED_DIM=${IMAGE_EMBED_DIM:-1536}
IMAGE_CAPTION_EMBED_DIM=${IMAGE_CAPTION_EMBED_DIM:-1536}
VIDEO_CAPTION_EMBED_DIM=${VIDEO_CAPTION_EMBED_DIM:-4096}

# ImageNet zero-shot eval config (webdataset format)
# During training, zero-shot classification uses the text encoder of timm/ViT-gopt-16-SigLIP2-384
# Leave empty to skip ImageNet zero-shot evaluation during training
IMAGENET_VAL_WDS="${IMAGENET_VAL_WDS:-}"
IMAGENET_VAL_NUM_SAMPLES=${IMAGENET_VAL_NUM_SAMPLES:-50000}
ZEROSHOT_FREQUENCY=${ZEROSHOT_FREQUENCY:-1}
ZEROSHOT_ARGS=()
if [ -n "$IMAGENET_VAL_WDS" ]; then
    ZEROSHOT_ARGS=(--imagenet-val-wds "$IMAGENET_VAL_WDS"
                   --imagenet-val-num-samples "$IMAGENET_VAL_NUM_SAMPLES"
                   --zeroshot-frequency "$ZEROSHOT_FREQUENCY")
fi

# Logging and checkpoint config
LOGS_DIR="${LOGS_DIR:-logs/covisco_siglip_wds}"
REPORT_TO="${REPORT_TO:-tensorboard}"
IMAGE_WORKERS=${IMAGE_WORKERS:-8}
VIDEO_WORKERS=${VIDEO_WORKERS:-8}  # Reduced to prevent OOM during video decoding
IMAGE_VIDCAP_WORKERS=${IMAGE_VIDCAP_WORKERS:-8}
DISABLE_VIDEO_AUG=${DISABLE_VIDEO_AUG:-true}
VIDEO_AUG_ARGS=""
if [ "$DISABLE_VIDEO_AUG" = "true" ]; then
    VIDEO_AUG_ARGS="--disable-video-aug"
fi

SAVE_STEPS=${SAVE_STEPS:-100}

# Leave empty to train from scratch; set to a checkpoint path to resume
RESUME="${RESUME:-}"
RESUME_ARGS=()
if [ -n "$RESUME" ]; then
    RESUME_ARGS=(--resume "$RESUME")
fi
# Build the --use-reconstruction argument
RECONSTRUCTION_ARGS=""
if [ "$USE_RECONSTRUCTION" = "true" ]; then
    RECONSTRUCTION_ARGS="--use-reconstruction"
fi

python -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -- \
    -m open_clip_train.main_covisco \
    --dataset-type webdataset_embedding \
    $IMAGE_DATA_ARGS \
    $VIDEO_DATA_ARGS \
    $IMAGE_VIDCAP_DATA_ARGS \
    --image-samples-per-shard $IMAGE_SAMPLES_PER_SHARD \
    --video-samples-per-shard $VIDEO_SAMPLES_PER_SHARD \
    --image-vidcap-samples-per-shard $IMAGE_VIDCAP_SAMPLES_PER_SHARD \
    --image-num-samples $IMAGE_NUM_SAMPLES \
    --video-num-samples $VIDEO_NUM_SAMPLES \
    --image-vidcap-num-samples $IMAGE_VIDCAP_NUM_SAMPLES \
    --image-batch-size $IMAGE_BATCH_SIZE \
    --video-batch-size $VIDEO_BATCH_SIZE \
    --image-vidcap-batch-size $IMAGE_VIDCAP_BATCH_SIZE \
    --force-image-size  448\
    --video-size  224 \
    --dynamic-resolution \
    --dynamic-resolutions 224 336 448\
    --image-embed-dim $IMAGE_EMBED_DIM \
    --image-caption-embed-dim $IMAGE_CAPTION_EMBED_DIM \
    --video-caption-embed-dim $VIDEO_CAPTION_EMBED_DIM \
    --model $MODEL_NAME \
    --loss-dist-impl $SIGLIP_DIST_IMPL \
    $RECONSTRUCTION_ARGS \
    $VIDEO_AUG_ARGS \
    --reconstruction-weight $RECONSTRUCTION_WEIGHT \
    --decoder-layers $DECODER_LAYERS \
    --decoder-num-image-queries $DECODER_NUM_QUERIES \
    --decoder-num-video-queries $DECODER_NUM_QUERIES \
    --decoder-num-heads $DECODER_NUM_HEADS \
    --init-logit-scale-image-caption $INIT_SCALE_IMAGE_CAPTION \
    --init-logit-scale-image-image $INIT_SCALE_IMAGE_IMAGE \
    --init-logit-scale-video-caption $INIT_SCALE_VIDEO_CAPTION \
    --init-logit-bias-image-caption $INIT_BIAS_IMAGE_CAPTION \
    --init-logit-bias-image-image $INIT_BIAS_IMAGE_IMAGE \
    --init-logit-bias-video-caption $INIT_BIAS_VIDEO_CAPTION \
    --image-caption-weight $IMAGE_CAPTION_WEIGHT \
    --image-image-weight $IMAGE_IMAGE_WEIGHT \
    --video-caption-weight $VIDEO_CAPTION_WEIGHT \
    --image-video-caption-weight $IMAGE_VIDEO_CAPTION_WEIGHT \
    "${ZEROSHOT_ARGS[@]}" \
    --batch-size $IMAGE_BATCH_SIZE \
    --accum-freq $ACCUM_FREQ \
    --accum-freq-image $IMG_ACCUM_FREQ \
    --accum-freq-video  $VID_ACCUM_FREQ \
    --accum-freq-image-vidcap $IMAGE_VIDCAP_ACCUM_FREQ \
    --epochs $EPOCHS \
    --lr $LR \
    --wd $WD \
    --warmup $WARMUP_STEPS \
    --image-workers $IMAGE_WORKERS \
    --video-workers $VIDEO_WORKERS \
    --image-vidcap-workers $IMAGE_VIDCAP_WORKERS \
    --precision amp_bfloat16 \
    --grad-checkpointing \
    --logs "$LOGS_DIR" \
    --report-to $REPORT_TO \
    --save-frequency 1 \
    --save-steps $SAVE_STEPS \
    "${RESUME_ARGS[@]}" \
    --beta1 0.9 \
    --beta2 0.95 \
    --eps 1e-8 \
    --grad-clip-norm 1.0 \
    --lr-scheduler const \
    --random-uniform-frame-sample \
     --frame-concat \
    --log-every-n-steps 10 \
    --seed 63 
    # --aug-cfg scale='(0.8, 1.0)' ratio='(0.75, 1.333)'

