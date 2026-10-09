#!/usr/bin/env bash
# Run MMEB-V2 evaluation (video_ret / video_cls / video_mret / image_cls /
# visdoc) loading the released CoVisco vision encoder (config.json +
# model.safetensors) instead of a .pt checkpoint. See eval_mme_v2_all_hf.py.
set -euo pipefail

# --------------------------------------------------------------------------- #
# Optional HTTP(S) proxy (HuggingFace annotations / snapshot_download may need
# network access). Leave unset for direct network access.
# --------------------------------------------------------------------------- #
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

# --------------------------------------------------------------------------- #
# Config (override via env), e.g.:
#   HF_MODEL=<user>/CoVisco-L-14 IMAGE_SIZE=384 bash run_mme_v2_all_qwen3-vl_hf.sh smoke
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Hugging Face Hub repo id (default) or a local release dir (set HF_TOKEN for a private repo).
HF_MODEL="${HF_MODEL:-ernie-research/CoVisco-L-14}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the MMEB-V2 data root}"
TEXT_MODEL="${TEXT_MODEL:?set TEXT_MODEL to the Qwen3-VL-Embedding model path}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/mme_v2_hf_results}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"       # square input resolution
VIDEO_BATCH="${VIDEO_BATCH:-2}"
IMAGE_BATCH="${IMAGE_BATCH:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_FRAMES="${NUM_FRAMES:-64}"        # frames uniformly sampled per clip (get_segments path)
SEGMENT_T_SIZE="${SEGMENT_T_SIZE:-16}" # temporal segment size; NUM_FRAMES must be a multiple of it
RAW_LABELS="${RAW_LABELS:-0}"         # set to 1 to disable label cleaning (A/B)
NATIVE_RES="${NATIVE_RES:-1}"         # 1 (default): native-resolution preprocessing; set 0 to disable
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-1400}"  # long-side cap (px) when NATIVE_RES=1
# Set DISTRIBUTED=1 to let this wrapper launch torchrun. For multi-node runs,
# also set DIST_NNODES, DIST_NODE_RANK, DIST_MASTER_ADDR and DIST_MASTER_PORT.
DIST_NPROC_PER_NODE="${DIST_NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
DIST_NNODES="${DIST_NNODES:-1}"
DIST_NODE_RANK="${DIST_NODE_RANK:-0}"
DIST_MASTER_ADDR="${DIST_MASTER_ADDR:-127.0.0.1}"
DIST_MASTER_PORT="${DIST_MASTER_PORT:-29501}"
mkdir -p "${OUT_DIR}"

COMMON=(--hf_model "${HF_MODEL}" --data_root "${DATA_ROOT}" --text_model_path "${TEXT_MODEL}"
        --image_size "${IMAGE_SIZE}" --video_batch_size "${VIDEO_BATCH}"
        --image_batch_size "${IMAGE_BATCH}" --num_workers "${NUM_WORKERS}"
        --num_frames "${NUM_FRAMES}" --segment_t_size "${SEGMENT_T_SIZE}")
if [[ "${RAW_LABELS}" == "1" ]]; then COMMON+=(--raw_labels); fi
if [[ "${NATIVE_RES}" == "1" ]]; then COMMON+=(--native_resolution --max_image_side "${MAX_IMAGE_SIDE}"); fi

run() {
  echo -e "\n>>> $*"
  if [[ "${DISTRIBUTED:-0}" == "1" ]]; then
    python -m torch.distributed.run \
      --nproc_per_node="${DIST_NPROC_PER_NODE}" \
      --nnodes="${DIST_NNODES}" \
      --node_rank="${DIST_NODE_RANK}" \
      --master_addr="${DIST_MASTER_ADDR}" \
      --master_port="${DIST_MASTER_PORT}" \
      "${SCRIPT_DIR}/eval_mme_v2_all_hf.py" "$@"
  else
    python "${SCRIPT_DIR}/eval_mme_v2_all_hf.py" "$@"
  fi
}

# --------------------------------------------------------------------------- #
# Pick what to run with the first arg: smoke | video | image | visdoc | all (default)
# --------------------------------------------------------------------------- #
MODE="${1:-all}"

case "${MODE}" in
  # Quick end-to-end sanity check on one small dataset per task type.
  smoke)
    run "${COMMON[@]}" --tasks video_cls  --datasets UCF101      --output "${OUT_DIR}/smoke_video_cls.json"
    run "${COMMON[@]}" --tasks video_mret --datasets Charades-STA --output "${OUT_DIR}/smoke_video_mret.json"
    run "${COMMON[@]}" --tasks image_cls  --datasets VOC2007     --output "${OUT_DIR}/smoke_image_cls.json"
    run "${COMMON[@]}" --tasks visdoc     --datasets ViDoRe_arxivqa --output "${OUT_DIR}/smoke_visdoc.json"
    ;;

  # All video meta-tasks.
  video)
    run "${COMMON[@]}" --tasks video_ret  --output "${OUT_DIR}/video_all.json"
    ;;

  # All image classification datasets.
  image)
    run "${COMMON[@]}" --tasks image_cls --output "${OUT_DIR}/image_cls_all.json"
    ;;

  # All visual document retrieval datasets (ViDoRe / VisRAG).
  visdoc)
    run "${COMMON[@]}" --tasks visdoc --output "${OUT_DIR}/visdoc_all.json"
    ;;

  # Everything, in one run.
  all)
    run "${COMMON[@]}" --tasks video_ret video_cls video_mret image_cls visdoc \
        --output "${OUT_DIR}/mme_v2_all.json"
    ;;

  *)
    echo "usage: bash run_mme_v2_all_qwen3-vl_hf.sh [smoke|video|image|visdoc|all]" >&2
    exit 1
    ;;
esac

echo -e "\nDone. Results in ${OUT_DIR}/"

# --------------------------------------------------------------------------- #
# Env knobs: HF_MODEL DATA_ROOT TEXT_MODEL OUT_DIR IMAGE_SIZE VIDEO_BATCH
#            IMAGE_BATCH NUM_WORKERS NUM_FRAMES SEGMENT_T_SIZE RAW_LABELS
#            NATIVE_RES MAX_IMAGE_SIDE
#            DISTRIBUTED DIST_NPROC_PER_NODE DIST_NNODES DIST_NODE_RANK
#            DIST_MASTER_ADDR DIST_MASTER_PORT
# Examples:
#   bash run_mme_v2_all_qwen3-vl_hf.sh smoke
#   HF_MODEL=<your-hf-user>/CoVisco-L-14 bash run_mme_v2_all_qwen3-vl_hf.sh video
#   IMAGE_SIZE=384 bash run_mme_v2_all_qwen3-vl_hf.sh image
#   NUM_FRAMES=32 SEGMENT_T_SIZE=8 bash run_mme_v2_all_qwen3-vl_hf.sh video  # 4 segments of 8
#   NATIVE_RES=1 MAX_IMAGE_SIDE=1400 bash run_mme_v2_all_qwen3-vl_hf.sh visdoc
#   DISTRIBUTED=1 DIST_NPROC_PER_NODE=8 bash run_mme_v2_all_qwen3-vl_hf.sh video
# --------------------------------------------------------------------------- #
