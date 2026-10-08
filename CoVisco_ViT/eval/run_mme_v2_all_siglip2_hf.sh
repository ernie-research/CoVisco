#!/usr/bin/env bash
# MME-V2 evaluation using SigLIP2 text encoder + to_image_caption, loading the
# released CoVisco vision encoder (config.json + model.safetensors) instead of a
# .pt checkpoint. See eval_mme_v2_all_siglip2_hf.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

# Hugging Face Hub repo id (default) or a local release dir (set HF_TOKEN for a private repo).
HF_MODEL="${HF_MODEL:-ernie-research/CoVisco-L-14}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the MMEB-V2 data root}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/mme_v2_siglip2_hf_results}"
SIGLIP2_DIR="${SIGLIP2_DIR:-}"
SIGLIP2_PRETRAINED="${SIGLIP2_PRETRAINED:-webli}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
VIDEO_BATCH="${VIDEO_BATCH:-64}"
IMAGE_BATCH="${IMAGE_BATCH:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NATIVE_RES="${NATIVE_RES:-1}"                 # 1: preserve image/page aspect ratio
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-1400}"      # long-side cap when NATIVE_RES=1

DIST_NPROC_PER_NODE="${DIST_NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
DIST_NNODES="${DIST_NNODES:-1}"
DIST_NODE_RANK="${DIST_NODE_RANK:-0}"
DIST_MASTER_ADDR="${DIST_MASTER_ADDR:-127.0.0.1}"
DIST_MASTER_PORT="${DIST_MASTER_PORT:-29502}"
mkdir -p "${OUT_DIR}"

COMMON=(
  --hf_model "${HF_MODEL}"
  --data_root "${DATA_ROOT}"
  --image_size "${IMAGE_SIZE}"
  --video_batch_size "${VIDEO_BATCH}"
  --image_batch_size "${IMAGE_BATCH}"
  --num_workers "${NUM_WORKERS}"
  --siglip2_pretrained "${SIGLIP2_PRETRAINED}"
)
if [[ "${NATIVE_RES}" == "1" ]]; then
  COMMON+=(--native_resolution --max_image_side "${MAX_IMAGE_SIDE}")
fi
if [[ -n "${SIGLIP2_DIR}" ]]; then
  COMMON+=(--siglip2_dir "${SIGLIP2_DIR}")
fi

run() {
  echo -e "\n>>> $*"
  if [[ "${DISTRIBUTED:-0}" == "1" ]]; then
    python -m torch.distributed.run \
      --nproc_per_node="${DIST_NPROC_PER_NODE}" \
      --nnodes="${DIST_NNODES}" \
      --node_rank="${DIST_NODE_RANK}" \
      --master_addr="${DIST_MASTER_ADDR}" \
      --master_port="${DIST_MASTER_PORT}" \
      "${SCRIPT_DIR}/eval_mme_v2_all_siglip2_hf.py" "$@"
  else
    python "${SCRIPT_DIR}/eval_mme_v2_all_siglip2_hf.py" "$@"
  fi
}

MODE="${1:-all}"
case "${MODE}" in
  smoke)
    run "${COMMON[@]}" --tasks video_cls --datasets UCF101 \
      --output "${OUT_DIR}/smoke_video_cls.json"
    run "${COMMON[@]}" --tasks video_mret --datasets Charades-STA \
      --output "${OUT_DIR}/smoke_video_mret.json"
    run "${COMMON[@]}" --tasks image_cls --datasets VOC2007 \
      --output "${OUT_DIR}/smoke_image_cls.json"
    run "${COMMON[@]}" --tasks visdoc --datasets ViDoRe_arxivqa \
      --output "${OUT_DIR}/smoke_visdoc.json"
    ;;
  video)
    run "${COMMON[@]}" --tasks video_ret video_cls video_mret \
      --output "${OUT_DIR}/video_all.json"
    ;;
  image)
    run "${COMMON[@]}" --tasks image_cls --output "${OUT_DIR}/image_cls_all.json"
    ;;
  visdoc)
    run "${COMMON[@]}" --tasks visdoc --output "${OUT_DIR}/visdoc_all.json"
    ;;
  all)
    run "${COMMON[@]}" --tasks video_ret video_cls video_mret image_cls visdoc \
      --output "${OUT_DIR}/mme_v2_all.json"
    ;;
  *)
    echo "usage: bash run_mme_v2_all_siglip2_hf.sh [smoke|video|image|visdoc|all]" >&2
    exit 1
    ;;
esac

echo -e "\nDone. Results in ${OUT_DIR}/"

# Examples:
#   bash run_mme_v2_all_siglip2_hf.sh smoke
#   HF_MODEL=<your-hf-user>/CoVisco-L-14 bash run_mme_v2_all_siglip2_hf.sh image
#   NATIVE_RES=1 MAX_IMAGE_SIDE=1400 bash run_mme_v2_all_siglip2_hf.sh visdoc
