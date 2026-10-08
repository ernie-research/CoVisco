#!/usr/bin/env bash
# Zero-shot evaluation (SigLIP2 text encoder / to_image_caption branch) for the
# CoVisco ViT (SigLIP) checkpoint. See eval_zeroshot_siglip2.py for details.
#
# Datasets:
#   classification : imagenet1k  imagenetv2  objectnet(113-cls)  imagenet_real
#   retrieval      : coco  flickr  xm3600(36 langs)
set -euo pipefail

# --------------------------------------------------------------------------- #
# Optional HTTP(S) proxy (HuggingFace: clip-benchmark WDS, floschne/xm3600,
# SigLIP2 weights, github raw for real.json). Leave unset for direct access.
# --------------------------------------------------------------------------- #
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="${CKPT:?set CKPT to the CoVisco ViT checkpoint (.pt) path}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Native resolution: preserve aspect ratio (no square crop), long side capped by
# MAX_IMAGE_SIDE and rounded to a multiple of 14. Forces batch size 1.
NATIVE_RES="${NATIVE_RES:-0}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-1400}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/zeroshot_results}"
# ImageNet-1k validation webdataset shards (brace pattern like /path/{0..6}.tar).
IMAGENET1K_WDS="${IMAGENET1K_WDS:?set IMAGENET1K_WDS to the ImageNet-1k val webdataset shards}"
# ImageNet-ReaL source. Preferred: timm/imagenet-1k-wds validation shards
# (keyed by ILSVRC2012_val_XXXXXXXX). Alt: raw val JPEG dir. Empty -> skip ReaL.
IMAGENET_REAL_WDS="${IMAGENET_REAL_WDS:-}"
IMAGENET_REAL_DIR="${IMAGENET_REAL_DIR:-}"
mkdir -p "${OUT_DIR}"

COMMON=(--ckpt "${CKPT}" --image_size "${IMAGE_SIZE}"
        --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}"
        --imagenet1k_wds "${IMAGENET1K_WDS}")
if [[ -n "${IMAGENET_REAL_WDS}" && -d "${IMAGENET_REAL_WDS}" ]]; then
  COMMON+=(--imagenet_real_wds "${IMAGENET_REAL_WDS}")
fi
if [[ -n "${IMAGENET_REAL_DIR}" ]]; then
  COMMON+=(--imagenet_real_dir "${IMAGENET_REAL_DIR}")
fi
if [[ "${NATIVE_RES}" == "1" ]]; then
  COMMON+=(--native_resolution --max_image_side "${MAX_IMAGE_SIDE}")
fi

run() { echo -e "\n>>> $*"; python "${SCRIPT_DIR}/eval_zeroshot_siglip2.py" "$@"; }

MODE="${1:-all}"
case "${MODE}" in
  smoke)   # quickest sanity check: flickr retrieval (1k) only
    run "${COMMON[@]}" --tasks retrieval --ret_datasets flickr \
        --output "${OUT_DIR}/smoke_flickr.json" ;;
  cls)     # all classification datasets
    run "${COMMON[@]}" --tasks classification \
        --output "${OUT_DIR}/classification.json" ;;
  retrieval)
    run "${COMMON[@]}" --tasks retrieval \
        --output "${OUT_DIR}/retrieval.json" ;;
  imagenet)  # ImageNet family only
    run "${COMMON[@]}" --tasks classification \
        --cls_datasets imagenet1k imagenetv2 objectnet imagenet_real \
        --output "${OUT_DIR}/imagenet_family.json" ;;
  all)
    run "${COMMON[@]}" --tasks classification retrieval \
        --output "${OUT_DIR}/zeroshot_all.json" ;;
  *)
    echo "usage: bash run_zeroshot_siglip2.sh [smoke|cls|retrieval|imagenet|all]" >&2
    exit 1 ;;
esac

echo -e "\nDone. Results in ${OUT_DIR}/"

# --------------------------------------------------------------------------- #
# Env knobs: CKPT IMAGE_SIZE BATCH_SIZE NUM_WORKERS OUT_DIR IMAGENET1K_WDS
#            IMAGENET_REAL_WDS IMAGENET_REAL_DIR NATIVE_RES MAX_IMAGE_SIDE
# Examples:
#   bash run_zeroshot_siglip2.sh smoke
#   IMAGE_SIZE=384 bash run_zeroshot_siglip2.sh cls
#   IMAGENET_REAL_WDS=/path/to/timm_val_shards bash run_zeroshot_siglip2.sh imagenet
#   NATIVE_RES=1 bash run_zeroshot_siglip2.sh retrieval   # native res (batch=1)
#   NATIVE_RES=1 MAX_IMAGE_SIDE=1008 bash run_zeroshot_siglip2.sh smoke
# --------------------------------------------------------------------------- #
