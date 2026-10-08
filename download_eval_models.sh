#!/usr/bin/env bash
# ===========================================================================
# Download the public base models needed to EVALUATE CoVisco.
#
# These are the third-party weights the eval code loads; your own trained
# artifacts (CoVisco-L-14 encoder, the SFT/ViT .pt checkpoints) are NOT
# downloaded here.
#
# Models by sub-project:
#   CoVisco_ViT  (video branch) : Qwen/Qwen3-VL-Embedding-8B   -> text encoder
#   CoVisco_ViT  (SigLIP2 branch): timm/ViT-gopt-16-SigLIP2-384 -> text tower
#                 (open_clip can also auto-fetch this via pretrained=webli)
#   CoVisco_sft  (eval LLMs)    : Qwen/Qwen3-4B-Instruct-2507  -> 4B eval config
#                                 Qwen/Qwen3-1.7B              -> 1.7B config / token-selector eval
#
# Usage:
#   bash download_eval_models.sh            # download everything for eval (default)
#   bash download_eval_models.sh vit        # only CoVisco_ViT eval models
#   bash download_eval_models.sh sft        # only CoVisco_sft eval models
#
# Environment variables:
#   MODELS_DIR   target dir (default: ./models). Each model goes to
#                $MODELS_DIR/<model-name>.
#   HF_TOKEN     Hugging Face token (export it if any repo is gated / rate-limited).
#   HF_ENDPOINT  set to https://hf-mirror.com to use the mirror if huggingface.co
#                is blocked; leave unset to go direct (optionally via a proxy).
#   http_proxy / https_proxy   set if you reach huggingface.co through a proxy.
# ===========================================================================
set -euo pipefail

WHICH="${1:-all}"                      # all | vit | sft
MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"

# Resolve a python interpreter with huggingface_hub installed.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null; then
    echo "[ERROR] huggingface_hub not found for '$PYTHON_BIN'." >&2
    echo "        Install it first:  pip install -U 'huggingface_hub[hf_transfer]'" >&2
    exit 1
fi

mkdir -p "$MODELS_DIR"

# download <repo_id> [local_name]
download() {
    local repo="$1"
    local name="${2:-$(basename "$repo")}"
    local dest="$MODELS_DIR/$name"
    echo "---------------------------------------------------------------"
    echo "[download] $repo  ->  $dest"
    "$PYTHON_BIN" - "$repo" "$dest" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo,
    local_dir=dest,
    token=os.environ.get("HF_TOKEN") or None,
)
print(f"[ok] {repo}")
PY
}

download_vit() {
    download "Qwen/Qwen3-VL-Embedding-8B"
    download "timm/ViT-gopt-16-SigLIP2-384"
}

download_sft() {
    download "Qwen/Qwen3-4B-Instruct-2507"
    download "Qwen/Qwen3-1.7B"
}

case "$WHICH" in
    vit) download_vit ;;
    sft) download_sft ;;
    all) download_vit; download_sft ;;
    *)   echo "usage: bash download_eval_models.sh [all|vit|sft]" >&2; exit 1 ;;
esac

cat <<EOF

===============================================================================
Done. Models are under: $MODELS_DIR

Point the eval entry points at them:

  CoVisco_ViT (video branch):
    TEXT_MODEL=$MODELS_DIR/Qwen3-VL-Embedding-8B bash eval/run_mme_v2_all_qwen3-vl.sh
    # or: python eval/eval_mmeb_video_retrieval.py --text_model_path $MODELS_DIR/Qwen3-VL-Embedding-8B ...

  CoVisco_ViT (SigLIP2 branch):
    python eval/eval_zeroshot_siglip2.py --siglip2_dir $MODELS_DIR/ViT-gopt-16-SigLIP2-384 ...
    # (optional; open_clip auto-downloads timm/ViT-gopt-16-SigLIP2-384 via pretrained=webli)

  CoVisco_sft:
    set  llm.path: $MODELS_DIR/Qwen3-4B-Instruct-2507  in covisco_qwen3_4b_instruct2507*.yaml
    set  llm.path: $MODELS_DIR/Qwen3-1.7B              in covisco_qwen3_1.7b.yaml

Your own CoVisco encoder (CoVisco-L-14.pt) and the trained .pt checkpoints are
separate and not downloaded by this script.
===============================================================================
EOF
