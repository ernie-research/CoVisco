#!/bin/bash
# ============================================================================
# Precompute codec visidx for a video benchmark (one-time setup)
# ============================================================================
# Follows OneVision-Encoder/llava_next's
#   scripts/precompute_codec_patch/preprocess_video_benchmark.sh
# but produces visidx (instead of mosaic), matching this repo's ViT visidx interface.
#
# Usage:
#   bash scripts/precompute_codec_visidx/preprocess_video_benchmark.sh videomme
#   bash scripts/precompute_codec_visidx/preprocess_video_benchmark.sh perceptiontest_val_mc
#
# Flow:
#   1) Use lmms-eval to trigger the benchmark's video download, and resolve each video's
#      local path exactly the way the evaluator does (task.doc_to_visual), writing to jsonl;
#   2) Call precompute_video_visidx.py to compute codec visidx per video, saved to
#      ${OFFLINE_ROOT}/<video_stem>/visidx.npy.
# ============================================================================
set -euo pipefail

TASK="${1:?usage: $0 <benchmark_task> (e.g. videomme, perceptiontest_val_mc)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script lives in scripts/precompute_codec_visidx/; go up two levels to reach the repo root
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Proxy / HF (consistent with eval_covisco.sh) ----
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
export HF_TOKEN="${HF_TOKEN:-}"
unset HF_ENDPOINT

LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-${REPO_DIR}/third_party/lmms-eval}"
if [ ! -d "$LMMS_EVAL_DIR" ]; then
    echo "[ERROR] lmms-eval not found: $LMMS_EVAL_DIR"; exit 1
fi
export PYTHONPATH="${REPO_DIR}:${LMMS_EVAL_DIR}"
export HF_HOME="${HF_HOME:?set HF_HOME to a local directory for the datasets/hub cache}"
mkdir -p "$HF_HOME/datasets" "$HF_HOME/hub"

# ---- codec / grid parameters (must match the evaluator eval_covisco_codec.sh) ----
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-64}"
SEGMENT_T_SIZE="${SEGMENT_T_SIZE:-16}"
IMAGE_SIZE_VIDEO="${IMAGE_SIZE_VIDEO:-224}"
PATCH_SIZE="${PATCH_SIZE:-14}"
KEEP_RATIO="${KEEP_RATIO:-0.57}"
# Whether each segment forcibly keeps all patches of the first frame (then top-K fills the rest, so each segment's total stays equal). Enabled by default.
KEEP_FIRST_PER_SEGMENT="${KEEP_FIRST_PER_SEGMENT:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# codec tool directory (where the cv_reader scoring functions live)
export OV_ENCODER_TOOL_DIR="${OV_ENCODER_TOOL_DIR:?set OV_ENCODER_TOOL_DIR to the OneVision-Encoder Compressed_Video_Reader/tool directory}"

OFFLINE_BASE="${OFFLINE_BASE:-${REPO_DIR}/codec_visidx_cache}"
OFFLINE_ROOT="${OFFLINE_BASE}/${TASK}"
JSONL="${OFFLINE_BASE}/${TASK}_videos.jsonl"
mkdir -p "$OFFLINE_ROOT"

echo "========================================"
echo "Precompute codec visidx"
echo "  TASK=${TASK}"
echo "  grid=${NUM_VIDEO_FRAMES}f x (${IMAGE_SIZE_VIDEO}/${PATCH_SIZE})^2  seg_t=${SEGMENT_T_SIZE}  keep_ratio=${KEEP_RATIO}"
echo "  OFFLINE_ROOT=${OFFLINE_ROOT}"
echo "  cv_reader tool=${OV_ENCODER_TOOL_DIR}"
echo "========================================"

# ---- Step 1/2: resolve local video paths via lmms-eval -> jsonl ----
echo "[1/2] Resolving video paths via lmms-eval (this also triggers download)..."
TASK="$TASK" OUT_JSONL="$JSONL" python - <<'PY'
import os, sys, json
from pathlib import Path

task_name = os.environ["TASK"]
out_jsonl = os.environ["OUT_JSONL"]

from lmms_eval.tasks import get_task_dict

def _load_task():
    try:
        return get_task_dict([task_name])
    except TypeError:
        return get_task_dict([task_name], None)

td = _load_task()

def _iter_tasks(obj):
    """Recursively unpack the return of get_task_dict (may be a nested group -> {subtask: Task} / tuple / list),
    yielding task objects that have doc_to_visual. Groups like mvbench expand into multiple subtasks."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tasks(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tasks(v)
    elif hasattr(obj, "doc_to_visual"):
        yield obj

tasks = list(_iter_tasks(td))
if not tasks:
    print(f"[ERROR] no runnable task resolved for {task_name} (td keys={list(td.keys()) if isinstance(td, dict) else type(td)})", file=sys.stderr)
    sys.exit(3)
print(f"[info] resolved {len(tasks)} (sub)task(s) for {task_name}", file=sys.stderr)

seen = set()
n = 0
with open(out_jsonl, "w") as f:
    for task in tasks:
        docs = None
        for meth in ("test_docs", "validation_docs"):
            fn = getattr(task, meth, None)
            if fn is None:
                continue
            try:
                d = fn()
            except Exception:
                d = None
            if d is not None and len(d) > 0:
                docs = d
                break
        if docs is None:
            continue
        for i, doc in enumerate(docs):
            try:
                vis = task.doc_to_visual(doc)
            except Exception as e:
                print(f"[warn] doc_to_visual failed: {e}", file=sys.stderr)
                continue
            if vis is None:
                continue
            if not isinstance(vis, list):
                vis = [vis]
            for v in vis:
                if isinstance(v, str) and os.path.exists(v):
                    key = Path(v).stem
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(json.dumps({"video": v, "key": key}) + "\n")
                    n += 1
print(f"[ok] wrote {n} unique videos -> {out_jsonl}")
PY

NUM=$(wc -l < "$JSONL" | tr -d ' ')
if [ "$NUM" = "0" ]; then
    echo "[ERROR] No video paths resolved, jsonl is empty: $JSONL"; exit 4
fi
echo "[1/2] videos=${NUM}  jsonl=${JSONL}"

# ---- Step 2/2: compute visidx ----
echo "[2/2] Computing codec visidx..."
KEEP_FIRST_FLAG="--keep_first_per_segment"
if [ "$KEEP_FIRST_PER_SEGMENT" = "0" ] || [ "$KEEP_FIRST_PER_SEGMENT" = "false" ]; then
    KEEP_FIRST_FLAG="--no-keep_first_per_segment"
fi
python "${SCRIPT_DIR}/precompute_video_visidx.py" \
    --jsonl "$JSONL" \
    --out_root "$OFFLINE_ROOT" \
    --num_video_frames "$NUM_VIDEO_FRAMES" \
    --segment_t_size "$SEGMENT_T_SIZE" \
    --image_size "$IMAGE_SIZE_VIDEO" \
    --patch_size "$PATCH_SIZE" \
    --keep_ratio "$KEEP_RATIO" \
    ${KEEP_FIRST_FLAG} \
    --num_workers "$NUM_WORKERS"

echo "========================================"
echo "Done. visidx assets at: ${OFFLINE_ROOT}"
echo "For evaluation set:  CODEC_VISIDX_ROOT=${OFFLINE_ROOT}"
echo "========================================"
