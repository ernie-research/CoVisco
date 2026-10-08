#!/usr/bin/env bash
# ===========================================================================
# Setup the CoVisco_sft evaluation environment (lmms-eval based).
#
# What it does:
#   1. (optional) create a venv
#   2. install base deps from requirements.txt  (pins transformers>=4.51,<4.52)
#   3. install lmms-eval runtime deps from requirements-eval.txt, under a
#      constraint that protects the transformers pin
#   4. install the vendored third_party/lmms-eval editable, --no-deps
#      (so torch / transformers are never touched)
#   5. download the NLTK data a few tasks need
#   6. verify the install
#
# Usage:
#   bash scripts/setup_eval_env.sh                 # create .env-eval and install
#   CREATE_VENV=0 bash scripts/setup_eval_env.sh   # install into current env
#   VENV_DIR=/path/to/venv bash scripts/setup_eval_env.sh
#   PYTHON_BIN=python3.11 bash scripts/setup_eval_env.sh
#
# NOTE on torch: requirements.txt installs `torch>=2.0` from the default index.
# If you need a specific CUDA build, install it FIRST, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# then run this script (already-satisfied torch will not be reinstalled).
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-${REPO_DIR}/third_party/lmms-eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.env-eval}"
TRANSFORMERS_PIN="transformers>=4.51.0,<4.52.0"

if [ ! -d "$LMMS_EVAL_DIR" ]; then
    echo "[ERROR] vendored lmms-eval not found: $LMMS_EVAL_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. virtual environment
# ---------------------------------------------------------------------------
if [ "${CREATE_VENV:-1}" = "1" ]; then
    echo "[1/6] Creating venv at ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    python -m pip install -U pip
else
    echo "[1/6] Using current Python environment ($(command -v python))"
fi

# ---------------------------------------------------------------------------
# 2. base deps (pins transformers for Qwen3)
# ---------------------------------------------------------------------------
echo "[2/6] Installing base deps (requirements.txt)"
pip install -r "${REPO_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# 3. eval deps, protecting the transformers pin via a constraints file
# ---------------------------------------------------------------------------
echo "[3/6] Installing eval deps (requirements-eval.txt)"
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
echo "$TRANSFORMERS_PIN" > "$CONSTRAINTS"
pip install -c "$CONSTRAINTS" -r "${REPO_DIR}/requirements-eval.txt"

# ---------------------------------------------------------------------------
# 4. vendored lmms-eval (editable, no deps -> keeps torch/transformers intact)
# ---------------------------------------------------------------------------
echo "[4/6] Installing vendored lmms-eval (--no-deps)"
pip install -e "$LMMS_EVAL_DIR" --no-deps --no-build-isolation

# ---------------------------------------------------------------------------
# 5. NLTK data (nextqa stopwords, tokenizers used by some tasks)
# ---------------------------------------------------------------------------
echo "[5/6] Downloading NLTK data"
python - <<'PY'
import nltk
for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as e:
        print(f"[warn] nltk download '{pkg}' failed: {e}")
PY

# ---------------------------------------------------------------------------
# 6. verify
# ---------------------------------------------------------------------------
echo "[6/6] Verifying install"
python - <<'PY'
import transformers, accelerate, cv2, lmms_eval  # noqa: F401
v = transformers.__version__
print("transformers:", v)
print("accelerate  :", accelerate.__version__)
print("opencv      :", cv2.__version__)
print("lmms_eval   : import OK")
major, minor = (int(x) for x in v.split(".")[:2])
assert (major, minor) == (4, 51), f"transformers must stay 4.51.x for Qwen3, got {v}"
print("transformers pin OK")
PY

cat <<EOF

Done. Environment ready.
EOF
if [ "${CREATE_VENV:-1}" = "1" ]; then
    echo "Activate with: source ${VENV_DIR}/bin/activate"
fi
cat <<'EOF'

Runtime requirements NOT handled by this script:
  - GPU(s) + a CUDA-enabled torch build (eval runs multi-GPU via accelerate).
  - Pre-downloaded benchmark data; set HF_HOME / HF_DATASETS_CACHE as in the
    eval_covisco*.sh scripts (HF_DATASETS_CACHE must be on a local, non-FUSE disk).
  - A system ffmpeg so OpenCV can decode video (.mp4) benchmarks.
  - HF_TOKEN for gated datasets.
  - CODEC eval only (eval_covisco_codec.sh): the `cv_reader` extension compiled
    from OneVision-Encoder/Compressed_Video_Reader (set OV_ENCODER_TOOL_DIR),
    plus a one-time visidx precompute. Not installable via pip.
EOF
