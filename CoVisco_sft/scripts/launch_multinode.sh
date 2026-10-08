#!/bin/bash
# Multi-node launcher: parses the hostfile, then SSHes into each node to start the training script in parallel
# Usage:   bash scripts/launch_multinode.sh <train_script.sh> [extra env=val ...]
# Example: bash scripts/launch_multinode.sh scripts/run_sft_4b_instruct2507.sh NUM_EPOCHS=2
set -euo pipefail

HOSTFILE=${HOSTFILE:?set HOSTFILE to the path of your hostfile (one host per line)}
TRAIN_SCRIPT=${1:?"Usage: $0 <train_script.sh> [KEY=VAL ...]"}
shift
EXTRA_ENVS=("$@")   # remaining args are extra environment variables, in KEY=VAL format

# ---------- Parse hostfile ----------
if [ ! -f "$HOSTFILE" ]; then
    echo "[ERROR] hostfile does not exist: $HOSTFILE" >&2
    exit 1
fi

HOSTS=()
while read -r line; do
    # Skip blank lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue
    ip=$(echo "$line" | awk '{print $1}')
    HOSTS+=("$ip")
done < "$HOSTFILE"

NNODES=${#HOSTS[@]}
MASTER_ADDR=${HOSTS[0]}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

echo "=============================="
echo " Total nodes   : $NNODES"
echo " Master        : $MASTER_ADDR:$MASTER_PORT"
echo " GPUs per node : $NPROC_PER_NODE"
echo " Train script  : $TRAIN_SCRIPT"
[ ${#EXTRA_ENVS[@]} -gt 0 ] && echo " Extra args    : ${EXTRA_ENVS[*]}"
echo "=============================="

# ---------- Build the pass-through environment variable string ----------
ENV_STR="NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NPROC_PER_NODE=$NPROC_PER_NODE"
for kv in "${EXTRA_ENVS[@]}"; do
    ENV_STR="$ENV_STR $kv"
done

# Absolute script path (must be identical on every node; naturally satisfied with shared storage)
SCRIPT_ABS=$(realpath "$TRAIN_SCRIPT")
WORK_DIR=$(dirname "$SCRIPT_ABS")/..
LOG_DIR=$(dirname "$SCRIPT_ABS")/../logs
mkdir -p "$LOG_DIR"

# ---------- Start all nodes in parallel ----------
PIDS=()
for i in "${!HOSTS[@]}"; do
    HOST=${HOSTS[$i]}
    LOG_FILE="$LOG_DIR/node${i}_$(basename "$TRAIN_SCRIPT" .sh).log"

    echo "[INFO] Starting node${i} ($HOST) -> $LOG_FILE"

    ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$HOST" \
        "cd $WORK_DIR && NODE_RANK=$i $ENV_STR bash $SCRIPT_ABS" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

# ---------- Wait for all nodes and report results ----------
FAILED=0
for i in "${!PIDS[@]}"; do
    HOST=${HOSTS[$i]}
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "[OK]   node${i} ($HOST) training finished"
    else
        echo "[FAIL] node${i} ($HOST) exited abnormally, see logs/node${i}_*.log" >&2
        FAILED=$((FAILED + 1))
    fi
done

[ "$FAILED" -gt 0 ] && exit 1
echo "All nodes finished training."
