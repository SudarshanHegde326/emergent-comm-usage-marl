#!/usr/bin/env bash
# Day 16 — 3-seed run of attention-comm with bandwidth=8 (low-rank).

set -euo pipefail

BEST_MSG_DIM=8
BEST_NUM_HEADS=2
CHANNEL_TYPE="low-rank"
BANDWIDTH=8
STEPS=200000
SEEDS=(0 1 2)

# Resolve repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../../src/comms/attention_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -f "$SCRIPT_DIR/../../../repo/src/comms/attention_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../../repo" && pwd )"
else
    echo "ERROR: cannot locate repo root from $SCRIPT_DIR"
    exit 1
fi

TRAINER="$REPO_ROOT/scripts/trainers/04_attncomm_bandwidth_mappo.py"
if [ ! -f "$TRAINER" ]; then
    echo "ERROR: trainer not found at $TRAINER"
    exit 1
fi

if [ $((BEST_MSG_DIM % BEST_NUM_HEADS)) -ne 0 ]; then
    echo "ERROR: BEST_MSG_DIM=$BEST_MSG_DIM not divisible by BEST_NUM_HEADS=$BEST_NUM_HEADS"
    exit 1
fi

OUT_BASE="$REPO_ROOT/artifacts/checkpoints/attncomm_bw${BANDWIDTH}"
mkdir -p "$OUT_BASE"

echo "==> 3-Seed Bandwidth Runner Initialization"
echo "    channel_type = $CHANNEL_TYPE"
echo "    bandwidth    = $BANDWIDTH"
echo "    msg_dim      = $BEST_MSG_DIM"
echo "    num_heads    = $BEST_NUM_HEADS"
echo "    steps        = $STEPS"
echo "    seeds        = ${SEEDS[*]}"
echo "    started      = $(date)"
echo

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="$OUT_BASE/seed${SEED}"
    mkdir -p "$RUN_DIR"
    if [ -f "$RUN_DIR/final.pt" ]; then
        echo "==> SKIP seed $SEED (already has final.pt)"
        continue
    fi

    echo "==> Launching optimization loops for Seed $SEED @ $(date +%H:%M:%S)"
    python "$TRAINER" \
        --seed "$SEED" \
        --total-steps "$STEPS" \
        --channel-type "$CHANNEL_TYPE" \
        --bandwidth "$BANDWIDTH" \
        --msg-dim "$BEST_MSG_DIM" \
        --num-heads "$BEST_NUM_HEADS" \
        --hidden 128 \
        --no-wandb \
        || { echo "RUN FAILED for seed $SEED — continuing downstream loop entries"; continue; }

    if [ -f "$RUN_DIR/final.pt" ]; then
        echo "    Seed $SEED checkpoint verified -> $RUN_DIR/final.pt"
    else
        echo "    WARN: $RUN_DIR/final.pt missing after run execution"
    fi
done

echo
echo "==> 3-seed bandwidth runs completed at $(date)"
ls -lh "$OUT_BASE"/seed{0,1,2}/final.pt 2>/dev/null || true
