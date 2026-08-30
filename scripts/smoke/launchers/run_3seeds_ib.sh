#!/usr/bin/env bash
# Day 17 — Multi-seed stochastic bottleneck loop sequence orchestration.

set -euo pipefail

BEST_MSG_DIM=8
BEST_NUM_HEADS=2
BETA=0.01
STEPS=200000
SEEDS=(0 1 2)

BETA_TAG="${BETA//./p}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../../src/comms/attention_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -f "$SCRIPT_DIR/../../../repo/src/comms/attention_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../../repo" && pwd )"
else
    echo "ERROR: cannot locate repo root from $SCRIPT_DIR"
    exit 1
fi

TRAINER="$REPO_ROOT/scripts/trainers/05_attncomm_ib_mappo.py"
if [ ! -f "$TRAINER" ]; then
    echo "ERROR: trainer script not found at $TRAINER"
    exit 1
fi

OUT_BASE="$REPO_ROOT/experiments/attncomm_ib_b${BETA_TAG}"
mkdir -p "$OUT_BASE"

echo "==> Deploying Multi-Seed Information Bottleneck Pipeline"
echo "    Beta parameter:  $BETA"
echo "    Message Space:   $BEST_MSG_DIM"
echo "    Atari Step Bounds: $STEPS"
echo "    Target Seed Array: ${SEEDS[*]}"
echo "    Launch Window:   $(date)"
echo

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="$OUT_BASE/seed${SEED}"
    mkdir -p "$RUN_DIR"
    if [ -f "$RUN_DIR/final.pt" ]; then
        echo "==> SKIP seed $SEED (checkpoint discovered on local system)"
        continue
    fi

    echo "==> Launching optimization loops for Seed $SEED @ $(date +%H:%M:%S)"
    python "$TRAINER" \
        --seed "$SEED" \
        --total-steps "$STEPS" \
        --beta "$BETA" \
        --msg-dim "$BEST_MSG_DIM" \
        --num-heads "$BEST_NUM_HEADS" \
        --hidden 128 \
        --no-wandb \
        || { echo "TRAINER CRASHED for seed $SEED — advancing downstream loops"; continue; }

    if [ -f "$RUN_DIR/final.pt" ]; then
        echo "    Seed $SEED isolated -> $RUN_DIR/final.pt"
    else
        echo "    WARN: expected checkpoint missing at $RUN_DIR/final.pt"
    fi
done

echo
echo "==> 3-seed information bottleneck sequence execution terminated cleanly at $(date)"
ls -lh "$OUT_BASE"/seed{0,1,2}/final.pt 2>/dev/null || true
