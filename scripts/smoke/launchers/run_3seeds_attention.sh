#!/usr/bin/env bash
# Day 14 — Run attention-comm seeds 1 and 2 at the best-cell hyperparameters.

set -euo pipefail

BEST_MSG_DIM=8
BEST_NUM_HEADS=2
STEPS=200000
SEEDS_TO_RUN=(1 2)

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

TRAINER="$REPO_ROOT/scripts/trainers/03_attentioncomm_mappo_simple_spread.py"
if [ ! -f "$TRAINER" ]; then
    echo "ERROR: trainer not found at $TRAINER"
    exit 1
fi

# Confirm seed 0 exists — if not, abort
SEED0_PATH="$REPO_ROOT/experiments/attncomm_official/seed0/final.pt"
if [ ! -f "$SEED0_PATH" ]; then
    echo "ERROR: seed-0 official run not found at $SEED0_PATH"
    echo "       Complete Day 13's run first."
    exit 1
fi

OUT_BASE="$REPO_ROOT/experiments/attncomm_official"
mkdir -p "$OUT_BASE"

if [ $((BEST_MSG_DIM % BEST_NUM_HEADS)) -ne 0 ]; then
    echo "ERROR: BEST_MSG_DIM=$BEST_MSG_DIM not divisible by BEST_NUM_HEADS=$BEST_NUM_HEADS"
    exit 1
fi

echo "==> 3-seed attention-comm runner"
echo "    msg_dim   = $BEST_MSG_DIM"
echo "    num_heads = $BEST_NUM_HEADS"
echo "    steps     = $STEPS"
echo "    seeds     = ${SEEDS_TO_RUN[*]} (seed 0 already done)"
echo "    started   = $(date)"
echo

for SEED in "${SEEDS_TO_RUN[@]}"; do
    RUN_DIR="$OUT_BASE/seed${SEED}"
    mkdir -p "$RUN_DIR"

    if [ -f "$RUN_DIR/final.pt" ]; then
        echo "==> SKIP seed $SEED (already has final.pt)"
        continue
    fi

    echo "==> launching seed $SEED @ $(date +%H:%M:%S)"
    python "$TRAINER" \
        --seed "$SEED" \
        --total-steps "$STEPS" \
        --msg-dim "$BEST_MSG_DIM" \
        --num-heads "$BEST_NUM_HEADS" \
        --hidden 128 \
        || { echo "RUN FAILED for seed $SEED — see error above; continuing"; continue; }

    # Move checkpoint out of standard directory into experiments layout
    SRC="$REPO_ROOT/experiments/attncomm_mappo/seed${SEED}/final.pt"
    if [ -f "$SRC" ]; then
        mv "$SRC" "$RUN_DIR/final.pt"
        echo "    seed $SEED saved -> $RUN_DIR/final.pt"
    else
        echo "    WARN: expected checkpoint not found at $SRC"
    fi
done

echo
echo "==> 3-seed run complete at $(date)"
ls -lh "$OUT_BASE"/seed{0,1,2}/final.pt 2>/dev/null || true
