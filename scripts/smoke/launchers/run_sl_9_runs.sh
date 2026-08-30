#!/usr/bin/env bash
# Day 20 — 9-run Speaker-Listener sweep.
set -euo pipefail

COMM_TYPES=(none fc attn)
SEEDS=(0 1 2)
STEPS=200000
MSG_DIM=8
NUM_HEADS=2

# Resolve repo root (current dir)
REPO_ROOT="$(pwd)"
TRAINER="$REPO_ROOT/scripts/trainers/07_sl_mappo.py"

echo "==> Speaker-Listener 9-run sweep started $(date)"

for CT in "${COMM_TYPES[@]}"; do
    OUT_BASE="$REPO_ROOT/experiments/sl_${CT}"
    mkdir -p "$OUT_BASE"

    for SEED in "${SEEDS[@]}"; do
        RUN_DIR="$OUT_BASE/seed${SEED}"
        mkdir -p "$RUN_DIR"

        if [ -f "$RUN_DIR/final.pt" ]; then
            echo "==> SKIP comm=$CT seed=$SEED (already has final.pt)"
            continue
        fi

        echo "==> launching comm=$CT seed=$SEED"
        python "$TRAINER" \
            --comm-type "$CT" \
            --seed "$SEED" \
            --total-steps "$STEPS" \
            --msg-dim "$MSG_DIM" \
            --num-heads "$NUM_HEADS" \
            --hidden 128
    done
done
echo "==> sweep complete at $(date)."