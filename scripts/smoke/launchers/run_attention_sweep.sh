#!/usr/bin/env bash
# Day 11 sweep — msg_dim × num_heads grid for attention-comm.
#
# 3 × 3 = 9 jobs, single seed each, 200k steps each.
# Cells where msg_dim is not divisible by num_heads are skipped automatically.
#
# Usage (from repo root):
#     nohup bash scripts/run_attention_sweep.sh > artifacts/checkpoints/attncomm_sweep/sweep.log 2>&1 &
#
# Output layout:
#     artifacts/checkpoints/attncomm_sweep/msg{D}_h{H}_seed0/final.pt
#
# Expected runtime:
#   - CPU:  4–6 hours total (9 sequential runs)
#   - GPU:  1.5–2.5 hours total

set -euo pipefail

# Resolve repo root (this script can live in repo/scripts/ or in Day11_28May_2026/code/)
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
    echo "ERROR: trainer not found at $TRAINER — copy it in first (see Day 10)."
    exit 1
fi

OUT_BASE="$REPO_ROOT/artifacts/checkpoints/attncomm_sweep"
mkdir -p "$OUT_BASE"

# The grid
MSG_DIMS=(4 8 16)
NUM_HEADS=(1 2 4)
SEED=0
STEPS=200000

echo "==> sweep launching at $(date)"
echo "    repo root  = $REPO_ROOT"
echo "    trainer    = $TRAINER"
echo "    output dir = $OUT_BASE"
echo

# Counter for completed runs (cosmetic)
DONE_COUNT=0
TOTAL=0
for D in "${MSG_DIMS[@]}"; do
    for H in "${NUM_HEADS[@]}"; do
        if [ $((D % H)) -eq 0 ]; then
            TOTAL=$((TOTAL + 1))
        fi
    done
done

for D in "${MSG_DIMS[@]}"; do
    for H in "${NUM_HEADS[@]}"; do
        # Skip cells where divisibility fails — the trainer would refuse anyway
        if [ $((D % H)) -ne 0 ]; then
            echo "--> SKIP msg=$D heads=$H (not divisible)"
            continue
        fi

        RUN_NAME="msg${D}_h${H}_seed${SEED}"
        RUN_DIR="$OUT_BASE/$RUN_NAME"
        mkdir -p "$RUN_DIR"

        # Skip if already finished (idempotent restart)
        if [ -f "$RUN_DIR/final.pt" ]; then
            DONE_COUNT=$((DONE_COUNT + 1))
            echo "--> SKIP $RUN_NAME (already has final.pt) [$DONE_COUNT/$TOTAL]"
            continue
        fi

        echo
        echo "==> [$((DONE_COUNT + 1))/$TOTAL] msg_dim=$D num_heads=$H @ $(date +%H:%M:%S)"

        # NOTE: we set EXPERIMENTS_DIR via env so the trainer writes here.
        # The trainer's save path is hard-coded to artifacts/checkpoints/attncomm_mappo/seed{N},
        # so after each run we MOVE the checkpoint into the sweep tree.
        python "$TRAINER" \
            --seed "$SEED" \
            --total-steps "$STEPS" \
            --msg-dim "$D" \
            --num-heads "$H" \
            --hidden 128 \
            || { echo "RUN FAILED: msg=$D heads=$H — see log above"; continue; }

        # Move the checkpoint into the sweep tree
        SRC="$REPO_ROOT/artifacts/checkpoints/attncomm_mappo/seed${SEED}/final.pt"
        if [ -f "$SRC" ]; then
            mv "$SRC" "$RUN_DIR/final.pt"
            echo "    saved -> $RUN_DIR/final.pt"
            DONE_COUNT=$((DONE_COUNT + 1))
        else
            echo "    WARN: $SRC not found after run"
        fi
    done
done

echo
echo "==> sweep complete at $(date). $DONE_COUNT/$TOTAL runs finished."
echo "    next step: python scripts/analyze_attention_weights.py --checkpoint <each>/final.pt"
