#!/usr/bin/env bash
#  Canonical single-seed attention-comm training execution script.

set -euo pipefail

# Configuring verified Day-11 sweep champion variables
BEST_MSG_DIM=8
BEST_NUM_HEADS=2
SEED=0
STEPS=200000

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
TRAINER="$REPO_ROOT/scripts/trainers/03_attentioncomm_mappo_simple_spread.py"

if [ ! -f "$TRAINER" ]; then
    echo "ERROR: Trainer script not found at $TRAINER"
    exit 1


fi

OUT_BASE="$REPO_ROOT/experiments/attncomm_official"
RUN_DIR="$OUT_BASE/seed${SEED}"
mkdir -p "$RUN_DIR"

if [ -f "$RUN_DIR/final.pt" ]; then
    echo "REFUSING TO OVERWRITE: $RUN_DIR/final.pt already exists."
    exit 1
fi

echo "==> Launching Canonical Attention-Comm Policy Training Run"
echo "    Repo Axis:  $REPO_ROOT"
echo "    Message Space: $BEST_MSG_DIM | Attention Heads: $BEST_NUM_HEADS"
echo "    Target Steps:  $STEPS"

python "$TRAINER" \
    --seed "$SEED" \
    --total-steps "$STEPS" \
    --msg-dim "$BEST_MSG_DIM" \
    --num-heads "$BEST_NUM_HEADS" \
    --hidden 128

SRC="$REPO_ROOT/experiments/attncomm_mappo/seed${SEED}/final.pt"
if [ -f "$SRC" ]; then
    mv "$SRC" "$RUN_DIR/final.pt"
    echo "==> Done. Checkpoint saved securely to: $RUN_DIR/final.pt"
else
    echo "WARN: Expected model checkpoint file not found at $SRC"
    exit 1
fi
