#!/usr/bin/env bash
# Day 24 — first GNN-comm training run (single seed, 200k steps, with wandb).
#
# Usage (from repo root):
#     nohup bash scripts/run_first_gnn_training.sh \
#         > artifacts/checkpoints/gnncomm_mappo/training.log 2>&1 &

set -euo pipefail

# ============================================================
# DEFAULTS — match spec v2 (L=2, D=8)
# ============================================================
SEED=0
STEPS=200000
MSG_DIM=8
NUM_LAYERS=2
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../../src/comms/gnn_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -f "$SCRIPT_DIR/../../../repo/src/comms/gnn_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../../repo" && pwd )"
else
    echo "ERROR: cannot locate repo root"
    exit 1
fi

TRAINER="$REPO_ROOT/scripts/trainers/08_gnncomm_mappo_simple_spread.py"
if [ ! -f "$TRAINER" ]; then
    echo "ERROR: trainer not found at $TRAINER"
    exit 1
fi

RUN_DIR="$REPO_ROOT/artifacts/checkpoints/gnncomm_mappo/seed${SEED}"
mkdir -p "$RUN_DIR"

# Refuse to overwrite an existing run
if [ -f "$RUN_DIR/final.pt" ]; then
    echo "REFUSING TO OVERWRITE: $RUN_DIR/final.pt already exists."
    echo "If you really want to re-run, mv it first:"
    echo "    mv $RUN_DIR/final.pt $RUN_DIR/final_old_$(date +%Y%m%d_%H%M).pt"
    exit 1
fi

echo "==> first GNN-comm training run"
echo "    seed       = $SEED"
echo "    steps      = $STEPS"
echo "    msg_dim    = $MSG_DIM"
echo "    num_layers = $NUM_LAYERS"
echo "    output     = $RUN_DIR/final.pt"
echo "    started    = $(date)"
echo

python "$TRAINER" \
    --seed "$SEED" \
    --total-steps "$STEPS" \
    --msg-dim "$MSG_DIM" \
    --num-layers "$NUM_LAYERS" \
    --hidden 128

echo
echo "==> done. finished at $(date)"
ls -lh "$RUN_DIR/final.pt"