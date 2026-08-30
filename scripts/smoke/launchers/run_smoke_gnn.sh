#!/usr/bin/env bash
# Day 24 — 2k-step GNN-comm smoke. Verifies trainer wiring before launching
# the full 200k. Should finish in 60–120 seconds.
#
# Usage (from repo root):
#     bash scripts/run_smoke_gnn.sh

set -euo pipefail

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
    TRAINER="$SCRIPT_DIR/08_gnncomm_mappo_simple_spread.py"
    echo "Note: using trainer at $TRAINER (not yet copied into repo)."
fi

echo "==> GNN-comm smoke (seed=999, 2000 steps, no wandb)"
python "$TRAINER" \
    --seed 999 \
    --total-steps 2000 \
    --rollout-len 256 \
    --minibatch-size 64 \
    --ppo-epochs 2 \
    --msg-dim 8 \
    --num-layers 2 \
    --hidden 64 \
    --no-wandb

echo
echo "==> smoke complete. Inspect last 'per_layer_L2' — should be 3 finite values."