#!/usr/bin/env bash
# Day 39 — smoke GNN-comm SMACv2 trainer.
#
# 2k-step smoke (no wandb), ~5 min.
#
# Usage (from repo root):
#     bash scripts/smoke_smacv2_gnn.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../../src/envs/smacv2_env.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -f "$SCRIPT_DIR/../../../repo/src/envs/smacv2_env.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../../repo" && pwd )"
else
    echo "ERROR: cannot locate repo root"
    exit 1
fi

TRAINER="$REPO_ROOT/scripts/trainers/12_gnncomm_mappo_smacv2.py"
if [ ! -f "$TRAINER" ]; then
    echo "ERROR: trainer not found at $TRAINER"
    exit 1
fi

echo "=========================================================="
echo "==> SMOKE: $(basename "$TRAINER") (seed=999, 2000 steps, L=2)"
echo "=========================================================="
/home/users/hegdehes/msc_marl/.venv/bin/python "$TRAINER" \
    --seed 999 \
    --total-steps 2000 \
    --rollout-len 256 \
    --minibatch-size 64 \
    --ppo-epochs 2 \
    --hidden 64 \
    --num-layers 2 \
    --no-wandb

echo
echo "=========================================================="
echo "==> SMOKE COMPLETE — sanity checklist"
echo "=========================================================="
echo "    1. action_valid_frac MUST be 1.0000"
echo "    2. per_layer_L2 MUST be [a, b, c] with 3 finite numbers"
echo "    3. mean_ret MUST be finite (any value, even very low)"
echo
echo "    If any of the above fails, DO NOT launch the 3-seed run;"
echo "    fix the trainer first."