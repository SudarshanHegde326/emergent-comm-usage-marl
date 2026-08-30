#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
TRAINER="$REPO_ROOT/scripts/trainers/09_nocomm_mappo_smacv2.py"

echo "==> SMACv2 no-comm smoke (seed=999, 2000 steps, no wandb)"
python "$TRAINER" \
    --seed 999 \
    --total-steps 2000 \
    --rollout-len 256 \
    --minibatch-size 64 \
    --ppo-epochs 2 \
    --hidden 64 \
    --no-wandb

echo
echo "==> smoke complete."
echo "    Look at the last log line — action_valid_frac MUST be 1.0000"
