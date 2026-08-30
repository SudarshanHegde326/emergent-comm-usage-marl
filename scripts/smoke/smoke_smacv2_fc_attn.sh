#!/usr/bin/env bash
# Smoke test for the FC-comm (10) and attention-comm (11) SMACv2 trainers.
# Runs each for 2000 steps with --no-wandb and checks it completes + saves.
#
# (Previous version of this file mistakenly ran the aggregator instead of
#  smoking the trainers. This file now does what its name says.)
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

FC_TRAINER="$REPO_ROOT/scripts/trainers/10_fccomm_mappo_smacv2.py"
ATTN_TRAINER="$REPO_ROOT/scripts/trainers/11_attncomm_mappo_smacv2.py"

for t in "$FC_TRAINER" "$ATTN_TRAINER"; do
    if [ ! -f "$t" ]; then
        echo "ERROR: trainer not found at $t"
        exit 1
    fi
done

echo "============================================================"
echo "==> SMACv2 FC-comm smoke (seed=999, 2000 steps, no wandb)"
echo "============================================================"
python "$FC_TRAINER" \
    --seed 999 \
    --total-steps 2000 \
    --rollout-len 256 \
    --minibatch-size 64 \
    --ppo-epochs 2 \
    --msg-dim 8 \
    --hidden 64 \
    --no-wandb

echo
echo "============================================================"
echo "==> SMACv2 attention-comm smoke (seed=999, 2000 steps, no wandb)"
echo "============================================================"
python "$ATTN_TRAINER" \
    --seed 999 \
    --total-steps 2000 \
    --rollout-len 256 \
    --minibatch-size 64 \
    --ppo-epochs 2 \
    --msg-dim 8 \
    --num-heads 2 \
    --hidden 64 \
    --no-wandb

echo
echo "==> FC + attention smoke complete."
echo "    For BOTH: the last log line's action_valid_frac MUST be 1.0000,"
echo "    ep_return must be finite (no nan), and a final.pt must be saved."