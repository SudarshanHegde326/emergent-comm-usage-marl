#!/usr/bin/env bash
# Day 39 — W6 SMACv2 progress checker.
#
# One-screen status across all 12 SMACv2 runs (4 conditions x 3 seeds).
# Cheap and read-only — runs in ~1 second. Run it any time during W6.
#
# Usage (from repo root):
#     bash scripts/check_w6_progress.sh

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../../src/envs/smacv2_env.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -f "$SCRIPT_DIR/../../../repo/src/envs/smacv2_env.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../../repo" && pwd )"
else
    echo "ERROR: cannot locate repo root"
    exit 1
fi

CONDITIONS=("smacv2_nocomm" "smacv2_fccomm" "smacv2_attncomm" "smacv2_gnncomm")
SEEDS=(0 1 2)
TOTAL_DONE=0
TOTAL=$((${#CONDITIONS[@]} * ${#SEEDS[@]}))

echo "=========================================================="
echo "W6 SMACv2 progress @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
printf "%-22s %-10s %-10s %-10s\n" "condition" "seed 0" "seed 1" "seed 2"
echo "----------------------------------------------------------"

for COND in "${CONDITIONS[@]}"; do
    BASE_DIR="$REPO_ROOT/artifacts/checkpoints/$COND"
    row="$COND"
    for SEED in "${SEEDS[@]}"; do
        FINAL="$BASE_DIR/seed${SEED}/final.pt"
        if [ -f "$FINAL" ]; then
            row="$row | DONE  "
            TOTAL_DONE=$((TOTAL_DONE + 1))
        else
            # check for ANY log/checkpoint indicating it's in progress
            if [ -d "$BASE_DIR/seed${SEED}" ] && \
               find "$BASE_DIR/seed${SEED}" -type f 2>/dev/null | grep -q .; then
                row="$row | RUN   "
            else
                row="$row | -     "
            fi
        fi
    done
    echo "$row"
done

echo "----------------------------------------------------------"
echo "Total finished: $TOTAL_DONE / $TOTAL"
echo

# Tail the active background logs if they exist
echo "=========================================================="
echo "Recent log activity (last 5 lines per active log)"
echo "=========================================================="
# Log locations differ by launcher: the no-comm runner writes inside the
# condition folder; the FC/attn and GNN runners write at artifacts/checkpoints/ root.
for LOGPATH in \
    "$REPO_ROOT/artifacts/checkpoints/smacv2_nocomm/3seeds.log" \
    "$REPO_ROOT/artifacts/checkpoints/smacv2_nocomm_3seeds.log" \
    "$REPO_ROOT/artifacts/checkpoints/smacv2_fc_attn_6seeds.log" \
    "$REPO_ROOT/artifacts/checkpoints/smacv2_gnn_3seeds.log"; do
    if [ -f "$LOGPATH" ]; then
        echo
        echo "--- $(basename "$(dirname "$LOGPATH")")/$(basename "$LOGPATH") ---"
        tail -n 5 "$LOGPATH"
    fi
done

echo
echo "=========================================================="
echo "GPU snapshot (nvidia-smi top line)"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
           --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"

echo
if [ "$TOTAL_DONE" -lt "$TOTAL" ]; then
    REMAINING=$((TOTAL - TOTAL_DONE))
    echo "==> $REMAINING runs still pending. Re-run this script any time."
else
    echo "==> ALL $TOTAL SMACv2 runs complete. Ready to aggregate."
fi