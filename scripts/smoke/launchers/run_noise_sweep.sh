#!/usr/bin/env bash
# Day 18 — Noise sweep coordinator script.

set -euo pipefail

BEST_MSG_DIM=8
BEST_NUM_HEADS=2
SIGMAS=(0.10 0.25 0.50)
SEEDS=(0 1 2)
STEPS=200000

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../../src/comms/attention_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -f "$SCRIPT_DIR/../../../repo/src/comms/attention_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../../repo" && pwd )"
else
    echo "ERROR: cannot locate repo root"
    exit 1
fi

TRAINER="$REPO_ROOT/scripts/trainers/06_attncomm_noise_mappo.py"
if [ ! -f "$TRAINER" ]; then
    echo "ERROR: trainer not found at $TRAINER"
    exit 1
fi

sigma_tag () {
    awk -v s="$1" 'BEGIN { printf "s%03d", int(s*100 + 0.5) }'
}

TOTAL_RUNS=$(( ${#SIGMAS[@]} * ${#SEEDS[@]} ))
DONE_COUNT=0

echo "==> Initiating 9-Seed Gaussian Noise Sweep"
echo "    Sigmas:   ${SIGMAS[*]}"
echo "    Seeds:    ${SEEDS[*]}"
echo "    Total:    $TOTAL_RUNS optimization tracks"
echo "    Horizon:  $STEPS environment updates"
echo "    Started:  $(date)"
echo

for SIGMA in "${SIGMAS[@]}"; do
    TAG="$(sigma_tag "$SIGMA")"
    OUT_BASE="$REPO_ROOT/experiments/attncomm_noise_${TAG}"
    mkdir -p "$OUT_BASE"

    for SEED in "${SEEDS[@]}"; do
        RUN_DIR="$OUT_BASE/seed${SEED}"
        mkdir -p "$RUN_DIR"

        if [ -f "$RUN_DIR/final.pt" ]; then
            DONE_COUNT=$((DONE_COUNT + 1))
            echo "==> SKIP σ=$SIGMA seed $SEED (checkpoint verified) [$DONE_COUNT/$TOTAL_RUNS]"
            continue
        fi

        echo "==> [$(( DONE_COUNT + 1 ))/$TOTAL_RUNS] Launching σ=$SIGMA seed=$SEED @ $(date +%H:%M:%S)"
        python "$TRAINER" \
            --seed "$SEED" \
            --total-steps "$STEPS" \
            --sigma "$SIGMA" \
            --msg-dim "$BEST_MSG_DIM" \
            --num-heads "$BEST_NUM_HEADS" \
            --hidden 128 \
            || { echo "TRACK CRASHED σ=$SIGMA seed=$SEED — skipping"; continue; }

        if [ -f "$RUN_DIR/final.pt" ]; then
            DONE_COUNT=$((DONE_COUNT + 1))
            echo "    σ=$SIGMA seed=$SEED -> $RUN_DIR/final.pt"
        else
            echo "    WARN: expected model missing at $RUN_DIR/final.pt"
        fi
    done
done

echo
echo "==> Noise parameter sweep completed at $(date)."
