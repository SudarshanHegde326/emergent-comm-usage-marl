#!/usr/bin/env bash
# File: run_3seeds_fccomm.sh
# Description: Launch 3 independent FC-comm MAPPO seeds in the background on bityi.

set -u

TOTAL_STEPS=300000
SCRIPT="scripts/trainers/02_fccomm_mappo_simple_spread.py"
LOG_DIR="logs"
MSG_DIM=8
AGG_MODE="mean"
SEEDS=(0 1 2)

if [[ ! -f "$SCRIPT" ]]; then
    echo "[ERROR] Cannot find $SCRIPT relative to $(pwd)"
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "[launch] Starting ${#SEEDS[@]} background FC-comm seeds (msg_dim=$MSG_DIM, agg=$AGG_MODE)..."

for SEED in "${SEEDS[@]}"; do
    LOG_FILE="$LOG_DIR/fccomm_seed${SEED}.log"
    PID_FILE="$LOG_DIR/fccomm_seed${SEED}.pid"

    nohup python -u "$SCRIPT" \
        --seed "$SEED" \
        --total-steps "$TOTAL_STEPS" \
        --msg-dim "$MSG_DIM" \
        --agg-mode "$AGG_MODE" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "  -> FC Seed $SEED initialized under Process ID (PID): $(cat $PID_FILE)"
done

echo ""
echo "[launch] All background communication tasks submitted successfully! "
echo "To monitor progress live, run: tail -f artifacts/logs/fccomm_seed0.log"
