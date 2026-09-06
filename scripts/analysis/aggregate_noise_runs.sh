#!/usr/bin/env bash
# Day 19 — aggregate completed noise-sweep conditions dynamically.

set -euo pipefail

# Resolve repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../../src/comms/attention_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -f "$SCRIPT_DIR/../../../repo/src/comms/attention_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../../repo" && pwd )"
else
    echo "ERROR: cannot locate repo root"
    exit 1
fi

cd "$REPO_ROOT"

SIGMAS=(0.10 0.25 0.50)
TAGS=(s010 s025 s050)

echo "==> Scanning Gaussian Noise Sweep Matrix Conditions..."

ANY_DONE=0
for i in "${!SIGMAS[@]}"; do
    SIGMA="${SIGMAS[$i]}"
    TAG="${TAGS[$i]}"
    BASE="artifacts/checkpoints/attncomm_noise_${TAG}"

    # Check if this configuration has files on disk before running evaluations
    if [ ! -d "$BASE" ]; then
        echo "  σ=$SIGMA ($TAG):  SKIP — directory entry missing"
        continue
    fi

    # Check that all 3 seeds for this sigma are present
    MISSING=""
    for S in 0 1 2; do
        if [ ! -f "$BASE/seed${S}/final.pt" ]; then
            MISSING="$MISSING $S"
        fi
    done

    if [ -n "$MISSING" ]; then
        echo "  σ=$SIGMA ($TAG):  SKIP — incomplete optimization sweep. Missing seeds:$MISSING"
        continue
    fi

    # Skip seamlessly if aggregate already exists (idempotency preservation)
    if [ -f "$BASE/aggregate.json" ]; then
        echo "  σ=$SIGMA ($TAG):  Verified. Data node parsed -> $BASE/aggregate.json"
        ANY_DONE=$((ANY_DONE + 1))
        continue
    fi

    echo "  σ=$SIGMA ($TAG):  Aggregating 3 seeds across 20 evaluation rollouts..."
    python scripts/analysis/aggregate_seeds.py \
        --condition "attn-noise-${TAG}" \
        --checkpoints "$BASE/seed0/final.pt" \
                      "$BASE/seed1/final.pt" \
                      "$BASE/seed2/final.pt" \
        --eval-episodes 20 \
        --out "$BASE/aggregate.json"

    ANY_DONE=$((ANY_DONE + 1))
done

echo
echo "==> Data node parsing loop completed. Active records processed: $ANY_DONE"
