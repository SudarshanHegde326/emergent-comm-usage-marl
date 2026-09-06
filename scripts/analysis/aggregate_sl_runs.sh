#!/usr/bin/env bash
# Day 21 — aggregate yesterday's 9 Speaker-Listener runs into 3 per-comm-type
# JSONs.
#
# Idempotent: each (comm, 3 seeds) cell is skipped silently if it does not
# have all 3 final.pt files, and skipped quietly if the aggregate already
# exists. Mirrors the pattern of Day 19's noise-aggregator.
#
# Usage (from repo root):
#     bash scripts/aggregate_sl_runs.sh

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

COMM_TYPES=(none fc attn)

echo "==> aggregating Speaker-Listener runs"

ANY_DONE=0
for CT in "${COMM_TYPES[@]}"; do
    BASE="artifacts/checkpoints/sl_${CT}"

    # Check all 3 seeds present
    MISSING=""
    for S in 0 1 2; do
        if [ ! -f "$BASE/seed${S}/final.pt" ]; then
            MISSING="$MISSING $S"
        fi
    done

    if [ -n "$MISSING" ]; then
        echo "  comm=$CT:  SKIP — missing seed(s):$MISSING"
        continue
    fi

    if [ -f "$BASE/aggregate.json" ]; then
        echo "  comm=$CT:  already aggregated -> $BASE/aggregate.json"
        ANY_DONE=$((ANY_DONE + 1))
        continue
    fi

    echo "  comm=$CT:  aggregating 3 seeds..."
    # NOTE: uses scripts/aggregate_sl_seeds.py — the SL-specific aggregator
    #       (the Day-14 aggregate_seeds.py is hardcoded to simple_spread_v3
    #        and would fail on SL checkpoints).
    python scripts/analysis/aggregate_sl_seeds.py \
        --condition "sl-${CT}" \
        --checkpoints "$BASE/seed0/final.pt" \
                      "$BASE/seed1/final.pt" \
                      "$BASE/seed2/final.pt" \
        --eval-episodes 20 \
        --out "$BASE/aggregate.json" \
        || { echo "    aggregation FAILED — continuing"; continue; }
    ANY_DONE=$((ANY_DONE + 1))
done

echo
if [ "$ANY_DONE" -eq 0 ]; then
    echo "WARN: no aggregates produced. Check the sweep log first:"
    echo "       tail -50 artifacts/checkpoints/sl_sweep.log"
    exit 1
fi
echo "==> done. $ANY_DONE / ${#COMM_TYPES[@]} comm-types have aggregates."