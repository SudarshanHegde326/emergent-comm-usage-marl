#!/usr/bin/env bash
# Day 40 — smoke the partial aggregator pipeline end-to-end.
#
# Runs `smacv2_partial_aggregator.py` then `generate_w6_partial_results.py`
# on whatever is currently in `experiments/smacv2_*/`. Safe to run any
# time during W6 — partial-aware by design.
#
# Usage (from repo root):
#     bash scripts/smoke_partial_aggregator.sh

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

AGG="$REPO_ROOT/scripts/analysis/smacv2_partial_aggregator.py"
MD="$REPO_ROOT/scripts/analysis/generate_w6_partial_results.py"
for f in "$AGG" "$MD"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

echo "=========================================================="
echo "==> STEP 1/2: partial aggregator"
echo "=========================================================="
python "$AGG"

echo
echo "=========================================================="
echo "==> STEP 2/2: markdown generator"
echo "=========================================================="
python "$MD"

echo
echo "=========================================================="
echo "==> DONE"
echo "=========================================================="
echo "    JSON: $REPO_ROOT/experiments/aggregates/smacv2_partial.json"
echo "    MD:   $REPO_ROOT/docs/W6_SMACv2_Partial_Results.md"
echo
echo "    Re-run this script any time more seeds finish."
echo "    Final run = Monday after all 12 seeds land."
