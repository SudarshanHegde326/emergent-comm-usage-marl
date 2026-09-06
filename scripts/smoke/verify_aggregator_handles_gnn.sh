#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
AGG="$REPO_ROOT/scripts/analysis/aggregate_seeds.py"

DETECT_OK=0
LOAD_OK=0

if grep -qE '(gnn-comm|GNNCommPipeline|agg\.layer_norms)' "$AGG"; then
    DETECT_OK=1
fi

if grep -qE 'GNNCommPipeline' "$AGG"; then
    LOAD_OK=1
fi

echo "==> verifying $AGG"
echo "    detect_condition knows gnn-comm: $([ $DETECT_OK -eq 1 ] && echo yes || echo no)"
echo "    load_pipeline builds GNNCommPipeline: $([ $LOAD_OK -eq 1 ] && echo yes || echo no)"

if [ $DETECT_OK -eq 1 ] && [ $LOAD_OK -eq 1 ]; then
    echo
    echo "OK — gnn-comm branch present. Safe to run finish_gnn.sh."
    exit 0
fi

echo
echo "PATCH NEEDED — at least one piece is missing."
exit 1