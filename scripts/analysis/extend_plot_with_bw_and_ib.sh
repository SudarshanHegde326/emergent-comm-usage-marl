#!/usr/bin/env bash
# Day 17 — Extends the polished presentation bar chart with constraint conditions.

set -euo pipefail

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

JSON_ARGS=()
LABEL_ARGS=()

add_if_exists() {
    local jpath="$1"
    local label="$2"
    if [ -f "$jpath" ]; then
        JSON_ARGS+=("$jpath")
        LABEL_ARGS+=("$label")
        echo "    + including $jpath ($label)"
    else
        echo "    - missing   $jpath ($label) -- skipped"
    fi
}

echo "==> Assembling existing data nodes..."
add_if_exists experiments/nocomm_mappo/aggregate.json                 "No comm"
add_if_exists experiments/fccomm_mappo/aggregate.json                 "FC comm"
add_if_exists experiments/attncomm_official/aggregate.json            "Attention comm"
add_if_exists experiments/attncomm_bw8/aggregate.json                 "Attn + bw=8"
add_if_exists experiments/attncomm_ib_b0p01/aggregate.json            "Attn + IB β=0.01"

if [ ${#JSON_ARGS[@]} -lt 2 ]; then
    echo "ERROR: need at least 2 aggregates to plot a comparison."
    exit 1
fi

echo
echo "==> Rendering ${#JSON_ARGS[@]} comparative conditions"
python scripts/analysis/polish_3way_plot.py \
    --aggregates "${JSON_ARGS[@]}" \
    --labels     "${LABEL_ARGS[@]}" \
    --out        figures/fig_constraints_v1.png \
    --out-pdf    figures/fig_constraints_v1.pdf \
    --title      "Communication Conditions on Simple Spread (N=3, 200k steps)"

echo
echo "==> Plotting pipeline execution complete."
