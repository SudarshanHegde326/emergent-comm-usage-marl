#!/usr/bin/env bash
# Day 21 — produce the Speaker-Listener 3-bar comparison plot.
# Re-uses the Day-15 polish_3way_plot.py with SL aggregates.
#
# Output: figures/fig_sl_3way_comparison.{png,pdf}
#
# Usage (from repo root):
#     bash scripts/plot_sl_3way.sh

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

JSON_ARGS=()
LABEL_ARGS=()

add_if_exists () {
    local jpath="$1"
    local label="$2"
    if [ -f "$jpath" ]; then
        JSON_ARGS+=("$jpath")
        LABEL_ARGS+=("$label")
        echo "  + $jpath  ($label)"
    else
        echo "  - missing $jpath  ($label) -- skipped"
    fi
}

echo "==> assembling SL aggregates..."
add_if_exists artifacts/checkpoints/sl_none/aggregate.json   "No comm"
add_if_exists artifacts/checkpoints/sl_fc/aggregate.json     "FC comm"
add_if_exists artifacts/checkpoints/sl_attn/aggregate.json   "Attention comm"

if [ ${#JSON_ARGS[@]} -lt 2 ]; then
    echo "ERROR: need at least 2 aggregates to plot. Run aggregate_sl_runs.sh first."
    exit 1
fi

echo
echo "==> plotting ${#JSON_ARGS[@]} conditions on Speaker-Listener"
python scripts/analysis/polish_3way_plot.py \
    --aggregates "${JSON_ARGS[@]}" \
    --labels     "${LABEL_ARGS[@]}" \
    --out        figures/fig_sl_3way_comparison.png \
    --out-pdf    figures/fig_sl_3way_comparison.pdf \
    --title      "Communication conditions on Simple Speaker-Listener (N=2, 200k steps)"

echo
echo "==> done."
echo "    PNG: figures/fig_sl_3way_comparison.png"
echo "    PDF: figures/fig_sl_3way_comparison.pdf"
