#!/usr/bin/env bash
# Day 26 — extend the W3 8-bar canonical plot to 9 bars by adding the
# GNN-comm condition.
#
# Output: figures/fig_constraints_v3_9bars.{png,pdf}
#
# Skips silently if any of the 9 aggregates is missing.
#
# Usage (from repo root):
#     bash scripts/update_canonical_plot_to_9bars.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -d "$SCRIPT_DIR/../../experiments" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -d "$SCRIPT_DIR/../../../repo/experiments" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../../repo" && pwd )"
elif [ -d "$(pwd)/experiments" ]; then
    REPO_ROOT="$(pwd)"
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

echo "==> assembling 9-bar lineup..."
add_if_exists experiments/nocomm_mappo/aggregate.json                 "No comm"
add_if_exists experiments/fccomm_mappo/aggregate.json                 "FC comm"
add_if_exists experiments/attncomm_mappo/aggregate.json               "Attention"
add_if_exists experiments/attncomm_bw8/aggregate.json                 "Attn + bw=8"
add_if_exists experiments/attncomm_ib_b0p01/aggregate.json            "Attn + IB β=0.01"
add_if_exists experiments/attncomm_noise_s010/aggregate.json          "Attn + σ=0.10"
add_if_exists experiments/attncomm_noise_s025/aggregate.json          "Attn + σ=0.25"
add_if_exists experiments/attncomm_noise_s050/aggregate.json          "Attn + σ=0.50"
add_if_exists experiments/gnncomm_mappo/aggregate.json                "GNN-comm"

if [ ${#JSON_ARGS[@]} -lt 2 ]; then
    echo "ERROR: need at least 2 aggregates."
    exit 1
fi

echo
echo "==> plotting ${#JSON_ARGS[@]} conditions"
python scripts/analysis/polish_3way_plot.py \
    --aggregates "${JSON_ARGS[@]}" \
    --labels     "${LABEL_ARGS[@]}" \
    --out        figures/fig_constraints_v3_9bars.png \
    --out-pdf    figures/fig_constraints_v3_9bars.pdf \
    --title      "All conditions on Simple Spread (N=3, 200k steps, ddof=1 error bars)"

echo
echo "==> done."
echo "    PNG: figures/fig_constraints_v3_9bars.png"
echo "    PDF: figures/fig_constraints_v3_9bars.pdf"
echo
echo "==> compare to v2 8-bar (W3) to see the GNN bar lands in the comm-condition cluster:"
echo "    diff -u figures/fig_constraints_v2_8bars.png figures/fig_constraints_v3_9bars.png \\"
echo "        | head -1 ; # binary diff — open both images to compare visually."
