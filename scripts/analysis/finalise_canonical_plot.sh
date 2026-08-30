#!/usr/bin/env bash
# Day 19 — maps the grand milestone canonical chart overview.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../../src/comms/attention_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
else
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
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
        echo "  + including data node: $label"
    else
        echo "  - skipping missing data node: $label"
    fi
}

echo "==> Assembling existing metrics into 8-bar layout lineup..."
add_if_exists experiments/nocomm_mappo/aggregate.json                 "No comm"
add_if_exists experiments/fccomm_mappo/aggregate.json                 "FC comm"
add_if_exists experiments/attncomm_official/aggregate.json            "Attention"
add_if_exists experiments/attncomm_bw8/aggregate.json                 "Attn + bw=8"
add_if_exists experiments/attncomm_ib_b0p01/aggregate.json            "Attn + IB β=0.01"
add_if_exists experiments/attncomm_noise_s010/aggregate.json          "Attn + σ=0.10"
add_if_exists experiments/attncomm_noise_s025/aggregate.json          "Attn + σ=0.25"
add_if_exists experiments/attncomm_noise_s050/aggregate.json          "Attn + σ=0.50"

python scripts/analysis/polish_3way_plot.py \
    --aggregates "${JSON_ARGS[@]}" \
    --labels     "${LABEL_ARGS[@]}" \
    --out        figures/fig_constraints_v2_8bars.png \
    --out-pdf    figures/fig_constraints_v2_8bars.pdf \
    --title      "Cooperative Multi-Agent Condition Comparison Matrix (200k steps)"
