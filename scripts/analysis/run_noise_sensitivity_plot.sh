#!/usr/bin/env bash
# Day 19 — maps return-vs-σ performance degradation curves.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../../src/comms/attention_comm.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
else
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
fi
cd "$REPO_ROOT"

NOISE_ARGS=()
add_if_exists () {
    local sigma="$1"
    local path="$2"
    if [ -f "$path" ]; then
        NOISE_ARGS+=("$sigma" "$path")
        echo "  + adding noise threshold node σ=$sigma ($path)"
    else
        echo "  - skipping missing threshold data node σ=$sigma ($path)"
    fi
}

echo "==> Inspecting available noise aggregates..."
add_if_exists 0.10 experiments/attncomm_noise_s010/aggregate.json
add_if_exists 0.25 experiments/attncomm_noise_s025/aggregate.json
add_if_exists 0.50 experiments/attncomm_noise_s050/aggregate.json

BASE_ATTN="experiments/attncomm_official/aggregate.json"
if [ ! -f "$BASE_ATTN" ]; then
    echo "ERROR: baseline attention aggregate not found at $BASE_ATTN"
    exit 1
fi

python scripts/analysis/polish_noise_plot.py \
    --baseline-attn  "$BASE_ATTN" \
    --noise-aggregates "${NOISE_ARGS[@]}" \
    --out figures/fig_noise_sensitivity_v1.png \
    --out-pdf figures/fig_noise_sensitivity_v1.pdf \
    --title "Attention Routing Performance Sensitivity to Channel Noise"