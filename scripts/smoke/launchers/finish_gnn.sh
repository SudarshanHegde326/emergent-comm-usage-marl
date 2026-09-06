#!/usr/bin/env bash
set -uo pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

STEPS=200000
GNN_DIR="artifacts/checkpoints/gnncomm_mappo"

rm -rf "$GNN_DIR/seed999"

for s in 1 2; do
    if [[ -f "$GNN_DIR/seed$s/final.pt" ]]; then
        echo "==> seed$s already exists, skipping"
    else
        echo "==> training GNN seed$s ($STEPS steps)"
        python -u scripts/trainers/08_gnncomm_mappo_simple_spread.py \
            --seed "$s" --total-steps "$STEPS" --num-layers 2 --no-wandb
    fi
done

n=0
for s in 0 1 2; do [[ -f "$GNN_DIR/seed$s/final.pt" ]] && n=$((n+1)); done
if [[ "$n" -ne 3 ]]; then
    echo "GATE BLOCK: only $n/3 GNN seeds present — not aggregating."
    exit 1
fi

echo "==> aggregating GNN (3 seeds)"
python scripts/analysis/aggregate_seeds.py \
    --condition gnn-comm \
    --checkpoints "$GNN_DIR/seed0/final.pt" "$GNN_DIR/seed1/final.pt" "$GNN_DIR/seed2/final.pt" \
    --eval-episodes 20 \
    --out "$GNN_DIR/aggregate.json"

echo
echo "==> GNN complete at n=3. Result:"
python - <<'PY'
import json, statistics as st
d=json.load(open("artifacts/checkpoints/gnncomm_mappo/aggregate.json"))
m=d.get("per_seed_mean_return") or d.get("per_seed_returns")
print(f"   gnn-comm: {sum(m)/len(m):.2f} ± {st.pstdev(m):.2f}  (n={len(m)})")
PY
