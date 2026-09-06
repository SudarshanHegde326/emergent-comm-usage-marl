#!/usr/bin/env bash
# Day 26 — verify GNN-comm is at n=3 and aggregate.json is valid.
#
# Three outcomes:
#   OK                       — 3 seeds + aggregate.json + parseable JSON
#   AGGREGATE MISSING        — 3 seeds present but no aggregate; emits the
#                              one-liner you need to run
#   INCOMPLETE               — fewer than 3 seeds; re-launch finish_gnn.sh
#
# Usage (from repo root):
#     bash scripts/check_gnn_n3_status.sh

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -d "$SCRIPT_DIR/../../experiments" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -d "$SCRIPT_DIR/../../../repo/experiments" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../../repo" && pwd )"
elif [ -d "$(pwd)/experiments" ]; then
    REPO_ROOT="$(pwd)"
else
    echo "ERROR: cannot locate repo root from $SCRIPT_DIR"
    exit 2
fi

GNN_DIR="$REPO_ROOT/artifacts/checkpoints/gnncomm_mappo"
AGG="$GNN_DIR/aggregate.json"

# 1. Count seeds
n=0
present=()
missing=()
for s in 0 1 2; do
    if [ -f "$GNN_DIR/seed$s/final.pt" ]; then
        n=$((n + 1))
        present+=("$s")
    else
        missing+=("$s")
    fi
done

echo "==> GNN status @ $(date +%H:%M:%S)"
echo "    repo  = $REPO_ROOT"
echo "    seeds present = ${present[*]:-(none)}"
echo "    seeds missing = ${missing[*]:-(none)}"
echo "    n = $n / 3"

if [ "$n" -lt 3 ]; then
    echo
    echo "INCOMPLETE — only $n/3 seeds present."
    echo "    Re-launch the runner to fill the gap:"
    echo "        nohup bash scripts/finish_gnn.sh \\"
    echo "             > artifacts/checkpoints/gnncomm_mappo/finish_resume.log 2>&1 &"
    exit 1
fi

# 2. Aggregate file present?
if [ ! -f "$AGG" ]; then
    echo
    echo "AGGREGATE MISSING — 3 seeds present but no aggregate.json."
    echo "    Run the one-liner:"
    echo "        python scripts/aggregate_seeds.py \\"
    echo "            --condition gnn-comm \\"
    echo "            --checkpoints $GNN_DIR/seed0/final.pt $GNN_DIR/seed1/final.pt $GNN_DIR/seed2/final.pt \\"
    echo "            --eval-episodes 20 \\"
    echo "            --out $AGG"
    exit 1
fi

# 3. JSON parseable + has the expected fields?
python3 - "$AGG" <<'PY' || exit 1
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception as e:
    print(f"    AGGREGATE INVALID — failed to parse: {e}")
    sys.exit(1)
required = ["condition", "num_seeds", "mean_return_across_seeds",
            "std_return_across_seeds", "per_seed_mean_return"]
miss = [k for k in required if k not in d]
if miss:
    print(f"    AGGREGATE INVALID — missing fields: {miss}")
    sys.exit(1)
if d["num_seeds"] != 3:
    print(f"    AGGREGATE PARTIAL — num_seeds={d['num_seeds']} (expected 3)")
    sys.exit(1)
m, s, n = d["mean_return_across_seeds"], d["std_return_across_seeds"], d["num_seeds"]
print(f"    aggregate: condition={d['condition']!r}  "
      f"mean={m:+.2f}  std(ddof=1)={s:.2f}  n={n}")
PY
rc=$?
if [ "$rc" -ne 0 ]; then
    exit 1
fi

echo
echo "OK — GNN at n=3 with valid aggregate.json. Safe to update canonical plot."
