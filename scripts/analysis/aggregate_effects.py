"""Aggregate per-run analysis CSVs into a summary with effect sizes.

Reads every CSV produced by analyze_checkpoint.py (one row per
map/mode/comm/seed), then for each (map, mode) group reports, per metric:

  * mean +/- std over seeds, per condition
  * Hedges' g (small-sample corrected) of each communicating condition vs the
    no-comm baseline, with a bootstrap 95% CI

Honest statistics note (also state this in the write-up): with ~3 seeds these
effect sizes are DESCRIPTIVE, not significance tests. The bootstrap CI over so
few seeds is wide by construction; report it and resist over-claiming.

Usage:
    python aggregate_effects.py --glob "artifacts/metrics/*.csv" --out summary.csv
"""

from __future__ import annotations
import argparse, csv, glob, os, sys
from collections import defaultdict
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from src.analysis.metrics import hedges_g, bootstrap_ci

# Metrics compared against the no-comm baseline.
METRICS = ["test_win_rate", "mine_I_M_obs", "mine_I_M_enemy", "infonce_I_M_enemy",
           "hoyer_sparsity", "activation_sparsity", "gate_open_rate",
           "abl_zero_winrate_drop", "abl_shuffle_winrate_drop"]
BASELINE = "none"


def load_rows(pattern):
    rows = []
    for path in glob.glob(pattern):
        with open(path) as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def fval(row, key):
    try:
        v = float(row.get(key, "nan"))
        return v
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help='e.g. "artifacts/metrics/*.csv"')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(args.glob)
    if not rows:
        print("no CSVs matched", args.glob); return

    # group[(map, mode, comm)][metric] = list of per-seed values
    group = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r.get("map"), r.get("mode"), r.get("comm"))
        for m in METRICS:
            v = fval(r, m)
            if not np.isnan(v):
                group[key][m].append(v)

    out_rows = []
    # unique (map, mode) pairs
    scenes = sorted({(k[0], k[1]) for k in group})
    for mp, mode in scenes:
        base = {m: group.get((mp, mode, BASELINE), {}).get(m, []) for m in METRICS}
        comms = sorted({k[2] for k in group if k[0] == mp and k[1] == mode})
        for comm in comms:
            for m in METRICS:
                vals = group[(mp, mode, comm)].get(m, [])
                if not vals:
                    continue
                mean, std, n = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0, len(vals)
                out = {"map": mp, "mode": mode, "comm": comm, "metric": m,
                       "n_seeds": n, "mean": round(mean, 5), "std": round(std, 5)}
                if comm != BASELINE and base.get(m):
                    g = hedges_g(np.array(vals), np.array(base[m]))
                    lo, hi = bootstrap_ci(np.array(vals), np.array(base[m]))
                    out["hedges_g_vs_none"] = round(g, 4)
                    out["g_ci_lo"] = round(lo, 4)
                    out["g_ci_hi"] = round(hi, 4)
                out_rows.append(out)

    fields = ["map", "mode", "comm", "metric", "n_seeds", "mean", "std",
              "hedges_g_vs_none", "g_ci_lo", "g_ci_hi"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {args.out}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
