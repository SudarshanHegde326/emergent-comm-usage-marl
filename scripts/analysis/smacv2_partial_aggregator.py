from __future__ import annotations
import argparse, json, sys, torch, numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any

CONDITIONS = ["smacv2_nocomm", "smacv2_fccomm", "smacv2_attncomm", "smacv2_gnncomm"]
METRICS_OF_INTEREST = ["final_win_rate_20", "action_valid_frac"]
MIN_SEEDS_FOR_STATS = 2
ACTION_MASK_WARNING_THRESHOLD = 0.999

def find_seed_checkpoints(condition_dir: Path) -> list[tuple[int, Path]]:
    out = []
    if not condition_dir.exists(): return out
    for seed_dir in sorted(condition_dir.iterdir()):
        if seed_dir.is_dir() and seed_dir.name.startswith("seed"):
            final_pt = seed_dir / "final.pt"
            if final_pt.exists():
                out.append((int(seed_dir.name[4:]), final_pt))
    return out

def aggregate_condition(condition_dir: Path) -> dict[str, Any]:
    seed_ckpts = find_seed_checkpoints(condition_dir)
    per_seed = {idx: {k: torch.load(p, map_location="cpu", weights_only=False).get(k) 
                for k in METRICS_OF_INTEREST} for idx, p in seed_ckpts}
    
    aggregates = {}
    for metric in METRICS_OF_INTEREST:
        vals = [m[metric] for m in per_seed.values() if m.get(metric) is not None]
        mean = float(np.mean(vals)) if vals else None
        std = float(np.std(vals, ddof=1)) if len(vals) >= MIN_SEEDS_FOR_STATS else None
        aggregates[metric] = {"mean": mean, "std": std, "n": len(vals)}
    
    return {"n_seeds": len(seed_ckpts), "seeds": [s for s, _ in seed_ckpts], 
            "per_seed": per_seed, "aggregates": aggregates, "flags": []}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=str, default="experiments")
    args = parser.parse_args()
    exp_dir = Path(args.experiments_dir)
    out_path = exp_dir / "aggregates" / "smacv2_partial.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    blob = {"generated_at": datetime.now().isoformat(), "by_condition": {}}
    for cond in CONDITIONS:
        blob["by_condition"][cond] = aggregate_condition(exp_dir / cond)
    
    with out_path.open("w") as f:
        json.dump(blob, f, indent=2, default=lambda o: o.tolist() if isinstance(o, torch.Tensor) else str(o))
    print(f"Aggregator finished. Wrote {out_path}")

if __name__ == "__main__": main()