from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def fmt(agg):
    mean, std, n = agg.get("mean"), agg.get("std"), agg.get("n", 0)
    if mean is None: return "—"
    if std is None: return f"{mean:.3f}  (N={n} provisional)"
    return f"{mean:.3f} ± {std:.3f}  (N={n})"

def main():
    path = Path("artifacts/checkpoints/aggregates/smacv2_partial.json")
    with path.open() as f: blob = json.load(f)
    
    md = ["# W6 SMACv2 — Partial Results\n", "| Condition | Win rate | AVF |", "|---|---|---|"]
    for cond, data in blob["by_condition"].items():
        win = fmt(data["aggregates"]["final_win_rate_20"])
        avf = fmt(data["aggregates"]["action_valid_frac"])
        md.append(f"| {cond} | {win} | {avf} |")
    
    out_file = Path("docs/W6_SMACv2_Partial_Results.md")
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text("\n".join(md))
    print(f"Generated {out_file}")

if __name__ == "__main__": main()