"""
Quick comparison plot: no-comm vs FC-comm reward curves.
File: scripts/compare_runs.py

"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_log(file: Path) -> tuple[np.ndarray, np.ndarray]:
    pat = re.compile(r"(?:steps|Steps)=\s*(\d+)\s+(?:mean_ret|MeanReturn)[^=]*=\s*([\-\d.]+)")
    steps, rets = [], []
    with file.open() as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                rets.append(float(m.group(2)))
    return np.asarray(steps), np.asarray(rets)

def aggregate_seeds(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = [parse_log(f) for f in files if f.stat().st_size > 0]
    series = [(s, r) for s, r in series if len(s) > 0]
    if not series:
        raise RuntimeError(f"No usable data found in provided files: {files}")

    # Anchor the grid to the shortest execution path to prevent extrapolation artifacts
    max_step = min(s.max() for s, _ in series)
    grid = np.linspace(0, max_step, 100)

    interp = []
    for s, r in series:
        interp.append(np.interp(grid, s, r))
    arr = np.stack(interp, axis=0)

    # Calculate standard deviation envelope bounds if multiple seeds exist
    std_envelope = arr.std(axis=0) if arr.shape[0] > 1 else np.zeros_like(grid)
    return grid, arr.mean(axis=0), std_envelope

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs", help="Folder containing experimental execution log files")
    ap.add_argument("--out", default="figures/day3_nocomm_vs_fccomm.png")
    ap.add_argument("--title", default="Decentralized No-Comm vs Broadcast FC-Comm (Simple Spread)")
    args = ap.parse_args()

    log_dir = Path(args.logs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, color in [("nocomm", "tab:blue"), ("fccomm", "tab:orange")]:
        files = sorted(log_dir.glob(f"{name}_seed*.log"))
        if not files:
            print(f"[plot] No logs discovered for configuration category: {name}, skipping.")
            continue
        try:
            grid, mean, std = aggregate_seeds(files)
            ax.plot(grid, mean, label=f"{name} (n={len(files)} seeds)", color=color, lw=2)
            if len(files) > 1:
                ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)
        except RuntimeError as e:
            print(f"[plot] Skipping evaluation node for {name}: {e}")
            continue

    ax.set_xlabel("Cumulative Environment Steps")
    ax.set_ylabel("Mean Team Episode Return")
    ax.set_title(args.title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")
    fig.tight_layout()

    fig.savefig(out, dpi=150)
    print(f"[plot] Success! Comparison visualization generated at: {out} ")

if __name__ == "__main__":
    main()
