"""
Publication-style comparison plot: no-comm vs FC-comm.
File: scripts/make_comparison_plot.py
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Apply clean professional styling constraints
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16
})

LOG_PATTERN = re.compile(r"(?:steps|Steps)=\s*(\d+)\s+(?:mean_ret|MeanReturn)[^=]*=\s*([\-\d.]+)")

def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a training stdout log and extract (steps, mean_return) arrays."""
    steps, rets = [], []
    with path.open() as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                steps.append(int(m.group(1)))
                rets.append(float(m.group(2)))
    return np.asarray(steps), np.asarray(rets)

def aggregate(files: list[Path], n_grid: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Interpolates individual seed timelines onto a uniform grid for variance bounding."""
    series = []
    for f in files:
        if f.stat().st_size == 0:
            continue
        s, r = parse_log(f)
        if len(s) > 1:
            series.append((s, r))
    if not series:
        raise RuntimeError(f"No usable performance data located inside: {files}")
        
    common_max = min(s.max() for s, _ in series)
    grid = np.linspace(0, common_max, n_grid)
    interp = np.stack([np.interp(grid, s, r) for s, r in series], axis=0)
    
    std_dev = interp.std(axis=0) if interp.shape[0] > 1 else np.zeros_like(grid)
    return grid, interp.mean(axis=0), std_dev, len(series)

def smooth(y: np.ndarray, k: int = 5) -> np.ndarray:
    """Applies a gentle boxcar filter window for cleaner thesis visualizations."""
    if k <= 1 or len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")

# Consistent color palette matching future attention/GNN extensions
COLOURS = {
    "nocomm":   "#3b6ea5",   # Blue
    "fccomm":   "#e07b39",   # Orange
    "attncomm": "#2f9e44",   # Green
    "gnncomm":  "#9c36b5"    # Purple
}

PRETTY_NAMES = {
    "nocomm":   "Independent No-Comm Baseline",
    "fccomm":   "Fully-Connected Comm Baseline",
    "attncomm": "Selective Attention Comm",
    "gnncomm":  "Relational GNN Comm"
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--methods", nargs="+", default=["nocomm", "fccomm"])
    ap.add_argument("--out", default="figures/day4_nocomm_vs_fccomm.pdf")
    ap.add_argument("--title", default="Multi-Agent Performance Trajectory (Simple Spread)")
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    out_pdf = Path(args.out).with_suffix(".pdf")
    out_png = Path(args.out).with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    plotted_lines = 0
    for method in args.methods:
        files = sorted(logs_dir.glob(f"{method}_seed*.log"))
        if not files:
            continue
        try:
            grid, mean, std, n = aggregate(files)
        except RuntimeError:
            continue
            
        mean_smoothed = smooth(mean, args.smooth)
        colour = COLOURS.get(method, None)
        label = f"{PRETTY_NAMES.get(method, method)} (n={n})"
        
        ax.plot(grid, mean_smoothed, label=label, color=colour, lw=2.5)
        if n > 1:
            ax.fill_between(grid, mean_smoothed - std, mean_smoothed + std, color=colour, alpha=0.15)
        plotted_lines += 1

    if plotted_lines == 0:
        print("[warn] No completed baseline log files found yet to draw lines from.")
        return

    ax.set_xlabel("Cumulative Environment Steps")
    ax.set_ylabel("Mean Episode Team Reward")
    ax.set_title(args.title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"[plot] Success! Wrote LaTeX-ready PDF to {out_pdf} and preview to {out_png} ")

if __name__ == "__main__":
    main()
