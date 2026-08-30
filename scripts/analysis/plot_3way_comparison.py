"""Plot a 3-way (or N-way) comparison of communication conditions.

Consumes the JSON files produced by `aggregate_seeds.py`. For each
condition, draws:
  - A bar at the cross-seed mean return
  - An error bar at ± 1 cross-seed std
  - Faint dots for each individual seed's mean (transparency = 0.55)
  - Annotation: "n=K seeds, K_eval x K_seeds episodes"

This is the figure for Chapter 5 of the dissertation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless mode for cluster execution
import matplotlib.pyplot as plt
import numpy as np


CONDITION_COLORS = {
    "no-comm": "#888888",
    "fc-comm": "#1f77b4",
    "attn-comm": "#2ca02c",
    "gnn-comm": "#d62728",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregates", required=True, nargs="+", type=Path,
                        help="One JSON per condition (from aggregate_seeds.py).")
    parser.add_argument("--labels", required=True, nargs="+", type=str,
                        help="Display labels, one per --aggregates entry.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="Communication conditions",
                        type=str)
    args = parser.parse_args()

    if len(args.aggregates) != len(args.labels):
        raise SystemExit("--aggregates and --labels must have the same length")

    data = []
    for jf, label in zip(args.aggregates, args.labels):
        if not jf.exists():
            print(f"[plot] SKIP missing aggregate file path: {jf}")
            continue
        with jf.open() as f:
            d = json.load(f)
        d["__label__"] = label
        data.append(d)

    if not data:
        raise SystemExit("No metrics aggregates loaded successfully.")

    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(data) + 2), 4.5))

    xs = np.arange(len(data))
    means = [d["mean_return_across_seeds"] for d in data]
    stds = [d["std_return_across_seeds"] for d in data]
    n_seeds = [d["num_seeds"] for d in data]
    per_seed = [d["per_seed_mean_return"] for d in data]
    labels = [d["__label__"] for d in data]

    colors = [CONDITION_COLORS.get(lbl, "#666666") for lbl in labels]
    bars = ax.bar(xs, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.8, alpha=0.85,
                  error_kw={"elinewidth": 1.2})

    for i, seed_means in enumerate(per_seed):
        if len(seed_means) > 0:
            ax.scatter([i] * len(seed_means), seed_means,
                       color="black", alpha=0.55, s=22, zorder=3)

    for i, (m, s, n) in enumerate(zip(means, stds, n_seeds)):
        ax.text(i, m + (abs(s) + 1.5) * (1 if m >= 0 else -1),
                f"{m:+.1f} ± {s:.1f}\n(n={n} seed{'s' if n != 1 else ''})",
                ha="center", va="bottom" if m >= 0 else "top",
                fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean episode return (eval, greedy)")
    ax.set_title(args.title, fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Technical footer annotation
    fig.text(0.99, 0.01,
             "Error bars: ±1 std across seeds. Black dots: per-seed means.",
             ha="right", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"[plot] Figure successfully generated and saved -> {args.out}")

    # Console output summary table
    print("\n--- Summary Metrics ---")
    for d in data:
        print(f"  {d['__label__']:>10s} : {d['mean_return_across_seeds']:+.2f} "
              f"± {d['std_return_across_seeds']:.2f}  (n={d['num_seeds']})")


if __name__ == "__main__":
    main()
