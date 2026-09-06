"""Polished, dissertation-grade version of the 3-way comparison bar chart.

Consumes the JSON aggregation structures from aggregate_seeds.py and outputs
publication-quality PNG and vector-space PDF formats for LaTeX compilation.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless cluster environment rendering guard
import matplotlib.pyplot as plt
import numpy as np

# Colour-blind-friendly and grayscale-safe palette layout (Wong, 2011)
CONDITION_COLORS = {
    "No comm":         "#999999",
    "FC comm":         "#0072B2",
    "Attention comm":  "#009E73",
    "GNN comm":        "#D55E00",
    "no-comm":         "#999999",
    "fc-comm":         "#0072B2",
    "attn-comm":       "#009E73",
    "gnn-comm":        "#D55E00",
}

# Two-sided 95% critical t-distributions table for df = n - 1
_T_CRIT_95 = {1: math.nan, 2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}

def t_ci95_halfwidth(std_across_seeds: float, n: int) -> float:
    """Compute the approximate half-width of a 95% confidence interval."""
    if n < 2:
        return float("nan")
    t = _T_CRIT_95.get(n, 1.96)
    return t * (std_across_seeds / math.sqrt(n))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregates", required=True, nargs="+", type=Path)
    parser.add_argument("--labels", required=True, nargs="+", type=str)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-pdf", required=False, type=Path)
    parser.add_argument("--title", default="Communication conditions", type=str)
    parser.add_argument("--y-label", default="Mean episode return", type=str)
    args = parser.parse_args()

    if len(args.aggregates) != len(args.labels):
        raise SystemExit("--aggregates and --labels must match in sequence counts.")

    data = []
    for jf, label in zip(args.aggregates, args.labels):
        if not jf.exists():
            print(f"[polish] SKIP missing data node: {jf}")
            continue
        with jf.open() as f:
            d = json.load(f)
        d["__label__"] = label
        d["__ci95_half__"] = t_ci95_halfwidth(d["std_return_across_seeds"], d["num_seeds"])
        data.append(d)

    if not data:
        raise SystemExit("Zero aggregated metric structures discovered on disk.")

    # High-fidelity typography styling parameters matched to dissertation templates
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(1.4 * len(data) + 3.2, 4.2))

    xs = np.arange(len(data))
    means = [d["mean_return_across_seeds"] for d in data]
    stds = [d["std_return_across_seeds"] for d in data]
    n_seeds = [d["num_seeds"] for d in data]
    per_seed = [d["per_seed_mean_return"] for d in data]
    labels = [d["__label__"] for d in data]
    colours = [CONDITION_COLORS.get(lbl, "#666666") for lbl in labels]

    # Render descriptive bars with sample-standard deviation error boundaries
    ax.bar(xs, means, yerr=stds, capsize=5,
           color=colours, edgecolor="black", linewidth=0.7, alpha=0.9,
           error_kw={"elinewidth": 1.2, "ecolor": "black"})

    # Overlay individual seed returns using a subtle x-axis scatter jitter to separate overlapping runs
    rng = np.random.default_rng(0)
    for i, seed_means in enumerate(per_seed):
        if not seed_means:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=len(seed_means))
        ax.scatter(np.full(len(seed_means), i) + jitter, seed_means,
                   color="black", alpha=0.7, s=20, zorder=3,
                   edgecolor="white", linewidth=0.4)

    # Attach exact text metrics directly above or below each comparative bar axis
    for i, (m, s, n) in enumerate(zip(means, stds, n_seeds)):
        sign = 1 if m >= 0 else -1
        ax.text(i, m + sign * (abs(s) + 1.2),
                f"{m:+.1f} ± {s:.1f}\n(n={n})",
                ha="center", va="bottom" if m >= 0 else "top",
                fontsize=9, linespacing=1.1)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(args.y_label)
    ax.set_title(args.title)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Compile contextual footnote data
    total_eval = sum(d["num_seeds"] * d["eval_episodes_per_seed"] for d in data)
    n_text = ", ".join(f"{d['__label__']}: n={d['num_seeds']}" for d in data)
    fig.text(
        0.99, 0.005,
        f"Error bars: ±1 std across seeds. Dots: per-seed means. "
        f"Seeds — {n_text}. Total eval episodes: {total_eval}.",
        ha="right", va="bottom", fontsize=6, color="gray",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    print(f"[polish] High-resolution PNG generated and saved -> {args.out}")

    if args.out_pdf is not None:
        args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_pdf)
        print(f"[polish] Publication vector-space PDF generated and saved -> {args.out_pdf}")

    print("\n--- Final Academic Summary Table ---")
    for d in data:
        ci = d["__ci95_half__"]
        ci_str = f"±{ci:.2f} (95% CI ≈)" if not math.isnan(ci) else "(n=1, no CI available)"
        print(f"  {d['__label__']:>18s} : {d['mean_return_across_seeds']:+7.2f} "
              f"± {d['std_return_across_seeds']:5.2f}  "
              f"[n={d['num_seeds']}]   {ci_str}")

if __name__ == "__main__":
    main()
