"""Generates the Chapter 5 Section 5.4 Noise Sensitivity Curve."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load_agg_json(path_str: str | None):
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-attn", type=str, required=True)
    parser.add_argument("--noise-aggregates", nargs="+", required=True,
                        help="Pairs of: sigma json_path sigma json_path ...")
    parser.add_argument("--nocomm-ref", type=str, default=None)
    parser.add_argument("--fccomm-ref", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/fig_noise_sensitivity_final.png")
    parser.add_argument("--out-pdf", type=str, default="figures/fig_noise_sensitivity_final.pdf")
    parser.add_argument("--title", type=str, default="Attention Robustness to Additive Gaussian Channel Noise")
    args = parser.parse_args()

    if len(args.noise_aggregates) % 2 != 0:
        print("ERROR: --noise-aggregates must be passed in pairs of: [sigma path]")
        sys.exit(1)

    sigmas = [0.0]
    json_paths = [args.baseline_attn]

    for i in range(0, len(args.noise_aggregates), 2):
        sigmas.append(float(args.noise_aggregates[i]))
        json_paths.append(args.noise_aggregates[i+1])

    # Sort based on sigma values for clean line plotting
    sorted_idx = np.argsort(sigmas)
    sigmas = [sigmas[idx] for idx in sorted_idx]
    json_paths = [json_paths[idx] for idx in sorted_idx]

    xs = []
    ys = []
    y_stds = []
    all_seeds_data = []

    print("\n--- Parsing Noise Sensitivity Sweep Parameters ---")
    for s, path in zip(sigmas, json_paths):
        data = load_agg_json(path)
        if data is None:
            print(f"  σ = {s:.2f} : MISSING node entry -> {path}")
            continue

        m = data["mean_return_across_seeds"]
        std_val = data["std_return_across_seeds"]
        # FIX 2: Changed 'per_seed_means' to 'per_seed_mean_return' to match the database file structure
        per_seed = data["per_seed_mean_return"]

        xs.append(s)
        ys.append(m)
        y_stds.append(std_val)
        all_seeds_data.append(per_seed)
        print(f"  σ = {s:.2f} : Found. Mean = {m:7.2f} ± {std_val:.2f} (n={len(per_seed)})")

    if len(xs) < 1:
        print("ERROR: No files found to render.")
        sys.exit(1)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    xs = np.array(xs)
    ys = np.array(ys)
    y_stds = np.array(y_stds)

    ax.plot(xs, ys, marker="o", color="#D55E00", linewidth=2, label="Attention Comm", zorder=3)
    ax.fill_between(xs, ys - y_stds, ys + y_stds, color="#D55E00", alpha=0.15, zorder=2)

    rng = np.random.default_rng(seed=42)
    for idx, per_seed in enumerate(all_seeds_data):
        x_jitter = xs[idx] + rng.uniform(-0.01, 0.01, size=len(per_seed))
        ax.scatter(x_jitter, per_seed, color="#D55E00", alpha=0.5, edgecolors="none", s=25, zorder=4)
    ref_configs = [
        (args.nocomm_ref, "No Communication Reference", "black"),
        (args.fccomm_ref, "Fully-Connected Reference", "#0072B2")
    ]
    for path, label, color in ref_configs:
        ref_data = load_agg_json(path)
        if ref_data:
            ref_m = ref_data["mean_return_across_seeds"]
            ax.axhline(ref_m, color=color, linestyle="--", linewidth=1.2, alpha=0.8,
                       label=f"{label} ({ref_m:+.1f})")

    ax.set_xlabel("Channel Noise Standard Deviation (σ)")
    ax.set_ylabel("Mean Episode Return")
    ax.set_title(args.title, fontsize=11, fontweight="bold", pad=12)
    ax.grid(linestyle=":", alpha=0.5)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_xticks(xs)
    ax.legend(loc="best", fontsize=9, frameon=False)

    # Add descriptive footer
    fig.text(0.99, 0.01, "Error bars/bounds: ±1 sample std dev across seeds. Dots: individual seed means.",
             ha="right", va="bottom", fontsize=7, color="gray")

    # Save high-resolution figures
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")

    out_pdf_path = Path(args.out_pdf)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf_path, bbox_inches="tight")
    print(f"\n[plot] Figures saved:\n  -> {out_path}\n  -> {out_pdf_path}")

if __name__ == "__main__":
    main()