"""Regenerate RESULTS_TABLE.md from aggregate JSONs using ddof=1.

Reason for existing
-------------------
The current RESULTS_TABLE.md reports population std (ddof=0) but the
methodology note (Day 14) committed to sample std (ddof=1, "unbiased").
The aggregate JSONs already store ddof=1 stds in
`std_return_across_seeds`. This script reads them directly and rebuilds
the tables — the prose interpretation paragraphs are preserved verbatim.

What it does
------------
1. Walk `artifacts/checkpoints/` looking for `aggregate.json` files.
2. Categorise each by condition (auto-detected from the JSON `condition`
   field) — Simple Spread vs Speaker-Listener vs GNN.
3. Write a fresh `RESULTS_TABLE.md` with:
   - Header matching the previous file.
   - Tables regenerated from JSON values.
   - Interpretation paragraphs preserved from the old file (if found).
4. Print a summary of changes to stdout.

What it does NOT do
-------------------
- Does not change any aggregate JSON.
- Does not change any checkpoint or other artefact.
- Does not modify the interpretation paragraphs.
- Does not auto-commit (you commit manually after eyeballing the diff).

USAGE
-----
    python scripts/regenerate_results_table.py \
        --experiments-dir experiments \
        --out RESULTS_TABLE.md \
        [--preserve-interpretation-from RESULTS_TABLE.md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------
# Condition labels — map raw JSON "condition" to human-readable row name
# ----------------------------------------------------------------------

# Simple Spread (N=3) conditions
SS_LABELS = {
    "no-comm":          "no-comm",
    "fc-comm":          "fc-comm",
    "attn-comm":        "attention-comm",
    "attn-bw8":         "bandwidth (bw8)",
    "attn-ib-b0p01":    "info-bottleneck",
    "attn-noise-s010":  "noise σ=0.1",
    "attn-noise-s025":  "noise σ=0.25",
    "attn-noise-s050":  "noise σ=0.5",
    "gnn-comm":         "GNN-comm",
}
# Display order
SS_ORDER = [
    "no-comm", "fc-comm", "attn-comm",
    "attn-bw8", "attn-ib-b0p01",
    "attn-noise-s010", "attn-noise-s025", "attn-noise-s050",
    "gnn-comm",
]

# Some aggregate JSONs were written with the folder name as the
# `condition` field (e.g. "nocomm_mappo") instead of the canonical
# label (e.g. "no-comm"). Normalise those so every row is found.
CONDITION_ALIASES = {
    "nocomm_mappo":  "no-comm",
    "fccomm_mappo":  "fc-comm",
    "attncomm_mappo": "attn-comm",
    "gnncomm_mappo": "gnn-comm",
}

# Speaker-Listener conditions
SL_LABELS = {
    "sl-none": "sl-none",
    "sl-fc":   "sl-fc",
    "sl-attn": "sl-attn",
}
SL_ORDER = ["sl-none", "sl-fc", "sl-attn"]


# ----------------------------------------------------------------------
# Walker — find every aggregate.json and index by condition
# ----------------------------------------------------------------------


def find_aggregates(root: Path) -> dict[str, dict]:
    """Return a mapping condition -> aggregate-json-dict for every
    `aggregate.json` found under `root`."""
    out: dict[str, dict] = {}
    for p in sorted(root.rglob("aggregate.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"  SKIP {p}: cannot parse ({e})")
            continue
        cond = d.get("condition")
        if not cond:
            print(f"  SKIP {p}: no 'condition' field")
            continue
        cond = CONDITION_ALIASES.get(cond, cond)  # normalise folder-name labels
        if cond in out:
            # Two files for same condition — keep newer mtime
            existing_mtime = (out[cond]["__path__"]).stat().st_mtime
            new_mtime = p.stat().st_mtime
            if new_mtime <= existing_mtime:
                continue
        d["__path__"] = p
        out[cond] = d
    return out


# ----------------------------------------------------------------------
# Row formatter
# ----------------------------------------------------------------------


def format_row(label: str, agg: Optional[dict]) -> str:
    if agg is None:
        return f"| {label:<18s} | _missing_ | – | 0 | – |"
    mean = agg["mean_return_across_seeds"]
    std = agg["std_return_across_seeds"]
    n = agg["num_seeds"]
    notes = "" if n == 3 else f"only n={n} so far"
    return f"| {label:<18s} | {mean:+.2f}   | {std:.2f} | {n}     | {notes} |"


# ----------------------------------------------------------------------
# Interpretation extraction (preserve old prose)
# ----------------------------------------------------------------------


_INTERPRET_HEADER_RE = re.compile(
    r"^## What the numbers say.*$", re.MULTILINE
)
_STATUS_HEADER_RE = re.compile(r"^## Status.*$", re.MULTILINE)


def extract_interpretation(old_text: str) -> str:
    """Pull the '## What the numbers say' + '## Status' sections from the
    old file so we can paste them at the bottom unchanged."""
    if not old_text:
        return ""
    m = _INTERPRET_HEADER_RE.search(old_text)
    if not m:
        return ""
    return old_text[m.start():]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preserve-interpretation-from", type=Path, default=None,
                   help="Existing RESULTS_TABLE.md to copy the interpretation "
                        "paragraphs from. Defaults to --out if it exists.")
    args = p.parse_args()

    if not args.experiments_dir.is_dir():
        raise SystemExit(f"experiments dir not found: {args.experiments_dir}")

    print(f"[regen] scanning {args.experiments_dir}")
    aggs = find_aggregates(args.experiments_dir)
    print(f"[regen] found {len(aggs)} aggregate.json files")
    for c in sorted(aggs):
        print(f"        {c:<22s} -> {aggs[c]['__path__']}")

    # Preserve interpretation
    pres_src = args.preserve_interpretation_from or args.out
    old_text = ""
    if pres_src.exists():
        old_text = pres_src.read_text()
        print(f"[regen] preserving interpretation from {pres_src}")
    interpretation = extract_interpretation(old_text)

    # Build markdown
    lines = []
    lines.append("# Final Results — post-fix, clean re-run")
    lines.append("")
    lines.append("All numbers are mean episode return ± std across seeds "
                 "(higher = better; Simple Spread returns are negative by "
                 "construction). Stds are ddof=1 (unbiased sample std) to "
                 "match the methodology note in §4.")
    lines.append("")
    lines.append("## Simple Spread (N=3)")
    lines.append("")
    lines.append("| Condition          | Return  | ±    | seeds | Notes |")
    lines.append("|--------------------|---------|------|-------|-------|")
    for cond_key in SS_ORDER:
        agg = aggs.get(cond_key)
        label = SS_LABELS[cond_key]
        lines.append(format_row(label, agg))
    lines.append("")

    lines.append("## Speaker-Listener (N=2)")
    lines.append("")
    lines.append("| Condition          | Return  | ±    | seeds | Notes |")
    lines.append("|--------------------|---------|------|-------|-------|")
    for cond_key in SL_ORDER:
        agg = aggs.get(cond_key)
        label = SL_LABELS[cond_key]
        lines.append(format_row(label, agg))
    lines.append("")

    # Append preserved interpretation if present
    if interpretation:
        lines.append(interpretation.rstrip())
    else:
        lines.append("## What the numbers say")
        lines.append("")
        lines.append("_(Interpretation paragraphs to be re-added by hand from "
                     "the previous version of this file.)_")
        lines.append("")

    args.out.write_text("\n".join(lines) + "\n")
    print(f"\n[regen] wrote {args.out}")
    print(f"[regen] {sum(1 for k in SS_ORDER if k in aggs)} / "
          f"{len(SS_ORDER)} Simple-Spread rows populated")
    print(f"[regen] {sum(1 for k in SL_ORDER if k in aggs)} / "
          f"{len(SL_ORDER)} SL rows populated")
    print("\nNext step: review with  git diff RESULTS_TABLE.md")


if __name__ == "__main__":
    main()
