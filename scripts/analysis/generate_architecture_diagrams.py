"""Generate the four §3.2 architecture diagrams.

Produces consistent, dissertation-grade box-and-arrow figures
for each comm condition:

    figures/arch_no_comm.{png,pdf}
    figures/arch_fc_comm.{png,pdf}
    figures/arch_attention_comm.{png,pdf}
    figures/arch_gnn_comm.{png,pdf}

Style is enforced by construction — same font, same colours, same
arrow style, same DPI across all four diagrams. See
``writing/02_Diagram_Specifications.md`` for the spec.

USAGE
-----
    python scripts/generate_architecture_diagrams.py --out-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

STYLE = {
    "font_family":      "DejaVu Sans",
    "box_face":         "#EEEEEE",
    "box_edge":         "#222222",
    "box_lw":           1.4,
    "expand_face":      "#FFFFFF",
    "arrow_color":      "#222222",
    "arrow_lw":         1.6,
    "arrow_size":       12,
    "label_font_size":  10,
    "shape_font_size":  8,
    "shape_color":      "#555555",
    "shape_font":       "monospace",
    "title_size":       11,
    "aux_arrow_style":  "dashed",
    "aux_arrow_color":  "#777777",
    "dpi":              300,
}


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def draw_box(ax, x, y, w, h, label, face=None, edge=None, fontsize=None):
    face = face if face is not None else STYLE["box_face"]
    edge = edge if edge is not None else STYLE["box_edge"]
    fontsize = fontsize if fontsize is not None else STYLE["label_font_size"]
    bbox = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=face,
        edgecolor=edge,
        linewidth=STYLE["box_lw"],
    )
    ax.add_patch(bbox)
    ax.text(x, y, label, ha="center", va="center",
            family=STYLE["font_family"], fontsize=fontsize)


def draw_arrow(ax, x1, y1, x2, y2, label=None, style="solid"):
    color = STYLE["arrow_color"] if style == "solid" else STYLE["aux_arrow_color"]
    linestyle = "-" if style == "solid" else "--"
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f"->,head_length={STYLE['arrow_size']},head_width=8",
        color=color,
        linewidth=STYLE["arrow_lw"],
        linestyle=linestyle,
        mutation_scale=1.0,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)
    if label is not None:
        midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(midx, midy + 0.55, label, ha="center", va="bottom",
                family=STYLE["shape_font"], fontsize=STYLE["shape_font_size"],
                color=STYLE["shape_color"])


def setup_axes(figsize, xlim=(0, 10), ylim=(0, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = out_path.with_suffix(".png")
    pdf = out_path.with_suffix(".pdf")
    fig.savefig(png, dpi=STYLE["dpi"], bbox_inches="tight")
    fig.savefig(pdf, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {png} + {pdf.name}")


# ---------------------------------------------------------------------------
# Diagram A — no-comm
# ---------------------------------------------------------------------------


def diagram_no_comm(out_dir: Path):
    fig, ax = setup_axes(figsize=(7.0, 3.0), ylim=(0, 5))
    y = 2.0
    draw_box(ax, 1.0, y, 1.6, 1.0, "obs")
    draw_box(ax, 5.0, y, 2.4, 1.0, "Actor (MLP)")
    draw_box(ax, 9.0, y, 1.6, 1.0, "logits")
    draw_arrow(ax, 1.8, y, 3.8, y, label="(B, N, obs_dim)")
    draw_arrow(ax, 6.2, y, 8.2, y, label="(B, N, action_dim)")
    ax.set_title("Figure 3.1 — No-comm baseline", fontsize=STYLE["title_size"])
    save(fig, out_dir / "arch_no_comm")


# ---------------------------------------------------------------------------
# Diagram B — FC-comm
# ---------------------------------------------------------------------------


def diagram_fc_comm(out_dir: Path):
    fig, ax = setup_axes(figsize=(10.5, 4.0), xlim=(0, 11.6), ylim=(0, 5))
    y_main = 3.2
    pos = [
        (0.8,  "obs",                       1.0),
        (3.0,  "MessageHead\n(MLP)",        1.8),
        (5.6,  "FCAggregator\n(mean, no self)", 2.0),
        (8.2,  "CommActor\n(MLP)",          1.8),
        (10.4, "logits",                    1.2),
    ]
    for (x, label, w) in pos:
        draw_box(ax, x, y_main, w, 0.9, label)
    draw_arrow(ax, 1.3, y_main, 2.1, y_main)
    draw_arrow(ax, 3.9, y_main, 4.6, y_main, label="m: (B, N, msg_dim)")
    draw_arrow(ax, 6.6, y_main, 7.3, y_main, label="agg: (B, N, msg_dim)")
    draw_arrow(ax, 9.1, y_main, 9.8, y_main, label="(B, N, action_dim)")
    # skip obs -> CommActor
    draw_arrow(ax, 0.8, y_main - 0.45, 0.8, 1.5, style="solid")
    draw_arrow(ax, 0.8, 1.5, 8.2, 1.5, style="solid")
    draw_arrow(ax, 8.2, 1.5, 8.2, y_main - 0.45, style="solid")
    ax.text(4.5, 1.7, "obs (skip connection)", ha="center", va="bottom",
            family=STYLE["shape_font"], fontsize=STYLE["shape_font_size"],
            color=STYLE["shape_color"])
    ax.set_title("Figure 3.2 — FC-comm architecture",
                 fontsize=STYLE["title_size"])
    save(fig, out_dir / "arch_fc_comm")


# ---------------------------------------------------------------------------
# Diagram C — attention-comm
# ---------------------------------------------------------------------------


def diagram_attention_comm(out_dir: Path):
    fig, ax = setup_axes(figsize=(9.5, 4.5), xlim=(0, 11.5), ylim=(0, 6))
    y_main = 4.0
    pos = [
        (0.8,  "obs",                          1.0),
        (3.0,  "MessageHead\n(MLP)",           1.8),
        (5.7,  "MultiHeadAttn\nAggregator",    2.4),
        (8.4,  "CommActor\n(MLP)",             1.8),
        (10.6, "logits",                       1.2),
    ]
    for (x, label, w) in pos:
        draw_box(ax, x, y_main, w, 0.9, label)
    draw_arrow(ax, 1.3, y_main, 2.1, y_main)
    draw_arrow(ax, 3.9, y_main, 4.5, y_main, label="m: (B, N, msg_dim)")
    draw_arrow(ax, 6.9, y_main, 7.5, y_main, label="agg: (B, N, msg_dim)")
    draw_arrow(ax, 9.3, y_main, 10.0, y_main, label="(B, N, action_dim)")
    # skip obs -> CommActor
    draw_arrow(ax, 0.8, y_main - 0.45, 0.8, 2.7, style="solid")
    draw_arrow(ax, 0.8, 2.7, 8.4, 2.7, style="solid")
    draw_arrow(ax, 8.4, 2.7, 8.4, y_main - 0.45, style="solid")
    ax.text(4.6, 2.9, "obs (skip connection)", ha="center", va="bottom",
            family=STYLE["shape_font"], fontsize=STYLE["shape_font_size"],
            color=STYLE["shape_color"])
    # Inner detail below the boxes
    ax.text(5.7, 1.7,
            "Q, K, V projections -> softmax (self-masked)\n"
            "-> weighted sum -> W_O output projection",
            ha="center", va="top",
            family=STYLE["font_family"], fontsize=8,
            color="#333333")
    # Aux output above the attention box
    draw_arrow(ax, 6.8, y_main + 0.45, 6.8, 5.5, style="dashed")
    ax.text(6.85, 5.6,
            "attn weights (B, H, N, N) -> Ch.5 diagnostics",
            ha="left", va="bottom",
            family=STYLE["shape_font"], fontsize=STYLE["shape_font_size"],
            color=STYLE["shape_color"])
    ax.set_title("Figure 3.3 — Attention-comm architecture",
                 fontsize=STYLE["title_size"])
    save(fig, out_dir / "arch_attention_comm")


# ---------------------------------------------------------------------------
# Diagram D — GNN-comm
# ---------------------------------------------------------------------------


def diagram_gnn_comm(out_dir: Path):
    fig, ax = setup_axes(figsize=(13.5, 5.0), xlim=(0, 13.6), ylim=(0, 6))
    y_main = 4.0
    pos = [
        (0.8,  "obs",                         1.0),
        (3.0,  "MessageHead",                 1.6),
        (5.2,  "Channel\n(Identity)",         1.4),
        (7.8,  "GNNAggregator\n(L=2 layers)", 2.4),
        (10.6, "CommActor",                   1.6),
        (12.8, "logits",                      1.2),
    ]
    for (x, label, w) in pos:
        draw_box(ax, x, y_main, w, 0.9, label)
    draw_arrow(ax, 1.3, y_main, 2.1, y_main)
    draw_arrow(ax, 3.9, y_main, 4.4, y_main, label="m: (B,N,msg_dim)")
    draw_arrow(ax, 6.0, y_main, 6.5, y_main, label="m': same")
    draw_arrow(ax, 9.1, y_main, 9.7, y_main, label="agg: (B,N,msg_dim)")
    draw_arrow(ax, 11.5, y_main, 12.1, y_main, label="(B,N,act_dim)")
    # skip obs -> CommActor
    draw_arrow(ax, 0.8, y_main - 0.45, 0.8, 2.7, style="solid")
    draw_arrow(ax, 0.8, 2.7, 10.6, 2.7, style="solid")
    draw_arrow(ax, 10.6, 2.7, 10.6, y_main - 0.45, style="solid")
    ax.text(5.4, 2.9, "obs (skip connection)", ha="center", va="bottom",
            family=STYLE["shape_font"], fontsize=STYLE["shape_font_size"],
            color=STYLE["shape_color"])
    # Inner expansion of GNN aggregator
    ax.text(7.8, 1.7,
            "Layer 1: mean -> f_1(MLP) -> residual + LayerNorm\n"
            "Layer 2: mean -> f_2(MLP) -> residual + LayerNorm",
            ha="center", va="top",
            family=STYLE["font_family"], fontsize=8,
            color="#333333")
    # Channel-slot note (above the Channel box)
    ax.text(5.2, 5.5,
            "(constraint conditions of §3.3 instantiate this slot)",
            ha="center", va="bottom",
            family=STYLE["shape_font"], fontsize=STYLE["shape_font_size"],
            color=STYLE["shape_color"])
    # Aux output above the GNN aggregator
    draw_arrow(ax, 7.8, y_main + 0.45, 7.8, 5.5, style="dashed")
    ax.text(7.85, 5.6,
            "per-layer embeds [h^(0), h^(1), h^(2)] -> Ch.5 diagnostics",
            ha="left", va="bottom",
            family=STYLE["shape_font"], fontsize=STYLE["shape_font_size"],
            color=STYLE["shape_color"])
    ax.set_title("Figure 3.4 — GNN-comm architecture (vanilla, multi-hop)",
                 fontsize=STYLE["title_size"])
    save(fig, out_dir / "arch_gnn_comm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("figures"),
                   help="Directory to write the 8 output files into.")
    args = p.parse_args()

    print(f"[generator] writing to {args.out_dir}/")
    diagram_no_comm(args.out_dir)
    diagram_fc_comm(args.out_dir)
    diagram_attention_comm(args.out_dir)
    diagram_gnn_comm(args.out_dir)
    print("[generator] done — 4 PNG + 4 PDF generated")


if __name__ == "__main__":
    main()
