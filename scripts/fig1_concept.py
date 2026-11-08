"""Draft of Figure 1: incremental spectral merging.

Orthogonal layout, two-ink print logic (mono-color design system):
  substrate  Cool Gray #E9E9E5 (technology / research)
  dominant   Cobalt    #2148B8  -> states, chain, all type          (~80 % of ink)
  accent     Terracotta #C65F38 -> the merge event only              (~20 % of ink)
Ink density carries hierarchy: fine-tuned states are paper knockouts with a cobalt
outline; deployed merged states are solid cobalt with paper-coloured labels.

Run:  python scripts/fig1_concept.py            -> assets/figures/fig1_concept.pdf/.png
      python scripts/fig1_concept.py --compare  -> assets/figures/fig1_concept_compare.png
                                                   (solid vs outlined M boxes, side by side)
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
})

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "figures")
os.makedirs(OUT, exist_ok=True)

PAPER = "#E9E9E5"        # substrate: Cool Gray
COBALT = "#2148B8"       # dominant ink
COBALT_60 = "#7A91D4"    # 60 % screen of cobalt (chain arrows)
COBALT_20 = "#D3DBF1"    # 20 % screen (fill of outlined M variant)
TERRA = "#C65F38"        # accent ink

T = 4
D = 1.30
Y_FT, Y_M = 1.95, 0.62
BW, BH = 0.84, 0.52
R = 0.24
LABELS = ["0", "1", "2", "3", "T"]

X_W = [2 * D * t for t in range(T + 1)]          # W_0 .. W_T, even spacing
X_OP = {t: X_W[t] for t in range(2, T + 1)}
X_M = {1: X_W[1]}
for t in range(2, T + 1):
    X_M[t] = X_OP[t] + D


def draw(ax, solid_M=True):
    ax.set_facecolor(PAPER)
    ax.set_xlim(X_W[0] - 0.55, X_M[T] + 0.50)
    ax.set_ylim(-0.02, 2.62)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x, y, text, kind):
        if kind == "solid":
            fc, tc, lw = COBALT, PAPER, 0
        elif kind == "tint":
            fc, tc, lw = COBALT_20, COBALT, 2.6
        else:
            fc, tc, lw = PAPER, COBALT, 1.9
        ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                                    boxstyle="round,pad=0.02,rounding_size=0.11",
                                    ec=COBALT, fc=fc, lw=lw, zorder=3))
        ax.text(x, y, text, ha="center", va="center_baseline", fontsize=12.5,
                color=tc, zorder=4)

    def arrow(x0, y0, x1, y1, color, lw=1.6):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, color=color, lw=lw,
                                     shrinkA=0, shrinkB=0, zorder=2))

    def junction(x, y):
        ax.add_patch(Circle((x, y), R, fc=PAPER, ec=TERRA, lw=2.2, zorder=5))
        ax.plot([x - R, x + R], [y, y], color=TERRA, lw=1.7, zorder=6)
        ax.plot([x, x], [y - R, y + R], color=TERRA, lw=1.7, zorder=6)

    # top row: sequential fine-tuning (paper knockouts)
    for t in range(T + 1):
        box(X_W[t], Y_FT, rf"$W_{{{LABELS[t]}}}$", "paper")
        if t < T:
            arrow(X_W[t] + BW / 2, Y_FT, X_W[t + 1] - BW / 2, Y_FT, COBALT_60)
            ax.text((X_W[t] + X_W[t + 1]) / 2, Y_FT + 0.09,
                    f"task {LABELS[t + 1]}", ha="center", va="bottom", fontsize=10, color=COBALT)
    ax.text(X_W[0], Y_FT - BH / 2 - 0.09, "initial model", ha="center", va="top",
            fontsize=9.5, color=COBALT)

    # bottom row: merged, deployed states
    for t in range(1, T + 1):
        box(X_M[t], Y_M, rf"$M_{{{LABELS[t]}}}$", "solid" if solid_M else "tint")
        if t == 1:
            arrow(X_W[1], Y_FT - BH / 2, X_M[1], Y_M + BH / 2, COBALT)
            ax.text(X_M[1] + 0.12, (Y_FT + Y_M) / 2, r"$M_1 = W_1$", ha="left", va="center",
                    fontsize=10, color=COBALT)
        else:
            xo = X_OP[t]
            junction(xo, Y_M)
            arrow(X_M[t - 1] + BW / 2, Y_M, xo - R, Y_M, COBALT, lw=2.0)
            arrow(X_W[t], Y_FT - BH / 2, xo, Y_M + R, COBALT, lw=1.9)
            arrow(xo + R, Y_M, X_M[t] - BW / 2, Y_M, COBALT, lw=2.0)
            ax.text(xo, Y_M - R - 0.10,
                    rf"$\alpha\,\mathrm{{trunc}}_p(W_{{{LABELS[t]}}} - M_{{{LABELS[t-1]}}})$",
                    ha="center", va="top", fontsize=10, color=TERRA)


def save(fig, name):
    fig.patch.set_facecolor(PAPER)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 170})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.03, facecolor=fig.get_facecolor(), **kw)


if __name__ == "__main__":
    if "--compare" in sys.argv:
        fig, axes = plt.subplots(1, 2, figsize=(19, 3.2))
        draw(axes[0], solid_M=True)
        draw(axes[1], solid_M=False)
        axes[0].set_title("solid", fontsize=11, color=COBALT, loc="left")
        axes[1].set_title("outlined", fontsize=11, color=COBALT, loc="left")
        fig.subplots_adjust(wspace=0.06)
        save(fig, "fig1_concept_compare")
    else:
        fig, ax = plt.subplots(figsize=(9.6, 2.5))
        draw(ax, solid_M=True)
        save(fig, "fig1_concept")
    print("saved to", os.path.abspath(OUT))
