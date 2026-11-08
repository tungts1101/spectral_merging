"""Draft of Figure 2: drift compensation after a merge.

Colour logic shared by every diagram in the paper (see CLAUDE.md):
  substrate         Cool Gray  #E9E9E5
  cobalt, solid     model / backbone states (M_t, the head h_t)
  cobalt, outline   data, features, statistics
  terracotta        the operation this phase performs and its parameters
                    (Fig. 1: merge junction; here: drift map W)

Run:  python scripts/fig2_drift.py  ->  assets/figures/fig2_drift.pdf/.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
})

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "figures")
os.makedirs(OUT, exist_ok=True)

PAPER = "#E9E9E5"
COBALT = "#2148B8"
TERRA = "#C65F38"
TERRA_20 = "#F1D7CC"

BH = 0.52
Y_TOP, Y_MID, Y_BOT = 2.25, 1.40, 0.55

fig, ax = plt.subplots(figsize=(10.4, 2.9))
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)
ax.set_xlim(-0.35, 10.85)
ax.set_ylim(-0.12, 2.92)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, text, kind="paper", w=0.84, fs=12):
    if kind == "solid":
        fc, tc, ec, lw = COBALT, PAPER, COBALT, 0
    elif kind == "accent":
        fc, tc, ec, lw = TERRA_20, TERRA, TERRA, 2.2
    else:
        fc, tc, ec, lw = PAPER, COBALT, COBALT, 1.9
    ax.add_patch(FancyBboxPatch((x - w / 2, y - BH / 2), w, BH,
                                boxstyle="round,pad=0.02,rounding_size=0.11",
                                ec=ec, fc=fc, lw=lw, zorder=3))
    ax.text(x, y, text, ha="center", va="center_baseline", fontsize=fs, color=tc, zorder=4)


def arrow(x0, y0, x1, y1, color, lw=1.7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=2))


def line(x0, y0, x1, y1, color, lw=1.7):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="round", zorder=2)


def note(x, y, text, color=COBALT, ha="center", va="bottom", fs=9.5):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color)


# ------------------------------------------------------------ 1. paired features
x_data, x_ext, x_feat = 0.5, 2.5, 4.5
box(x_data, Y_MID, r"$x$", w=0.62)
note(x_data, Y_MID - BH / 2 - 0.09, "current-task data", va="top")

x_split = x_data + 0.31 + 0.5
line(x_data + 0.31, Y_MID, x_split, Y_MID, COBALT)
line(x_split, Y_BOT, x_split, Y_TOP, COBALT)
arrow(x_split, Y_TOP, x_ext - 0.42, Y_TOP, COBALT)
arrow(x_split, Y_BOT, x_ext - 0.42, Y_BOT, COBALT)

box(x_ext, Y_TOP, r"$M_{t-1}$", kind="solid")
box(x_ext, Y_BOT, r"$M_t$", kind="solid")
note(x_ext, Y_TOP + BH / 2 + 0.09, "previous extractor")
note(x_ext, Y_BOT - BH / 2 - 0.09, "current extractor", va="top")

arrow(x_ext + 0.42, Y_TOP, x_feat - 0.55, Y_TOP, COBALT)
arrow(x_ext + 0.42, Y_BOT, x_feat - 0.55, Y_BOT, COBALT)
box(x_feat, Y_TOP, r"$F_{t-1}$", w=1.10)
box(x_feat, Y_BOT, r"$F_t$", w=1.10)
note(x_feat, Y_TOP + BH / 2 + 0.09, "features")
note(x_feat, Y_BOT - BH / 2 - 0.09, "features", va="top")

# ------------------------------------------------------------ 2. drift map W
x_join = x_feat + 0.55 + 0.5
x_W = 7.1
line(x_feat + 0.55, Y_TOP, x_join, Y_TOP, COBALT)
line(x_feat + 0.55, Y_BOT, x_join, Y_BOT, COBALT)
line(x_join, Y_BOT, x_join, Y_TOP, COBALT)
arrow(x_join, Y_MID, x_W - 0.55, Y_MID, COBALT)
note((x_join + x_W - 0.55) / 2, Y_MID + 0.09, "fit")

box(x_W, Y_MID, r"$T_t$", kind="accent", w=1.10)
note(x_W, Y_MID - BH / 2 - 0.10,
     r"$\min_T \|F_{t-1}T^{\!\top} - F_t\|^2 + \lambda\|T - I\|^2$",
     color=TERRA, va="top", fs=9.5)

box(x_W, Y_TOP, r"$\mathcal{N}(\mu_c,\,\Sigma_c)$", w=1.55)
note(x_W, Y_TOP + BH / 2 + 0.09, "old-class statistics (stale)")
arrow(x_W, Y_TOP - BH / 2, x_W, Y_MID + BH / 2, COBALT)
note(x_W + 0.10, (Y_TOP + Y_MID) / 2, "apply", ha="left", va="center")

# ------------------------------------------------------------ 3. transported statistics
x_st = 9.75
W_ST = 2.10
arrow(x_W + 0.55, Y_MID, x_st - W_ST / 2, Y_MID, COBALT)
box(x_st, Y_MID, r"$\mathcal{N}(T_t\mu_c,\,T_t\Sigma_c T_t^{\!\top})$", w=W_ST, fs=11)
note(x_st, Y_MID + BH / 2 + 0.09, "old classes, transported")

fig.savefig(os.path.join(OUT, "fig2_drift.pdf"), bbox_inches="tight", pad_inches=0.03, facecolor=PAPER)
fig.savefig(os.path.join(OUT, "fig2_drift.png"), bbox_inches="tight", pad_inches=0.03, dpi=170, facecolor=PAPER)
print("saved to", os.path.abspath(OUT))
