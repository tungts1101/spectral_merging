"""Figure 5: merge sweep on ImageNet-R, variant x p x alpha (final accuracy, 3 seeds).

Data: figures_data/mc3d_imagenetr.csv (RESULTS.md "Merge 3-D ablation").
Four heatmaps, one per truncation variant; rows = energy fraction p, columns = damping alpha.
Cobalt sequential colour scale on the Cool Gray substrate, STIX font, cell values printed.
The best cell of each panel is outlined in terracotta.

Run:  python scripts/fig5_merge_sweep.py  ->  assets/figures/fig5_merge_sweep.{pdf,png}
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "assets", "figures")
os.makedirs(OUT, exist_ok=True)

PAPER, COBALT, TERRA = "#E9E9E5", "#2148B8", "#C65F38"
CMAP = LinearSegmentedColormap.from_list("cobalt", [PAPER, "#C9D2EE", "#7A91D4", COBALT])

VARIANTS = [("only_A", "Down-projection only (default)"), ("only_B", "Up-projection only"),
            ("separate", "Both factors separately"), ("combine", "Product $\\mathbf{BA}$")]
GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

rows = list(csv.DictReader(open(os.path.join(ROOT, "figures_data", "mc3d_imagenetr.csv"), encoding="utf-8")))
assert all(r["complete"] == "True" and r["n_seeds"] == "3" for r in rows), "incomplete cells"
data = {(r["variant"], float(r["p"]), float(r["alpha"])): float(r["fa_mean"]) for r in rows}
assert len(data) == 4 * 36, len(data)

vmin = min(data.values())
vmax = max(data.values())

fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.95))
fig.patch.set_facecolor(PAPER)
for ax, (key, title) in zip(axes, VARIANTS):
    M = np.array([[data[(key, p, a)] for a in GRID] for p in GRID])
    ax.set_facecolor(PAPER)
    ax.imshow(M, cmap=CMAP, vmin=vmin, vmax=vmax, aspect="equal")
    for i in range(6):
        for j in range(6):
            v = M[i, j]
            frac = (v - vmin) / (vmax - vmin)
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.6,
                    color=PAPER if frac > 0.6 else COBALT)
    bi, bj = np.unravel_index(M.argmax(), M.shape)
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=TERRA, lw=1.4, zorder=5))
    ax.set_xticks(range(6)); ax.set_xticklabels([str(a) for a in GRID], color=COBALT, fontsize=6)
    ax.set_yticks(range(6)); ax.set_yticklabels([str(p) for p in GRID], color=COBALT, fontsize=6)
    ax.set_xlabel(r"damping $\alpha$", color=COBALT, fontsize=6.5, labelpad=2)
    ax.set_title(title, color=COBALT, fontsize=6.8, pad=3)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_ylabel(r"energy fraction $p$", color=COBALT, fontsize=6.5, labelpad=2)
for ax in axes[1:]:
    ax.set_yticklabels([])
fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.17, wspace=0.12)
fig.savefig(os.path.join(OUT, "fig5_merge_sweep.pdf"), facecolor=PAPER)
fig.savefig(os.path.join(OUT, "fig5_merge_sweep.png"), facecolor=PAPER, dpi=200)
best = max(data.items(), key=lambda kv: kv[1])
print("range", vmin, vmax, "| best", best, "| no-merge corner only_A:", data[("only_A", 1.0, 1.0)])
