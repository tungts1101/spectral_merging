"""Robustness bar figures on CIFAR-100-C (top) and CIFAR-100-P (bottom).

  fig3_robustness        : ours vs external methods (E2-LoRA, TUNA, MOS, SLCA, EASE)
                           data: figures_data/cifar_c_per_category.csv, cifar_p_per_category.csv
                           (single seed, shared inference harness; RESULTS.md section 2)
  fig4_robustness_merge  : ours vs classical merge rules inside our framework
                           (task arithmetic / average, TIES, max-abs)
                           data: figures_data/merge_cifar_c_per_category.csv,
                           merge_cifar_p_per_category.csv (3 seeds; RESULTS.md
                           "Robustness of the merge operators")

Design (paper figure rules): Cool Gray substrate, STIX font, cobalt for ours,
terracotta reserved for the strongest competitor, cobalt screens for the rest.
Above each category the signed gap to the strongest competitor is printed, defined
so that a positive number always means we gain (accuracy: ours - ref; flip
probability: ref - ours).

CIFAR-C panels show the 15 standard corruptions; the four held-out ones (gaussian_blur,
saturate, spatter, speckle_noise) are not drawn. The legend of the external figure gives
means over the 15 drawn corruptions; the merge figure carries no legend means, since its
table in the paper reports the 19-corruption means with std.

Run:  python scripts/fig3_robustness.py  ->  assets/figures/fig3_robustness.{pdf,png}
                                             assets/figures/fig4_robustness_merge.{pdf,png}
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Drawn at the ICLR text width (5.5 in) so that font sizes below are the printed sizes.
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 6.5})
FS_TICK, FS_GAP, FS_TITLE, FS_LEG, FS_FAM = 5.5, 5.5, 6.5, 6.0, 6.0

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "assets", "figures")
os.makedirs(OUT, exist_ok=True)

PAPER = "#E9E9E5"
COBALT = "#2148B8"
TERRA = "#C65F38"
C60, C40, C25 = "#7A91D4", "#A6B5E3", "#C9D2EE"
INK = COBALT

# (csv key, label, face, edge); first = ours, second = reference competitor (terracotta)
EXTERNAL = [
    ("ours",   "ISM (ours)",  COBALT, COBALT),
    ("e2lora", "E$^2$-LoRA",  TERRA,  TERRA),
    ("tuna",   "TUNA",        C60,    COBALT),
    ("mos",    "MOS",         C40,    COBALT),
    ("slca",   "SLCA",        C25,    COBALT),
    ("ease",   "EASE",        PAPER,  COBALT),
]
MERGE = [
    ("spectral_ours",   "ISM (ours)",                COBALT, COBALT),
    ("task_arithmetic", "Task Arithmetic (average)", TERRA,  TERRA),
    ("ties",            "TIES",                      C60,    COBALT),
    ("max_abs_magmax",  "Max-abs (MagMax)",          C25,    COBALT),
]

C_FAMILIES = [
    ("Noise",   ["gaussian_noise", "shot_noise", "impulse_noise"]),
    ("Blur",    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"]),
    ("Weather", ["snow", "frost", "fog", "brightness"]),
    ("Digital", ["contrast", "elastic_transform", "pixelate", "jpeg_compression"]),
]
HELD_OUT = {"gaussian_blur", "saturate", "spatter", "speckle_noise"}
P_ORDER = ["gaussian_noise", "shot_noise", "motion_blur", "zoom_blur", "snow",
           "brightness", "translate", "rotate", "tilt", "scale"]
PRETTY = {
    "gaussian_noise": "Gaussian\nnoise", "shot_noise": "Shot\nnoise", "impulse_noise": "Impulse\nnoise",
    "defocus_blur": "Defocus\nblur", "glass_blur": "Glass\nblur", "motion_blur": "Motion\nblur",
    "zoom_blur": "Zoom\nblur", "snow": "Snow", "frost": "Frost", "fog": "Fog", "brightness": "Bright-\nness",
    "contrast": "Contrast", "elastic_transform": "Elastic", "pixelate": "Pixelate",
    "jpeg_compression": "JPEG", "translate": "Translate", "rotate": "Rotate", "tilt": "Tilt", "scale": "Scale",
}
C_CATS = [c for _, fam in C_FAMILIES for c in fam]


def load(name, key):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "figures_data", name), encoding="utf-8")))
    data = {r[key]: {m: float(r[m]) for m in r if m != key} for r in rows}
    return data


def check(c_data, p_data):
    assert set(C_CATS) == set(c_data) - HELD_OUT, "unexpected CIFAR-C categories"
    assert set(P_ORDER) == set(p_data), "unexpected CIFAR-P categories"


def bars(ax, cats, data, methods, ylim, ylabel, better_sign, decimals):
    n = len(methods)
    w = min(0.13, 0.8 / n)
    ours, ref = methods[0][0], methods[1][0]
    ax.set_facecolor(PAPER)
    xs = list(range(len(cats)))
    for i, (m, label, fc, ec) in enumerate(methods):
        off = (i - (n - 1) / 2) * w
        ax.bar([x + off for x in xs], [data[c][m] for c in cats], width=w,
               color=fc, edgecolor=ec, linewidth=0.4, zorder=3)
    for x, c in zip(xs, cats):
        gap = better_sign * (data[c][ours] - data[c][ref])
        top = max(data[c][m] for m, *_ in methods)
        ax.text(x, top + (ylim[1] - ylim[0]) * 0.025, f"{gap:+.{decimals}f}", ha="center",
                va="bottom", fontsize=FS_GAP, color=COBALT if gap >= 0 else TERRA, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([PRETTY[c] for c in cats], fontsize=FS_TICK, color=INK)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=INK)
    ax.tick_params(colors=INK, length=2)
    ax.yaxis.grid(True, color="#FFFFFF", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK)
        ax.spines[sp].set_linewidth(0.8)


def make_figure(methods, out_name, c_data, p_data, ylim_c=(30, 100), legend_means=True):
    check(c_data, p_data)
    ref_label = methods[1][1]
    c_mean = {m: sum(c_data[c][m] for c in C_CATS) / len(C_CATS) for m, *_ in methods}
    p_mean = {m: sum(p_data[c][m] for c in P_ORDER) / len(P_ORDER) for m, *_ in methods}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 3.6), gridspec_kw={"height_ratios": [1.15, 1]})
    fig.patch.set_facecolor(PAPER)

    bars(ax1, C_CATS, c_data, methods, ylim_c, "Accuracy (%)  ↑", +1, 1)
    y0, start = ylim_c[0], 0
    span = ylim_c[1] - ylim_c[0]
    for fam, members in C_FAMILIES:
        end = start + len(members) - 1
        ax1.annotate("", xy=(start - 0.42, y0 - span * 0.30), xytext=(end + 0.42, y0 - span * 0.30),
                     arrowprops=dict(arrowstyle="-", color=INK, lw=0.6), annotation_clip=False)
        ax1.text((start + end) / 2, y0 - span * 0.33, fam, ha="center", va="top", fontsize=FS_FAM,
                 color=INK, clip_on=False)
        start = end + 1
    ax1.set_title(f"CIFAR-100-C: accuracy per corruption (mean over 5 severities); "
                  f"number above each group = ours $-$ {ref_label}", fontsize=FS_TITLE, color=INK, loc="left")

    bars(ax2, P_ORDER, p_data, methods, (0, 13), "Flip probability (%)  ↓", -1, 2)
    ax2.set_title(f"CIFAR-100-P: flip probability per perturbation (lower is better); "
                  f"number above each group = {ref_label} $-$ ours", fontsize=FS_TITLE, color=INK, loc="left")

    if legend_means:
        handles = [Patch(facecolor=fc, edgecolor=ec, linewidth=0.6,
                         label=f"{label}  ({c_mean[m]:.2f} / {p_mean[m]:.2f})")
                   for m, label, fc, ec in methods]
        title = "mean CIFAR-100-C accuracy / mean CIFAR-100-P flip probability"
    else:
        handles = [Patch(facecolor=fc, edgecolor=ec, linewidth=0.6, label=label)
                   for m, label, fc, ec in methods]
        title = None
    leg = fig.legend(handles=handles, loc="lower center", ncol=2 if len(methods) == 4 else 3, frameon=False,
                     fontsize=FS_LEG, bbox_to_anchor=(0.5, 0.0), handlelength=1.0, columnspacing=0.9,
                     title=title, title_fontsize=FS_LEG)
    for t in leg.get_texts():
        t.set_color(INK)
    if title:
        leg.get_title().set_color(INK)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.95, bottom=0.185 if legend_means else 0.16, hspace=0.72)
    fig.savefig(os.path.join(OUT, f"{out_name}.pdf"), facecolor=PAPER)
    fig.savefig(os.path.join(OUT, f"{out_name}.png"), facecolor=PAPER, dpi=170)
    plt.close(fig)
    print(out_name, "| CIFAR-C(15):", {m: round(v, 2) for m, v in c_mean.items()},
          "| CIFAR-P:", {m: round(v, 2) for m, v in p_mean.items()})


if __name__ == "__main__":
    make_figure(EXTERNAL, "fig3_robustness",
                load("cifar_c_per_category.csv", "corruption"), load("cifar_p_per_category.csv", "perturbation"))
    make_figure(MERGE, "fig4_robustness_merge",
                load("merge_cifar_c_per_category.csv", "corruption"), load("merge_cifar_p_per_category.csv", "perturbation"),
                ylim_c=(45, 100), legend_means=False)
    print("saved to", os.path.abspath(OUT))
