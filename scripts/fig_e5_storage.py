"""Figure E5: final accuracy vs storage of the class statistics for the five
covariance modes (full / diagonal / low-rank k16 / low-rank k64 / shared) on the
four benchmarks (rebuttal A4: memory of the per-class Gaussians).

Data: figures_data/e5_storage_accuracy.csv written by `python parse_logs.py e5`
(the one producer): one row per dataset x mode with the code-derived storage
columns (lora_MB, head_MB, stats_MB, total_MB; the memory_table.py formulas) next
to the parsed cov_* cells of run_covmode.py (n_seeds, fa, fa_sd, aa, aa_sd, ffm,
ffm_sd).  The parser names the full covariance by its config value, mode
"class"; it is read here as memory_table's "full".

Four panels, x = total storage in MB (LoRA module + head + statistics, log
scale), y = FA with +- std error bars.  The full covariance (the paper) is the
solid cobalt point, the cheap variants are hollow cobalt markers labelled by
mode, and the dotted vertical line marks the LoRA module alone (33.0 MB, the
part that does not grow with the number of classes).  Every point must have 3
seeds (--allow-incomplete to draw anyway).

Run:  python scripts/fig_e5_storage.py  ->  assets/figures/fig_e5_storage.{pdf,png}
"""
import argparse
import csv
import os
import sys

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER, COBALT, TERRA = "#E9E9E5", "#2148B8", "#C65F38"
C60 = "#7A91D4"

PANELS = [("cifar224", "CIFAR-100"), ("cub", "CUB"), ("imagenetr", "ImageNet-R"), ("cars", "Cars")]
# (mode, rank) -> (label, marker)
MODES = [(("full", None), ("full covariance", "o")), (("lowrank", 64), ("low-rank 64", "D")),
         (("lowrank", 16), ("low-rank 16", "s")), (("shared", None), ("shared", "p")),
         (("diag", None), ("diagonal", "^"))]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path, allow_incomplete=False):
    """{(dataset, mode, rank): row} with floats; asserts 3 seeds per point."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        if r.get("kind", "ours") != "ours" or _f(r.get("fa")) is None:
            continue
        rank = int(float(r["rank"])) if r.get("rank") else None
        mode = {"class": "full"}.get(r["mode"], r["mode"])      # parse_logs e5 writes the config value
        n = int(_f(r.get("n_seeds")) or 0)
        if n != 3:
            msg = f"{r['dataset']} {mode} k={rank}: n_seeds={n}"
            if not allow_incomplete:
                raise AssertionError("point without 3 seeds: " + msg)
            print("WARNING (drawn anyway):", msg)
        out[(r["dataset"], mode, rank)] = {k: _f(v) for k, v in r.items()} | {"dataset": r["dataset"]}
    if not out:
        raise AssertionError(f"no result rows in {path}")
    return out


def style_axis(ax):
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=COBALT, length=2, labelsize=6, which="both")
    ax.yaxis.grid(True, color="#FFFFFF", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(COBALT)
        ax.spines[sp].set_linewidth(0.8)


def draw(data, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.95))
    fig.patch.set_facecolor(PAPER)
    lora_mb = next(iter(data.values()))["lora_MB"]
    for ax, (ds, title) in zip(axes, PANELS):
        style_axis(ax)
        pts = [(key, data[key]) for key in data if key[0] == ds]
        if not pts:
            ax.set_title(title + " (no data)", color=COBALT, fontsize=7, pad=3)
            continue
        ax.axvline(lora_mb, color=C60, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        drawn = []
        for (mode, rank), (label, marker) in MODES:
            key = (ds, mode, rank)
            if key not in data:
                continue
            r = data[key]
            full = mode == "full"
            ax.errorbar([r["total_MB"]], [r["fa"]], yerr=[r["fa_sd"] or 0.0], color=COBALT, marker=marker,
                        ms=4.2 if full else 3.6, mfc=COBALT if full else PAPER, mec=COBALT, mew=0.8, ls="none",
                        capsize=1.5, elinewidth=0.7, zorder=4)
            drawn.append((label, full, r["total_MB"], r["fa"]))
        ax.set_xscale("log")
        ys = [y for _, _, _, y in drawn]
        lo, hi = min(ys), max(ys)
        pad = max(0.6, 0.3 * (hi - lo))
        ylo, yhi = lo - pad, hi + pad
        xlo, xhi = lora_mb * 0.8, max(x for _, _, x, _ in drawn) * 1.6
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(xlo, xhi)
        ticks = [v for v in (30, 50, 100, 200, 500) if xlo <= v <= xhi]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        # labels: the full point to the left of its marker, the cheap variants to the
        # right, pushed apart vertically (axes fraction) when they would overlap
        for label, full, x, y in drawn:
            if full:
                ax.annotate(label, (x, y), xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                            fontsize=5.2, color=COBALT, zorder=5)
        cheap = sorted([(y, x, label) for label, full, x, y in drawn if not full])
        yf = [(y - ylo) / (yhi - ylo) for y, _, _ in cheap]
        gap = 0.1
        for k in range(1, len(yf)):
            yf[k] = max(yf[k], yf[k - 1] + gap)
        for (y, x, label), f in zip(cheap, yf):
            xf = (math.log(x) - math.log(xlo)) / (math.log(xhi) - math.log(xlo))
            moved = abs(f - (y - ylo) / (yhi - ylo)) > 0.02
            ax.annotate(label, (x, y), xycoords="data", xytext=(xf + 0.06, f), textcoords="axes fraction",
                        ha="left", va="center", fontsize=5.2, color=COBALT, zorder=5,
                        arrowprops=dict(arrowstyle="-", color=C60, lw=0.4, shrinkA=0, shrinkB=2) if moved else None)
        ax.set_xlabel("storage (MB)", color=COBALT, fontsize=6.5, labelpad=2)
        ax.set_title(title, color=COBALT, fontsize=7, pad=3)
    axes[0].set_ylabel("final accuracy FA (%)", color=COBALT, fontsize=6.5, labelpad=2)
    handles = [Line2D([], [], color=COBALT, marker="o", ms=4.2, ls="none", label="full covariance (paper)"),
               Line2D([], [], color=COBALT, marker="s", ms=3.6, mfc=PAPER, ls="none", label="cheaper statistics (labelled)"),
               Line2D([], [], color=C60, lw=0.8, ls=(0, (1, 1.5)), label=f"LoRA module alone ({lora_mb:.0f} MB)")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6,
                     bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(COBALT)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.33, wspace=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + ".pdf", facecolor=PAPER)
    fig.savefig(out_path + ".png", facecolor=PAPER, dpi=200)
    plt.close(fig)
    print("saved", out_path + ".{pdf,png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "figures_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "assets", "figures"))
    ap.add_argument("--allow-incomplete", action="store_true", help="draw points with fewer than 3 seeds")
    args = ap.parse_args(argv)
    data = load(os.path.join(args.data_dir, "e5_storage_accuracy.csv"), args.allow_incomplete)
    for key in sorted(data, key=lambda k: (k[0], k[1], k[2] or 0)):
        r = data[key]
        print(f"{key[0]:<10} {key[1]:<8} k={key[2] or '-':<4} total {r['total_MB']:8.2f} MB  FA {r['fa']:.2f} +- {r['fa_sd'] or 0:.2f}")
    draw(data, os.path.join(args.out_dir, "fig_e5_storage"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
