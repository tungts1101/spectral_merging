"""Figure 7 (rebuttal E12): the other merge rules under an equal per-dataset tuning
budget, final accuracy against the merge coefficient on the four benchmarks (3 seeds).

Data: figures_data/e12_merge_grid.csv (parse_logs.py e12), one row per (dataset, rule,
coefficient) with n_seeds, fa, fa_sd, aa, aa_sd, ffm, ffm_sd, old_mean, new_acc,
first5_mean; the column names ds|dataset, rule|train_merge|method, coef|train_merge_coef
and fa|faa are all accepted.  ISM reference per dataset: a row whose rule is 'ism' (or
'pspectral') in the same CSV when the grid ran with --with-ism, else the headline only_A
p = 0.9 cell of figures_data/mc3d_<ds>_full.csv (mc3d_cars_extra.csv holds the Cars 0.2 pair).
1 x 4 panels, FA vs coefficient: classical rules (Task Arithmetic, TIES, max-abs) in full
cobalt, published rules (DARE, DELLA, KnOTS, Breadcrumbs) in cobalt 60 %, distinct markers,
direct labels to the right of the last panel; ISM as a full-ink horizontal line; each rule's
best cell is ringed in terracotta.  STIX font, Cool Gray substrate, no legend (caption).

Run:  python scripts/fig7_merge_grid.py [--data-dir figures_data] [--out-dir assets/figures]
      [--allow-partial]  -> assets/figures/fig7_merge_grid.{pdf,png}
Asserts the full 7 rules x 6 coefficients x 4 datasets grid with 3 seeds per cell (and an
ISM reference per dataset) unless --allow-partial, which draws what exists while runs are
in flight.
"""
import argparse
import csv
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAPER, COBALT, COBALT60, TERRA = "#E9E9E5", "#2148B8", "#7A91D4", "#C65F38"

DATASETS = [("cifar224", "CIFAR-100"), ("cub", "CUB"), ("imagenetr", "ImageNet-R"), ("cars", "Cars")]
COEFS = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2]                    # run_e12_merge_grid.COEFS
# rule -> (ink, marker, label); classical rules full ink, published rules cobalt 60 %
STYLE = OrderedDict([
    ("avg", (COBALT, "o", "Task Arithmetic")),
    ("ties", (COBALT, "s", "TIES")),
    ("max_abs", (COBALT, "^", "max-abs")),
    ("dare", (COBALT60, "o", "DARE")),
    ("della", (COBALT60, "s", "DELLA")),
    ("knots", (COBALT60, "^", "KnOTS")),
    ("breadcrumbs", (COBALT60, "D", "Breadcrumbs")),
])
ISM_RULES = ("ism", "pspectral")
# Headline alpha per dataset (run_best.CONFIGS; rebuttal_common.HEADLINE_ALPHA is canonical)
HEADLINE = {"cifar224": 0.15, "cub": 0.10, "imagenetr": 0.30, "cars": 0.20}
CUBES = {"cifar224": ("mc3d_cifar224_full.csv",), "cub": ("mc3d_cub_full.csv",),
         "imagenetr": ("mc3d_imagenetr_full.csv",), "cars": ("mc3d_cars_full.csv", "mc3d_cars_extra.csv")}
N_SEEDS = 3


def _get(row, *names):
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            return v
    return None


def load_grid(path):
    """({(ds, rule, coef): {fa, fa_sd, n_seeds}}, {ds: (fa, fa_sd)} ISM rows)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is missing (parse_logs.py e12 writes it)")
    cells, ism = {}, {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            ds = _get(r, "ds", "dataset")
            rule = _get(r, "rule", "train_merge", "method")
            fa = _get(r, "fa", "faa", "fa_mean")
            if ds is None or rule is None or fa is None:
                continue
            rec = {"fa": float(fa), "fa_sd": float(_get(r, "fa_sd", "faa_sd") or 0.0),
                   "n_seeds": int(float(_get(r, "n_seeds") or 0))}
            if rule in ISM_RULES:
                ism[ds] = (rec["fa"], rec["fa_sd"])
                continue
            coef = _get(r, "coef", "train_merge_coef", "c")
            if coef is None:
                continue
            cells[(ds, rule, round(float(coef), 4))] = rec
    return cells, ism


def ism_from_cubes(data_dir, ds):
    """FA of the headline only_A p = 0.9 cell from the mc3d CSVs, or None."""
    found = None
    for f in CUBES[ds]:
        path = os.path.join(data_dir, f)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if (r.get("variant", "only_A") == "only_A" and round(float(r["p"]), 4) == 0.9
                        and round(float(r["alpha"]), 4) == HEADLINE[ds]):
                    found = (float(_get(r, "fa", "fa_mean")), float(_get(r, "fa_sd") or 0.0))
    return found


def style_axis(ax):
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COBALT)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=COBALT, labelsize=5.6, length=2, width=0.5)
    ax.set_xticks(COEFS)
    # 0.85 keeps its tick but no label: "0.7" and "0.85" collide at this panel width
    ax.set_xticklabels(["" if c == 0.85 else f"{c:g}" for c in COEFS])
    ax.set_xlim(0.22, 1.28)


def spread(values, gap):
    """Shift label positions apart so neighbours are at least `gap` apart."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        if out[hi] - out[lo] < gap:
            out[hi] = out[lo] + gap
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "figures_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "assets", "figures"))
    ap.add_argument("--allow-partial", action="store_true",
                    help="draw the cells that exist instead of asserting the full grid")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    cells, ism = load_grid(os.path.join(args.data_dir, "e12_merge_grid.csv"))
    for ds, _ in DATASETS:
        if ds not in ism:
            ref = ism_from_cubes(args.data_dir, ds)
            if ref:
                ism[ds] = ref
    problems = []
    for ds, _ in DATASETS:
        for rule in STYLE:
            for c in COEFS:
                rec = cells.get((ds, rule, c))
                if rec is None:
                    problems.append(f"{ds} e12_{rule}_c{c} missing")
                elif rec["n_seeds"] != N_SEEDS:
                    problems.append(f"{ds} e12_{rule}_c{c} n_seeds={rec['n_seeds']}")
        if ds not in ism:
            problems.append(f"{ds}: no ISM reference (rule 'ism' row or mc3d cube cell)")
    if problems:
        msg = f"{len(problems)} incomplete cell(s): " + "; ".join(problems[:12]) + (" ..." if len(problems) > 12 else "")
        if not args.allow_partial:
            raise AssertionError(msg)
        print("WARNING (--allow-partial):", msg)

    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.9))
    fig.patch.set_facecolor(PAPER)
    for ax, (ds, title) in zip(axes, DATASETS):
        style_axis(ax)
        ends = []                                   # (label, y, colour) for the direct labels
        for rule, (ink, marker, label) in STYLE.items():
            pts = [(c, cells[(ds, rule, c)]) for c in COEFS if (ds, rule, c) in cells]
            if not pts:
                continue
            xs = [c for c, _ in pts]
            ys = [r["fa"] for _, r in pts]
            ax.plot(xs, ys, color=ink, lw=0.9, marker=marker, ms=2.4, mfc=ink, mec=ink, mew=0.6, zorder=3)
            bi = max(range(len(ys)), key=lambda i: ys[i])
            ax.plot([xs[bi]], [ys[bi]], ls="none", marker="o", ms=6.2, mfc="none", mec=TERRA, mew=1.0, zorder=5)
            ends.append((label, ys[-1], ink))
            print(f"{title:<10} {label:<16} best c={xs[bi]:<5g} FA {ys[bi]:.2f}"
                  + (f"  (ISM {ism[ds][0]:.2f}, gap {ys[bi] - ism[ds][0]:+.2f})" if ds in ism else ""))
        if ds in ism:
            ax.axhline(ism[ds][0], color=COBALT, lw=1.0, zorder=2)
            ends.append(("ISM", ism[ds][0], COBALT))
        ax.set_title(title, color=COBALT, fontsize=6.8, pad=3)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.06 * (hi - lo))          # headroom above the ISM line
        if ax is axes[-1] and ends:
            lo, hi = ax.get_ylim()
            ys = spread([y for _, y, _ in ends], 0.055 * (hi - lo))
            for (label, _, ink), y in zip(ends, ys):
                ax.annotate(label, xy=(1.02, y), xycoords=("axes fraction", "data"),
                            color=ink, fontsize=5.2, ha="left", va="center_baseline",
                            annotation_clip=False)
    axes[0].set_ylabel("final accuracy (%)", color=COBALT, fontsize=6.5, labelpad=2)
    fig.text(0.5, 0.015, r"merge coefficient $c$", color=COBALT, fontsize=6.5, ha="center", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.86, top=0.90, bottom=0.2, wspace=0.34)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"fig7_merge_grid.{ext}"), facecolor=PAPER,
                    **({"dpi": 200} if ext == "png" else {}))
    print(f"wrote {os.path.join(args.out_dir, 'fig7_merge_grid.{pdf,png}')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
