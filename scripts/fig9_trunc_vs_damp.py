"""Figure 9 (rebuttal E8): truncation (p = 0.9) versus damped interpolation (p = 1.0),
final accuracy and forgetting against the damping alpha on the four benchmarks (3 seeds).

Data (schema of export_mc3d_full.py: variant,p,alpha,n_seeds,fa,fa_sd,aa,aa_sd,ffm,
ffm_sd,old_mean,new_acc,first5_mean):
  figures_data/mc3d_cifar224_full.csv, mc3d_cub_full.csv       parse_logs.py e8 on the E8 logs
  figures_data/mc3d_imagenetr_full.csv + mc3d_imagenetr_extra.csv  the committed cube (alpha
                                                               0.3 pair) + any new IN-R cells
  figures_data/mc3d_cars_full.csv + mc3d_cars_extra.csv        cube + the E8 headline pair (0.2)
The _extra.csv files are read when present (parse_logs.py e8 writes them for the cells a
committed cube lacks); a later file overrides the cells of an earlier one.
2 x 4 panels: top FA, bottom FFM vs alpha; p = 0.9 in cobalt (filled markers), p = 1.0 in
cobalt 60 % (open markers), thin +-sd bands; the headline alpha of each dataset is the
terracotta tick on the x axis (the chosen operating point; operator colour).  STIX font,
Cool Gray substrate, no legend (the caption carries it).

Run:  python scripts/fig9_trunc_vs_damp.py [--data-dir figures_data] [--out-dir assets/figures]
      -> assets/figures/fig9_trunc_vs_damp.{pdf,png}
Asserts every (p, alpha) cell of the two curves exists with 3 seeds before drawing
(--allow-partial downgrades that to a warning and draws the cells that exist), and prints the
p = 0.9 vs p = 1.0 table plus the large-step cells (p 0.1 / 0.5 at alpha 1.0) when present.
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAPER, COBALT, COBALT60, TERRA = "#E9E9E5", "#2148B8", "#7A91D4", "#C65F38"

# (dataset, panel title, CSV files; a later file overrides cells of an earlier one).  The
# first file is required; the _extra.csv files are optional (read when present).
DATASETS = [("cifar224", "CIFAR-100", ("mc3d_cifar224_full.csv",)),
            ("cub", "CUB", ("mc3d_cub_full.csv",)),
            ("imagenetr", "ImageNet-R", ("mc3d_imagenetr_full.csv", "mc3d_imagenetr_extra.csv")),
            ("cars", "Cars", ("mc3d_cars_full.csv", "mc3d_cars_extra.csv"))]
# Headline alpha per dataset (run_best.CONFIGS; rebuttal_common.HEADLINE_ALPHA is the
# canonical copy -- the figure scripts stay self-contained like fig5).
HEADLINE = {"cifar224": 0.15, "cub": 0.10, "imagenetr": 0.30, "cars": 0.20}
GRID_A = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
PS = (0.9, 1.0)                         # truncation, interpolation
EXTRA_P = (0.1, 0.5)                    # large-step claim cells at alpha 1.0
N_SEEDS = 3


def _get(row, *names):
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            return float(v)
    return None


def load_cells(data_dir, files, allow_partial=False):
    """{(p, alpha): {fa, fa_sd, ffm, ffm_sd, n_seeds}} of the only_A rows.  The first file
    must exist (unless allow_partial); the others (_extra.csv) are read when present."""
    cells = {}
    for i, f in enumerate(files):
        path = os.path.join(data_dir, f)
        if not os.path.isfile(path):
            if i == 0 and not allow_partial:
                raise FileNotFoundError(f"{path} is missing (parse_logs.py e8 writes the E8 CSVs)")
            print(f"  ({path} not found; " + ("skipped" if i else "panel drawn from nothing") + ")")
            continue
        with open(path, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("variant", "only_A") != "only_A":
                    continue
                key = (round(float(r["p"]), 4), round(float(r["alpha"]), 4))
                cells[key] = {"fa": _get(r, "fa", "fa_mean", "faa"), "fa_sd": _get(r, "fa_sd", "faa_sd") or 0.0,
                              "ffm": _get(r, "ffm"), "ffm_sd": _get(r, "ffm_sd") or 0.0,
                              "n_seeds": int(float(r.get("n_seeds", 0) or 0)),
                              "complete": r.get("complete", "True")}
    return cells


def _drawable(c):
    return c is not None and c["fa"] is not None and c["ffm"] is not None


def check_complete(ds, cells, allow_partial=False):
    """Every (p, alpha) of the two curves with 3 seeds; returns the alpha list.  With
    allow_partial the problems are printed as a warning instead of raised."""
    alphas = sorted(set(GRID_A) | {HEADLINE[ds]})
    bad = []
    for p in PS:
        for a in alphas:
            c = cells.get((p, a))
            if not _drawable(c):
                bad.append(f"mc3d_only_A_p{p}_a{a} missing")
            elif c["n_seeds"] != N_SEEDS or str(c["complete"]) not in ("True", "1", "true"):
                bad.append(f"mc3d_only_A_p{p}_a{a} has n_seeds={c['n_seeds']} complete={c['complete']}")
    if bad:
        msg = f"{ds}: incomplete cells: " + "; ".join(bad)
        if not allow_partial:
            raise AssertionError(msg)
        print("WARNING (drawn anyway):", msg)
    return alphas


def style_axis(ax):
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COBALT)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=COBALT, labelsize=5.8, length=2, width=0.5)
    ax.set_xticks(GRID_A)
    ax.set_xticklabels([f"{a:g}" for a in GRID_A])
    ax.set_xlim(0.03, 1.07)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "figures_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "assets", "figures"))
    ap.add_argument("--allow-partial", action="store_true",
                    help="draw the cells that exist when some are missing or have fewer than 3 seeds "
                         "(warning instead of an error)")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    def num(c, key):
        return f"{c[key]:8.2f}" if _drawable(c) else f"{'-':>8}"

    fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.6))
    fig.patch.set_facecolor(PAPER)
    for j, (ds, title, files) in enumerate(DATASETS):
        cells = load_cells(args.data_dir, files, args.allow_partial)
        alphas = check_complete(ds, cells, args.allow_partial)
        ax_fa, ax_ff = axes[0, j], axes[1, j]
        for p, color, mfc in ((0.9, COBALT, COBALT), (1.0, COBALT60, PAPER)):
            have = [a for a in alphas if _drawable(cells.get((p, a)))]   # all of them unless partial
            if not have:
                continue
            fa = [cells[(p, a)]["fa"] for a in have]
            fs = [cells[(p, a)]["fa_sd"] for a in have]
            ff = [cells[(p, a)]["ffm"] for a in have]
            fsd = [cells[(p, a)]["ffm_sd"] for a in have]
            for ax, ys, sd in ((ax_fa, fa, fs), (ax_ff, ff, fsd)):
                ax.fill_between(have, [y - s for y, s in zip(ys, sd)], [y + s for y, s in zip(ys, sd)],
                                color=color, alpha=0.18, lw=0)
                ax.plot(have, ys, color=color, lw=1.1, marker="o", ms=2.6, mfc=mfc, mec=color, mew=0.8,
                        zorder=3)
        for ax in (ax_fa, ax_ff):
            style_axis(ax)
            ax.axvline(HEADLINE[ds], ymin=0, ymax=0.09, color=TERRA, lw=1.4, zorder=4)   # operating point
        ax_fa.set_title(title, color=COBALT, fontsize=6.8, pad=3)
        ax_fa.set_xticklabels([])
        ax_ff.set_xlabel(r"damping $\alpha$", color=COBALT, fontsize=6.5, labelpad=2)

        # text summary for the rebuttal ('-' marks a missing cell under --allow-partial)
        h = HEADLINE[ds]
        print(f"{title}: alpha  FA(p0.9)  FA(p1.0)  dFA   FFM(p0.9)  FFM(p1.0)")
        for a in alphas:
            c9, c1 = cells.get((0.9, a)), cells.get((1.0, a))
            mark = " <- headline" if a == h else ""
            dfa = f"{c9['fa'] - c1['fa']:+5.2f}" if _drawable(c9) and _drawable(c1) else f"{'-':>5}"
            print(f"  {a:>5g}  {num(c9, 'fa')}  {num(c1, 'fa')}  {dfa}"
                  f"  {num(c9, 'ffm'):>9}  {num(c1, 'ffm'):>9}{mark}")
        extra = [(p, cells[(p, 1.0)]) for p in EXTRA_P if _drawable(cells.get((p, 1.0)))]
        if extra:
            print("  large-step cells at alpha 1.0: " + ", ".join(
                f"p{p}: FA {c['fa']:.2f} FFM {c['ffm']:.2f} (n={c['n_seeds']})" for p, c in extra)
                + "".join(f"; p{p}: FA {cells[(p, 1.0)]['fa']:.2f} FFM {cells[(p, 1.0)]['ffm']:.2f}"
                          for p in PS if _drawable(cells.get((p, 1.0)))))
    axes[0, 0].set_ylabel("final accuracy (%)", color=COBALT, fontsize=6.5, labelpad=2)
    axes[1, 0].set_ylabel("forgetting (%)", color=COBALT, fontsize=6.5, labelpad=2)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.92, bottom=0.14, wspace=0.32, hspace=0.18)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"fig9_trunc_vs_damp.{ext}"), facecolor=PAPER,
                    **({"dpi": 200} if ext == "png" else {}))
    print(f"wrote {os.path.join(args.out_dir, 'fig9_trunc_vs_damp.{pdf,png}')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
