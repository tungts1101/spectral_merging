"""Figure E9: estimation error of the old-class statistics vs task index, with
and without the drift correction, next to the one-step identity and oracle
references (rebuttal A6 / xkNt-Q2: does the map fitted on the new task's data
reduce the old-class error, how far is it from an oracle, does it accumulate).

Data: figures_data/e9_driftdiag_<ds>_<split>.csv written by `python parse_logs.py
e9` (the one collector: seed means under the bare metric names and <metric>_sd
std of the per-task rows that main.Learner.drift_diagnostics appends to
logs_main/<ds>/driftdiag_<split>.csv), group 'all' = every old class.  Top row:
relative error of the class means
||mu_est - mu_ref|| / ||mu_ref|| under the deployed extractor M_t; bottom row:
relative Frobenius error of the covariances.  Curves: statistics never
corrected (raw, cobalt 40 %), the stored statistics after the transport
(corrected, solid cobalt), the one-step identity map (the true M_{t-1}
statistics used as they are, cobalt 60 %, dash-dot) and the one-step oracle
map fitted on the old-class pairs themselves (terracotta, dashed);
--with-onestep-w adds the one-step fitted map (dotted cobalt).  Bands = +- std
over seeds; every point must have 3 seeds (--allow-incomplete to draw anyway).

Run:  python scripts/fig_e9_driftdiag.py [--split t10]
      ->  assets/figures/fig_e9_driftdiag[_<split>].{pdf,png}
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER, COBALT, TERRA = "#E9E9E5", "#2148B8", "#C65F38"
C60, C40 = "#7A91D4", "#A6B5E3"

DATASETS = [("cifar224", "CIFAR-100"), ("cub", "CUB"), ("imagenetr", "ImageNet-R"), ("cars", "Cars")]
# (column stem, label, colour, linestyle); the stem is completed with the metric
CURVES = [("{m}_raw", "never corrected", C40, "-"),
          ("os_{m}_I", "one-step identity", C60, "-."),
          ("os_{m}_oracle", "one-step oracle $W$", TERRA, "--"),
          ("{m}_after", "corrected (ISM)", COBALT, "-")]
ONESTEP_W = ("os_{m}_W", "one-step fitted $W$", COBALT, ":")
ROWS = [("mean_rel", "rel. error of class means"), ("cov_relF", "rel. Frobenius error of covariances")]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(path, allow_incomplete=False):
    """Rows of group 'all' sorted by task, as dicts of floats; asserts 3 seeds."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["group"] == "all"]
    rows.sort(key=lambda r: int(float(r["task"])))
    bad = [int(float(r["task"])) for r in rows if int(float(r["n_seeds"])) != 3]
    if bad:
        msg = f"{os.path.basename(path)}: tasks {bad} have n_seeds != 3"
        if not allow_incomplete:
            raise AssertionError(msg)
        print("WARNING (drawn anyway):", msg)
    return rows


def style_axis(ax):
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=COBALT, length=2, labelsize=6)
    ax.yaxis.grid(True, color="#FFFFFF", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(COBALT)
        ax.spines[sp].set_linewidth(0.8)


def draw(data, curves, out_path):
    n = len(data)
    fig, axes = plt.subplots(2, n, figsize=(5.5, 3.0), squeeze=False)
    fig.patch.set_facecolor(PAPER)
    for j, (ds, title, rows) in enumerate(data):
        tasks = np.array([float(r["task"]) for r in rows])
        for i, (metric, ylabel) in enumerate(ROWS):
            ax = axes[i][j]
            style_axis(ax)
            for stem, label, colour, ls in curves:
                col = stem.format(m=metric)
                y = np.array([_f(r.get(col, "")) for r in rows])
                sd = np.array([_f(r.get(col + "_sd", "")) for r in rows])
                ok = np.isfinite(y)
                if not ok.any():
                    continue
                ax.plot(tasks[ok], y[ok], color=colour, ls=ls, lw=1.2 if colour == COBALT and ls == "-" else 0.9,
                        marker="o", ms=1.8, zorder=4 if colour == COBALT else 3)
                band = np.where(np.isfinite(sd), sd, 0.0)
                ax.fill_between(tasks[ok], (y - band)[ok], (y + band)[ok], color=colour, alpha=0.18, lw=0, zorder=2)
            ax.set_xlim(tasks.min() - 0.3, tasks.max() + 0.3)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            if i == 0:
                ax.set_title(title, color=COBALT, fontsize=7, pad=3)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("task $t$", color=COBALT, fontsize=6.5, labelpad=2)
            if j == 0:
                ax.set_ylabel(ylabel, color=COBALT, fontsize=6.2, labelpad=2)
    handles = [Line2D([], [], color=c, ls=ls, lw=1.2 if c == COBALT and ls == "-" else 0.9, marker="o", ms=1.8,
                      label=lab) for _, lab, c, ls in curves]
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(curves), frameon=False, fontsize=6,
                     bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(COBALT)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.2, wspace=0.32, hspace=0.18)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + ".pdf", facecolor=PAPER)
    fig.savefig(out_path + ".png", facecolor=PAPER, dpi=200)
    plt.close(fig)
    print("saved", out_path + ".{pdf,png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "figures_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "assets", "figures"))
    ap.add_argument("--split", default="t10", help="t10 (default), t20 or t50 (file suffix for non-t10)")
    ap.add_argument("--datasets", nargs="+", default=[d for d, _ in DATASETS])
    ap.add_argument("--with-onestep-w", action="store_true", help="add the one-step fitted-W curve")
    ap.add_argument("--allow-incomplete", action="store_true", help="draw points with fewer than 3 seeds")
    args = ap.parse_args(argv)

    data = []
    for ds, title in DATASETS:
        if ds not in args.datasets:
            continue
        path = os.path.join(args.data_dir, f"e9_driftdiag_{ds}_{args.split}.csv")
        if not os.path.isfile(path):
            print(f"(no {path}; panel skipped)")
            continue
        rows = load(path, args.allow_incomplete)
        last = rows[-1]
        print(f"{ds} {args.split}: {len(rows)} tasks; last task mean_rel raw {_f(last['mean_rel_raw']):.4f} -> "
              f"after {_f(last['mean_rel_after']):.4f}; one-step I / W / oracle "
              f"{_f(last.get('os_mean_rel_I', '')):.4f} / {_f(last.get('os_mean_rel_W', '')):.4f} / "
              f"{_f(last.get('os_mean_rel_oracle', '')):.4f}")
        data.append((ds, title, rows))
    if not data:
        raise SystemExit("no e9_driftdiag_<ds>_<split>.csv found; run `python parse_logs.py e9` first")
    curves = CURVES[:3] + ([ONESTEP_W] if args.with_onestep_w else []) + CURVES[3:]
    suffix = "" if args.split == "t10" else f"_{args.split}"
    draw(data, curves, os.path.join(args.out_dir, "fig_e9_driftdiag" + suffix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
