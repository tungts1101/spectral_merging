"""Figure 8: what the discarded directions carry (rebuttal E6; 3 seeds).

Data: figures_data/e6_subspace_<ds>.csv (analyze_subspace.py), one panel column per
benchmark. Row 1: overlap (mean cos^2 of the principal angles) of the kept (cobalt,
solid) and discarded (cobalt 60 %, solid) right-singular directions of the lora_A
update with the old-task layer-input subspace, the same against the new task's
subspace (dashed), and the chance level m / d (dotted), vs task. Row 2: response
share of the discarded directions divided by their weight-energy share, on old
(cobalt) and new (cobalt 60 %) inputs, vs task, with the reference line at 1.
Means over the 36 layers and 3 seeds; the band is the std over seeds of the
layer mean. The caption carries the explanation; no legend text inside.

Run:  python scripts/fig8_subspace.py  ->  assets/figures/fig8_subspace.{pdf,png}
"""
import argparse
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAPER, COBALT, COBALT60, TERRA = "#E9E9E5", "#2148B8", "#7A91D4", "#C65F38"
DATASETS = [("cifar224", "CIFAR-100"), ("cub", "CUB"), ("imagenetr", "ImageNet-R"), ("cars", "Cars")]
N_SEEDS, N_TASKS, N_LAYERS = 3, 9, 36


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def seed_task_means(rows, col):
    """(tasks, mean over seeds of the layer mean, std over seeds) of column `col`."""
    tasks = sorted({int(r["task"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    per = np.full((len(seeds), len(tasks)), np.nan)
    for i, s in enumerate(seeds):
        for j, t in enumerate(tasks):
            vals = [float(r[col]) for r in rows
                    if int(r["seed"]) == s and int(r["task"]) == t and r.get(col, "") != ""]
            if vals:
                per[i, j] = np.mean(vals)
    return np.array(tasks), np.nanmean(per, axis=0), np.nanstd(per, axis=0)


def style(ax):
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=COBALT, labelsize=5.5, length=2, width=0.5)
    for sp in ax.spines.values():
        sp.set_color(COBALT)
        sp.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "figures_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "assets", "figures"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--datasets", nargs="+", default=[d for d, _ in DATASETS])
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    labels = dict(DATASETS)
    datasets = [(d, labels.get(d, d)) for d in args.datasets]

    data = []
    for ds, title in datasets:
        rows = read_csv(os.path.join(args.data_dir, f"e6_subspace_{ds}{args.tag}.csv"))
        seeds = sorted({int(r["seed"]) for r in rows})
        tasks = sorted({int(r["task"]) for r in rows})
        complete = len(seeds) == N_SEEDS and len(tasks) == N_TASKS and len(rows) == N_SEEDS * N_TASKS * N_LAYERS
        if not complete and not args.allow_partial:
            raise SystemExit(f"{ds}: incomplete data (seeds {seeds}, tasks {tasks}, {len(rows)} rows); "
                             f"pass --allow-partial to draw anyway")
        data.append((ds, title, rows))

    n = len(data)
    fig, axes = plt.subplots(2, n, figsize=(5.5, 2.9), squeeze=False)
    fig.patch.set_facecolor(PAPER)
    for col, (ds, title, rows) in enumerate(data):
        ax = axes[0, col]
        style(ax)
        ax.set_title(title, color=COBALT, fontsize=6.8, pad=3)
        for colname, color, ls in (("overlap_keep_old", COBALT, "-"), ("overlap_drop_old", COBALT60, "-"),
                                   ("overlap_keep_new", COBALT, (0, (3, 1.5))),
                                   ("overlap_drop_new", COBALT60, (0, (3, 1.5)))):
            t, m, s = seed_task_means(rows, colname)
            ax.plot(t, m, color=color, ls=ls, lw=0.9)
            ax.fill_between(t, m - s, m + s, color=color, alpha=0.18, lw=0)
        for colname in ("chance_old", "chance_new"):
            t, m, _ = seed_task_means(rows, colname)
            ax.plot(t, m, color=COBALT, ls=(0, (1, 1.5)), lw=0.7)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(1, N_TASKS + 1))
        ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("overlap with input subspace", color=COBALT, fontsize=6.3, labelpad=2)
        else:
            ax.set_yticklabels([])

        ax2 = axes[1, col]
        style(ax2)
        for colname, color in (("ratio_old", COBALT), ("ratio_new", COBALT60)):
            t, m, s = seed_task_means(rows, colname)
            ax2.plot(t, m, color=color, lw=0.9)
            ax2.fill_between(t, m - s, m + s, color=color, alpha=0.18, lw=0)
        ax2.axhline(1.0, color=TERRA, lw=0.6, ls=(0, (2, 1.5)))
        ax2.set_xticks(range(1, N_TASKS + 1))
        ax2.set_xticklabels([str(t) for t in range(1, N_TASKS + 1)], fontsize=5.5)
        ax2.set_xlabel("task $t$", color=COBALT, fontsize=6.5, labelpad=1.5)
        if col == 0:
            ax2.set_ylabel("response / energy share\n(discarded)", color=COBALT, fontsize=6.3, labelpad=2)
        else:
            ax2.set_yticklabels([])
    ylo = min(ax.get_ylim()[0] for ax in axes[1])
    yhi = max(ax.get_ylim()[1] for ax in axes[1])
    for ax in axes[1]:
        ax.set_ylim(min(0.0, ylo), yhi)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.93, bottom=0.12, hspace=0.2, wspace=0.14)
    out = os.path.join(args.out_dir, f"fig8_subspace{args.tag}")
    fig.savefig(out + ".pdf", facecolor=PAPER)
    fig.savefig(out + ".png", facecolor=PAPER, dpi=200)
    for ds, title, rows in data:
        line = []
        for colname in ("overlap_keep_old", "overlap_drop_old", "chance_old", "ratio_old", "ratio_new"):
            _, m, _ = seed_task_means(rows, colname)
            line.append(f"{colname}={np.nanmean(m):.3f}")
        print(ds, " ".join(line))
    print("wrote", out + ".pdf/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
