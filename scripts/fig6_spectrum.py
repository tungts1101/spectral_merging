"""Figure 6: spectrum of the merge update on the four benchmarks (rebuttal E4; 3 seeds).

Data: figures_data/e4_spectrum_<ds>.csv and e4_singvals_<ds>.csv (analyze_spectrum.py).
Row 1: heatmap of the kept rank k at p = 0.9 (exact truncate_energy rule) of the
lora_A update dA per layer (rows: block 0..11 x qkv / fc1 / fc2) and task 1..9, seed
mean, cobalt scale on the Cool Gray substrate. Row 2: normalised singular values
sigma_i / sigma_1 of dA (A space), mean over layers and seeds, for tasks 1, 3, 5, 7,
9 in cobalt and its screens; the tick marks the mean kept k of that task.
The caption carries the explanation; no legend text inside.

Run:  python scripts/fig6_spectrum.py  ->  assets/figures/fig6_spectrum.{pdf,png}
      python scripts/fig6_spectrum.py --tag _r8 --allow-partial   # an E11 chain
"""
import argparse
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAPER, COBALT, TERRA = "#E9E9E5", "#2148B8", "#C65F38"
SCREENS = ["#C9D2EE", "#A6B5E3", "#7A91D4", "#4A66C4", COBALT]
CMAP = LinearSegmentedColormap.from_list("cobalt", [PAPER, "#C9D2EE", "#7A91D4", COBALT])

DATASETS = [("cifar224", "CIFAR-100"), ("cub", "CUB"), ("imagenetr", "ImageNet-R"), ("cars", "Cars")]
MODULES = ["qkv", "fc1", "fc2"]
CURVE_TASKS = [1, 3, 5, 7, 9]
N_SEEDS, N_BLOCKS = 3, 12


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def kept_matrix(rows, p_col):
    """(k_mean [36 layers x T tasks], tasks, n_seeds) in A space, seed mean per cell."""
    rows = [r for r in rows if r["space"] == "A" and r.get(p_col, "") != ""]
    tasks = sorted({int(r["task"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    layers = [(b, m) for b in range(N_BLOCKS) for m in MODULES]
    M = np.full((len(layers), len(tasks)), np.nan)
    for i, (b, m) in enumerate(layers):
        for j, t in enumerate(tasks):
            vals = [float(r[p_col]) for r in rows
                    if int(r["block"]) == b and r["module"] == m and int(r["task"]) == t]
            if vals:
                M[i, j] = np.mean(vals)
    return M, tasks, seeds


def curves(sv_rows, tasks):
    """{task: (mean sigma_i / sigma_1 over layers and seeds, by idx)} in A space."""
    out = {}
    for t in tasks:
        acc = {}
        for r in sv_rows:
            if r["space"] != "A" or int(r["task"]) != t:
                continue
            acc.setdefault(int(r["idx"]), []).append(float(r["sigma_norm"]))
        if acc:
            idx = sorted(acc)
            out[t] = np.array([np.mean(acc[i]) for i in idx])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "figures_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "assets", "figures"))
    ap.add_argument("--tag", default="", help="CSV suffix, e.g. _r8 (E11 chains)")
    ap.add_argument("--p", type=float, default=0.9)
    ap.add_argument("--datasets", nargs="+", default=[d for d, _ in DATASETS])
    ap.add_argument("--allow-partial", action="store_true",
                    help="draw even when a dataset has fewer than 3 seeds x 9 tasks x 36 layers")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    p_col = f"k_p{args.p:g}"
    labels = dict(DATASETS)
    datasets = [(d, labels.get(d, d)) for d in args.datasets]

    panels = []
    for ds, title in datasets:
        rows = read_csv(os.path.join(args.data_dir, f"e4_spectrum_{ds}{args.tag}.csv"))
        sv_path = os.path.join(args.data_dir, f"e4_singvals_{ds}{args.tag}.csv")
        sv_rows = read_csv(sv_path) if os.path.exists(sv_path) else []
        M, tasks, seeds = kept_matrix(rows, p_col)
        n_rows_A = sum(1 for r in rows if r["space"] == "A")
        complete = (len(seeds) == N_SEEDS and len(tasks) == 9 and n_rows_A == N_SEEDS * 9 * 36
                    and not np.isnan(M).any())
        if not complete and not args.allow_partial:
            raise SystemExit(f"{ds}: incomplete data (seeds {seeds}, tasks {tasks}, {n_rows_A} A rows); "
                             f"pass --allow-partial to draw anyway")
        n_sv = max(int(r["n_sv"]) for r in rows if r["space"] == "A")
        panels.append((ds, title, M, tasks, curves(sv_rows, CURVE_TASKS), n_sv))

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(5.5, 3.1), squeeze=False)
    fig.patch.set_facecolor(PAPER)
    # colour scale 1 .. max observed k (shared by the panels; printed below for the caption)
    vmax = max(2.0, float(np.ceil(max(np.nanmax(p[2]) for p in panels))))
    for col, (ds, title, M, tasks, cur, n_sv) in enumerate(panels):
        ax = axes[0, col]
        ax.set_facecolor(PAPER)
        ax.imshow(M, cmap=CMAP, vmin=1, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(title, color=COBALT, fontsize=6.8, pad=3)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([str(t) for t in tasks], color=COBALT, fontsize=5.5)
        ax.set_yticks([1 + 3 * b for b in range(N_BLOCKS)])
        ax.set_yticklabels([str(b) for b in range(N_BLOCKS)] if col == 0 else [], color=COBALT, fontsize=5.5)
        for b in range(1, N_BLOCKS):
            ax.axhline(3 * b - 0.5, color=PAPER, lw=0.4)
        ax.set_xlabel("task $t$", color=COBALT, fontsize=6.5, labelpad=1.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if col == 0:
            ax.set_ylabel("block (qkv, fc1, fc2)", color=COBALT, fontsize=6.5, labelpad=2)

        ax2 = axes[1, col]
        ax2.set_facecolor(PAPER)
        for i, t in enumerate(CURVE_TASKS):
            if t not in cur:
                continue
            y = cur[t]
            x = np.arange(1, len(y) + 1)
            ax2.plot(x, y, color=SCREENS[i], lw=0.9, solid_capstyle="round")
            j = tasks.index(t) if t in tasks else None
            if j is not None:
                k_mean = float(np.nanmean(M[:, j]))
                ax2.plot([k_mean], [np.interp(k_mean, x, y)], marker="|", ms=4, mew=0.8,
                         color=SCREENS[i], ls="none")
        ax2.set_xlim(1, n_sv)
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("singular index $i$", color=COBALT, fontsize=6.5, labelpad=1.5)
        ax2.tick_params(colors=COBALT, labelsize=5.5, length=2, width=0.5)
        for sp in ax2.spines.values():
            sp.set_color(COBALT)
            sp.set_linewidth(0.5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        if col == 0:
            ax2.set_ylabel(r"$\sigma_i/\sigma_1$ of $\Delta A$", color=COBALT, fontsize=6.5, labelpad=2)
        else:
            ax2.set_yticklabels([])
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.11, hspace=0.5, wspace=0.14)
    out = os.path.join(args.out_dir, f"fig6_spectrum{args.tag}")
    fig.savefig(out + ".pdf", facecolor=PAPER)
    fig.savefig(out + ".png", facecolor=PAPER, dpi=200)
    for ds, title, M, tasks, _, n_sv in panels:
        print(f"{ds}: mean k(p={args.p:g}) per task", [f"{np.nanmean(M[:, j]):.1f}" for j in range(len(tasks))],
              f"| overall {np.nanmean(M):.2f} of {n_sv}")
    print(f"heatmap colour scale 1..{vmax:g}; wrote", out + ".pdf/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
