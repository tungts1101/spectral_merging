"""Figure 10: LoRA rank sweep on ImageNet-R (rebuttal E11; 3 seeds per rank).

Data: figures_data/e11_rank_imagenetr.csv (parse_logs.py e11: one row per rank with
r, lora_alpha, trainable_params, n_seeds, fa, fa_sd, aa, aa_sd, ffm, ffm_sd and,
when present, k_mean (kept rank at p = 0.9 from the [MergeStat] lines or the E4
replay), frac_k_eq_r, merge_time_s (GPU wall-clock of the pspectral call per merge)
and svd_ms_per_merge_cpu / svd_ms_per_merge_gpu from bench_svd.py). Missing kept-k
columns fall back to figures_data/e4_spectrum_imagenetr_r<r>.csv (analyze_spectrum.py
--tag _r<r>), the bench curve to figures_data/e11_svd_time.csv (variant only_A).
Three panels: FA (cobalt) and forgetting FFM (cobalt 60 %, right axis) vs r on a
log2 axis; mean kept k at p = 0.9 vs r with the k = r diagonal dotted; merge time
per task vs r (GPU, from [MergeStat]) with the CPU bench curve in cobalt 40 %.
The caption carries the explanation; no legend text inside.

Run:  python scripts/fig10_rank.py  ->  assets/figures/fig10_rank.{pdf,png}
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
PAPER, COBALT, COBALT60, COBALT40, TERRA = "#E9E9E5", "#2148B8", "#7A91D4", "#A6B5E3", "#C65F38"
N_SEEDS = 3
K_ALIASES = ("k_mean", "k_mean_p0.9", "kept_k_mean", "k_p0.9_mean")
TIME_ALIASES = ("merge_time_s", "merge_time_mean", "merge_time")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def first_col(row, names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return float(row[n])
    return np.nan


def kept_k_from_e4(data_dir, r, p_col="k_p0.9"):
    path = os.path.join(data_dir, f"e4_spectrum_imagenetr_r{r}.csv")
    if not os.path.exists(path):
        return np.nan
    vals = [float(x[p_col]) for x in read_csv(path) if x["space"] == "A" and x.get(p_col, "") != ""]
    return float(np.mean(vals)) if vals else np.nan


def bench_curve(data_dir, device="cpu", variant="only_A"):
    path = os.path.join(data_dir, "e11_svd_time.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    for x in read_csv(path):
        if x["device"].startswith(device) and x["variant"] == variant:
            out[int(x["r"])] = float(x["ms_per_merge"]) / 1e3
    return out


def style(ax):
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=COBALT, labelsize=5.5, length=2, width=0.5, which="both")
    for sp in ax.spines.values():
        sp.set_color(COBALT)
        sp.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "figures_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "assets", "figures"))
    ap.add_argument("--csv", default="e11_rank_imagenetr.csv")
    ap.add_argument("--allow-partial", action="store_true", help="draw ranks with fewer than 3 seeds")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_csv(os.path.join(args.data_dir, args.csv))
    rows.sort(key=lambda x: int(float(x["r"])))
    if not rows:
        raise SystemExit("no ranks in the CSV")
    bad = [x["r"] for x in rows if int(float(x.get("n_seeds", 0) or 0)) != N_SEEDS]
    if bad and not args.allow_partial:
        raise SystemExit(f"ranks without 3 seeds: {bad}; pass --allow-partial to draw anyway")
    r = np.array([int(float(x["r"])) for x in rows])
    fa = np.array([float(x["fa"]) for x in rows])
    fa_sd = np.array([float(x.get("fa_sd", 0) or 0) for x in rows])
    ffm = np.array([float(x["ffm"]) for x in rows])
    ffm_sd = np.array([float(x.get("ffm_sd", 0) or 0) for x in rows])
    k = np.array([first_col(x, K_ALIASES) for x in rows])
    for i, rr in enumerate(r):
        if np.isnan(k[i]):
            k[i] = kept_k_from_e4(args.data_dir, rr)
    mt = np.array([first_col(x, TIME_ALIASES) for x in rows])
    bench = bench_curve(args.data_dir, "cpu")
    bench_gpu = bench_curve(args.data_dir, "cuda")

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.8))
    fig.patch.set_facecolor(PAPER)
    # panel 1: FA and FFM vs r
    ax = axes[0]
    style(ax)
    ax.set_xscale("log", base=2)
    ax.errorbar(r, fa, yerr=fa_sd, color=COBALT, lw=0.9, marker="o", ms=2.5, capsize=1.5, elinewidth=0.6)
    ax.set_ylabel("final accuracy (%)", color=COBALT, fontsize=6.5, labelpad=2)
    axr = ax.twinx()
    axr.set_facecolor(PAPER)
    axr.errorbar(r, ffm, yerr=ffm_sd, color=COBALT60, lw=0.9, marker="s", ms=2.3, capsize=1.5, elinewidth=0.6)
    axr.tick_params(colors=COBALT60, labelsize=5.5, length=2, width=0.5)
    for sp in axr.spines.values():
        sp.set_color(COBALT60)
        sp.set_linewidth(0.5)
    axr.spines["top"].set_visible(False)
    axr.spines["left"].set_visible(False)
    axr.set_ylabel("forgetting (%)", color=COBALT60, fontsize=6.5, labelpad=2)
    ax.spines["right"].set_visible(False)
    # panel 2: kept k vs r
    ax = axes[1]
    style(ax)
    ax.spines["right"].set_visible(False)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.plot(r, r, color=COBALT, ls=(0, (1, 1.5)), lw=0.7)
    ax.plot(r, k, color=COBALT, lw=0.9, marker="o", ms=2.5)
    ax.set_ylabel(r"mean kept $k$ at $p=0.9$", color=COBALT, fontsize=6.5, labelpad=2)
    # panel 3: merge time vs r
    ax = axes[2]
    style(ax)
    ax.spines["right"].set_visible(False)
    ax.set_xscale("log", base=2)
    if np.isfinite(mt).any():
        ax.plot(r, mt, color=COBALT, lw=0.9, marker="o", ms=2.5)
    if bench_gpu:
        rb = sorted(bench_gpu)
        ax.plot(rb, [bench_gpu[x] for x in rb], color=COBALT60, lw=0.8, ls=(0, (3, 1.5)))
    if bench:
        rb = sorted(bench)
        ax.plot(rb, [bench[x] for x in rb], color=COBALT40, lw=0.9, marker="^", ms=2.3)
    ax.set_ylabel("merge time per task (s)", color=COBALT, fontsize=6.5, labelpad=2)
    for ax in axes:
        ax.set_xticks(r)
        ax.set_xticklabels([str(x) for x in r])
        ax.minorticks_off()
        ax.set_xlabel("LoRA rank $r$", color=COBALT, fontsize=6.5, labelpad=1.5)
    fig.subplots_adjust(left=0.085, right=0.955, top=0.95, bottom=0.2, wspace=0.55)
    out = os.path.join(args.out_dir, "fig10_rank")
    fig.savefig(out + ".pdf", facecolor=PAPER)
    fig.savefig(out + ".png", facecolor=PAPER, dpi=200)
    for i, rr in enumerate(r):
        print(f"r={rr}: FA {fa[i]:.2f} +- {fa_sd[i]:.2f} | FFM {ffm[i]:.2f} | kept k {k[i]:.1f} | "
              f"merge {mt[i]:.3f}s | bench cpu {bench.get(rr, float('nan')):.3f}s")
    print("wrote", out + ".pdf/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
