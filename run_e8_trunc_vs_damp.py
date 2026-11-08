"""E8: truncation (p = 0.9) versus damped interpolation (p = 1.0) at every damping
alpha on CIFAR-100 and CUB, the headline pair on Cars, and the paper's large-step
claim (p in {0.1, 0.5} at alpha = 1.0), all cache-served on the main_*_r64_qkvfc
chains (Stage-2-only re-runs, about 10 min per seed).

Why config-only: helper.truncate_energy keeps every direction at p = 1.0, so
pspectral only_A at p = 1.0 is the plain interpolation M_{t-1} + alpha (W_t -
M_{t-1}) on A, B and norm.* (up to float32 SVD round-off); p = 0.9 is ISM's
truncated step.  Both are pspectral configs, nothing else changes.

Cells (config names mc3d_only_A_p{p}_a{a}, the naming export_mc3d_full.py and
parse_logs.py e8 expect; the set is de-duplicated, so CUB's headline 0.10 is the
grid's 0.1 and appears once):
  cifar224   p in {0.9, 1.0} x alpha in {0.1, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0}   14
             + p in {0.1, 0.5} at alpha 1.0 (large-step claim)                  2
  cub        p in {0.9, 1.0} x alpha in {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}         12
             + p in {0.1, 0.5} at alpha 1.0                                     2
  cars       p in {0.9, 1.0} at the headline alpha 0.2                           2
  imagenetr  nothing by default: the cube figures_data/mc3d_imagenetr_full.csv
             already holds the alpha-0.3 pair; `--datasets imagenetr` runs it.
Total 32 cells x 3 seeds = 96 cached runs (about 16 GPU-h).

Robustness exports: the CIFAR cells at alpha 0.15 (both p) set
final_model_dir = final_models/e8, so run_single_experiment writes
cifar224_mc3d_only_A_p{0.9,1.0}_a0.15_s{seed}.pt (backbone, classifier, class
Gaussians) for the server-side CIFAR-100-C/-P harness: a paired comparison on
one chain.

Outputs: logs_main/<ds>/mc3d_only_A_p{p}_a{a}.log, one per cell (3 seeds,
SUMMARY with FFM, [MergeStat] lines with the realised step norms).  Then
    python parse_logs.py e8                 # -> figures_data/mc3d_cifar224_full.csv,
                                            #    mc3d_cub_full.csv, mc3d_cars_extra.csv
    python scripts/fig9_trunc_vs_damp.py    # FA and forgetting vs alpha, p 0.9 vs 1.0
Do not re-run export_mc3d_full.py for Cars or ImageNet-R: it rebuilds the committed
cube CSVs from whatever logs it finds.

Run:
    python run_e8_trunc_vs_damp.py --dry-run
    python run_e8_trunc_vs_damp.py --smoke                  # one cell, 2 tasks, 1 epoch
    python run_e8_trunc_vs_damp.py --datasets cifar224
    python run_e8_trunc_vs_damp.py --datasets cub cars
Each dataset runs in its own directory checkpoints_main/rebuttal/e8_<ds> (private
copies of the task-0 files, links to the cached W_t), so two datasets may run as
two processes.  Never start two processes on the same directory.
"""
import argparse
import os
import sys
import time
from collections import OrderedDict

import rebuttal_common as rc

GRID_A = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)      # the cube's alpha axis (RESULTS.md, Fig. 5)
PS = (0.9, 1.0)                               # truncation vs interpolation
EXTRA_P = (0.1, 0.5)                          # paper's large-step claim, tested at ...
EXTRA_P_ALPHA = 1.0                           # ... alpha = 1.0 on the full-grid datasets
FULL_GRID_DATASETS = ("cifar224", "cub")      # grid + headline; the others run the pair only
DEFAULT_DATASETS = ("cifar224", "cub", "cars")
EXPORT_ALPHA = {"cifar224": 0.15}             # final_model_dir cells (both p)
FINAL_MODEL_DIR = os.path.join("final_models", "e8")
JOB = "e8"                                    # checkpoints_main/rebuttal/e8_<ds>


def _f(x):
    """Canonical float so 0.10 and 0.1 give one cell and one tag ('a0.1')."""
    return float(f"{float(x):.6g}")


def cell_name(p, a):
    """Config name in export_mc3d_full's pattern mc3d_(\\w+?)_p([\\d.]+)_a([\\d.]+)."""
    return f"mc3d_only_A_p{p}_a{a}"


def build_cells(ds, alphas=GRID_A, ps=PS, extra_p=EXTRA_P, with_export=True, job_suffix=""):
    """OrderedDict name -> config for `ds`: the headline pair first (the paired
    evidence), then the grid by alpha, then the large-step cells.  Names are
    de-duplicated by construction."""
    headline = _f(rc.HEADLINE_ALPHA[ds])
    ps = [_f(p) for p in ps]
    if ds in FULL_GRID_DATASETS:
        grid = sorted({_f(a) for a in alphas} | {headline})
    else:
        grid = [headline]
    order = [(p, headline) for p in ps]
    order += [(p, a) for a in grid if a != headline for p in ps]
    if ds in FULL_GRID_DATASETS:
        order += [(_f(p), _f(EXTRA_P_ALPHA)) for p in extra_p]
    ckpt = rc.isolated_ckpt_path(f"{JOB}_{ds}{job_suffix}")
    cells = OrderedDict()
    for p, a in order:
        name = cell_name(p, a)
        if name in cells:
            continue
        cfg = rc.cfg(ds, pspectral_p=p, train_merge_alpha=a, merge_log_stats=True,
                     checkpoint_dir=ckpt)
        if with_export and ds in EXPORT_ALPHA and a == _f(EXPORT_ALPHA[ds]):
            cfg["final_model_dir"] = FINAL_MODEL_DIR
        cells[name] = cfg
    return cells


def smoke_subset(per_ds):
    """One representative cell: the CIFAR p = 1.0 export cell (exercises the
    interpolation path and the final_model_dir export), else the first cell."""
    if "cifar224" in per_ds:
        name = cell_name(_f(1.0), _f(EXPORT_ALPHA["cifar224"]))
        if name in per_ds["cifar224"]:
            return OrderedDict([("cifar224", OrderedDict([(name, per_ds["cifar224"][name])]))])
    for ds, cells in per_ds.items():
        if cells:
            name = next(iter(cells))
            return OrderedDict([(ds, OrderedDict([(name, cells[name])]))])
    return OrderedDict()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, datasets=DEFAULT_DATASETS)
    ap.add_argument("--alphas", nargs="+", type=float, default=list(GRID_A),
                    help="alpha grid for the full-grid datasets (the headline alpha is always added)")
    ap.add_argument("--ps", nargs="+", type=float, default=list(PS),
                    help="energy fractions of the main comparison (default 0.9 1.0)")
    ap.add_argument("--extra-p", nargs="*", type=float, default=list(EXTRA_P),
                    help=f"extra p values at alpha {EXTRA_P_ALPHA} on CIFAR/CUB (large-step claim); "
                         "pass the flag without values to skip them")
    ap.add_argument("--no-export", action="store_true",
                    help="do not set final_model_dir on the CIFAR alpha-0.15 cells")
    ap.add_argument("--smoke-all", action="store_true",
                    help="with --smoke: smoke every cell instead of the one representative cell")
    ap.add_argument("--job-suffix", default="",
                    help="suffix of the checkpoint directory name (run two processes on one dataset)")
    args = rc.apply_common_args(ap.parse_args(argv))

    per_ds = OrderedDict(
        (ds, build_cells(ds, args.alphas, args.ps, args.extra_p, not args.no_export, args.job_suffix))
        for ds in args.datasets)
    if args.smoke and not args.smoke_all:
        per_ds = smoke_subset(per_ds)
    if args.smoke:
        for cells in per_ds.values():
            for cfg in cells.values():
                if "final_model_dir" in cfg:          # keep smoke exports out of final_models/e8
                    cfg["final_model_dir"] = FINAL_MODEL_DIR + "_smoke"

    total = sum(len(c) for c in per_ds.values())
    n_seeds = 1 if args.smoke else len(args.seeds)
    for ds, cells in per_ds.items():
        n_exp = sum(1 for c in cells.values() if c.get("final_model_dir"))
        print(f"[{JOB}] {ds}: {len(cells)} cells x {n_seeds} seed(s)"
              + (f", {n_exp} with final_model_dir exports" if n_exp else ""))

    t0 = time.time()
    for ds, cells in per_ds.items():
        if not cells:
            continue
        if not args.smoke:
            # Private directory per (experiment, dataset): copies of the files the
            # job rewrites, links to the cached W_t; the cleanup glob stays inside.
            prefix = next(iter(cells.values()))["train_prefix"]
            rc.isolated_ckpt_dir(ds, prefix, args.seeds, f"{JOB}_{ds}{args.job_suffix}",
                                 src_dir=args.ckpt_src, dry_run=args.dry_run)
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))

    if args.dry_run:
        rc.print_cells(total)
        return 0
    print(f"[{JOB}] done: {total} cells. Next: python parse_logs.py e8 ; "
          f"python scripts/fig9_trunc_vs_damp.py")
    print(f"Total: {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
