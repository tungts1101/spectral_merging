"""Validation-split selection of the Stage-2 hyper-parameters (rebuttal E14):
choose damping alpha, ridge lambda and the local-term weight on a held-out
10 % carve of the training data instead of on the test set (b28C-W2, xkNt-Q1).

Protocol. `data_val_frac 0.1` carves 10 % of every class's training images
(stratified, at least one image in each split per class, drawn with a private
np.random.RandomState(run seed)) BEFORE any training, so Stage 1 (LoRA
fine-tuning), the drift fit and the class statistics only ever see the
remaining 90 %. The carved images are evaluated after every task with a
private generator ('[Evaluation-val] Total Acc / Grouped' and
'[Evaluation-val] FAA / FFM / FFD / ASA' lines, one '  [val] ...' line after the
unchanged SUMMARY), so the test pass and the training stream are untouched.
Selection = argmax of the validation FAA per seed (and pooled over seeds); the
report quotes the TEST FA / AA / FF of the selected cell next to the headline
cell on the same 90 % chain, the test-oracle cell and the full-data headline
(parse_logs.py e14 -> figures_data/e14_valsplit_<ds>.csv, e14_valsplit_selected.csv).

Cells per dataset, config name e14_val10_a{alpha}_l{lambda}_rw{rw}:
  1. the headline cell (tab:hparams alpha / lambda / local-term weight) runs
     FIRST: it trains the 90 % chain under train_prefix <prefix>_val10
     (main_cifar224_r64_qkvfc_val10, ...) in checkpoints_main/rebuttal/e14, a
     fresh directory without links to the shared cache;
  2. the cached grid, Stage 2 only on that chain: alpha in {0.1, 0.2, 0.3} plus
     the headline alpha  x  lambda in {10, 100}  x  train_ca_robust_weight in
     {0.1, 0.2}: CIFAR-100 16 cells (alpha 0.15 added), CUB / ImageNet-R / Cars
     12 each, 52 in all. Cells are de-duplicated by their formatted name, so
     the headline is never run twice.
Weight decay stays at its test-selected value (5e-4 / 1e-3 / 2e-4 / 1e-3): it
needs one chain per value and is stated as a caveat, not re-selected. CUB
(~30 images per class) and Cars (~41) lose 3-4 images per class to the carve,
so the 90 %-chain headline sits a few tenths below the full-data headline;
compare within the val chain.

Isolation. All cells of one dataset share the val10 prefix, and every dataset
shares the directory (plan, ground rule 3: one process per directory, cells
run sequentially; merged tags carry alpha, not lambda or the local-term weight,
and reset_merge recomputes them). For two parallel processes give each its own
--ckpt-dir. Logs: logs_main/<dataset>/e14_val10_a*_l*_rw*.log. Smoke: --smoke
(2 tasks x 1 epoch, first seed, `_smoke` suffix on name, prefix and directory,
logs under logs_main/smoke); the usual check is `--smoke --ds cub --headline-only`.
Resume: a finished cell is skipped; a partial log needs --redo or --append.

Cost (plan estimate, re-cost from the E2 time table): 12 chain trainings (the
headline cells) plus 48 cached cells x 3 seeds x ~10 min ~ 24 GPU-h; the minimum
grid --alphas 0.1 0.2 0.3 --rws <headline value> is 7 (CIFAR) / 6 cells per
dataset.

Usage:
    python run_e14_valsplit.py --dry-run                    # 52 cells; no torch needed
    python run_e14_valsplit.py --smoke --ds cub --headline-only
    python run_e14_valsplit.py --headline-only              # train the four 90 % chains
    python run_e14_valsplit.py                              # chains first, then the grid
    python run_e14_valsplit.py --ds imagenetr --alphas 0.1 0.2 0.3 --lambdas 10 100 --rws 0.1
"""
import argparse
import os
import sys
import time
from collections import OrderedDict

import rebuttal_common as rc   # torch-free: posture, launcher, resume rule

GRID_ALPHAS = (0.1, 0.2, 0.3)          # plus the dataset's headline alpha
GRID_LAMBDAS = (10.0, 100.0)
GRID_RW = (0.1, 0.2)                   # train_ca_robust_weight (local-term weight)
VAL_FRAC = 0.1
CKPT_DIR = rc.isolated_ckpt_path("e14")   # checkpoints_main/rebuttal/e14


# ----------------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------------

def val_token(frac):
    """'val10' for a 10 % carve (name and prefix token)."""
    return f"val{int(round(frac * 100))}"


def cell_name(frac, alpha, lam, rw):
    return f"e14_{val_token(frac)}_a{alpha:g}_l{lam:g}_rw{rw:g}"


def headline(ds):
    """(alpha, lambda, local-term weight) of tab:hparams for `ds`."""
    return (rc.HEADLINE_ALPHA[ds], rc.HEADLINE_LAMBDA[ds], rc.POSTURE[ds]["train_ca_robust_weight"])


def _key(point):
    return tuple(f"{v:g}" for v in point)


def grid_points(ds, alphas=GRID_ALPHAS, lambdas=GRID_LAMBDAS, rws=GRID_RW, headline_only=False):
    """Launch order of the (alpha, lambda, rw) cells of one dataset: the headline
    first (it trains the chain), then the grid with the headline alpha added,
    de-duplicated by the formatted name (critic fix: never count a cell twice)."""
    head = headline(ds)
    points = [head]
    if headline_only:
        return points
    seen = {_key(head)}
    for a in sorted({float(a) for a in alphas} | {float(head[0])}):
        for lam in sorted({float(v) for v in lambdas}):
            for rw in sorted({float(v) for v in rws}):
                p = (a, lam, rw)
                if _key(p) in seen:
                    continue
                seen.add(_key(p))
                points.append(p)
    return points


def build_cfg(ds, alpha, lam, rw, frac=VAL_FRAC, ckpt_dir=CKPT_DIR):
    """ISM headline config of `ds` with the grid values, the val10 prefix, the
    fresh directory and the carve; data_val_seed stays unset (= the run seed)."""
    return rc.cfg(ds, train_merge_alpha=alpha, train_drift_lambda=lam,
                  train_ca_robust_weight=rw,
                  train_prefix=f"{rc.prefix(ds)}_{val_token(frac)}",
                  checkpoint_dir=ckpt_dir,
                  data_val_frac=frac)


def build_cells(datasets, alphas, lambdas, rws, frac=VAL_FRAC, ckpt_dir=CKPT_DIR,
                headline_only=False):
    """{ds: OrderedDict(config_name -> cfg)} in launch order."""
    out = OrderedDict()
    for ds in datasets:
        configs = OrderedDict()
        for a, lam, rw in grid_points(ds, alphas, lambdas, rws, headline_only):
            configs[cell_name(frac, a, lam, rw)] = build_cfg(ds, a, lam, rw, frac, ckpt_dir)
        out[ds] = configs
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validation-split selection campaign (rebuttal E14); see the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap)
    ap.add_argument("--alphas", nargs="+", type=float, default=list(GRID_ALPHAS),
                    help="damping alphas of the grid (the headline alpha is always added)")
    ap.add_argument("--lambdas", nargs="+", type=float, default=list(GRID_LAMBDAS),
                    help="ridge lambdas of the grid")
    ap.add_argument("--rws", nargs="+", type=float, default=list(GRID_RW),
                    help="local-term weights (train_ca_robust_weight) of the grid")
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC,
                    help="carved fraction per class (default 0.1 -> val10)")
    ap.add_argument("--headline-only", action="store_true",
                    help="run only the headline cell per dataset (trains the 90 %% chains)")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR,
                    help=f"fresh checkpoint directory of this job (default {CKPT_DIR})")
    args = rc.apply_common_args(ap.parse_args(argv))
    if not 0.0 < args.val_frac < 1.0:
        raise SystemExit(f"--val-frac must lie in (0, 1), got {args.val_frac}")
    if os.path.normpath(args.ckpt_dir) == os.path.normpath(rc.SEQFT_CKPT_DIR):
        raise SystemExit("--ckpt-dir must not be the shared cache checkpoints_main/seqft")

    cells = build_cells(args.datasets, args.alphas, args.lambdas, args.rws,
                        frac=args.val_frac, ckpt_dir=args.ckpt_dir,
                        headline_only=args.headline_only)
    kw = rc.launch_kwargs(args)
    if os.path.normpath(kw["ckpt_src"]) == os.path.normpath(rc.SEQFT_CKPT_DIR):
        kw["ckpt_src"] = args.ckpt_dir        # the val chains live in the E14 directory
    if len(args.datasets) < len(rc.DATASETS) and not args.dry_run:
        print(f"[note] one process per directory: another process on {args.ckpt_dir} "
              "would share the merged-checkpoint cleanup; use --ckpt-dir for parallel runs")

    t0 = time.time()
    total = 0
    outcome = {}
    for ds, configs in cells.items():
        first = next(iter(configs))
        print(f"[e14] {ds}: {len(configs)} cell(s); {first} first "
              f"(trains the {100 - args.val_frac * 100:g} % chain "
              f"{configs[first]['train_prefix']} in {args.ckpt_dir})")
        outcome[ds] = rc.launch(ds, configs, args.seeds, **kw)
        total += len(configs)
    if args.dry_run:
        rc.print_cells(total)
        return 0
    for ds, res in outcome.items():
        for name, status in res.items():
            print(f"{ds} | {name}: {status}")
    print(f"Total: {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
