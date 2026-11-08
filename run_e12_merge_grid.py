"""E12: equal tuning budget for the other merge rules, a per-dataset coefficient
grid on the cached main_*_r64_qkvfc chains (Stage-2-only re-runs, about 10 min
per seed).

Rules (the Tab. 6 settings from RESULTS.md) and the knob they share, the merge
coefficient train_merge_coef (lamb):
  avg          Task Arithmetic: helper.merge, pairwise chain (M_{t-1}, W_t)
  ties         TIES, topk 100 = no trim (Tab. 6 setting)            via
  max_abs      max-magnitude selection (MagMax)                      train_merge_incremental
  dare         merging_sota, q 0.5
  della        merging_sota, q 0.5
  knots        KnOTS-TIES, topk 100 (full rank)
  breadcrumbs  Model Breadcrumbs, beta 0.5, gamma 0.01 (the strongest published
               rule in RESULTS.md, so it is in the default grid)
Coefficients {0.3, 0.5, 0.7, 0.85, 1.0, 1.2}: for the pairwise rules c = 1 gives
(M + W)/2 and c scales the mean task vector, so the grid samples the likely
optimum densely with one amplifying value.  Note: with c < 1 these rules shrink
the accumulated module toward W_0; the tuned damped-interpolation member of the
Task Arithmetic family is pspectral at p = 1 (E8), not these cells.

Cells: 7 rules x 6 coefficients x 4 datasets = 168 configs, e12_{rule}_c{c},
3 seeds each = 504 cached runs (about 84 GPU-h; ImageNet-R and CUB first, they
decide the "ahead of the best published rule" claim).  The merged tag is
{method}_c{coef}; cells inside one job run sequentially in the job's private
directory checkpoints_main/rebuttal/e12_<ds>, so tags cannot collide.
The classical rules set train_merge_incremental True: the log must show
"[Merging] Merging 2 task checkpoints" (the parser asserts it).

Outputs: logs_main/<ds>/e12_{rule}_c{c}.log (SUMMARY with FFM, Grouped lines,
reduced [MergeStat] lines).  Then
    python parse_logs.py e12            # -> figures_data/e12_merge_grid.csv + two
                                        #    tables (per-dataset best c; IN-R-selected c)
    python scripts/fig7_merge_grid.py
Both tables are test-selected; the validation-selected variant belongs to E14.

Run:
    python run_e12_merge_grid.py --dry-run
    python run_e12_merge_grid.py --smoke                      # e12_avg_c0.5 on ImageNet-R, 2 tasks
    python run_e12_merge_grid.py --datasets imagenetr cub
    python run_e12_merge_grid.py --datasets cifar224 cars
    python run_e12_merge_grid.py --rules avg ties --coefs 0.5 1.0 --job-suffix _classical
`--with-ism` adds the ISM headline cell (e12_ism) per dataset as the paired
reference on the same chain and directory.  Two processes may run on different
datasets (or on one dataset with different --job-suffix values); never on one
directory.
"""
import argparse
import sys
import time
from collections import OrderedDict

import rebuttal_common as rc

RULES = OrderedDict([
    ("avg", {"train_merge": "avg", "train_merge_incremental": True, "train_merge_topk": 100}),
    ("ties", {"train_merge": "ties", "train_merge_incremental": True, "train_merge_topk": 100}),
    ("max_abs", {"train_merge": "max_abs", "train_merge_incremental": True, "train_merge_topk": 100}),
    ("dare", {"train_merge": "dare", "merge_dare_q": 0.5}),
    ("della", {"train_merge": "della", "merge_della_q": 0.5}),
    ("knots", {"train_merge": "knots", "merge_knots_topk": 100.0}),
    ("breadcrumbs", {"train_merge": "breadcrumbs", "merge_bc_beta": 0.5, "merge_bc_gamma": 0.01}),
])
CLASSICAL = ("avg", "ties", "max_abs")          # helper.merge rules (pairwise only when incremental)
COEFS = (0.3, 0.5, 0.7, 0.85, 1.0, 1.2)
DATASET_ORDER = ("imagenetr", "cub", "cifar224", "cars")
SMOKE_CELL = ("imagenetr", "avg", 0.5)
JOB = "e12"


def _f(x):
    """Canonical float so 1 and 1.0 give one cell and the tag 'c1.0'."""
    return float(f"{float(x):.6g}")


def cell_name(rule, c):
    return f"e12_{rule}_c{c}"


def build_cells(ds, rules=tuple(RULES), coefs=COEFS, with_ism=False, job_suffix=""):
    """OrderedDict name -> config for `ds`, rule-major, coefficient ascending."""
    ckpt = rc.isolated_ckpt_path(f"{JOB}_{ds}{job_suffix}")
    cells = OrderedDict()
    if with_ism:
        cells["e12_ism"] = rc.cfg(ds, merge_log_stats=True, checkpoint_dir=ckpt)
    for rule in rules:
        if rule not in RULES:
            raise SystemExit(f"unknown rule {rule!r}; known: {list(RULES)}")
        for c in sorted({_f(c) for c in coefs}):
            name = cell_name(rule, c)
            if name in cells:
                continue
            cells[name] = rc.cfg(ds, **RULES[rule], train_merge_coef=c,
                                 merge_log_stats=True, checkpoint_dir=ckpt)
    for name, cfg in cells.items():
        if cfg["train_merge"] in CLASSICAL:
            assert cfg.get("train_merge_incremental") is True, name   # pairwise protocol
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, datasets=DATASET_ORDER)
    ap.add_argument("--rules", nargs="+", default=list(RULES),
                    help=f"rules to run (default {' '.join(RULES)})")
    ap.add_argument("--coefs", nargs="+", type=float, default=list(COEFS),
                    help="train_merge_coef grid (default 0.3 0.5 0.7 0.85 1.0 1.2)")
    ap.add_argument("--with-ism", action="store_true",
                    help="add the ISM headline cell e12_ism per dataset (paired reference)")
    ap.add_argument("--smoke-all", action="store_true",
                    help="with --smoke: smoke every cell instead of the representative one")
    ap.add_argument("--job-suffix", default="",
                    help="suffix of the checkpoint directory name (run two processes on one dataset)")
    args = rc.apply_common_args(ap.parse_args(argv))

    per_ds = OrderedDict((ds, build_cells(ds, args.rules, args.coefs, args.with_ism, args.job_suffix))
                         for ds in args.datasets)
    if args.smoke and not args.smoke_all:
        ds0, rule0, c0 = SMOKE_CELL
        if ds0 in per_ds and cell_name(rule0, _f(c0)) in per_ds[ds0]:
            name = cell_name(rule0, _f(c0))
            per_ds = OrderedDict([(ds0, OrderedDict([(name, per_ds[ds0][name])]))])
        else:
            ds0 = next(iter(per_ds))
            name = next(iter(per_ds[ds0]))
            per_ds = OrderedDict([(ds0, OrderedDict([(name, per_ds[ds0][name])]))])

    total = sum(len(c) for c in per_ds.values())
    for ds, cells in per_ds.items():
        print(f"[{JOB}] {ds}: {len(cells)} cells x {1 if args.smoke else len(args.seeds)} seed(s)")

    t0 = time.time()
    for ds, cells in per_ds.items():
        if not cells:
            continue
        if not args.smoke:
            prefix = next(iter(cells.values()))["train_prefix"]
            rc.isolated_ckpt_dir(ds, prefix, args.seeds, f"{JOB}_{ds}{args.job_suffix}",
                                 src_dir=args.ckpt_src, dry_run=args.dry_run)
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))

    if args.dry_run:
        rc.print_cells(total)
        return 0
    print(f"[{JOB}] done: {total} cells. Next: python parse_logs.py e12 ; python scripts/fig7_merge_grid.py")
    print(f"Total: {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
