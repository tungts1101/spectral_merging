"""E12: equal tuning budget for the merge rules of Tab. 6, selected per dataset on
the test split (the chains are trained in the job directory when no cache exists,
e.g. on bio / greenhorn).
Final protocol (2026-11-11, later decision): --budget 4 (default) and --proto.
Every method, ISM included, gets 4 combinations per dataset (GRID4: the 2 x 2 shape
of the original Tab. 6 grids; Model Stock 1 analytic cell), evaluated with the
cosine-to-prototype head only (no drift map, no alignment, no classifier); 3 seeds;
no full-protocol re-run.  One method runs on one machine for all four datasets
(scripts/e12p_machine.sh: bio = ism nufilt ties knots modelstock, greenhorn = dare
della breadcrumbs avg max_abs).  Parse with  python parse_logs.py e12 --pattern '^e12p_'.
Earlier 36-cell design (--budget 36), kept for reference:
Rules and grids (RULES): two-knob rules get 6 coefficients x 6 values of their own
knob (TIES trim ratio, DARE / DELLA drop rate, Breadcrumbs beta, KnOTS rank budget,
NUFILT r_p); one-knob rules (Task Arithmetic, max-abs) 36 coefficients 0.05 .. 1.80;
Model Stock is analytic (1 cell). Only same-meaning knobs share ISM's value set
(NUFILT's damping coefficient = alpha's role); the other rules' coefficient c scales
the mean task vector from W_0 and keeps its own spread. --with-ism adds ISM's
headline cell and its 6 x 6 alpha x p grid on the same chain.
Cells per dataset: 8 rules x 36 + Model Stock 1 = 289 (+ 37 ISM); each cell is a
Stage-2 re-run (5-10 min per seed) once the chain exists.  Config names
e12_{rule}[_{k|q|b|rp}{value}]_c{coef} and e12_ism_p{p}_a{alpha}; the merged tag
carries the knob when it leaves its Tab. 6 value (main.merge_tag), so cells never
share merged files.  parse_logs.py e12 selects the best cell per rule and dataset.
Original single-axis description (7 rules x 6 coefficients, 2026-11-08):
E12: equal tuning budget for the other merge rules, a per-dataset coefficient
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
import os
import time
from collections import OrderedDict

import rebuttal_common as rc

# Value sets. Only knobs with the SAME MEANING as an ISM knob share ISM's set:
# NUFILT's damping coefficient acts on the new step from M_{t-1} like alpha, so it
# takes COEF_ISM. The classical / sparsification rules' coefficient c scales the
# MEAN task vector from W_0 (c = 1 is the plain mean (M + W)/2), a different
# quantity, so it keeps its own spread around 1. Budget is matched by count:
# 36 cells per method and dataset (6 x 6 for two-knob rules, 36 coefficients for
# one-knob rules); Model Stock is analytic and has no knob (1 cell, stated).
COEF_ISM = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)          # ISM's alpha set; also its p set
COEF_RULE = (0.3, 0.5, 0.7, 0.85, 1.0, 1.2)        # coefficient on the mean task vector
COEF_DENSE = tuple(round(0.05 * i, 2) for i in range(1, 37))   # one-knob rules, 36 values
P_ISM = COEF_ISM

# rule -> (fixed config, knob (key, tag letter, values) or None, coefficient values)
RULES = OrderedDict([
    ("avg", ({"train_merge": "avg", "train_merge_incremental": True, "train_merge_topk": 100}, None, COEF_DENSE)),
    ("max_abs", ({"train_merge": "max_abs", "train_merge_incremental": True, "train_merge_topk": 100}, None, COEF_DENSE)),
    ("modelstock", ({"train_merge": "modelstock"}, None, (1.0,))),          # analytic t; ignores c
    ("ties", ({"train_merge": "ties", "train_merge_incremental": True},
              ("train_merge_topk", "k", (10, 20, 40, 60, 80, 100)), COEF_RULE)),
    ("dare", ({"train_merge": "dare"}, ("merge_dare_q", "q", (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)), COEF_RULE)),
    ("della", ({"train_merge": "della"}, ("merge_della_q", "q", (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)), COEF_RULE)),
    ("knots", ({"train_merge": "knots"}, ("merge_knots_topk", "k", (10.0, 20.0, 40.0, 60.0, 80.0, 100.0)), COEF_RULE)),
    ("breadcrumbs", ({"train_merge": "breadcrumbs", "merge_bc_gamma": 0.01},
                     ("merge_bc_beta", "b", (0.3, 0.5, 0.7, 0.85, 0.9, 0.95)), COEF_RULE)),
    # NUFILT (E15b rule): r_p x damping coefficient, adaptive filter off. r_p clamps to the
    # rank of M_{t-1} - W_0 (64 for a rank-64 LoRA factor), so 64 = the paper's 128 (full
    # removal of the old row space).
    ("nufilt", ({"train_merge": "nufilt", "merge_nufilt_adapt": False},
                ("merge_nufilt_rp", "rp", (8, 16, 24, 32, 48, 64)), COEF_ISM)),
])
# Budget 4 (user decision 2026-11-11, proxy protocol): 4 combinations per method and
# dataset, the 2 x 2 shape of the original Tab. 6 grids (run_sota_merge.GRIDS values
# kept for DARE / DELLA / Breadcrumbs / KnOTS); ISM is one method among the others
# (alpha x p, no extra headline cell); Model Stock stays its single analytic cell.
# rule -> (knob values or None, coefficient values); ISM: (p values, alpha values).
GRID4 = OrderedDict([
    ("ism", ((0.7, 0.9), (0.1, 0.3))),
    ("nufilt", ((32, 64), (0.1, 0.3))),
    ("ties", ((20, 100), (0.3, 1.0))),
    ("dare", ((0.5, 0.9), (0.3, 1.0))),
    ("della", ((0.5, 0.9), (0.3, 1.0))),
    ("breadcrumbs", ((0.5, 0.85), (0.3, 1.0))),
    ("knots", ((50.0, 100.0), (0.3, 1.0))),
    ("avg", (None, (0.3, 0.5, 0.7, 1.0))),
    ("max_abs", (None, (0.3, 0.5, 0.7, 1.0))),
    ("modelstock", (None, (1.0,))),
])
BUDGETS = (4, 36)
DEFAULT_BUDGET = 4
METHODS = ("ism",) + tuple(RULES)                 # --rules accepts 'ism' (the alpha x p grid)
CLASSICAL = ("avg", "ties", "max_abs")          # helper.merge rules (pairwise only when incremental)
COEFS = COEF_RULE                               # --coefs overrides the two-knob rules' coefficient axis
DATASET_ORDER = ("imagenetr", "cub", "cifar224", "cars")
SMOKE_CELL = ("imagenetr", "avg", None, 0.5)
JOB = "e12"


def _f(x):
    """Canonical float so 1 and 1.0 give one cell and the tag 'c1.0'."""
    return float(f"{float(x):.6g}")


PROTO = {"eval_head": "proto_cos", "train_ca": False, "train_drift": False, "eval_proto_workers": 8}
PROTO_JOB = "e12p"      # proxy cells: own names / logs / directory (chains symlinked from e12_<ds>)


def cell_name(rule, knob_value, c, job=JOB):
    """{job}_{rule}[_{letter}{knob}]_c{c}; one-knob rules keep the {job}_{rule}_c{c} names."""
    knob = RULES[rule][1]
    mid = "" if knob is None or knob_value is None else f"_{knob[1]}{knob_value:g}"
    return f"{job}_{rule}{mid}_c{c}"


def ism_cell_name(p, a, job=JOB):
    return f"{job}_ism_p{p}_a{a}"


def build_cells(ds, rules=tuple(RULES), coefs=None, with_ism=False, job_suffix="", ism_grid=True, knobs=None,
                proto=False, budget=DEFAULT_BUDGET):
    """OrderedDict name -> config for `ds`, rule-major, knob-major, coefficient ascending.

    budget 4 (default): GRID4, 4 combinations per method; 'ism' in `rules` (or
    `with_ism`) adds ISM's 2 x 2 alpha x p cells and nothing else.
    budget 36: the 6 x 6 / 36-value grids of RULES; `coefs` overrides the coefficient
    axis of the two-knob rules only (NUFILT keeps COEF_ISM, the one-knob rules
    COEF_DENSE, Model Stock its single cell); with `with_ism` (or 'ism' in `rules`)
    the ISM headline cell comes first (it trains the chain when no cache exists) and,
    with `ism_grid`, ISM's own 6 x 6 grid alpha x p at the headline variant follows."""
    if budget not in BUDGETS:
        raise SystemExit(f"budget must be one of {BUDGETS}")
    job = PROTO_JOB if proto else JOB
    extra = dict(PROTO) if proto else {}
    ckpt = rc.isolated_ckpt_path(f"{job}_{ds}{job_suffix}")
    cells = OrderedDict()
    rules = list(rules)
    if "ism" in rules:
        with_ism = True
        rules = [r for r in rules if r != "ism"]
    if with_ism:
        if budget == 4:
            p_values, a_values = GRID4["ism"]
        else:
            cells[f"{job}_ism"] = rc.cfg(ds, merge_log_stats=True, checkpoint_dir=ckpt, **extra)
            p_values, a_values = (P_ISM, COEF_ISM) if ism_grid else ((), ())
        for p_ in p_values:
            for a in a_values:
                cells[ism_cell_name(p_, a, job)] = rc.cfg(ds, pspectral_p=p_, train_merge_alpha=a,
                                                          merge_log_stats=True, checkpoint_dir=ckpt, **extra)
    for rule in rules:
        if rule not in RULES:
            raise SystemExit(f"unknown rule {rule!r}; known: {list(METHODS)}")
        fixed, knob, rule_coefs = RULES[rule]
        knob_values4 = None
        if budget == 4:
            knob_values4, rule_coefs = GRID4[rule]
        if coefs is not None and knob is not None and rule != "nufilt":
            rule_coefs = coefs
        if knob is None:
            knob_values = [None]
        elif knobs is not None:   # --knobs: cast to the type of the rule's own values (int trim ratios stay ints)
            knob_values = [type(knob[2][0])(v) for v in knobs]
        elif budget == 4:
            knob_values = list(knob_values4)
        else:
            knob_values = list(knob[2])
        for kv in knob_values:
            for c in sorted({_f(c) for c in rule_coefs}):
                name = cell_name(rule, kv, c, job)
                if name in cells:
                    continue
                over = dict(fixed)
                if knob is not None:
                    over[knob[0]] = kv
                cells[name] = rc.cfg(ds, **over, train_merge_coef=c,
                                     merge_log_stats=True, checkpoint_dir=ckpt, **extra)
    for name, cfg in cells.items():
        if cfg["train_merge"] in CLASSICAL:
            assert cfg.get("train_merge_incremental") is True, name   # pairwise protocol
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, datasets=DATASET_ORDER)
    ap.add_argument("--rules", nargs="+", default=list(METHODS),
                    help=f"methods to run (default {' '.join(METHODS)}; 'ism' = the alpha x p grid)")
    ap.add_argument("--budget", type=int, choices=BUDGETS, default=DEFAULT_BUDGET,
                    help="cells per method and dataset: 4 (GRID4, default since 2026-11-11) or 36 (RULES)")
    ap.add_argument("--coefs", nargs="+", type=float, default=None,
                    help="coefficient axis of the two-knob rules (default 0.3 0.5 0.7 0.85 1.0 1.2); "
                         "NUFILT keeps ISM's set, the one-knob rules their 36 values")
    ap.add_argument("--with-ism", action="store_true",
                    help="add the ISM headline cell e12_ism (trains the chain when no cache exists) "
                         "and ISM's 6 x 6 alpha x p grid per dataset")
    ap.add_argument("--no-ism-grid", action="store_true",
                    help="with --with-ism: the headline cell only")
    ap.add_argument("--proto", action="store_true",
                    help="proxy protocol (2026-11-11): cosine-to-prototype head (RanPAC style), no drift map, "
                         "no alignment; names e12p_*, directory e12p_<ds> with the chain symlinked from e12_<ds>")
    ap.add_argument("--knobs", nargs="+", type=float, default=None,
                    help="values of the selected two-knob rules' own knob (default: the rule's 6 values); "
                         "e.g. --rules ties --knobs 20 --coefs 0.5 for one timing cell")
    ap.add_argument("--smoke-all", action="store_true",
                    help="with --smoke: smoke every cell instead of the representative one")
    ap.add_argument("--job-suffix", default="",
                    help="suffix of the checkpoint directory name (run two processes on one dataset)")
    args = rc.apply_common_args(ap.parse_args(argv))

    job = PROTO_JOB if args.proto else JOB
    per_ds = OrderedDict((ds, build_cells(ds, args.rules, args.coefs, args.with_ism, args.job_suffix,
                                          ism_grid=not args.no_ism_grid, knobs=args.knobs, proto=args.proto,
                                          budget=args.budget))
                         for ds in args.datasets)
    if args.smoke and not args.smoke_all:
        ds0, rule0, k0, c0 = SMOKE_CELL
        if ds0 in per_ds and cell_name(rule0, k0, _f(c0), job) in per_ds[ds0]:
            name = cell_name(rule0, k0, _f(c0), job)
            per_ds = OrderedDict([(ds0, OrderedDict([(name, per_ds[ds0][name])]))])
        else:
            ds0 = next(iter(per_ds))
            name = next(iter(per_ds[ds0]))
            per_ds = OrderedDict([(ds0, OrderedDict([(name, per_ds[ds0][name])]))])

    total = sum(len(c) for c in per_ds.values())
    for ds, cells in per_ds.items():
        print(f"[{job}] {ds}: {len(cells)} cells x {1 if args.smoke else len(args.seeds)} seed(s)")

    t0 = time.time()
    for ds, cells in per_ds.items():
        if not cells:
            continue
        if not args.smoke:
            prefix = next(iter(cells.values()))["train_prefix"]
            # proxy cells take the chain trained by the e12_<ds> job (symlinks) unless --ckpt-src is set
            src = args.ckpt_src
            if args.proto and src == rc.SEQFT_CKPT_DIR and os.path.isdir(rc.isolated_ckpt_path(f"{JOB}_{ds}")):
                src = rc.isolated_ckpt_path(f"{JOB}_{ds}")
            rc.isolated_ckpt_dir(ds, prefix, args.seeds, f"{job}_{ds}{args.job_suffix}",
                                 src_dir=src, dry_run=args.dry_run)
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))

    if args.dry_run:
        rc.print_cells(total)
        return 0
    print(f"[{JOB}] done: {total} cells. Next: python parse_logs.py e12 ; python scripts/fig7_merge_grid.py")
    print(f"Total: {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
