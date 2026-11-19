"""Long task sequences (rebuttal E1): the 20/50-task splits of CIFAR-100,
ImageNet-R and CUB, cache-served on the long chains when they exist (Stage 2
only) or trained otherwise, plus an optional alpha x lambda sensitivity grid
on the cached 50-task chains.

Splits (E2-LoRA Table 2 protocol; RESULTS.md section 4 "Long task sequences"):
    cifar224   t20 = 20 tasks x 5 classes     t50 = 50 tasks x 2 classes
    imagenetr  t20 = 20 tasks x 10 classes    t50 = 50 tasks x 4 classes
    cub        t20 = 20 tasks x 10 classes    (ISM-only row: no published baselines)
A split is a plain config override (dataset_num_task / dataset_init_cls /
dataset_increment, copied into the DataManager by main.run_single_experiment).
50-task chains use train_epochs = 5: a 2- or 4-class task is only 4-8
iterations per epoch and 10 epochs at lr 1e-2 diverged on one ImageNet-R seed
(RESULTS.md "Stability note"); the reported 50-task rows were trained with 5.

Cells (logs_main/<ds>/<cell>.log, run_config_sweep format)
    long_<split>                   headline posture of the dataset (rebuttal_common.cfg:
                                   pspectral only_A p 0.9, headline alpha / lambda /
                                   weight decay / local-term weight)
    long_<split>_a<a>_p<p>[_l<l>]  --cells A:P[:L]: one Stage-2 cell per spec at merge step
                                   alpha = A, energy fraction p = P and (optional) drift
                                   lambda = L, on the same cached chain; --tag <s> suffixes
                                   every cell name (a chain trained with other settings,
                                   e.g. --epochs 5 --prefix ..._e5 --tag _e5).
    long_<split>_lgcv              --lambda-gcv: headline alpha / p with lambda chosen per task by
                                   generalised cross-validation (train_drift_lambda_rule = 'gcv').
    long_<split>_plain<tag>        --plain: the sequential chain alone (train_merge none, no drift
                                   map, no alignment); with --lr / --epochs / --batch-size it trains
                                   and evaluates a chain under other training settings (own --prefix).
    long_<split>_a<alpha>_l<lam>   --grid: alpha in {headline, 0.1, 0.3} x lambda in
                                   {100, 1000} minus the headline pair (that is the
                                   cell above).  Stage 2 only on the cached chain, so
                                   alpha and lambda never touch the trained states.
                                   A TEST-SET sensitivity study (rebuttal concern A3),
                                   never new headline numbers.
Metrics: parse_logs.py e1 recomputes FA/AA/FF and the sample-weighted Last/Inc
totals (the E2-LoRA Table 2 metric) from the Grouped / Total Acc lines.

Chain prefix: --prefix; else the fallback <headline prefix>_<split>
(main_cifar224_r64_qkvfc_t50) when its chain is complete under --ckpt-src; else
the single complete chain of that length found by check_cache; else the
fallback, in which case the job TRAINS the chain (the dry-run says so).  Every
(dataset, split) runs in its own isolated checkpoint directory
checkpoints_main/rebuttal/long_<ds>_<split> (rebuttal_common.isolated_ckpt_dir:
private copies of the task-0 files, links to the rest), so the merged-checkpoint
cleanup and the base-file rewrite never touch the shared cache; the grid cells
share that directory and run one after the other.  Run one process per
directory.

Cost (plan E1): a cached 50-task Stage-2 cell is about 5x a 10-task run (eval
and LCA grow with T); measure one seed first.  CUB t20 trains a chain.

Usage
    python run_long.py --dry-run                               # every split of every dataset
    python run_long.py --ds cifar224 --split t50 --dry-run
    python run_long.py --ds cifar224 --split t50               # reproduction cell, 3 seeds
    python run_long.py --ds cifar224 imagenetr --split t50 --grid
    python run_long.py --ds cub --split t20 --prefix main_cub_r64_qkvfc_t20
    python run_long.py --ds cifar224 --split t50 --smoke       # 2 tasks x 1 epoch, logs_main/smoke
    python run_long.py --ds cifar224 --split t50 --small-task-epochs 5   # new chain: 5 epochs
                                                               # only for tasks with <= 4 classes
"""
import argparse
import os
import sys
from collections import OrderedDict

import check_cache
import rebuttal_common as rc

# split -> (num_task, init_cls, increment); DataManager rule: init + k * inc == total.
SPLITS = {
    "cifar224": {"t20": (20, 5, 5), "t50": (50, 2, 2)},
    "imagenetr": {"t20": (20, 10, 10), "t50": (50, 4, 4)},
    "cub": {"t20": (20, 10, 10)},
}
LONG_DATASETS = tuple(SPLITS)
LONG_SPLITS = ("t20", "t50")
EPOCHS_50 = 5                   # RESULTS.md section 4, "Stability note"
GRID_ALPHAS = (0.1, 0.3)        # plus the headline alpha of the dataset
GRID_LAMBDAS = (100.0, 1000.0)
# lambda candidates of the GCV rule (--lambda-gcv / --lambda-rule gcv): main.py's default grid stops
# at 1000, which the first 20-task CIFAR cell picked at the edge, so the long-split cells use a wider one.
GCV_LAMBDA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0)


# ----------------------------------------------------------------------------
# Splits and configs
# ----------------------------------------------------------------------------

def split_params(ds, split):
    """(T, init_cls, increment) of `split` on `ds`; SystemExit for unknown pairs."""
    if ds not in SPLITS:
        raise SystemExit(f"{ds} has no long split; datasets with long splits: {', '.join(LONG_DATASETS)}")
    if split not in SPLITS[ds]:
        raise SystemExit(f"no {split} split for {ds}; defined: {', '.join(sorted(SPLITS[ds]))}")
    T, init, inc = SPLITS[ds][split]
    assert rc.task_increments(rc.TOTAL_CLASSES[ds], init, inc) == [init] + [inc] * (T - 1), (ds, split)
    return T, init, inc


def split_desc(ds, split):
    T, init, inc = split_params(ds, split)
    return f"{T} tasks x {inc} classes" if init == inc else f"{init} + {T - 1} x {inc} classes"


def job_name(ds, split):
    """Name of the isolated checkpoint directory shared by the cells of (ds, split)."""
    return f"long_{ds}_{split}"


def long_ckpt_dir(ds, split):
    return rc.isolated_ckpt_path(job_name(ds, split))


def fallback_prefix(ds, split, small_task_epochs=None, family=None):
    """<headline prefix>_<split>[_st<N>]; the _st suffix marks a chain trained with
    train_epochs_small_task = N (a different chain, so a different cache key)."""
    pre = f"{rc.prefix(ds, family)}_{split}"
    if small_task_epochs is not None:
        pre += f"_st{int(small_task_epochs)}"
    return pre


def long_cfg(ds, split, prefix, alpha=None, lam=None, epochs=None, small_task_epochs=None, p=None,
             lr=None, batch_size=None, plain=False, lambda_rule=None, modules=None, **over):
    """Headline Stage-2 config of `ds` on the long split with the chain `prefix`.

    train_epochs is 5 for 50-task chains (EPOCHS_50) unless `epochs` is given;
    `small_task_epochs` sets train_epochs_small_task (main.train_backbone: that
    many epochs for a task with <= train_small_task_max_classes = 4 classes);
    `alpha` / `lam` / `p` override train_merge_alpha / train_drift_lambda /
    pspectral_p (grid and --cells cells).
    """
    T, init, inc = split_params(ds, split)
    c = rc.cfg(ds, train_prefix=prefix, dataset_num_task=T, dataset_init_cls=init,
               dataset_increment=inc)
    if T == 50:
        c["train_epochs"] = EPOCHS_50
    if epochs is not None:
        c["train_epochs"] = int(epochs)
    if small_task_epochs is not None:
        c["train_epochs_small_task"] = int(small_task_epochs)
    if lr is not None:
        c["train_base_lr"] = float(lr)      # head lr follows (train_head_lr unset)
    if batch_size is not None:
        c["train_batch_size"] = int(batch_size)
    if modules is not None:
        c["model_lora_target_modules"] = list(modules)
    if plain:
        c["train_merge"] = "none"
        c["train_ca"] = False
        c["train_drift"] = False
    if lambda_rule is not None:
        c["train_drift_lambda_rule"] = lambda_rule
        if lambda_rule == "gcv":
            c["train_drift_lambda_grid"] = list(GCV_LAMBDA_GRID)
    if alpha is not None:
        c["train_merge_alpha"] = float(alpha)
    if lam is not None:
        c["train_drift_lambda"] = float(lam)
    if p is not None:
        c["pspectral_p"] = float(p)
    c.update(over)
    return c


def grid_pairs(ds):
    """(alpha, lambda) pairs of the sensitivity grid of `ds`: {headline, 0.1, 0.3} x
    {100, 1000} without the headline pair (the reproduction cell)."""
    head_a, head_l = round(rc.HEADLINE_ALPHA[ds], 6), float(rc.HEADLINE_LAMBDA[ds])
    alphas = sorted({round(a, 6) for a in GRID_ALPHAS} | {head_a})
    return [(a, l) for a in alphas for l in GRID_LAMBDAS if not (a == head_a and l == head_l)]


def cell_name(split, alpha=None, lam=None, p=None, tag=""):
    """long_<split>[<tag>] for the reproduction cell; long_<split>_a<alpha>_l<lam> for
    the --grid cells (unchanged names); long_<split>_a<alpha>_p<p>[_l<lam>] for --cells."""
    if alpha is None:
        return f"long_{split}{tag}"
    name = f"long_{split}_a{alpha:g}"
    if p is not None:
        name += f"_p{p:g}"
    if lam is not None:
        name += f"_l{lam:g}"
    return name + tag


def parse_cell_spec(spec):
    """'A:P' or 'A:P:L' -> (alpha, p, lam or None); SystemExit on a malformed spec."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise SystemExit(f"--cells expects A:P or A:P:L, got {spec!r}")
    try:
        alpha, pp = float(parts[0]), float(parts[1])
        lam = float(parts[2]) if len(parts) == 3 else None
    except ValueError:
        raise SystemExit(f"--cells expects numbers, got {spec!r}")
    if not (0.0 < pp <= 1.0):
        raise SystemExit(f"--cells: p must be in (0, 1], got {pp}")
    return alpha, pp, lam


def build_cells(ds, split, prefix, ckpt_dir, grid=False, epochs=None, small_task_epochs=None,
                cell_specs=(), tag="", lr=None, batch_size=None, plain=False, lambda_gcv=False,
                no_base=False, lambda_rule=None, modules=None):
    """OrderedDict {cell name: config} for (ds, split): the reproduction cell, with
    grid=True the sensitivity cells, and one cell per (alpha, p, lam) of `cell_specs`."""
    common = dict(epochs=epochs, small_task_epochs=small_task_epochs, lr=lr, batch_size=batch_size,
                  modules=modules, checkpoint_dir=ckpt_dir)
    cells = OrderedDict()
    if plain:
        cells[f"long_{split}_plain{tag}"] = long_cfg(ds, split, prefix, plain=True, **common)
        return cells
    if not no_base:
        cells[cell_name(split, tag=tag)] = long_cfg(ds, split, prefix, **common)
    if lambda_gcv:
        cells[f"long_{split}_lgcv{tag}"] = long_cfg(ds, split, prefix, lambda_rule="gcv", **common)
    if grid:
        for a, l in grid_pairs(ds):
            cells[cell_name(split, a, l, tag=tag)] = long_cfg(ds, split, prefix, alpha=a, lam=l, **common)
    rule_tag = "" if lambda_rule is None else f"_l{lambda_rule}"
    for a, pp, l in cell_specs:
        if l is not None and lambda_rule is not None:
            raise SystemExit("--cells A:P:L and --lambda-rule exclude each other")
        name = cell_name(split, a, l, pp, rule_tag + tag)
        if name in cells:
            raise SystemExit(f"--cells: duplicate cell {name}")
        cells[name] = long_cfg(ds, split, prefix, alpha=a, lam=l, p=pp, lambda_rule=lambda_rule, **common)
    return cells


# ----------------------------------------------------------------------------
# Chain lookup (shared with run_driftdiag.py)
# ----------------------------------------------------------------------------

def chain_candidates(ds, seeds, T, ckpt_dir):
    """Prefixes under `ckpt_dir` whose seqft chains are complete for T tasks for
    every seed (check_cache inventory)."""
    if not os.path.isdir(ckpt_dir):
        return []
    chains, _ = check_cache.inventory(ckpt_dir)
    by_prefix = {}
    for ch in chains.values():
        if ch.ds == ds and ch.backbone == rc.MODEL_BACKBONE and ch.complete_for(T):
            by_prefix.setdefault(ch.prefix, set()).add(ch.seed)
    return sorted(p for p, got in by_prefix.items() if set(seeds) <= got)


def resolve_prefix(ds, split, seeds, override=None, ckpt_src=rc.SEQFT_CKPT_DIR,
                   small_task_epochs=None, family=None):
    """(train_prefix, note) of the long chain of (ds, split).

    Order: `override` (--prefix); the fallback name when its chain is complete
    under ckpt_src; the single complete T-task chain found there; else the
    fallback name, which means the job trains the chain.
    """
    T = split_params(ds, split)[0]
    fb = fallback_prefix(ds, split, small_task_epochs, family)
    if override:
        return override, "given by --prefix"
    cands = chain_candidates(ds, seeds, T, ckpt_src)
    if small_task_epochs is not None:
        # a cached chain trained without the small-task rule would be served as is
        # (cache hits skip training), so only the exact _st<N> chain qualifies
        cands = [c for c in cands if c == fb]
    if fb in cands:
        return fb, f"complete {T}-task chain under {ckpt_src}"
    if len(cands) == 1:
        return cands[0], f"the only complete {T}-task chain under {ckpt_src} (fallback {fb} absent)"
    if cands:
        return fb, f"several complete {T}-task chains under {ckpt_src} {cands}; pass --prefix to pick one"
    return fb, f"no complete {T}-task chain under {ckpt_src}: the job would TRAIN it"


def compact_status(status):
    """One line per check_cache.chain_status result: complete, or the per-seed
    counts of present files (the full missing-index lists are too long at T=50)."""
    T = status["T"]
    if status["complete"]:
        return f"[cache] {status['dir']} (T={T}): complete for seeds {sorted(status['seeds'])}"
    parts = []
    for seed, s in status["seeds"].items():
        parts.append(f"seed {seed}: base={'yes' if s['base'] else 'NO'} backbone {s['n_backbone']}/{T} "
                     f"classifier {T - len(s['classifier_missing'])}/{T}")
    return f"[cache] {status['dir']} (T={T}): INCOMPLETE -- " + "; ".join(parts)


def find_chain_dir(ds, prefix, seeds, T, candidates):
    """First directory of `candidates` holding the complete T-task chain of
    (ds, prefix) for every seed, or None."""
    for d in candidates:
        if d and os.path.isdir(d):
            if check_cache.chain_status(ds, prefix, seeds, T, ckpt_dir=d)["complete"]:
                return d
    return None


def prepare_ckpt_dir(ds, prefix, seeds, T, job, ckpt_src=rc.SEQFT_CKPT_DIR, dry_run=False,
                     smoke=False, verbose=True, extra_src=()):
    """Isolated checkpoint directory checkpoints_main/rebuttal/<job> for the chain
    (ds, prefix), filled from the first of (ckpt_src, extra_src..., the directory
    itself) that holds the complete chain (extra_src: e.g. the run_long directory
    that trained a long chain).  Prints check_cache's per-seed status.  With
    dry_run nothing is created; with smoke (and not dry_run) nothing is copied
    either, because smoke cells use <dir>_smoke and a _smoke prefix (they train
    two tasks).  Returns the directory path (always the same string)."""
    dst = rc.isolated_ckpt_path(job)
    src = find_chain_dir(ds, prefix, seeds, T, [ckpt_src, *extra_src, dst])
    if verbose:
        print(compact_status(check_cache.chain_status(ds, prefix, seeds, T, ckpt_dir=src or ckpt_src)))
    if src == dst:
        if verbose:
            print(f"[ckpt] {dst}: the {T}-task chain is already in the job directory")
        return dst
    if smoke and not dry_run:
        return dst
    rc.isolated_ckpt_dir(ds, prefix, seeds, job, src_dir=src or ckpt_src, dry_run=dry_run, verbose=verbose)
    return dst


# ----------------------------------------------------------------------------
# Side CSVs next to the logs (shared with run_covmode.py / run_driftdiag.py)
# ----------------------------------------------------------------------------

def side_csv_path(ds, stem, log_root=None, smoke=False):
    """logs_main/<ds>/<stem>.csv for the CSV side outputs of main.py
    (stage_timing_csv -> timing_<cell>.csv, drift_diag_csv -> driftdiag_<split>.csv);
    smoke runs write under logs_main/smoke with a _smoke suffix, like their logs."""
    root = rc.SMOKE_LOG_ROOT if smoke else (log_root or rc.LOG_DIR)
    return os.path.join(root, ds, f"{stem}{'_smoke' if smoke else ''}.csv")


def backup_side_csv(path, dry_run=False):
    """Move a stale side CSV to <path>.bakN before a --redo (main.py appends rows
    across runs, so a rerun must not inherit the old ones).  Returns the backup
    path, or None when there is nothing to move."""
    if not os.path.isfile(path):
        return None
    n = 1
    while os.path.exists(f"{path}.bak{n}"):
        n += 1
    dst = f"{path}.bak{n}"
    if not dry_run:
        os.replace(path, dst)
    print(f"[redo] stale {path} -> {dst}" + (" (dry-run: not moved)" if dry_run else ""))
    return dst


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, datasets=LONG_DATASETS)
    ap.add_argument("--split", nargs="+", choices=LONG_SPLITS, default=None,
                    help="splits to run (default: every split defined for the dataset)")
    ap.add_argument("--grid", action="store_true",
                    help="add the alpha x lambda sensitivity cells (test-set study)")
    ap.add_argument("--cells", nargs="+", default=[], metavar="A:P[:L]",
                    help="extra Stage-2 cells on the cached chain: merge step alpha A, energy "
                         "fraction p P, optional drift lambda L (default: headline lambda)")
    ap.add_argument("--tag", default="",
                    help="suffix for every cell name (log file), e.g. _e5 for a 5-epoch chain")
    ap.add_argument("--prefix", default=None,
                    help="train_prefix of the long chain (one dataset x split only); default: "
                         "check_cache lookup, fallback <headline prefix>_<split>")
    ap.add_argument("--epochs", type=int, default=None,
                    help="train_epochs override (default: BASE_CONFIG for 20 tasks, 5 for 50 tasks)")
    ap.add_argument("--plain", action="store_true",
                    help="the sequential chain alone: train_merge none, no drift map, no alignment "
                         "(one cell long_<split>_plain<tag>; trains the chain if it is not cached)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="train_batch_size override for the chain (trains a new chain; own --prefix/--tag)")
    ap.add_argument("--modules", nargs="+", default=None, choices=("qkv", "fc1", "fc2", "proj"),
                    help="model_lora_target_modules override, e.g. --modules qkv (trains a new chain; own --prefix/--tag)")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="config overrides applied to every cell (Stage-2 knobs such as "
                         "train_ca_robust_weight=0.3 train_ca_n_classes=20); give the cells a --tag")
    ap.add_argument("--lambda-gcv", action="store_true",
                    help="add the cell long_<split>_lgcv<tag>: headline alpha / p, lambda per task by GCV")
    ap.add_argument("--lambda-rule", choices=("gcv",), default=None,
                    help="drift lambda rule for every --cells cell (cell names get _lgcv)")
    ap.add_argument("--no-base", action="store_true",
                    help="skip the reproduction cell (when it already ran; --cells / --lambda-gcv only)")
    ap.add_argument("--lr", type=float, default=None,
                    help="train_base_lr override for the chain (trains a new chain; give it its own "
                         "--prefix and --tag, e.g. --lr 0.005 --prefix <headline>_t20_lr0.005 --tag _lr0.005)")
    ap.add_argument("--small-task-epochs", type=int, default=None,
                    help="train_epochs_small_task: epochs for tasks with <= 4 classes "
                         "(trains a new chain; the fallback prefix gets _st<N>)")
    args = rc.apply_common_args(ap.parse_args(argv))

    plan = []
    for ds in args.datasets:
        if ds not in SPLITS:
            raise SystemExit(f"{ds} has no long split; datasets with long splits: {', '.join(LONG_DATASETS)}")
        for split in (args.split or sorted(SPLITS[ds])):
            if split not in SPLITS[ds]:
                print(f"[skip] {ds}: no {split} split defined ({', '.join(sorted(SPLITS[ds]))} only)")
                continue
            plan.append((ds, split))
    if not plan:
        raise SystemExit("nothing to run")
    if args.prefix and len(plan) != 1:
        raise SystemExit("--prefix applies to exactly one dataset x split")

    total = 0
    for ds, split in plan:
        T = split_params(ds, split)[0]
        prefix, note = resolve_prefix(ds, split, args.seeds, args.prefix, args.ckpt_src,
                                      small_task_epochs=args.small_task_epochs)
        job = job_name(ds, split)
        print(f"[plan] {ds} {split} ({split_desc(ds, split)}) prefix={prefix} -- {note}")
        if T == 50 and args.dry_run:
            print(f"[plan] {ds} {split}: train_epochs={args.epochs if args.epochs is not None else EPOCHS_50}; "
                  f"a cached 50-task Stage-2 cell costs about 5x a 10-task run (measure one seed first)")
        ckpt_dir = prepare_ckpt_dir(ds, prefix, args.seeds, T, job, args.ckpt_src,
                                    dry_run=args.dry_run, smoke=args.smoke)
        cells = build_cells(ds, split, prefix, ckpt_dir, grid=args.grid, epochs=args.epochs,
                            small_task_epochs=args.small_task_epochs,
                            cell_specs=[parse_cell_spec(s) for s in args.cells], tag=args.tag,
                            lr=args.lr, batch_size=args.batch_size, plain=args.plain,
                            lambda_gcv=args.lambda_gcv, no_base=args.no_base,
                            lambda_rule=args.lambda_rule, modules=args.modules)
        if (args.lr is not None or args.batch_size is not None or args.modules is not None) and not args.prefix:
            raise SystemExit("--lr / --batch-size / --modules train a new chain: pass --prefix (and --tag) so it does not reuse the headline chain")
        if not cells:
            raise SystemExit("no cells (--no-base without --cells / --lambda-gcv)")
        for kv in args.set:
            if "=" not in kv:
                raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
            k, v = kv.split("=", 1)
            try:
                val = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                val = v
            for c in cells.values():
                c[k] = val
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))
        total += len(cells)
    if args.dry_run:
        rc.print_cells(total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
