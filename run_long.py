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


def long_cfg(ds, split, prefix, alpha=None, lam=None, epochs=None, small_task_epochs=None, **over):
    """Headline Stage-2 config of `ds` on the long split with the chain `prefix`.

    train_epochs is 5 for 50-task chains (EPOCHS_50) unless `epochs` is given;
    `small_task_epochs` sets train_epochs_small_task (main.train_backbone: that
    many epochs for a task with <= train_small_task_max_classes = 4 classes);
    `alpha` / `lam` override train_merge_alpha / train_drift_lambda (grid cells).
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
    if alpha is not None:
        c["train_merge_alpha"] = float(alpha)
    if lam is not None:
        c["train_drift_lambda"] = float(lam)
    c.update(over)
    return c


def grid_pairs(ds):
    """(alpha, lambda) pairs of the sensitivity grid of `ds`: {headline, 0.1, 0.3} x
    {100, 1000} without the headline pair (the reproduction cell)."""
    head_a, head_l = round(rc.HEADLINE_ALPHA[ds], 6), float(rc.HEADLINE_LAMBDA[ds])
    alphas = sorted({round(a, 6) for a in GRID_ALPHAS} | {head_a})
    return [(a, l) for a in alphas for l in GRID_LAMBDAS if not (a == head_a and l == head_l)]


def cell_name(split, alpha=None, lam=None):
    if alpha is None:
        return f"long_{split}"
    return f"long_{split}_a{alpha:g}_l{lam:g}"


def build_cells(ds, split, prefix, ckpt_dir, grid=False, epochs=None, small_task_epochs=None):
    """OrderedDict {cell name: config} for (ds, split): the reproduction cell and,
    with grid=True, the sensitivity cells."""
    common = dict(epochs=epochs, small_task_epochs=small_task_epochs, checkpoint_dir=ckpt_dir)
    cells = OrderedDict()
    cells[cell_name(split)] = long_cfg(ds, split, prefix, **common)
    if grid:
        for a, l in grid_pairs(ds):
            cells[cell_name(split, a, l)] = long_cfg(ds, split, prefix, alpha=a, lam=l, **common)
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
    ap.add_argument("--prefix", default=None,
                    help="train_prefix of the long chain (one dataset x split only); default: "
                         "check_cache lookup, fallback <headline prefix>_<split>")
    ap.add_argument("--epochs", type=int, default=None,
                    help="train_epochs override (default: BASE_CONFIG for 20 tasks, 5 for 50 tasks)")
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
                            small_task_epochs=args.small_task_epochs)
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))
        total += len(cells)
    if args.dry_run:
        rc.print_cells(total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
