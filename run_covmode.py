"""Cheaper class statistics for Stage 2 (rebuttal E5) with the per-stage timers
(E3): the five covariance modes of train_ca_cov_mode, cache-served on the
headline chains of the four benchmarks.

Cells (logs_main/<ds>/<cell>.log, run_config_sweep format), 4 datasets x 3 seeds
    cov_class         reference: one full covariance per class (the paper; the
                      BASE_CONFIG default, so this cell must reproduce the headline
                      log of the same chain)
    cov_diag          one variance vector per class                (train_ca_cov_mode diag)
    cov_lowrank_k16   rank-16 factor + residual diagonal per class (lowrank, train_ca_cov_rank 16)
    cov_lowrank_k64   rank-64 factor + residual diagonal per class (lowrank, rank 64)
    cov_shared        one pooled covariance for all classes + means (shared)
Everything else is the headline posture (rebuttal_common.cfg: pspectral only_A
p 0.9, per-dataset alpha / lambda / weight decay / local-term weight).  The
cheap modes fit with covstats.fit_diag / fit_lowrank / scatter, transport with
transport_diag / transport_lowrank_U / transport_full and sample in align through
covstats.make_dist; main.py logs '[Alignment] cov_mode=.. rank=.. stats_floats=..'
once per task.  Storage per mode comes from memory_table.py (printed in the
dry-run; the per-class counts are 590,592 / 1,536 / 13,824 / 50,688 / 768 floats).

Stage timers (E3 rides on the E5 reference runs, plan E3/E5): the cov_class cell
sets stage_timing=True and stage_timing_csv=logs_main/<ds>/timing_<cell>.csv by
default (--timing reference), one '[Timing] task t <stage>=..s ..' line per stage
(train, merge, drift, drift_features, drift_solve, transport, stats, align, eval)
and one CSV row per line (main.TIMING_CSV_HEADER).  The timers do not touch the
RNG, so the reference cell still reproduces the headline log's evaluation lines.
A bare --timing (= --timing all) times every cell; --timing none times nothing.
--solver woodbury sets train_drift_solver (the n x n push-through solve, E3) and
suffixes the cell names with _woodbury.

Each dataset runs in its own isolated checkpoint directory
checkpoints_main/rebuttal/covmode_<ds> (rebuttal_common.isolated_ckpt_dir); the
five cells share it and run one after the other, which is required because they
share the merged tag pspec_only_A_p0.9_a<alpha>.  Run one process per directory.

Outputs: python parse_logs.py e5        # -> figures_data/e5_storage_accuracy.csv (FA vs storage)
         python parse_logs.py e3        # -> the stage-time table from the [Timing] lines / CSVs
         python scripts/fig_e5_storage.py
Cost (plan E5): 20 cache-served Stage-2 runs x 3 seeds x ~10 min = ~10 GPU-h.

Usage
    python run_covmode.py --dry-run                        # 20 cells
    python run_covmode.py --ds cifar224                     # reference (timed) + variants
    python run_covmode.py --ds cifar224 --timing            # every cell timed
    python run_covmode.py --ds cub imagenetr cars
    python run_covmode.py --ds cars --modes cov_diag cov_shared
    python run_covmode.py --ds cifar224 --smoke             # 2 tasks x 1 epoch under logs_main/smoke
"""
import argparse
import sys
from collections import OrderedDict

import rebuttal_common as rc
import run_long

# cell name -> config overrides (train_ca_cov_mode values accepted by main.py after E5)
MODES = OrderedDict([
    ("cov_class", {"train_ca_cov_mode": "class"}),
    ("cov_diag", {"train_ca_cov_mode": "diag"}),
    ("cov_lowrank_k16", {"train_ca_cov_mode": "lowrank", "train_ca_cov_rank": 16}),
    ("cov_lowrank_k64", {"train_ca_cov_mode": "lowrank", "train_ca_cov_rank": 64}),
    ("cov_shared", {"train_ca_cov_mode": "shared"}),
])
# train_ca_cov_mode -> memory_table mode name
MT_MODE = {"class": "full", "diag": "diag", "lowrank": "lowrank", "shared": "shared"}


def job_name(ds):
    """Name of the isolated checkpoint directory shared by the cells of `ds`."""
    return f"covmode_{ds}"


TIMING_CHOICES = ("reference", "all", "none")   # which cells carry the E3 stage timers
REFERENCE_MODE = "cov_class"                     # the paper's statistics; the E3 timers ride on it


def timed(mode, timing):
    """True when the cell of `mode` gets stage_timing under the --timing choice."""
    if timing not in TIMING_CHOICES:
        raise ValueError(f"timing must be one of {TIMING_CHOICES}, got {timing!r}")
    return timing == "all" or (timing == "reference" and mode == REFERENCE_MODE)


def build_cells(ds, modes, ckpt_dir, timing="reference", solver=None, log_root=None, smoke=False):
    """OrderedDict {cell name: config} for `ds`; `timing` in TIMING_CHOICES."""
    cells = OrderedDict()
    for mode in modes:
        name = mode + (f"_{solver}" if solver and solver != "dense" else "")
        c = rc.cfg(ds, checkpoint_dir=ckpt_dir, **MODES[mode])
        if solver and solver != "dense":
            c["train_drift_solver"] = solver
        if timed(mode, timing):
            c["stage_timing"] = True
            c["stage_timing_csv"] = run_long.side_csv_path(ds, f"timing_{name}", log_root, smoke)
        cells[name] = c
    return cells


def storage_note(ds, mode):
    """One line with the statistics storage of `mode` on `ds` (memory_table formulas)."""
    import memory_table as mt
    over = MODES[mode]
    C = rc.TOTAL_CLASSES[ds]
    floats = mt.stats_floats(MT_MODE[over["train_ca_cov_mode"]], C, rank=over.get("train_ca_cov_rank"))
    return (f"{mode}: statistics {floats:,} floats = {mt.mb(floats):.2f} MB for C={C} "
            f"(LoRA module {mt.mb(mt.lora_floats()[0]):.2f} MB, head {mt.mb(mt.head_floats(C)):.2f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap)
    ap.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES),
                    help="cells to run (default: all five)")
    ap.add_argument("--timing", nargs="?", const="all", default="reference", choices=TIMING_CHOICES,
                    help="cells with stage_timing=True and a timing_<cell>.csv next to the log (E3): "
                         "'reference' (default) = the cov_class cell, 'all' (a bare --timing), 'none'")
    ap.add_argument("--solver", choices=("dense", "woodbury"), default=None,
                    help="train_drift_solver (default: unset = dense); woodbury suffixes the cell names")
    args = rc.apply_common_args(ap.parse_args(argv))

    total = 0
    for ds in args.datasets:
        prefix, T = rc.prefix(ds), rc.n_tasks(ds)
        timed_modes = [m for m in args.modes if timed(m, args.timing)]
        print(f"[plan] {ds}: {len(args.modes)} mode(s) on chain {prefix} ({T} tasks)"
              + (f" with stage timers on {', '.join(timed_modes)}" if timed_modes else ""))
        if args.dry_run:
            for mode in args.modes:
                print("[plan]   " + storage_note(ds, mode))
        ckpt_dir = run_long.prepare_ckpt_dir(ds, prefix, args.seeds, T, job_name(ds), args.ckpt_src,
                                             dry_run=args.dry_run, smoke=args.smoke)
        cells = build_cells(ds, args.modes, ckpt_dir, timing=args.timing, solver=args.solver,
                            log_root=args.log_root, smoke=args.smoke)
        if args.redo:
            for c in cells.values():
                if "stage_timing_csv" in c:
                    run_long.backup_side_csv(c["stage_timing_csv"], dry_run=args.dry_run)
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))
        total += len(cells)
    if args.dry_run:
        rc.print_cells(total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
