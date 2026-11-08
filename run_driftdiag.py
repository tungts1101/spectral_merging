"""Drift-map diagnostics (rebuttal E9): the headline configs with drift_diag=True
on the 10-task splits and, when their chains exist, on the 20/50-task splits of
run_long.py.

What the flag does (main.Learner.drift_diagnostics; read only when drift_diag is
True): after fit_drift_map and transport_prototypes at every task t >= 1 it
re-extracts a seeded subsample of old-class images under the deployed M_t (and
under M_{t-1} for the one-step view), fits reference Gaussians and reports the
error of the stored class statistics never corrected / before / after the
transport, plus the one-step comparison identity vs fitted W vs an oracle W
fitted on the old-class pairs (driftsolve.solve_drift), per old task and
overall: one '[DriftDiag]' line per task and one CSV row per (task, group) in
logs_main/<ds>/driftdiag_<split>.csv (main.DRIFTDIAG_CSV_HEADER, appended
across seeds).  Nothing is written back to the statistics and the RNG is
isolated (private RandomState / Generator, torch.random.fork_rng), so the
accuracy trajectory must be unchanged -- --validate checks exactly that.

Cells (logs_main/<ds>/<cell>.log, run_config_sweep format)
    driftdiag_t10                 headline posture (rebuttal_common.cfg) + drift_diag keys,
                                  4 datasets x 3 seeds
    driftdiag_t20 / driftdiag_t50 run_long.long_cfg(ds, split, prefix) + drift_diag keys;
                                  prefix from run_long.resolve_prefix (--prefix override);
                                  skipped when the long chain exists nowhere, unless
                                  --train-missing
    driftdiag_<split>_validate_on / _off   --validate: the first seed of the first dataset
                                  with and without drift_diag; the '[Evaluation] Total
                                  Acc' and '[Evaluation] FAA' lines of the two logs must
                                  be identical (exit code 1 otherwise)
Diagnostic keys from the CLI: drift_diag_source (--source train|test; 'train' = the
old classes' training images under the deterministic test transform, the same
images as compute_multivariate_normal, so the error isolates the transport;
'test' = held-out images), drift_diag_max_per_class (--max-per-class, 0 = all),
drift_diag_seed (--diag-seed), drift_diag_onestep (--no-onestep drops the
identity / W / oracle segment), drift_diag_kl (--kl adds symmetric KL columns).
Isolated checkpoint directories: checkpoints_main/rebuttal/driftdiag_<ds>_<split>[_validate].

Outputs: python parse_logs.py e9 --roots logs_main --out-dir figures_data
             (-> figures_data/e9_driftdiag_<ds>_<split>.csv, seed means + _sd, and the error table)
         python scripts/fig_e9_driftdiag.py
Cost (plan E9): ~13 min per 10-task seed (two extra feature passes over <= 100
images per old class), ~2.6 GPU-h for the four 10-task splits; the long splits
add ~6 GPU-h.

Usage
    python run_driftdiag.py --dry-run                      # 4 datasets x t10
    python run_driftdiag.py --ds cifar224 --validate       # RNG-isolation check (seed 1993)
    python run_driftdiag.py --splits t10
    python run_driftdiag.py --ds cifar224 imagenetr --splits t20 t50
    python run_driftdiag.py --ds cifar224 --splits t50 --prefix main_cifar224_r64_qkvfc_t50
    python run_driftdiag.py --source test --max-per-class 0 --kl
"""
import argparse
import os
import re
import sys
from collections import OrderedDict

import rebuttal_common as rc
import run_long

SPLIT_CHOICES = ("t10",) + run_long.LONG_SPLITS

# run_single_experiment header (main.run_config_sweep; the dash is U+2014) and the
# evaluation lines that must not change when the diagnostic is on.
_HEADER_RE = re.compile(r"Experiment: (?P<ds>\S+) [—-] (?P<cfg>.+?) [—-] seed (?P<seed>\d+)\s*$", re.M)
_EVAL_RE = re.compile(r"\[Evaluation\] (?:Total Acc|FAA): .*$")


def diag_keys(args, csv_path):
    """The drift_diag_* config keys from the CLI."""
    return {"drift_diag": True,
            "drift_diag_source": args.source,
            "drift_diag_max_per_class": int(args.max_per_class),
            "drift_diag_seed": int(args.diag_seed),
            "drift_diag_onestep": not args.no_onestep,
            "drift_diag_kl": bool(args.kl),
            "drift_diag_csv": csv_path}


def base_cfg(ds, split, prefix, ckpt_dir):
    """Headline config of (ds, split) on the chain `prefix` without the diagnostic."""
    if split == "t10":
        return rc.cfg(ds, checkpoint_dir=ckpt_dir)
    return run_long.long_cfg(ds, split, prefix, checkpoint_dir=ckpt_dir)


def chain_of(ds, split, seeds, args):
    """(prefix, T, note, extra_src) of the chain a cell runs on, or None when the
    dataset has no such split."""
    if split == "t10":
        return rc.prefix(ds), rc.n_tasks(ds), "headline chain", ()
    if ds not in run_long.SPLITS or split not in run_long.SPLITS[ds]:
        return None
    prefix, note = run_long.resolve_prefix(ds, split, seeds, args.prefix, args.ckpt_src)
    return prefix, run_long.split_params(ds, split)[0], note, (run_long.long_ckpt_dir(ds, split),)


def build_cells(ds, split, prefix, ckpt_dir, args, validate=False):
    """OrderedDict {cell name: config}: one diagnostic cell, or the on/off pair."""
    stem = f"driftdiag_{split}" + ("_validate" if validate else "")
    csv_path = run_long.side_csv_path(ds, stem, args.log_root, args.smoke)
    cells = OrderedDict()
    on = base_cfg(ds, split, prefix, ckpt_dir)
    on.update(diag_keys(args, csv_path))
    if not validate:
        cells[stem] = on
        return cells
    cells[stem + "_on"] = on
    cells[stem + "_off"] = base_cfg(ds, split, prefix, ckpt_dir)
    return cells


# ----------------------------------------------------------------------------
# --validate: the two logs must carry identical evaluation lines
# ----------------------------------------------------------------------------

def eval_lines(log_path, config_name, seed):
    """'[Evaluation] Total Acc ...' and '[Evaluation] FAA ...' lines (timestamp
    stripped) of the LAST block of (config_name, seed) in log_path; None when the
    log or the block is missing."""
    if not os.path.isfile(log_path):
        return None
    text = rc.read_log(log_path)
    heads = list(_HEADER_RE.finditer(text))
    block = None
    for i, m in enumerate(heads):
        if m.group("cfg").strip() == config_name and int(m.group("seed")) == int(seed):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            block = text[m.start():end]
    if block is None:
        return None
    return [m.group(0) for m in (_EVAL_RE.search(ln) for ln in block.splitlines()) if m]


def compare_validate(ds, name_on, name_off, seed, log_root):
    """Exit code 0 when the evaluation lines of the two cells are identical."""
    on = eval_lines(rc.log_path(ds, name_on, log_root), name_on, seed)
    off = eval_lines(rc.log_path(ds, name_off, log_root), name_off, seed)
    if not on or not off:
        print(f"[validate] FAIL: missing evaluation lines (on: {None if on is None else len(on)}, "
              f"off: {None if off is None else len(off)}) for seed {seed} under {log_root}/{ds}")
        return 1
    same = on == off
    print(f"[validate] {ds} seed {seed}: {len(on)} evaluation lines with drift_diag, {len(off)} without "
          f"-> {'IDENTICAL (PASS)' if same else 'DIFFERENT (FAIL)'}")
    if not same:
        for a, b in zip(on, off):
            if a != b:
                print(f"[validate]   first difference:\n    on : {a}\n    off: {b}")
                break
        if len(on) != len(off):
            print(f"[validate]   line counts differ ({len(on)} vs {len(off)})")
    return 0 if same else 1


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap)
    ap.add_argument("--splits", nargs="+", choices=SPLIT_CHOICES, default=["t10"],
                    help="splits to run (default t10; t20/t50 need the run_long chains)")
    ap.add_argument("--validate", action="store_true",
                    help="run the first seed of the first dataset with and without drift_diag "
                         "and compare the evaluation lines of the two logs")
    ap.add_argument("--source", choices=("train", "test"), default="train",
                    help="drift_diag_source (default train)")
    ap.add_argument("--max-per-class", type=int, default=100,
                    help="drift_diag_max_per_class (default 100; 0 = all images)")
    ap.add_argument("--diag-seed", type=int, default=0, help="drift_diag_seed (default 0)")
    ap.add_argument("--no-onestep", action="store_true", help="drift_diag_onestep=False")
    ap.add_argument("--kl", action="store_true", help="drift_diag_kl=True (symmetric KL columns)")
    ap.add_argument("--prefix", default=None,
                    help="train_prefix of a long chain (one dataset x long split only)")
    ap.add_argument("--train-missing", action="store_true",
                    help="run a long split even when its chain exists nowhere (the job trains it)")
    args = rc.apply_common_args(ap.parse_args(argv))

    if args.validate:
        ds, split, seeds = args.datasets[0], args.splits[0], args.seeds[:1]
        if len(args.datasets) > 1 or len(args.splits) > 1 or len(args.seeds) > 1:
            print(f"[validate] using {ds} {split} seed {seeds[0]} (first of the given datasets/splits/seeds)")
        chain = chain_of(ds, split, seeds, args)
        if chain is None:
            raise SystemExit(f"{ds} has no {split} split")
        prefix, T, note, extra = chain
        job = f"driftdiag_{ds}_{split}_validate"
        print(f"[plan] {ds} {split} validate: prefix={prefix} ({note})")
        ckpt_dir = run_long.prepare_ckpt_dir(ds, prefix, seeds, T, job, args.ckpt_src,
                                             dry_run=args.dry_run, smoke=args.smoke, extra_src=extra)
        cells = build_cells(ds, split, prefix, ckpt_dir, args, validate=True)
        if args.redo:
            run_long.backup_side_csv(next(iter(cells.values()))["drift_diag_csv"], dry_run=args.dry_run)
        rc.launch(ds, cells, seeds, **rc.launch_kwargs(args))
        if args.dry_run:
            rc.print_cells(len(cells))
            return 0
        log_root = rc.SMOKE_LOG_ROOT if args.smoke else (args.log_root or rc.LOG_DIR)
        names = [n + ("_smoke" if args.smoke else "") for n in cells]
        return compare_validate(ds, names[0], names[1], seeds[0], log_root)

    plan = []
    for ds in args.datasets:
        for split in args.splits:
            chain = chain_of(ds, split, args.seeds, args)
            if chain is None:
                print(f"[skip] {ds}: no {split} split (run_long.SPLITS)")
                continue
            plan.append((ds, split) + chain)
    long_cells = [p for p in plan if p[1] != "t10"]
    if args.prefix and len(long_cells) != 1:
        raise SystemExit("--prefix applies to exactly one dataset x long split")
    if not plan:
        raise SystemExit("nothing to run")

    total = 0
    for ds, split, prefix, T, note, extra in plan:
        job = f"driftdiag_{ds}_{split}"
        print(f"[plan] {ds} {split} ({T} tasks) prefix={prefix} -- {note}")
        if split != "t10" and not args.train_missing:
            if run_long.find_chain_dir(ds, prefix, args.seeds, T, [args.ckpt_src, *extra,
                                                                   rc.isolated_ckpt_path(job)]) is None:
                print(f"[skip] {ds} {split}: the {T}-task chain {prefix} exists nowhere "
                      f"(run run_long.py first, or pass --train-missing)")
                continue
        ckpt_dir = run_long.prepare_ckpt_dir(ds, prefix, args.seeds, T, job, args.ckpt_src,
                                             dry_run=args.dry_run, smoke=args.smoke, extra_src=extra)
        cells = build_cells(ds, split, prefix, ckpt_dir, args)
        if args.redo:
            for c in cells.values():
                run_long.backup_side_csv(c["drift_diag_csv"], dry_run=args.dry_run)
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))
        total += len(cells)
    if args.dry_run:
        rc.print_cells(total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
