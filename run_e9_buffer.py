"""Drift map fitted on a replay buffer (rebuttal E9, buffer arm; xkNt-Q2).

The paper fits the drift map T_t on the current task's training images under
M_{t-1} and M_t. Here it is fitted instead on a class-balanced buffer of stored
training images: train_drift_buffer_n images of EVERY class seen so far, the
current task included (main.Learner._drift_buffer_extend / _drift_buffer_loader;
the draw uses a private RandomState seeded by the run seed). Everything else is
the headline posture (rebuttal_common.cfg), cache-served on the headline chain,
so the only change against the paper's cell is the paired-feature source of the
ridge solve. Storing images violates the exemplar-free setting; the row is a
reference, stated as such in the appendix.

Cells (logs_main/<ds>/<cell>.log, run_config_sweep format), per dataset x 3 seeds
    e9b_n5, e9b_n10, e9b_n20   buffer of 5 / 10 / 20 images per class (--n)
The paper's own fit is the e12_ism / headline log of the same chain (reference
row; add --with-reference for a paired e9b_ref cell in the same directory).
Default dataset: ImageNet-R only (the appendix default).

    python run_e9_buffer.py --dry-run
    python run_e9_buffer.py --smoke
    python run_e9_buffer.py --ckpt-src checkpoints_main/rebuttal/e12_imagenetr
    python run_e9_buffer.py --n 5 10 20 --with-reference
"""
import argparse
import sys
from collections import OrderedDict

import rebuttal_common as rc
import run_long

DEFAULT_N = (5, 10, 20)
E9_DATASETS = ("imagenetr",)


def job_name(ds):
    """Name of the isolated checkpoint directory shared by the cells of `ds`."""
    return f"e9b_{ds}"


def build_cells(ds, ns, ckpt_dir, with_reference=False):
    """OrderedDict {cell name: config}: one cell per buffer size, optional reference."""
    cells = OrderedDict()
    if with_reference:
        cells["e9b_ref"] = rc.cfg(ds, checkpoint_dir=ckpt_dir)
    for n in ns:
        cells[f"e9b_n{int(n)}"] = rc.cfg(ds, checkpoint_dir=ckpt_dir, train_drift_buffer_n=int(n))
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, datasets=E9_DATASETS)
    ap.add_argument("--n", nargs="+", type=int, default=list(DEFAULT_N),
                    help="stored images per class (default 5 10 20)")
    ap.add_argument("--with-reference", action="store_true",
                    help="add the paper's own fit (current-task images) as cell e9b_ref")
    args = rc.apply_common_args(ap.parse_args(argv))
    if any(n <= 0 for n in args.n):
        raise SystemExit(f"--n must be positive integers, got {args.n}")

    total = 0
    for ds in args.datasets:
        prefix, T = rc.prefix(ds), rc.n_tasks(ds)
        print(f"[plan] {ds}: buffer sizes {args.n} on chain {prefix} ({T} tasks)"
              + (" + reference" if args.with_reference else ""))
        ckpt_dir = run_long.prepare_ckpt_dir(ds, prefix, args.seeds, T, job_name(ds), args.ckpt_src,
                                             dry_run=args.dry_run, smoke=args.smoke)
        cells = build_cells(ds, args.n, ckpt_dir, with_reference=args.with_reference)
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))
        total += len(cells)
    if args.dry_run:
        rc.print_cells(total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
