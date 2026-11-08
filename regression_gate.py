"""Default-path regression gate (plan, ground rule 2 and server day 0, step 3).

Re-runs ONE cached headline seed with the integrated code in an isolated
checkpoint directory and compares its per-task `Grouped:` vectors with the
reference block of the same chain in the existing logs.  The default cell is
ImageNet-R, seed 1993, the paper posture (pspectral only_A p 0.9 alpha 0.3,
lambda 100), i.e. rebuttal_common.cfg("imagenetr") with nothing else set.  If
the vectors differ at all, a default-on code path changed and NO rebuttal job
should start before the cause is found.

The gate log lands under logs_main/gate/<ds>/gate_<ds>_s<seed>.log (not in the
dataset folder, so parsers that read logs_main/<ds> never see it).  Reference
blocks are found with parse_logs: every finished block in logs_main/<ds>/*.log
whose seed and posture keys equal the gate cell's (train_prefix, train_merge,
variant, p, alpha, lambda, ridge, transport, local-term weight, weight decay,
cov mode, split) and whose task count matches.

Usage
    python regression_gate.py --dry-run           # show the cell, the dir and the references found
    python regression_gate.py                     # run (about 10 GPU-min) and compare
    python regression_gate.py --compare-only      # compare an already finished gate log
    python regression_gate.py --ds cub --seed 1994
    python regression_gate.py --ref-log logs_main/imagenetr/best_imagenetr.log
Exit code 0 = PASS (bit-identical vectors), 1 = FAIL or no reference, 2 = usage.
"""
import argparse
import glob
import os
import sys

import rebuttal_common as rc

GATE_LOG_ROOT = os.path.join(rc.LOG_DIR, "gate")
# posture keys that must agree between the gate cell and a reference block
POSTURE_KEYS = ("train_method", "train_prefix", "train_merge", "pspectral_p", "train_merge_alpha",
                "train_drift", "train_drift_ridge", "train_drift_lambda", "train_drift_transport",
                "train_ca", "train_ca_robust_weight", "train_ca_cov_shrinkage", "train_weight_decay",
                "model_lora_r", "model_lora_alpha", "dataset_num_task", "dataset_init_cls",
                "dataset_increment", "train_epochs")
# keys whose absence in an old log means the default (pre-rebuttal behaviour)
DEFAULT_IF_MISSING = {"train_ca_cov_mode": "class", "merge_update_op": "trunc",
                      "merge_alpha_rule": "fixed", "train_drift_solver": "dense",
                      "train_drift_lambda_rule": "fixed"}


def gate_name(ds, seed):
    return f"gate_{ds}_s{seed}"


def _eq(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return str(a) == str(b)


def posture_matches(cfg_run, cfg_gate):
    """True when every posture key of the gate cell equals the block's value."""
    for k in POSTURE_KEYS:
        if k not in cfg_gate:
            continue
        if k not in cfg_run:
            return False
        if not _eq(cfg_run[k], cfg_gate[k]):
            return False
    for k, default in DEFAULT_IF_MISSING.items():
        if str(cfg_run.get(k, default)) != str(cfg_gate.get(k, default)):
            return False
    v_run = cfg_run.get("spectral_variant", cfg_run.get("pspectral_variant", "only_A"))
    v_gate = cfg_gate.get("spectral_variant", cfg_gate.get("pspectral_variant", "only_A"))
    return str(v_run) == str(v_gate)


def find_references(ds, seed, cfg_gate, ref_log=None, exclude_path=None):
    """Finished reference blocks (parse_logs run dicts) of the gate cell's chain."""
    import parse_logs as pl
    if ref_log:
        files = [ref_log]
    else:
        files = sorted(glob.glob(os.path.join(rc.LOG_DIR, ds, "*.log")))
    refs = []
    for path in files:
        if exclude_path and os.path.abspath(path) == os.path.abspath(exclude_path):
            continue
        try:
            runs = pl.parse_file(path)
        except Exception as exc:  # unreadable or foreign file: skip, do not abort the gate
            print(f"[gate] skip {path}: {type(exc).__name__}: {exc}")
            continue
        for run in runs:
            if run.get("dataset") != ds or run.get("seed") != seed:
                continue
            if run.get("status") not in (None, "ok") and run.get("fail_reason"):
                continue
            if not run.get("rows"):
                continue
            if posture_matches(run.get("config") or {}, cfg_gate):
                refs.append(run)
    return refs


def compare_rows(gate_rows, ref_rows):
    """(identical: bool, max_abs_diff: float, n_tasks_compared: int)."""
    T = min(len(gate_rows), len(ref_rows))
    worst = 0.0
    for t in range(T):
        g, r = list(gate_rows[t]), list(ref_rows[t])
        if len(g) != len(r):
            return False, float("inf"), T
        worst = max(worst, max(abs(float(a) - float(b)) for a, b in zip(g, r)) if g else 0.0)
    identical = (len(gate_rows) == len(ref_rows)) and worst == 0.0
    return identical, worst, T


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", default="imagenetr", choices=rc.DATASETS)
    ap.add_argument("--seed", type=int, default=1993)
    ap.add_argument("--prefix-family", choices=rc.PREFIX_FAMILIES, default=None)
    ap.add_argument("--ckpt-src", default=rc.SEQFT_CKPT_DIR, help="directory of the cached chains")
    ap.add_argument("--ref-log", default=None, help="compare against this log only")
    ap.add_argument("--dry-run", action="store_true", help="show the cell and the references; run nothing")
    ap.add_argument("--compare-only", action="store_true", help="skip the run, compare the existing gate log")
    ap.add_argument("--redo", action="store_true", help="back up an existing gate log and rerun")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="tolerance on the Grouped entries (default 0 = bit-identical required)")
    args = ap.parse_args(argv)
    if args.prefix_family:
        rc.set_prefix_family(args.prefix_family)

    ds, seed = args.ds, args.seed
    name = gate_name(ds, seed)
    prefix, T = rc.prefix(ds), rc.n_tasks(ds)
    import run_long
    ckpt_dir = run_long.prepare_ckpt_dir(ds, prefix, [seed], T, name, args.ckpt_src,
                                         dry_run=args.dry_run or args.compare_only)
    cfg_gate = rc.cfg(ds, checkpoint_dir=ckpt_dir)
    log = rc.log_path(ds, name, GATE_LOG_ROOT)
    print(f"[gate] cell {name}: chain {prefix}, {T} tasks, merged tag {rc.merge_tag(cfg_gate)}, "
          f"log {log}, checkpoints {ckpt_dir}")

    refs = find_references(ds, seed, cfg_gate, ref_log=args.ref_log, exclude_path=log)
    print(f"[gate] {len(refs)} reference block(s) with the same posture and seed:")
    for r in refs:
        print(f"[gate]   {r.get('path')} :: {r.get('config_name')} ({len(r['rows'])} tasks, "
              f"FAA {r.get('faa')})")
    if args.dry_run:
        if not refs:
            print("[gate] WARNING: no reference block found; the gate could only report, not pass")
        return 0
    if not refs:
        print("[gate] FAIL: no reference block found in logs_main/%s; pass --ref-log" % ds)
        return 1

    if not args.compare_only:
        rc.launch(ds, {name: cfg_gate}, [seed], log_root=GATE_LOG_ROOT, redo=args.redo)
    import parse_logs as pl
    if not os.path.isfile(log):
        print(f"[gate] FAIL: gate log {log} missing")
        return 1
    gate_runs = [r for r in pl.parse_file(log) if r.get("seed") == seed and r.get("rows")]
    if not gate_runs:
        print(f"[gate] FAIL: no finished block for seed {seed} in {log}")
        return 1
    gate = gate_runs[-1]
    ok_any = False
    for r in refs:
        identical, worst, n = compare_rows(gate["rows"], r["rows"])
        passed = identical or (args.tol > 0 and worst <= args.tol and len(gate["rows"]) == len(r["rows"]))
        ok_any = ok_any or passed
        print(f"[gate] vs {r.get('path')} :: {r.get('config_name')}: "
              f"{'PASS' if passed else 'FAIL'} ({n} tasks, max |diff| {worst:.6g}, "
              f"FAA gate {gate.get('faa')} ref {r.get('faa')})")
    print("[gate] RESULT:", "PASS" if ok_any else "FAIL")
    return 0 if ok_any else 1


if __name__ == "__main__":
    sys.exit(main())
