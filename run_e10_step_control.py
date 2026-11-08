"""E10: truncation versus other step limits at matched spectral norm.

Merge-time operators (cache-served, Stage-2-only re-runs, about 10 min per seed).
helper.pspectral_merging(update_op=...) applies one of three steps to the lora_A
update dA = A_t - A_{M,t-1} (merge_ops.spectral_step):
  trunc      alpha * trunc_p(dA): ISM's step, p = 0.9 (the paired reference,
             re-run here with the [MergeStat] norms)
  clip       U min(S, alpha sigma_1) Vh: same spectral norm alpha sigma_1 as the
             truncated step, every direction kept, the tail untouched
  scale_fro  beta dA with beta = alpha ||trunc_p(dA)||_F / ||dA||_F:
             Frobenius-matched scaling of the full update
Full-update scaling at matched spectral norm is p = 1 at the same alpha, which
E8 and the cubes already cover.  B and norm.* keep the alpha interpolation in
every op; the [MergeStat] line documents the realised sigma_max / Frobenius norms
of the A-step, the effective step and the raw update, so "matched" is measured.

`--part merge` (default), 15 cells = 45 cached runs (about 7.5 GPU-h):
  e10_trunc_a{a}, e10_clip_a{a}, e10_scalefro_a{a} at the headline alpha on the
  four datasets (12), plus the clip curve on ImageNet-R at alpha in {0.3, 0.5,
  0.7, 1.0}: 0.3 is the headline cell itself (one cell, not two) and alpha 1.0
  is the identity step (clip at sigma_1 changes nothing, so this cell must equal
  the no-merge corner of the cube; a sanity cell the design asked for).

Training-time penalty (retrains the chain under a new prefix; the E2 time table
gives the cost).  l2m_weight > 0 adds l2m_weight * sum ||theta - theta_anchor||^2
/ numel over the trainable LoRA tensors to every batch loss of task t >= 1, with
anchor = the previously deployed module.  The E10 cell uses train_merge = "none",
so the deployed model is W_t, the anchor is W_{t-1}, and regularisation replaces
merging ("[StepNorm]" lines give the realised norms of W_t - W_{t-1}).
  --part l2m-pilot [--weights 0.1 1 10]   3 tasks (train_stop_at_task 2), first
                                          seed, prefixes <prefix>_l2m{w}
  --part l2m-report                       torch-free: prints the pilot's StepNorm
                                          sigma_max per task next to the
                                          e10_trunc MergeStat sigma_max and
                                          suggests the two bracketing weights
  --part l2m --weights w1 w2              full 10-task runs x 3 seeds under the
                                          same prefixes (the pilot's tasks 0-2
                                          are reused from the cache)
  --copy-task0                            copies the cached task-0 files of the
                                          headline chain to the l2m prefixes
                                          (task 0 is unpenalised, so W_0 is
                                          identical; saves one task of training)

Outputs: logs_main/<ds>/e10_{trunc,clip,scalefro}_a{a}.log and
logs_main/imagenetr/e10_l2m{w}[_pilot].log with [MergeStat] / [StepNorm] / [L2M]
lines; then `python parse_logs.py e10` -> figures_data/e10_step_control.csv.

Run:
    python run_e10_step_control.py --part merge --dry-run
    python run_e10_step_control.py --part merge --smoke        # clip on ImageNet-R, 2 tasks
    python run_e10_step_control.py --part merge --datasets imagenetr cifar224 cub cars
    python run_e10_step_control.py --part l2m-pilot
    python run_e10_step_control.py --part l2m-report
    python run_e10_step_control.py --part l2m --weights <w1> <w2> --copy-task0
Merge cells run in checkpoints_main/rebuttal/e10_<ds> (one process per
directory); the L2M chains live in checkpoints_main/rebuttal/e10_l2m_<ds>.
"""
import argparse
import os
import re
import shutil
import sys
import time
from collections import OrderedDict

import rebuttal_common as rc

OPS = OrderedDict([("trunc", "trunc"), ("clip", "clip"), ("scale_fro", "scalefro")])  # op -> name token
CLIP_CURVE_DATASET = "imagenetr"
CLIP_ALPHAS = (0.3, 0.5, 0.7, 1.0)      # 0.3 = headline cell, 1.0 = identity sanity cell
L2M_DATASET = "imagenetr"
PILOT_WEIGHTS = (0.1, 1.0, 10.0)
PILOT_STOP_AT_TASK = 2                   # tasks 0, 1, 2 (learn() breaks when task >= stop)
JOB = "e10"
PARTS = ("merge", "l2m-pilot", "l2m", "l2m-report")


def _f(x):
    """Canonical float so 0.30 and 0.3 give one cell and one tag."""
    return float(f"{float(x):.6g}")


# ----------------------------------------------------------------------------
# Merge-time cells
# ----------------------------------------------------------------------------

def merge_cells(ds, ops=tuple(OPS), clip_alphas=CLIP_ALPHAS, job_suffix=""):
    """OrderedDict name -> config: the three ops at the headline alpha, plus the
    clip curve on ImageNet-R.  De-duplicated by name."""
    headline = _f(rc.HEADLINE_ALPHA[ds])
    ckpt = rc.isolated_ckpt_path(f"{JOB}_{ds}{job_suffix}")
    cells = OrderedDict()

    def add(op, a):
        name = f"e10_{OPS[op]}_a{a}"
        if name not in cells:
            cells[name] = rc.cfg(ds, merge_update_op=op, train_merge_alpha=a,
                                 merge_log_stats=True, checkpoint_dir=ckpt)

    for op in ops:
        add(op, headline)
    if ds == CLIP_CURVE_DATASET and "clip" in ops:
        for a in clip_alphas:
            add("clip", _f(a))
    return cells


# ----------------------------------------------------------------------------
# L2-toward-anchor training cells
# ----------------------------------------------------------------------------

def l2m_prefix(ds, w):
    """New train_prefix per weight (the cache key), e.g. main_inr_r64_qkvfc_l2m1.0."""
    return f"{rc.prefix(ds)}_l2m{float(w)}"


def l2m_cells(ds, weights, pilot, job_suffix=""):
    """OrderedDict name -> config of the merge-less, L2M-penalised chains."""
    ckpt = rc.isolated_ckpt_path(f"{JOB}_l2m_{ds}{job_suffix}")
    cells = OrderedDict()
    for w in weights:
        w = _f(w)
        if w <= 0:
            raise SystemExit(f"l2m_weight must be positive, got {w}")
        name = f"e10_l2m{float(w)}" + ("_pilot" if pilot else "")
        over = dict(train_merge="none", l2m_weight=w, train_prefix=l2m_prefix(ds, w),
                    merge_log_stats=True, checkpoint_dir=ckpt)
        if pilot:
            over["train_stop_at_task"] = PILOT_STOP_AT_TASK
        cells[name] = rc.cfg(ds, **over)
    return cells


def copy_task0(ds, seeds, src_prefix, dst_prefixes, src_dir, dst_dir, dry_run):
    """Copy {stem}_backbone_base/_backbone_0/_classifier_0.pt of the headline chain
    to the l2m prefixes inside dst_dir.  Task 0 is identical (the penalty starts at
    task 1), so the l2m run then reuses it ("[Training] Reusing cached ...")."""
    if not dry_run:
        os.makedirs(dst_dir, exist_ok=True)
    for dst_prefix in dst_prefixes:
        for seed in seeds:
            s_stem = rc.chain_stem(seed, ds, src_prefix)
            d_stem = rc.chain_stem(seed, ds, dst_prefix)
            for suffix in ("_backbone_base.pt", "_backbone_0.pt", "_classifier_0.pt"):
                src = os.path.join(src_dir, s_stem + suffix)
                dst = os.path.join(dst_dir, d_stem + suffix)
                tag = "[copy-task0 dry-run]" if dry_run else "[copy-task0]"
                if os.path.exists(dst):
                    print(f"{tag} kept {dst}")
                elif not os.path.isfile(src):
                    print(f"{tag} MISSING {src} -> {dst} not created (task 0 will be trained)")
                elif dry_run:
                    print(f"{tag} would copy {src} -> {dst}")
                else:
                    shutil.copy2(src, dst)
                    print(f"{tag} copied {src} -> {dst}")


# ----------------------------------------------------------------------------
# Pilot report (torch-free): StepNorm of the l2m chains vs MergeStat of truncation
# ----------------------------------------------------------------------------

_KV_RE = re.compile(r"(\w+)=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def tag_rows(path, tag):
    """[{task, <numeric key>: value, ...}] for every '[tag] task N ...' line of a
    log (all seeds); non-numeric fields such as op=clip are skipped."""
    rows = []
    if not os.path.isfile(path):
        return rows
    head = re.compile(r"\[" + re.escape(tag) + r"\] task (\d+)")
    for line in rc.read_log(path).splitlines():
        m = head.search(line)
        if not m:
            continue
        kv = {k: float(v) for k, v in _KV_RE.findall(line[m.end():])}
        kv["task"] = int(m.group(1))
        rows.append(kv)
    return rows


def per_task_mean(rows, key):
    """{task: mean of key over the rows (seeds) of that task}."""
    acc = {}
    for r in rows:
        if key in r:
            acc.setdefault(r["task"], []).append(r[key])
    return {t: sum(v) / len(v) for t, v in sorted(acc.items())}


def l2m_report(ds, weights, log_root=None, key="eff_sigma_max_mean"):
    """Print the realised effective-step sigma_max per task of the truncation
    reference (e10_trunc_a{headline}, [MergeStat]) next to the l2m chains
    ([StepNorm]) and suggest the bracketing weights.  Returns the suggestion."""
    headline = _f(rc.HEADLINE_ALPHA[ds])
    ref_name = f"e10_{OPS['trunc']}_a{headline}"
    ref = per_task_mean(tag_rows(rc.log_path(ds, ref_name, log_root), "MergeStat"), key)
    runs = OrderedDict()
    for w in weights:
        w = _f(w)
        for name in (f"e10_l2m{float(w)}_pilot", f"e10_l2m{float(w)}"):
            rows = tag_rows(rc.log_path(ds, name, log_root), "StepNorm")
            if rows:
                runs[w] = (name, per_task_mean(rows, key))
                break
    print(f"[l2m-report] {ds}: realised effective-step sigma_max ({key}), mean over seeds")
    if not ref:
        print(f"  reference {rc.log_path(ds, ref_name, log_root)} has no [MergeStat] lines "
              f"(run --part merge on {ds} first)")
    if not runs:
        print("  no [StepNorm] lines found for weights " + ", ".join(str(_f(w)) for w in weights)
              + " (run --part l2m-pilot first)")
        return None
    tasks = sorted({t for t in ref} | {t for _, m in runs.values() for t in m})
    header = f"{'task':>4} {'trunc':>10} " + " ".join(f"{'l2m' + str(w):>10}" for w in runs)
    print("  " + header)
    for t in tasks:
        if t == 0:
            continue
        cells = [f"{ref.get(t, float('nan')):10.4g}"] + [f"{m.get(t, float('nan')):10.4g}" for _, m in runs.values()]
        print(f"  {t:>4} " + " ".join(cells))
    pilot_tasks = [t for t in tasks if 1 <= t <= PILOT_STOP_AT_TASK]
    if not ref or not pilot_tasks:
        return None
    target = sum(ref[t] for t in pilot_tasks if t in ref) / max(1, sum(1 for t in pilot_tasks if t in ref))
    means = {}
    for w, (_, m) in runs.items():
        vals = [m[t] for t in pilot_tasks if t in m]
        if vals:
            means[w] = sum(vals) / len(vals)
    print(f"  target (truncation, tasks {pilot_tasks[0]}-{pilot_tasks[-1]}): {target:.4g}")
    for w, v in means.items():
        print(f"  l2m{w}: {v:.4g} ({'larger' if v > target else 'smaller'} step than truncation)")
    ws = sorted(means)
    suggestion = None
    for lo, hi in zip(ws, ws[1:]):
        if (means[lo] - target) * (means[hi] - target) <= 0:
            suggestion = (lo, hi)
            break
    if suggestion:
        print(f"  suggestion: --weights {suggestion[0]} {suggestion[1]} bracket the truncation step")
    elif len(ws) >= 2 and all(means[w] > 0 for w in ws):
        # log-log fit of sigma vs weight, extrapolated to the target
        import math
        xs = [math.log(w) for w in ws]
        ys = [math.log(means[w]) for w in ws]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        if slope < 0:
            w_star = math.exp(mx + (math.log(target) - my) / slope)
            lo, hi = _f(w_star / 3), _f(w_star * 3)
            suggestion = (lo, hi)
            print(f"  no pilot weight brackets the target; log-log extrapolation gives w* = {w_star:.3g}; "
                  f"suggestion: --weights {lo} {hi}")
        else:
            print("  the step norm does not decrease with the weight; check the [L2M] lines before choosing")
    else:
        print("  not enough pilot weights to bracket the target")
    return suggestion


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap)
    ap.add_argument("--part", choices=PARTS, default="merge",
                    help="merge (default) | l2m-pilot | l2m-report | l2m")
    ap.add_argument("--ops", nargs="+", choices=list(OPS), default=list(OPS),
                    help="merge-time operators (default trunc clip scale_fro)")
    ap.add_argument("--clip-alphas", nargs="*", type=float, default=list(CLIP_ALPHAS),
                    help=f"clip curve on {CLIP_CURVE_DATASET} (default 0.3 0.5 0.7 1.0; "
                         "pass the flag without values for the headline cell only)")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="l2m_weight values (pilot default 0.1 1 10; required for --part l2m)")
    ap.add_argument("--copy-task0", action="store_true",
                    help="copy the headline chain's task-0 files to the l2m prefixes before running")
    ap.add_argument("--smoke-all", action="store_true",
                    help="with --smoke: smoke every cell instead of the representative one")
    ap.add_argument("--job-suffix", default="",
                    help="suffix of the checkpoint directory name (run two processes on one dataset)")
    args = rc.apply_common_args(ap.parse_args(argv))

    # The L2M parts retrain ImageNet-R only unless datasets are given explicitly.
    l2m_part = args.part.startswith("l2m")
    datasets = list(args.datasets)
    if l2m_part and datasets == list(rc.DATASETS):
        datasets = [L2M_DATASET]

    if args.part == "l2m-report":
        for ds in datasets:
            l2m_report(ds, args.weights or PILOT_WEIGHTS, args.log_root)
        return 0

    per_ds = OrderedDict()
    if args.part == "merge":
        for ds in datasets:
            per_ds[ds] = merge_cells(ds, args.ops, args.clip_alphas, args.job_suffix)
    else:
        pilot = args.part == "l2m-pilot"
        weights = args.weights or (list(PILOT_WEIGHTS) if pilot else None)
        if not weights:
            raise SystemExit("--part l2m needs --weights w1 w2 (take them from --part l2m-report)")
        if pilot:
            args.seeds = args.seeds[:1]           # the pilot is a one-seed calibration
        for ds in datasets:
            per_ds[ds] = l2m_cells(ds, weights, pilot, args.job_suffix)

    if args.smoke and not args.smoke_all:
        # one representative cell: clip on ImageNet-R, or the first l2m cell
        ds0 = CLIP_CURVE_DATASET if (args.part == "merge" and CLIP_CURVE_DATASET in per_ds) else next(iter(per_ds))
        cells = per_ds[ds0]
        want = f"e10_clip_a{_f(rc.HEADLINE_ALPHA[ds0])}" if args.part == "merge" else next(iter(cells))
        name = want if want in cells else next(iter(cells))
        per_ds = OrderedDict([(ds0, OrderedDict([(name, cells[name])]))])

    total = sum(len(c) for c in per_ds.values())
    for ds, cells in per_ds.items():
        print(f"[{JOB}] {args.part} {ds}: {len(cells)} cells x {1 if args.smoke else len(args.seeds)} seed(s)"
              + (" (training jobs, new prefixes)" if l2m_part else ""))

    t0 = time.time()
    for ds, cells in per_ds.items():
        if not cells:
            continue
        if not args.smoke:
            if args.part == "merge":
                prefix = next(iter(cells.values()))["train_prefix"]
                rc.isolated_ckpt_dir(ds, prefix, args.seeds, f"{JOB}_{ds}{args.job_suffix}",
                                     src_dir=args.ckpt_src, dry_run=args.dry_run)
            else:
                # Training jobs: a fresh directory without links; optional task-0 copies.
                ckpt = next(iter(cells.values()))["checkpoint_dir"]
                if not args.dry_run:
                    os.makedirs(ckpt, exist_ok=True)
                if args.copy_task0:
                    copy_task0(ds, args.seeds, rc.prefix(ds),
                               [c["train_prefix"] for c in cells.values()],
                               args.ckpt_src, ckpt, args.dry_run)
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))

    if args.dry_run:
        rc.print_cells(total)
        return 0
    if args.part == "l2m-pilot":
        for ds in per_ds:
            l2m_report(ds, [c["l2m_weight"] for c in per_ds[ds].values()], args.log_root)
    print(f"[{JOB}] done: {total} cells. Next: python parse_logs.py e10")
    print(f"Total: {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
