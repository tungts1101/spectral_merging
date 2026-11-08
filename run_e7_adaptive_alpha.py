"""Adaptive damping alpha (specnorm / featshift rules) and the data-free GCV
ridge lambda on the cached headline chains (rebuttal E7; q6ZZ-Q2).

The main.py keys (B1): merge_alpha_rule in {fixed, specnorm, featshift},
merge_alpha_param (tau / kappa), merge_alpha_stat in {eff, A}, merge_alpha_min,
with alpha_t = clip(param / s_t, merge_alpha_min, 1) and
  specnorm : s_t = mean over the 36 LoRA pairs of sigma_max of the raw effective
             update scale (B_t A_t - B_M A_M) (stat 'eff') or of dA (stat 'A'),
             computed from the states alone (no data);
  featshift: s_t = mean ||f_t - f_{t-1}|| / mean ||f_{t-1}|| over the current
             task's train images between M_{t-1} and W_t ([FeatShift] line).
train_drift_lambda_rule = 'gcv' picks lambda per task by closed-form generalised
cross-validation of the ridge on the current task's paired features over
train_drift_lambda_grid ({1, 3, 10, 30, 100, 300, 1000}).

Protocol (the paper's own): tau and kappa are set ONCE on ImageNet-R so the
realised alpha averages its headline 0.30, then the same values run on the four
datasets; the questions are whether the realised alphas land near 0.15 / 0.10 /
0.20 on CIFAR-100 / CUB / Cars and whether FA matches the paired headline cell.

Modes
  --calibrate   tau from figures_data/e4_spectrum_imagenetr.csv (analyze_spectrum.py,
                space 'eff' by default) by bisection on mean_t clip(tau / s_t) = target
                (seeds x tasks of the headline chain); kappa from the [FeatShift]
                lines of one cached calibration run e7_calib_featshift (ImageNet-R,
                fixed rule + merge_log_featshift, --calib-seeds, 1 cell); writes
                figures_data/e7_calibration.json and prints the offline preview of
                the specnorm alphas for every dataset whose E4 CSV exists.
                (The preview uses the headline chain's M_{t-1}, not the adaptive
                chain's, so it is an approximation of the realised values.)
  --run         2 rules x --datasets cells e7_{rule}_{param:g}, params from the json
                (or --tau / --kappa), merge_alpha_stat 'eff', merge_alpha_min 0.05,
                merge_log_stats True; --param-scale f1 f2 adds the scaled parameters
                as extra cells (sensitivity). Default mode when no mode flag is given.
  --lambda-gcv  one cell e7_lambda_gcv per dataset with train_drift_lambda_rule 'gcv'
                (headline alpha, fixed rule).
  --with-reference  adds the paired fixed-rule headline cell e7_fixed per dataset (the
                same chain; E8 / E10 produce the identical cell under their own names).
Every cell is cache-served (10-task chains under checkpoints_main/seqft) and runs
in its own isolated checkpoint directory checkpoints_main/rebuttal/<name>
(rebuttal_common.isolated_ckpt_dir); logs go to logs_main/<ds>/<name>.log and
parse_logs.py e7 reads the '[Merging] adaptive alpha ...' / '[FeatShift]' lines.

Run
    python run_e7_adaptive_alpha.py --dry-run                     # 8 cells, no torch needed
    python run_e7_adaptive_alpha.py --calibrate                   # after analyze_spectrum.py --ds imagenetr
    python run_e7_adaptive_alpha.py --run                         # 2 rules x 4 datasets x 3 seeds
    python run_e7_adaptive_alpha.py --run --param-scale 0.7 1.4   # sensitivity cells
    python run_e7_adaptive_alpha.py --lambda-gcv                  # 4 datasets x 3 seeds
    python run_e7_adaptive_alpha.py --smoke                       # 2-task smoke of every cell, seed 1993
Cost: calibration 1 cached run; --run 24 cached runs (about 4 GPU-h); --lambda-gcv
12 cached runs (about 2 GPU-h).
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict

import rebuttal_common as rc

RULES = ("specnorm", "featshift")
CALIB_DS = "imagenetr"
CALIB_NAME = "e7_calib_featshift"
GCV_NAME = "e7_lambda_gcv"
FIXED_NAME = "e7_fixed"
ALPHA_MIN, ALPHA_MAX = 0.05, 1.0
DEFAULT_STAT = "eff"
CALIB_JSON = os.path.join("figures_data", "e7_calibration.json")
E4_CSV = os.path.join("figures_data", "e4_spectrum_{ds}.csv")
# Placeholder parameters for a dry-run before the calibration exists.  A real
# --run on them is refused (check_params_ready) unless --allow-placeholder is set.
PLACEHOLDER_PARAMS = {"specnorm": 1.0, "featshift": 1.0}
PLACEHOLDER_SOURCE = "PLACEHOLDER (run --calibrate)"

# real line: '[FeatShift] task {t} shift={:.6g} n={n_images}' (main.py B1); the
# tests/make_fake_log fixture writes 'shift=0.0712 (rel) n=2400', so n= is optional here
_FEATSHIFT_RE = re.compile(r"\[FeatShift\] task (?P<task>\d+) shift=(?P<shift>[-+0-9.eE]+)")


# ---------------------------------------------------------------------------
# Calibration arithmetic (numpy only)
# ---------------------------------------------------------------------------

def clipped_alpha(param, stats, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX):
    """alpha_t = clip(param / s_t) for every statistic (merge_ops.alpha_from_stat)."""
    out = []
    for s in stats:
        s = float(s)
        out.append(alpha_max if not (s > 0.0) else min(alpha_max, max(alpha_min, param / s)))
    return out


def solve_param(stats, target, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX, tol=1e-9):
    """Smallest param with mean_t clip(param / s_t, alpha_min, alpha_max) >= target
    (the mean is non-decreasing and continuous in param, so bisection is exact).

    Returns (param, mean_alpha). target must lie in [alpha_min, alpha_max]; stats
    must be positive and finite. The unclipped solution target / mean(1 / s_t)
    (harmonic rule) is the starting bracket.
    """
    stats = [float(s) for s in stats if s == s and s > 0.0]
    if not stats:
        raise ValueError("no positive statistics to calibrate on")
    if not alpha_min <= target <= alpha_max:
        raise ValueError(f"target {target} outside [{alpha_min}, {alpha_max}]")

    def mean_alpha(param):
        return sum(clipped_alpha(param, stats, alpha_min, alpha_max)) / len(stats)

    lo, hi = 0.0, target * max(stats)        # at hi every alpha >= target
    while mean_alpha(hi) < target:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_alpha(mid) >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return hi, mean_alpha(hi)


def round_param(x, digits=4):
    """Round to `digits` significant digits so the merged-checkpoint tag
    (a{rule}{param}) stays short and the json, config and tag agree."""
    return float(f"{x:.{digits}g}")


def specnorm_stats(csv_path, stat=DEFAULT_STAT):
    """{(seed, task): mean over pairs of sigma_1} from an e4_spectrum CSV; stat
    'eff' = the effective update, 'A' = the lora_A update (the two
    merge_alpha_stat values)."""
    space = {"eff": "eff", "A": "A"}[stat]
    acc = {}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["space"] != space:
                continue
            acc.setdefault((int(r["seed"]), int(r["task"])), []).append(float(r["sigma1"]))
    return {k: sum(v) / len(v) for k, v in sorted(acc.items())}


def featshift_stats(log_path, config_name=CALIB_NAME):
    """{(seed, task): shift} from the '[FeatShift] task t shift=.. n=..' lines of
    the calibration log (last block per seed, through parse_logs when available)."""
    out = {}
    try:
        import parse_logs
        runs = parse_logs.parse_file(log_path)
        for run in runs:
            if run.get("config_name") not in (None, config_name):
                continue
            for line in run.get("tags", {}).get("FeatShift", []):
                m = _FEATSHIFT_RE.search(line)
                if m:
                    out[(int(run["seed"]), int(m.group("task")))] = float(m.group("shift"))
        if out:
            return out
    except Exception as exc:  # fall back to a plain scan of the file
        print(f"[E7] parse_logs unavailable for {log_path} ({type(exc).__name__}: {exc}); scanning")
    seed = None
    text = rc.read_log(log_path)
    for line in text.splitlines():
        hm = rc._HEADER_RE.search(line)
        if hm:
            seed = int(hm.group("seed")) if hm.group("cfg").strip() == config_name else None
            continue
        m = _FEATSHIFT_RE.search(line)
        if m and seed is not None:
            out[(seed, int(m.group("task")))] = float(m.group("shift"))
    return out


def preview(param, stats_by_key, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX):
    """(mean alpha, {task: mean over seeds of alpha_t}) for a parameter."""
    per_task = {}
    for (seed, task), s in stats_by_key.items():
        per_task.setdefault(task, []).append(clipped_alpha(param, [s], alpha_min, alpha_max)[0])
    task_means = {t: sum(v) / len(v) for t, v in sorted(per_task.items())}
    all_alpha = [a for v in per_task.values() for a in v]
    return (sum(all_alpha) / len(all_alpha) if all_alpha else float("nan")), task_means


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def cell_name(rule, param):
    return f"e7_{rule}_{param:g}"


def load_params(args):
    """{rule: param} from --tau/--kappa, else the calibration json, else placeholders."""
    params, source = {}, {}
    data = {}
    if os.path.isfile(args.calibration):
        with open(args.calibration, encoding="utf-8") as fh:
            data = json.load(fh)
    for rule, cli in (("specnorm", args.tau), ("featshift", args.kappa)):
        if cli is not None:
            params[rule], source[rule] = float(cli), "cli"
        elif rule in data and data[rule].get("param") is not None:
            params[rule], source[rule] = float(data[rule]["param"]), args.calibration
        else:
            params[rule], source[rule] = PLACEHOLDER_PARAMS[rule], PLACEHOLDER_SOURCE
    return params, source


def check_params_ready(args, source):
    """Refuse a real --run whose parameters are the placeholders: 24 cached runs at
    tau = kappa = 1.0 would be launched and tagged a{rule}1.0 without a calibration
    behind them.  Dry runs and smoke runs pass; --allow-placeholder overrides."""
    missing = [rule for rule in args.rules if source[rule] == PLACEHOLDER_SOURCE]
    if missing and not (args.dry_run or args.smoke or args.allow_placeholder):
        raise SystemExit(
            f"[E7] no calibrated parameter for {', '.join(missing)} (no {args.calibration} and no "
            f"--tau/--kappa): run `python run_e7_adaptive_alpha.py --calibrate` first, pass the "
            f"parameters explicitly, or pass --allow-placeholder to launch the placeholder 1.0 anyway")


def build_cells(args, params):
    """OrderedDict {ds: OrderedDict {name: config}} of the selected modes."""
    cells = OrderedDict((ds, OrderedDict()) for ds in args.datasets)
    stat = args.stat
    if args.run:
        for ds in args.datasets:
            for rule in args.rules:
                values = [params[rule]] + [round_param(params[rule] * f) for f in args.param_scale]
                for v in values:
                    name = cell_name(rule, v)
                    if name in cells[ds]:
                        continue
                    cells[ds][name] = rc.cfg(
                        ds, merge_alpha_rule=rule, merge_alpha_param=float(v),
                        merge_alpha_stat=stat, merge_alpha_min=ALPHA_MIN, merge_log_stats=True)
    if args.lambda_gcv:
        for ds in args.datasets:
            cells[ds][GCV_NAME] = rc.cfg(ds, train_drift_lambda_rule="gcv", merge_log_stats=True)
    if args.with_reference:
        # Paired fixed-rule headline cell on the same chain (the FA reference of the
        # e7 table); E8 / E10 produce the same cell under their own names, so it is opt-in.
        for ds in args.datasets:
            cells[ds][FIXED_NAME] = rc.cfg(ds, merge_log_stats=True)
    return cells


def calib_cell(args):
    """The featshift calibration cell: fixed rule + [FeatShift] lines on ImageNet-R."""
    return CALIB_NAME, rc.cfg(CALIB_DS, merge_log_featshift=True, merge_log_stats=True)


def with_ckpt_dir(ds, name, config, seeds, args):
    """Attach the isolated checkpoint directory of the cell (created unless --dry-run)."""
    config = dict(config)
    config["checkpoint_dir"] = rc.isolated_ckpt_dir(
        ds, config["train_prefix"], seeds, name, src_dir=args.ckpt_src, dry_run=args.dry_run)
    return config


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def do_calibrate(args):
    """tau from the E4 CSV, kappa from the calibration run; writes the json.
    Returns the number of launched cells (0 or 1)."""
    target = args.target if args.target is not None else rc.HEADLINE_ALPHA[CALIB_DS]
    result = {"dataset": CALIB_DS, "target_alpha": target, "alpha_min": ALPHA_MIN,
              "alpha_max": ALPHA_MAX, "stat": args.stat}
    e4 = args.e4_csv or E4_CSV.format(ds=CALIB_DS)
    if os.path.isfile(e4):
        for stat in ("eff", "A"):
            stats = specnorm_stats(e4, stat)
            param, mean_alpha = solve_param(stats.values(), target)
            param = round_param(param)
            mean_alpha, per_task = preview(param, stats)
            entry = {"param": param, "mean_alpha": mean_alpha, "n_stats": len(stats),
                     "mean_stat": sum(stats.values()) / len(stats), "per_task_alpha": per_task,
                     "source": e4}
            result["specnorm" if stat == args.stat else f"specnorm_{stat}"] = entry
            print(f"[E7] specnorm stat={stat}: tau = {param:g} -> mean alpha {mean_alpha:.4f} "
                  f"over {len(stats)} (seed, task) statistics of {e4}")
            print("      per task: " + " ".join(f"t{t}={a:.3f}" for t, a in per_task.items()))
    else:
        print(f"[E7] {e4} missing: run `python analyze_spectrum.py --ds {CALIB_DS}` first; "
              f"specnorm tau not calibrated")

    # featshift: one cached ImageNet-R run with the fixed rule and the [FeatShift] lines
    name, config = calib_cell(args)
    seeds = list(args.calib_seeds)
    launched = 0
    log = rc.log_path(CALIB_DS, name, args.log_root)
    if args.dry_run or not rc.has_summary(log, name, len(seeds)):
        config = with_ckpt_dir(CALIB_DS, name, config, seeds, args)
        rc.launch(CALIB_DS, OrderedDict([(name, config)]), seeds, reseed=True, **rc.launch_kwargs(args))
        launched = 1
    if not args.dry_run and os.path.isfile(log):
        shifts = featshift_stats(log, name)
        if shifts:
            param, mean_alpha = solve_param(shifts.values(), target)
            param = round_param(param)
            mean_alpha, per_task = preview(param, shifts)
            result["featshift"] = {"param": param, "mean_alpha": mean_alpha, "n_stats": len(shifts),
                                   "mean_stat": sum(shifts.values()) / len(shifts),
                                   "per_task_alpha": per_task, "source": log}
            print(f"[E7] featshift: kappa = {param:g} -> mean alpha {mean_alpha:.4f} over "
                  f"{len(shifts)} (seed, task) shifts of {log}")
        else:
            print(f"[E7] no [FeatShift] lines in {log}; kappa not calibrated")

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.calibration) or ".", exist_ok=True)
        with open(args.calibration, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print(f"[E7] wrote {args.calibration}")

    # offline preview of the specnorm rule on every dataset (headline chain M_{t-1})
    if "specnorm" in result:
        tau = result["specnorm"]["param"]
        print(f"\n[E7] offline preview, specnorm stat={args.stat}, tau={tau:g} "
              f"(statistics of the HEADLINE chains, so an approximation):")
        print("| dataset | headline alpha | mean realised alpha (preview) | per task |")
        print("|---|---:|---:|---|")
        for ds in rc.DATASETS:
            path = E4_CSV.format(ds=ds)
            if not os.path.isfile(path):
                print(f"| {ds} | {rc.HEADLINE_ALPHA[ds]:.2f} | (no {path}) | |")
                continue
            mean_alpha, per_task = preview(tau, specnorm_stats(path, args.stat))
            print(f"| {ds} | {rc.HEADLINE_ALPHA[ds]:.2f} | {mean_alpha:.3f} | "
                  + " ".join(f"{a:.2f}" for a in per_task.values()) + " |")
    return launched


def do_cells(args):
    """--run / --lambda-gcv cells through rebuttal_common.launch. Returns the count."""
    params, source = load_params(args)
    if args.run:
        for rule in args.rules:
            print(f"[E7] {rule}: param {params[rule]:g} from {source[rule]}")
        check_params_ready(args, source)
    cells = build_cells(args, params)
    n = 0
    for ds, configs in cells.items():
        if not configs:
            continue
        ready = OrderedDict()
        for name, config in configs.items():
            ready[name] = with_ckpt_dir(ds, name, config, args.seeds, args)
        rc.launch(ds, ready, args.seeds, reseed=True, **rc.launch_kwargs(args))
        n += len(ready)
    return n


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap)
    ap.add_argument("--calibrate", action="store_true", help="calibrate tau / kappa on ImageNet-R")
    ap.add_argument("--run", action="store_true", help="run the adaptive rules (default mode)")
    ap.add_argument("--lambda-gcv", action="store_true", help="run train_drift_lambda_rule='gcv'")
    ap.add_argument("--rules", nargs="+", choices=RULES, default=list(RULES))
    ap.add_argument("--stat", choices=("eff", "A"), default=DEFAULT_STAT,
                    help="merge_alpha_stat of the specnorm rule (default eff)")
    ap.add_argument("--tau", type=float, default=None, help="specnorm parameter (overrides the json)")
    ap.add_argument("--kappa", type=float, default=None, help="featshift parameter (overrides the json)")
    ap.add_argument("--param-scale", nargs="*", type=float, default=[],
                    help="extra cells at param x factor (e.g. 0.7 1.4)")
    ap.add_argument("--with-reference", action="store_true",
                    help="add the paired fixed-rule headline cell e7_fixed per dataset")
    ap.add_argument("--allow-placeholder", action="store_true",
                    help="let --run launch the placeholder parameters (1.0) when no calibration json "
                         "and no --tau/--kappa exist (refused otherwise)")
    ap.add_argument("--target", type=float, default=None,
                    help="target mean alpha of the calibration (default: ImageNet-R headline 0.30)")
    ap.add_argument("--calib-seeds", nargs="+", type=int, default=[rc.SEEDS[0]],
                    help="seeds of the featshift calibration run (default 1993)")
    ap.add_argument("--calibration", default=CALIB_JSON, help="calibration json path")
    ap.add_argument("--e4-csv", default=None, help="E4 spectrum CSV of ImageNet-R (default figures_data/e4_spectrum_imagenetr.csv)")
    return ap


def main(argv=None):
    args = rc.apply_common_args(build_parser().parse_args(argv))
    if not (args.calibrate or args.run or args.lambda_gcv):
        args.run = True
    n = 0
    if args.calibrate:
        n += do_calibrate(args)
    if args.run or args.lambda_gcv:
        n += do_cells(args)
    if args.dry_run:
        rc.print_cells(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
