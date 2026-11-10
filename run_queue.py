"""One-command queues for the rebuttal experiments (plan: "Server day 0" and
"Launch order").  Runs the drivers one after another as subprocesses, tees their
output to the console and to logs_main/queue/<stream>.txt (a .txt so the log
parsers never read it), records exit codes and durations, and continues with
the next step when one fails (the drivers resume by themselves: a cell whose
log already has a clean SUMMARY is skipped, so re-running a stream is cheap).

Streams
    day0      import check, cache inventory, posture check, tier-b unit tests,
              the regression gate, E4 spectrum replay (also the E7 preview) and
              the E2 tuning record / time table.  Read the time table before
              starting the GPU streams.
    smoke     every driver's --smoke cell (2 tasks x 1 epoch, logs_main/smoke),
              run_keeplora --smoke twice (second must hit the cache), main --smoke-keeplora.
              --clean-smoke removes the smoke directories afterwards.
    cached    Stage-2-only jobs on the cached headline chains, in the plan's
              order: E8, E10 merge ops, E15a (NUFILT), E7, E5 (+E3 timers), E9, E6, E12.
    training  jobs that train new chains, in the plan's order: E14, E13, KeepLoRA
              main stage, E11, E10 L2M pilot + report.
    optional  E1 long splits, E15a independent regime (--indep-prefix), KeepLoRA
              grid and IN21k-only backbone, E9 on long chains.
Plan change of 2026-11-10: the scheduled E15 baselines are KeepLoRA (run_keeplora.py)
and NUFILT; the InfLoRA port (run_inflora.py, main --smoke-inflora) and the OPCM
rule (run_e15_opcm_nufilt.py --rules opcm) stay in the repo but are not queued.
The cached and training streams are designed to run at the same time (one
process each): every cached job owns an isolated checkpoint directory, every
training job a fresh prefix.

Usage
    python run_queue.py day0 --dry-run              # print the commands
    python run_queue.py day0
    python run_queue.py smoke --clean-smoke
    nohup python run_queue.py cached   > queue_cached.out 2>&1 &
    nohup python run_queue.py training > queue_training.out 2>&1 &
    python run_queue.py cached --from e7            # resume from a step (name prefix or index)
    python run_queue.py cached --only e8 e10        # a subset
    python run_queue.py optional --indep-prefix <prefix of the Appendix E chain>
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import rebuttal_common as rc

QUEUE_LOG_DIR = os.path.join(rc.LOG_DIR, "queue")
PY = sys.executable or "python"


def step(name, argv, note=""):
    return {"name": name, "argv": list(argv), "note": note}


def build_streams(args):
    py = args.python
    ds_all = list(rc.DATASETS)
    s = {}

    s["day0"] = [
        step("import", [py, "-c", "import main, backbones, helper, merging_sota; print('import ok')"],
             "every tracked module imports (the six new modules must be in the checkout)"),
        step("cache", [py, "check_cache.py"], "which prefix family and long-split chains exist"),
        step("posture", [py, "rebuttal_common.py", "--check"], "POSTURE == run_best.CONFIGS / run_sota_merge.DS"),
        step("tierb", [py, "tests/smoke_rebuttal.py", "--tier", "b"],
             "CPU torch unit tests incl. the pspectral / sota regressions vs 0eab95e"),
        step("gate", [py, "regression_gate.py"], "cached ImageNet-R seed 1993 headline, Grouped vectors identical"),
    ] + [step(f"e4-{ds}", [py, "analyze_spectrum.py", "--ds", ds], "E4 spectrum replay (CPU)") for ds in ds_all] + [
        step("e2", [py, "parse_logs.py", "e2", "--roots", "logs_main"],
             "tuning record + per-dataset training times: re-cost the launch order from it"),
    ]

    s["smoke"] = [
        step("smoke-e8", [py, "run_e8_trunc_vs_damp.py", "--smoke"]),
        step("smoke-e10", [py, "run_e10_step_control.py", "--part", "merge", "--smoke"]),
        step("smoke-e15", [py, "run_e15_opcm_nufilt.py", "--stage", "1", "--smoke"], "NUFILT cells (default rules)"),
        step("smoke-e7", [py, "run_e7_adaptive_alpha.py", "--smoke"]),
        step("smoke-e5", [py, "run_covmode.py", "--smoke"]),
        step("smoke-e9", [py, "run_driftdiag.py", "--splits", "t10", "--smoke"]),
        step("smoke-e12", [py, "run_e12_merge_grid.py", "--smoke"]),
        step("smoke-e14", [py, "run_e14_valsplit.py", "--smoke"]),
        step("smoke-e13", [py, "run_e13_fewshot.py", "--smoke"]),
        step("smoke-e11", [py, "run_e11_rank_sweep.py", "--smoke"]),
        step("smoke-e1", [py, "run_long.py", "--ds", "cifar224", "--split", "t20", "--smoke"]),
        step("smoke-keeplora-1", [py, "run_keeplora.py", "--smoke"], "first pass trains 2 tasks"),
        step("smoke-keeplora-2", [py, "run_keeplora.py", "--smoke"], "second pass must hit the cache with identical Total Acc"),
        step("smoke-main", [py, "main.py", "--smoke-keeplora"], "legacy eight recipes + the KeepLoRA cell"),
        step("smoke-driftdiag-validate", [py, "run_driftdiag.py", "--validate", "--smoke"],
             "drift_diag on/off must give identical FAA lines"),
    ]

    s["cached"] = [
        step("e8", [py, "run_e8_trunc_vs_damp.py"], "A1 evidence: truncation vs damped interpolation, C/P exports"),
        step("e10-merge", [py, "run_e10_step_control.py", "--part", "merge"], "clip / scale_fro at matched norm"),
        step("e15-stage1", [py, "run_e15_opcm_nufilt.py", "--stage", "1"], "NUFILT grid on ImageNet-R (4 cells)"),
        step("e15-stage2", [py, "run_e15_opcm_nufilt.py", "--stage", "2"], "best NUFILT cell on the other datasets"),
        step("e7-calibrate", [py, "run_e7_adaptive_alpha.py", "--calibrate"], "tau / kappa so mean alpha = 0.30 on ImageNet-R"),
        step("e7-run", [py, "run_e7_adaptive_alpha.py", "--run", "--with-reference"], "specnorm + featshift, 4 datasets"),
        step("e7-gcv", [py, "run_e7_adaptive_alpha.py", "--lambda-gcv"], "data-free lambda"),
        step("e5", [py, "run_covmode.py"], "cheap statistics; the cov_class cell carries the E3 timers"),
        step("e9", [py, "run_driftdiag.py", "--splits", "t10"], "drift-map diagnostics, 10-task chains"),
    ] + [step(f"e6-{ds}", [py, "analyze_subspace.py", "--ds", ds], "discarded directions vs input subspaces") for ds in ds_all] + [
        step("e12-a", [py, "run_e12_merge_grid.py", "--ds", "imagenetr", "cub", "--with-ism"], "equal tuning budget, first two datasets"),
        step("e12-b", [py, "run_e12_merge_grid.py", "--ds", "cifar224", "cars", "--with-ism"], "remaining datasets"),
    ]

    s["training"] = [
        step("e14", [py, "run_e14_valsplit.py"], "90 % chains + validation-selected grid"),
        step("e13", [py, "run_e13_fewshot.py"], "few-shot per task, K 5 / 20"),
        step("keeplora-main", [py, "run_keeplora.py", "--stage", "main"], "KeepLoRA, 4 datasets x 3 seeds, our backbone"),
        step("e11", [py, "run_e11_rank_sweep.py"], "LoRA rank sweep on ImageNet-R"),
        step("e10-l2m-pilot", [py, "run_e10_step_control.py", "--part", "l2m-pilot"], "3-task pilot, weights 0.1 / 1 / 10"),
        step("e10-l2m-report", [py, "run_e10_step_control.py", "--part", "l2m-report"],
             "prints the two weights; then run: run_e10_step_control.py --part l2m --weights w1 w2"),
    ]

    opt = [
        step("e1-long", [py, "run_long.py"], "20 / 50-task reruns (train_epochs 5 at 50 tasks)"),
        step("e9-long", [py, "run_driftdiag.py", "--splits", "t20", "t50"], "needs the E1 chains"),
        step("keeplora-grid", [py, "run_keeplora.py", "--stage", "grid"], "one switch per cell, ImageNet-R / CIFAR-100, seed 1993"),
        step("keeplora-in21k", [py, "run_keeplora.py", "--stage", "in21k"], "KeepLoRA on the IN21k-only backbone"),
    ]
    if args.indep_prefix:
        opt.insert(0, step("e15-indep", [py, "run_e15_opcm_nufilt.py", "--stage", "1", "--regime",
                                         "independent", "--prefix", args.indep_prefix],
                           "native regime on the Appendix E chain"))
    else:
        opt.insert(0, step("e15-indep", [py, "-c", "print('skipped: pass --indep-prefix <Appendix E chain prefix>')"],
                           "needs --indep-prefix"))
    s["optional"] = opt
    return s


def select_steps(steps, args):
    if args.only:
        steps = [st for st in steps if any(st["name"] == o or st["name"].startswith(o) for o in args.only)]
    if args.skip:
        steps = [st for st in steps if not any(st["name"] == o or st["name"].startswith(o) for o in args.skip)]
    if args.from_step:
        idx = None
        if args.from_step.isdigit():
            idx = int(args.from_step)
        else:
            for i, st in enumerate(steps):
                if st["name"] == args.from_step or st["name"].startswith(args.from_step):
                    idx = i
                    break
        if idx is None:
            raise SystemExit(f"--from {args.from_step!r} matches no step of this stream")
        steps = steps[idx:]
    return steps


def run_step(st, log_fh, env):
    """Run one step, tee its output; return (exit_code, seconds)."""
    t0 = time.time()
    proc = subprocess.Popen(st["argv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        log_fh.write(line)
    proc.wait()
    log_fh.flush()
    return proc.returncode, time.time() - t0


SMOKE_PATTERNS = ("*smoke*",)


def clean_smoke(dry_run):
    """Remove the smoke artefacts the plan names: logs_main/smoke,
    checkpoints_main/rebuttal/*smoke*, checkpoints_main/inflora/*smoke*."""
    import glob
    targets = [rc.SMOKE_LOG_ROOT]
    targets += glob.glob(os.path.join(rc.REBUTTAL_CKPT_ROOT, "*smoke*"))
    targets += glob.glob(os.path.join(rc.CHECKPOINT_ROOT, "inflora", "*smoke*"))
    targets += glob.glob(os.path.join(rc.CHECKPOINT_ROOT, "keeplora", "*smoke*"))
    for t in targets:
        if not os.path.exists(t):
            continue
        print(f"[queue] {'would remove' if dry_run else 'removing'} {t}")
        if not dry_run:
            if os.path.isdir(t):
                shutil.rmtree(t, ignore_errors=True)
            else:
                os.remove(t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream", choices=("day0", "smoke", "cached", "training", "optional"))
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--from", dest="from_step", default=None, help="start at this step (name, name prefix or index)")
    ap.add_argument("--only", nargs="+", default=None, help="run only these steps (names or prefixes)")
    ap.add_argument("--skip", nargs="+", default=None, help="skip these steps (names or prefixes)")
    ap.add_argument("--stop-on-fail", action="store_true", help="stop the stream at the first non-zero exit")
    ap.add_argument("--python", default=PY, help="interpreter for the steps (default: this one)")
    ap.add_argument("--indep-prefix", default=None, help="train_prefix of the Appendix E independent-init chain (optional stream)")
    ap.add_argument("--clean-smoke", action="store_true", help="smoke stream: remove the smoke dirs when every step passed")
    ap.add_argument("--log", default=None, help="queue log (default logs_main/queue/<stream>.txt)")
    args = ap.parse_args(argv)

    streams = build_streams(args)
    steps = select_steps(streams[args.stream], args)
    print(f"[queue] stream {args.stream}: {len(steps)} step(s)")
    for i, st in enumerate(steps):
        note = f"   # {st['note']}" if st["note"] else ""
        print(f"[queue] {i:2d} {st['name']:<22s} {' '.join(st['argv'][1:])}{note}")
    if args.dry_run:
        if args.clean_smoke:
            clean_smoke(dry_run=True)
        print(f"STEPS: {len(steps)}")
        return 0

    log_path = args.log or os.path.join(QUEUE_LOG_DIR, f"{args.stream}.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    results = []
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== queue {args.stream} started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        for i, st in enumerate(steps):
            banner = f"\n===== [{i}] {st['name']}: {' '.join(st['argv'])}  ({time.strftime('%H:%M:%S')}) =====\n"
            sys.stdout.write(banner)
            fh.write(banner)
            code, sec = run_step(st, fh, env)
            results.append((st["name"], code, sec))
            line = f"===== [{i}] {st['name']} exit {code} after {sec / 60:.1f} min =====\n"
            sys.stdout.write(line)
            fh.write(line)
            fh.flush()
            if code != 0 and args.stop_on_fail:
                print("[queue] stopping at the first failure (--stop-on-fail)")
                break
        fh.write(f"===== queue {args.stream} finished {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    print(f"\n[queue] summary ({log_path}):")
    for name, code, sec in results:
        print(f"[queue]   {name:<22s} {'ok  ' if code == 0 else 'FAIL'} exit {code:<3d} {sec / 60:7.1f} min")
    failed = [n for n, c, _ in results if c != 0]
    if args.stream == "smoke" and args.clean_smoke:
        if failed:
            print("[queue] smoke failures: leaving the smoke directories in place for inspection")
        else:
            clean_smoke(dry_run=False)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
