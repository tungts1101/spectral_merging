"""Tests for regression_gate.py (reference matching, vector comparison) and the
run_queue.py dry-runs.  Torch-free; fixtures from tests/make_fake_log.py.

Usage:  python tests/test_regression_gate.py
"""
import copy
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import make_fake_log as mf          # noqa: E402
import rebuttal_common as rc        # noqa: E402
import regression_gate as rg        # noqa: E402

HEADLINE = dict(alpha=0.3, lam=100.0, prefix="main_inr_r64_qkvfc",
                extra_config={"train_weight_decay": 2e-4})
failures = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def write_log(d, name, **kw):
    out = mf.make_sweep_log("imagenetr", name, **kw)
    txt = out[0] if isinstance(out, tuple) else out
    p = os.path.join(d, name + ".log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(txt)
    return p


def test_matching(d):
    cfg_gate = rc.cfg("imagenetr", checkpoint_dir="x")
    good = write_log(d, "best_imagenetr", seeds=(1993, 1994), **HEADLINE)
    refs = rg.find_references("imagenetr", 1993, cfg_gate, ref_log=good)
    check(len(refs) == 1, "headline block of the right seed is found once")
    check(len(rg.find_references("imagenetr", 1995, cfg_gate, ref_log=good)) == 0,
          "a seed absent from the log yields no reference")
    for label, kw in [
        ("alpha 0.5", dict(HEADLINE, alpha=0.5)),
        ("lambda 10", dict(HEADLINE, lam=10.0)),
        ("cov diag", dict(HEADLINE, extra_config={"train_weight_decay": 2e-4, "train_ca_cov_mode": "diag"})),
        ("other prefix", dict(HEADLINE, prefix="best_imagenetr_r64")),
        ("p 0.5", dict(HEADLINE, p=0.5)),
    ]:
        p = write_log(d, label.replace(" ", "_"), seeds=(1993,), **kw)
        check(len(rg.find_references("imagenetr", 1993, cfg_gate, ref_log=p)) == 0,
              f"a block with {label} is not a reference")
    crashed = write_log(d, "crash", seeds=(1993,), crash_seed=1993, crash_at_task=4, **HEADLINE)
    check(len(rg.find_references("imagenetr", 1993, cfg_gate, ref_log=crashed)) == 0,
          "a crashed seed is not a reference")
    return refs[0]


def test_compare(ref):
    rows = ref["rows"]
    identical, worst, n = rg.compare_rows(rows, rows)
    check(identical and worst == 0.0 and n == len(rows), "identical vectors pass with max diff 0")
    rows2 = copy.deepcopy(rows)
    rows2[3][0] += 0.01
    identical, worst, _ = rg.compare_rows(rows2, rows)
    check(not identical and abs(worst - 0.01) < 1e-9, "a 0.01 change in one entry fails")
    identical, worst, n = rg.compare_rows(rows[:-1], rows)
    check(not identical and n == len(rows) - 1, "a missing task fails")
    rows3 = copy.deepcopy(rows)
    rows3[2] = rows3[2][:-1]
    identical, worst, _ = rg.compare_rows(rows3, rows)
    check(not identical and worst == float("inf"), "a vector of another length fails")


def test_cli_dry_runs():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "regression_gate.py", "--dry-run"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    check(r.returncode == 0 and "[gate] cell gate_imagenetr_s1993" in r.stdout and "Traceback" not in r.stdout + r.stderr,
          "regression_gate.py --dry-run exits 0 and names the cell")
    for stream in ("day0", "smoke", "cached", "training", "optional"):
        r = subprocess.run([sys.executable, "run_queue.py", stream, "--dry-run"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        steps = [ln for ln in r.stdout.splitlines() if ln.startswith("STEPS: ")]
        n = int(steps[0].split()[1]) if steps else 0
        check(r.returncode == 0 and n > 0 and "Traceback" not in r.stdout + r.stderr,
              f"run_queue.py {stream} --dry-run exits 0 with {n} steps")
    r = subprocess.run([sys.executable, "run_queue.py", "cached", "--from", "e7", "--dry-run"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    first = [ln for ln in r.stdout.splitlines() if ln.startswith("[queue]  0 ")]
    check(bool(first) and "e7-calibrate" in first[0], "--from e7 starts the cached stream at e7-calibrate")
    for d in ("logs_main", "checkpoints_main", "final_models"):
        check(not os.path.exists(os.path.join(ROOT, d)), f"dry-runs created no {d}/")


def main():
    d = tempfile.mkdtemp(prefix="gate_test_")
    try:
        ref = test_matching(d)
        test_compare(ref)
        test_cli_dry_runs()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
