"""Dry-run gate for the rebuttal drivers plus unit checks of the torch-free
launcher infrastructure (rebuttal_common.py, check_cache.py, memory_table.py).

Section A (always): task_increments, merge_tag legacy strings, cfg(), the
resume checks on a synthetic log written in the exact run_config_sweep format,
isolated_ckpt_dir() on a fake cache, launch(dry_run=True) output, the
check_cache file-name parser and the memory_table constants.

Section B: every driver present at the repo root (run_e*.py, run_long.py,
run_covmode.py, run_driftdiag.py, run_inflora.py) runs with --dry-run (plus the
extra arguments in EXPECTED) in a subprocess.  Required: exit code 0, no
Traceback, exactly one `CELLS: <n>` line with n > 0, unique (ds, name) pairs
among the `[dry-run]` lines printed by rebuttal_common.launch(), their count
equal to CELLS, no cell writing into the shared checkpoints_main/seqft, and
CELLS equal to the plan's number when EXPECTED gives one.  Scripts that do not
exist yet are skipped.  No torch or numpy is needed.

Usage:
    python tests/test_dryruns.py                 # everything
    python tests/test_dryruns.py --only run_e8_trunc_vs_damp.py
    python tests/test_dryruns.py --skip-scripts  # section A only
    python tests/test_dryruns.py --lenient       # CELLS mismatches become warnings
Environment: DRYRUN_LENIENT=1 (same as --lenient); REBUTTAL_TMP = temp root for
the fake cache and logs (default: the system temp directory).
"""
import argparse
import contextlib
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import rebuttal_common as rc            # noqa: E402
import check_cache                      # noqa: E402

# script -> (extra arguments before --dry-run, expected CELLS or None = any
# positive integer).  Numbers come from the plan; update them when a driver's
# default grid changes on purpose.
EXPECTED = {
    "run_e7_adaptive_alpha.py": ([], None),                 # 2 rules x 4 datasets (+ GCV cells)
    # CIFAR 14 + CUB 12 + Cars 2 grid cells, plus p in {0.1, 0.5} at alpha 1.0 on CIFAR and CUB
    # (2 p x 2 datasets = 4 cells; the plan's "8 cells" was a slip): 32 cells = 96 cached runs.
    "run_e8_trunc_vs_damp.py": ([], 32),
    # 3 ops x 4 datasets (12) + the clip curve on IN-R at alpha {0.5, 0.7, 1.0} beyond the headline
    # 0.3; alpha 1.0 is the identity sanity cell the driver adds to the plan's 14 (documented there).
    "run_e10_step_control.py": (["--part", "merge"], 15),
    "run_e11_rank_sweep.py": ([], 5),                       # r in {8,16,32,128} + r = 64 reference
    "run_e12_merge_grid.py": ([], 168),                     # 7 rules x 6 coefficients x 4 datasets
    "run_e13_fewshot.py": ([], 20),                         # 2 datasets x K {5,20} x 5 rows
    "run_e14_valsplit.py": ([], None),                      # headline + grid per dataset
    "run_e15_opcm_nufilt.py": (["--stage", "1"], 8),        # stage-1 grid on ImageNet-R
    "run_long.py": (["--ds", "cifar224", "--split", "t50"], 1),  # the plan's own example, no --grid
    "run_covmode.py": ([], 20),                             # 5 modes x 4 datasets
    "run_driftdiag.py": (["--splits", "t10"], 4),           # 4 datasets x 10-task split
    "run_inflora.py": ([], None),                           # stage-dependent
}
SCRIPT_GLOBS = ("run_e*.py", "run_long.py", "run_covmode.py", "run_driftdiag.py", "run_inflora.py")

LOG_PREFIX = "2026-11-08 10:00:00,123 [main.py] => "
DASH = "—"

failures, warnings = [], []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("FAIL:", msg)


def warn(msg):
    warnings.append(msg)
    print("WARN:", msg)


# ----------------------------------------------------------------------------
# Section A: infrastructure
# ----------------------------------------------------------------------------

def test_task_increments():
    ti = rc.task_increments
    check(ti(100, 10, 10) == [10] * 10, "cifar 10x10")
    check(ti(196, 16, 20) == [16] + [20] * 9, "cars 16 + 9x20")
    check(ti(200, 20, 20) == [20] * 10, "inr 10x20")
    check(ti(100, 2, 2) == [2] * 50, "cifar 50x2")
    check(ti(200, 4, 4) == [4] * 50, "inr 50x4")
    check(ti(100, 5, 5) == [5] * 20, "cifar 20x5")
    check(ti(200, 10, 10) == [10] * 20, "inr 20x10")
    check(ti(260, 65, 65) == [65] * 4, "officehome 4x65")
    check(ti(2070, 345, 345) == [345] * 6, "domainnet 6x345")
    check(len(ti(196, 16, 10)) == 19, "cars (16,10) gives 19 tasks (strict <)")
    check(rc.n_tasks("cars") == 10 and rc.n_tasks("cifar224", {"dataset_init_cls": 2, "dataset_increment": 2}) == 50,
          "n_tasks with split override")


def test_merge_tag():
    mt = rc.merge_tag
    check(mt(rc.cfg("imagenetr")) == "pspec_only_A_p0.9_a0.3", mt(rc.cfg("imagenetr")))
    check(mt(rc.cfg("cifar224")) == "pspec_only_A_p0.9_a0.15", mt(rc.cfg("cifar224")))
    check(mt(rc.cfg("cub")) == "pspec_only_A_p0.9_a0.1", mt(rc.cfg("cub")))
    check(mt(rc.cfg("cars")) == "pspec_only_A_p0.9_a0.2", mt(rc.cfg("cars")))
    check(mt(rc.cfg("cifar224", pspectral_p=1.0)) == "pspec_only_A_p1.0_a0.15", "p 1.0 keeps '1.0'")
    check(mt({"train_merge": "kspectral", "spectral_variant": "separate", "train_merge_k": 16,
              "train_merge_alpha": 1.0}) == "kspec_separate_k16_a1.0", "kspectral tag")
    check(mt({"train_merge": "avg", "train_merge_coef": 1.0}) == "avg_c1.0", "avg tag")
    check(mt({"train_merge": "dare", "train_merge_coef": 0.3, "merge_dare_q": 0.5}) == "dare_c0.3", "dare tag")
    check(mt({"train_merge": "pspectral"}) == "pspec_only_A_p0.9_a0.3", "BASE_CONFIG defaults")
    check(mt(rc.cfg("imagenetr", merge_update_op="clip")) == "pspec_only_A_p0.9_clip_a0.3", "update op token")
    # the plain cell's cleanup glob must not match a clip/scale_fro cell's files (shared directory)
    import fnmatch
    for op in ("clip", "scale_fro"):
        other = rc.merged_filename(1993, "imagenetr", rc.cfg("imagenetr", merge_update_op=op), 5)
        check(not fnmatch.fnmatch(other, f"*_merged_{mt(rc.cfg('imagenetr'))}_*.pt"),
              f"trunc cleanup glob matches the {op} file {other}")
    check(mt(rc.cfg("imagenetr", merge_alpha_rule="specnorm", merge_alpha_param=0.5))
          == "pspec_only_A_p0.9_aspecnorm0.5", "alpha rule tag")
    check(mt(rc.cfg("imagenetr", train_merge="opcm")) == "opcm_a0.5_adaptive", "opcm tag")
    check(mt(rc.cfg("imagenetr", train_merge="nufilt", merge_nufilt_adapt=True))
          == "nufilt_rp128_rv8_rl64_ad1", "nufilt tag")
    fn = rc.merged_filename(1993, "imagenetr", rc.cfg("imagenetr"), 3)
    check(fn == "1993_imagenetr_vit_base_patch16_224_lora_main_inr_r64_qkvfc_merged_pspec_only_A_p0.9_a0.3_3.pt", fn)


def test_cfg():
    c = rc.cfg("cub")
    want = {"train_method": "seqft", "model_lora_r": 64, "model_lora_alpha": 128,
            "model_lora_target_modules": ["qkv", "fc1", "fc2"], "train_prefix": "main_cub_r64_qkvfc",
            "train_ca": True, "train_drift": True, "train_drift_ridge": "identity",
            "train_drift_transport": "full", "train_ca_cov_shrinkage": 0.0, "train_ca_robust_weight": 0.1,
            "train_ca_n_classes": 40, "train_ca_k_samples": 64, "train_merge": "pspectral",
            "spectral_variant": "only_A", "pspectral_p": 0.9, "train_merge_alpha": 0.10,
            "train_weight_decay": 1e-3, "train_drift_lambda": 10.0, "cache_backbone": True,
            "reset_train": False, "reset_merge": True, "cleanup_merged": True}
    for k, v in want.items():
        check(c.get(k) == v, f"cfg(cub)[{k}] = {c.get(k)!r}, want {v!r}")
    check(set(c) == set(want), f"cfg(cub) keys differ: {sorted(set(c) ^ set(want))}")
    check("cleanup_scope" not in c and "checkpoint_dir" not in c, "cfg() must not set cleanup_scope/checkpoint_dir")
    check(rc.cfg("cars", prefix_family="best")["train_prefix"] == "best_cars_r64", "best family")
    check(rc.cfg("cars", train_merge_alpha=0.5, extra=1)["train_merge_alpha"] == 0.5, "override")
    c["model_lora_target_modules"].append("x")
    check(rc.cfg("cub")["model_lora_target_modules"] == ["qkv", "fc1", "fc2"], "lists are copied")
    rc.set_prefix_family("best")
    check(rc.cfg("imagenetr")["train_prefix"] == "best_imagenetr_r64", "set_prefix_family")
    rc.set_prefix_family("main")
    check(rc.HEADLINE_ALPHA == {"imagenetr": 0.30, "cifar224": 0.15, "cub": 0.10, "cars": 0.20}, "headline alphas")
    check(rc.HEADLINE_LAMBDA == {"imagenetr": 100.0, "cifar224": 100.0, "cub": 10.0, "cars": 100.0}, "headline lambdas")


def fake_log(path, config_name, ds, seeds, fail_seed=None, incomplete_seed=None, summary=True, append=False):
    """Write a log in the exact run_config_sweep / run_single_experiment format."""
    lines = []
    for s in seeds:
        lines += [LOG_PREFIX, "=" * 80,
                  LOG_PREFIX + f"Experiment: {ds} {DASH} {config_name} {DASH} seed {s}",
                  LOG_PREFIX + "Configuration:",
                  LOG_PREFIX + "  train_merge: pspectral",
                  LOG_PREFIX + "  train_prefix: main_cifar224_r64_qkvfc",
                  LOG_PREFIX + "[Drift] W shape (768, 768), ||W||=27.7, ||W-I||=0.9, "
                               "mean cos(W·f_(t-1), f_t)=0.99 over 5000 samples (λ=100.0, ridge=identity)"]
        if s == fail_seed:
            lines += [LOG_PREFIX + f"[Experiment {ds}_{config_name}] RuntimeError: boom",
                      LOG_PREFIX + "Time: 12.3s"]
            continue
        if s == incomplete_seed:
            lines += [LOG_PREFIX + "[Evaluation] Task 0"]
            continue
        lines += [LOG_PREFIX + "[Evaluation] Total Acc: 95.10, Grouped: [95.1]",
                  LOG_PREFIX + "[Evaluation] FAA: 95.10, FFM: 0.00, FFD: 0.00, ASA: 95.10",
                  LOG_PREFIX + "Time: 600.0s"]
    if summary:
        lines += [LOG_PREFIX, "=" * 80,
                  LOG_PREFIX + f"SUMMARY  {ds.upper()} {DASH} {config_name.upper()}",
                  LOG_PREFIX + "=" * 80,
                  LOG_PREFIX + "  ASA: 95.10 ± 0.00 | FAA: 95.10 ± 0.00 | FFM: 0.00 ± 0.00"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def test_log_checks(tmp):
    log = os.path.join(tmp, "logs", "cifar224", "cell.log")
    fake_log(log, "cell", "cifar224", [1993, 1994, 1995])
    check(rc.has_summary(log, "cell", 3), "complete log -> has_summary")
    check(not rc.has_summary(log, "cell", 4), "fewer headers than seeds")
    check(not rc.has_summary(log, "other", 3), "config name must match")
    check(rc.seed_status(log, "cell") == {1993: "ok", 1994: "ok", 1995: "ok"}, rc.seed_status(log, "cell"))
    check(not rc.has_summary(os.path.join(tmp, "none.log"), "cell", 1), "missing file")

    fake_log(log, "cell", "cifar224", [1993, 1994, 1995], fail_seed=1994)
    check(not rc.has_summary(log, "cell", 3), "error line -> not done")
    check(rc.seed_status(log, "cell") == {1993: "ok", 1994: "failed", 1995: "ok"}, "failed seed status")
    fake_log(log, "cell", "cifar224", [1994], summary=False, append=True)
    check(rc.seed_status(log, "cell")[1994] == "ok", "last block of a seed wins")
    check(not rc.has_summary(log, "cell", 3), "the old error line still blocks has_summary")

    fake_log(log, "cell", "cifar224", [1993, 1994], incomplete_seed=1994, summary=False)
    check(rc.seed_status(log, "cell") == {1993: "ok", 1994: "incomplete"}, "incomplete seed")
    check(not rc.has_summary(log, "cell", 2), "no SUMMARY -> not done")

    # a log holding two configs (sota_merge-style files are segmented by config name)
    fake_log(log, "cell", "cifar224", [1993])
    fake_log(log, "cell2", "cifar224", [1993, 1994], append=True)
    check(rc.has_summary(log, "cell", 1) and rc.has_summary(log, "cell2", 2) and not rc.has_summary(log, "cell", 2),
          "per-config header counting")


def make_fake_cache(src, ds, prefix, seeds, T):
    os.makedirs(src, exist_ok=True)
    for seed in seeds:
        stem = rc.chain_stem(seed, ds, prefix)
        files = [f"{stem}_backbone_base.pt"] + [f"{stem}_backbone_{t}.pt" for t in range(T)] \
            + [f"{stem}_classifier_{t}.pt" for t in range(T)]
        for f in files:
            with open(os.path.join(src, f), "w") as fh:
                fh.write(f)
    return src


def test_isolated_dir(tmp):
    src = make_fake_cache(os.path.join(tmp, "seqft"), "cifar224", "main_cifar224_r64_qkvfc", [1993], 3)
    stem = rc.chain_stem(1993, "cifar224", "main_cifar224_r64_qkvfc")
    with open(os.path.join(src, f"{stem}_merged_pspec_only_A_p0.9_a0.15_2.pt"), "w") as fh:
        fh.write("merged")
    root = os.path.join(tmp, "rebuttal")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dst = rc.isolated_ckpt_dir("cifar224", "main_cifar224_r64_qkvfc", [1993, 1994], "e8",
                                   src_dir=src, root=root)
    out = buf.getvalue()
    check(dst == os.path.join(root, "e8") and os.path.isdir(dst), f"isolated dir {dst}")
    for f in (f"{stem}_backbone_base.pt", f"{stem}_backbone_0.pt", f"{stem}_classifier_0.pt"):
        p = os.path.join(dst, f)
        check(os.path.isfile(p) and not os.path.islink(p), f"{f} must be a private copy")
    for f in (f"{stem}_backbone_1.pt", f"{stem}_backbone_2.pt", f"{stem}_classifier_1.pt", f"{stem}_classifier_2.pt"):
        p = os.path.join(dst, f)
        check(os.path.exists(p), f"{f} must be linked or copied")
    check(not os.path.exists(os.path.join(dst, f"{stem}_merged_pspec_only_A_p0.9_a0.15_2.pt")),
          "merged files are not carried over")
    check("MISSING" in out and "seed 1994" in out and "TRAIN this chain" in out, "missing seed is reported")
    n_links = sum(os.path.islink(os.path.join(dst, f)) for f in os.listdir(dst))
    print(f"  isolated dir: {n_links} symlink(s), {len(os.listdir(dst)) - n_links} file(s) "
          f"({'symlinks work' if n_links else 'copy fallback'} on this OS)")
    # the copy isolates the shared cache from the base-file rewrite
    with open(os.path.join(dst, f"{stem}_backbone_base.pt"), "w") as fh:
        fh.write("rewritten by Learner.__init__")
    with open(os.path.join(src, f"{stem}_backbone_base.pt")) as fh:
        check(fh.read() == f"{stem}_backbone_base.pt", "source base file untouched by a rewrite of the copy")
    # idempotent
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc.isolated_ckpt_dir("cifar224", "main_cifar224_r64_qkvfc", [1993], "e8", src_dir=src, root=root)
    check("copied 0, linked 0, copied-instead-of-linked 0, kept 7" in buf.getvalue(),
          f"second call keeps everything: {buf.getvalue().strip()}")
    # dangling link replaced
    link = os.path.join(dst, f"{stem}_backbone_2.pt")
    if os.path.islink(link):
        os.remove(link)
        os.symlink("does_not_exist.pt", link)
        with contextlib.redirect_stdout(io.StringIO()):
            rc.isolated_ckpt_dir("cifar224", "main_cifar224_r64_qkvfc", [1993], "e8", src_dir=src, root=root)
        check(os.path.exists(link), "dangling link replaced")
    # chain_status on the isolated dir and the source
    st = check_cache.chain_status("cifar224", "main_cifar224_r64_qkvfc", [1993, 1994], 3, ckpt_dir=dst)
    check(st["seeds"][1993]["complete"] and not st["complete"], "status per seed")
    check(st["seeds"][1994]["backbone_missing"] == [0, 1, 2] and not st["seeds"][1994]["base"], "missing seed status")
    st = check_cache.chain_status("cifar224", "main_cifar224_r64_qkvfc", [1993], 3, ckpt_dir=src)
    check(st["complete"], "source chain complete")
    st = check_cache.chain_status("cifar224", "main_cifar224_r64_qkvfc", [1993], 10, ckpt_dir=src)
    check(st["seeds"][1993]["backbone_missing"] == list(range(3, 10)), "T=10 on a 3-task chain")
    # dry_run creates nothing
    with contextlib.redirect_stdout(io.StringIO()):
        d2 = rc.isolated_ckpt_dir("cifar224", "main_cifar224_r64_qkvfc", [1993], "e9", src_dir=src, root=root, dry_run=True)
    check(not os.path.exists(d2), "dry_run must not create the directory")
    # inventory of the fake source
    chains, unparsed = check_cache.inventory(src)
    check(len(chains) == 1 and not unparsed, f"inventory: {len(chains)} chains, unparsed {unparsed}")
    ch = next(iter(chains.values()))
    check(ch.n_tasks == 3 and ch.base and ch.classifier_tasks == {0, 1, 2}, "chain counts")
    check(not ch.complete and ch.expected_T == 10 and ch.complete_for(3), "3 of 10 tasks")
    check(dict(ch.merged) == {"pspec_only_A_p0.9_a0.15": {2}}, f"merged tags {dict(ch.merged)}")


def test_smokify():
    name, c = rc.smokify("e8_cell", rc.cfg("cub", checkpoint_dir="checkpoints_main/rebuttal/e8"))
    check(name == "e8_cell_smoke", name)
    check(c["train_prefix"] == "main_cub_r64_qkvfc_smoke", c["train_prefix"])
    check(c["checkpoint_dir"] == "checkpoints_main/rebuttal/e8_smoke", c["checkpoint_dir"])
    check(c["train_stop_at_task"] == 1 and c["train_epochs"] == 1 and c["train_ca_epochs"] == 2, "smoke keys")
    _, c2 = rc.smokify("x", rc.cfg("cub"))
    check(c2["checkpoint_dir"] == os.path.join("checkpoints_main", "rebuttal", "smoke"), c2["checkpoint_dir"])
    check(rc.smoke_overrides() == {"train_stop_at_task": 1, "train_epochs": 1, "train_ca_epochs": 2}, "smoke_overrides")


def test_launch_dry_run(tmp):
    src = make_fake_cache(os.path.join(tmp, "seqft2"), "cars", "main_cars_r64_qkvfc", [1993, 1994], 10)
    configs = {"e8_a": rc.cfg("cars", checkpoint_dir="checkpoints_main/rebuttal/e8"),
               "e8_none": rc.cfg("cars", train_merge="none")}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = rc.launch("cars", configs, [1993, 1994, 1995], log_root=os.path.join(tmp, "logs"),
                        dry_run=True, ckpt_src=src)
    text = buf.getvalue()
    check("main" not in sys.modules, "dry-run must not import main")
    check(out == {"e8_a": "dry", "e8_none": "dry"}, str(out))
    lines = [ln for ln in text.splitlines() if ln.startswith("[dry-run] ")]
    check(len(lines) == 2, f"two [dry-run] lines, got {len(lines)}")
    kv = dict(re.findall(r"(\w+)=(\S+)", lines[0]))
    check(kv.get("ds") == "cars" and kv.get("name") == "e8_a" and kv.get("tag") == "pspec_only_A_p0.9_a0.2"
          and kv.get("prefix") == "main_cars_r64_qkvfc" and kv.get("ckpt") == "checkpoints_main/rebuttal/e8"
          and kv.get("status") == "todo" and kv.get("cache") == "2/3", f"dry-run fields: {kv}")
    check("no merged checkpoints" in lines[1], "train_merge none is annotated")
    check("'train_merge_alpha': 0.2" in text, "config dict is printed")
    # a finished log flips the status
    fake_log(rc.log_path("cars", "e8_a", os.path.join(tmp, "logs")), "e8_a", "cars", [1993, 1994, 1995])
    with contextlib.redirect_stdout(buf):
        rc.launch("cars", {"e8_a": configs["e8_a"]}, [1993, 1994, 1995], log_root=os.path.join(tmp, "logs"),
                  dry_run=True, ckpt_src=src, verbose=False)
    check("status=done" in buf.getvalue(), "finished log -> status=done")
    # smoke dry-run
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc.launch("cars", {"e8_a": configs["e8_a"]}, [1993, 1994], dry_run=True, smoke=True, ckpt_src=src, verbose=False)
    line = buf.getvalue().splitlines()[0]
    check("name=e8_a_smoke" in line and "prefix=main_cars_r64_qkvfc_smoke" in line
          and "ckpt=checkpoints_main/rebuttal/e8_smoke" in line
          and os.path.join("logs_main", "smoke") in line and "cache=0/1" in line, f"smoke line: {line}")
    # partial log without --redo/--append raises before importing main
    partial = rc.log_path("cars", "e8_b", os.path.join(tmp, "logs"))
    fake_log(partial, "e8_b", "cars", [1993, 1994], fail_seed=1994)
    try:
        rc.launch("cars", {"e8_b": configs["e8_a"]}, [1993, 1994, 1995], log_root=os.path.join(tmp, "logs"))
        check(False, "partial log must raise without --redo/--append")
    except RuntimeError as exc:
        check("--redo" in str(exc) and "--append" in str(exc), str(exc))
    # add_common_args round trip
    ap = rc.add_common_args(argparse.ArgumentParser())
    a = rc.apply_common_args(ap.parse_args(["--ds", "cub", "cars", "--seeds", "1993", "--prefix-family", "best", "--dry-run"]))
    check(a.datasets == ["cub", "cars"] and a.seeds == [1993] and rc.get_prefix_family() == "best", "common args")
    check(rc.launch_kwargs(a)["dry_run"] is True and rc.launch_kwargs(a)["ckpt_src"] == rc.SEQFT_CKPT_DIR, "launch_kwargs")
    rc.set_prefix_family("main")


def test_check_cache_parse():
    p = check_cache.parse_filename
    r = p("1993_imagenetr_vit_base_patch16_224_lora_main_inr_r64_qkvfc_backbone_base.pt")
    check(r == {"seed": 1993, "ds": "imagenetr", "backbone": "vit_base_patch16_224_lora",
                "prefix": "main_inr_r64_qkvfc", "kind": "backbone", "tag": None, "t": -1,
                "file": "1993_imagenetr_vit_base_patch16_224_lora_main_inr_r64_qkvfc_backbone_base.pt"}, str(r))
    r = p("1994_cars_vit_base_patch16_224_lora_main_cars_r64_qkvfc_merged_pspec_only_A_p0.9_a0.2_3.pt")
    check(r["kind"] == "merged" and r["tag"] == "pspec_only_A_p0.9_a0.2" and r["t"] == 3 and r["ds"] == "cars", str(r))
    r = p("1993_officehome_dil_vit_base_patch16_224_lora_main_oh_classifier_2.pt")
    check(r["ds"] == "officehome_dil" and r["prefix"] == "main_oh" and r["kind"] == "classifier" and r["t"] == 2, str(r))
    r = p("1993_cifar224_vit_base_patch16_224_lora_backbone_0.pt")
    check(r["prefix"] == "" and r["backbone"] == "vit_base_patch16_224_lora" and r["t"] == 0, str(r))
    r = p("1993_cifar224_e2lora_e13_fs5_state_1.pt")
    check(r["backbone"] == "e2lora" and r["prefix"] == "e13_fs5" and r["kind"] == "state", str(r))
    r = p("1995_imagenetr_vit_base_patch16_224_lora_main_inr_r64_qkvfc_merged_avg_c1.0_9.pt")
    check(r["tag"] == "avg_c1.0" and r["t"] == 9, str(r))
    check(p("random.pt") is None and p("README.md") is None, "non-matching names")


def test_memory_table():
    import memory_table as mt
    mt.self_check()
    check(mt.lora_floats()[0] == 8_259_072, "lora floats")
    check(mt.stats_per_class("full") == 590_592, "full stats per class")
    rows = mt.build_rows()
    check(len(rows) == 6 * 6, f"{len(rows)} storage rows")
    full = {(r["dataset"], r["mode"]): r for r in rows}
    check(full[("cifar224", "full")]["stats_MB"] == 236.24 and full[("cifar224", "full")]["lora_MB"] == 33.04, "CIFAR MB")
    check(full[("dndil", "full")]["C"] == 2070 and full[("dndil", "full")]["stats_floats"] == 2070 * 590_592, "DomainNet")
    base = mt.baseline_rows(["imagenetr"])
    ran = next(r for r in base if r["method"] == "RanPAC")
    check(ran["stats_floats"] == 10_000 ** 2 + 200 * 10_000, "RanPAC floats")
    check(any("TO VERIFY" in r["source"] for r in base if r["method"] == "SLDC"), "SLDC flagged")


# ----------------------------------------------------------------------------
# Section B: driver dry-runs
# ----------------------------------------------------------------------------

CELLS_RE = re.compile(r"^CELLS: (\d+)\s*$", re.M)
DRY_RE = re.compile(r"^\[dry-run\] (.*)$", re.M)


def run_script(script, extra):
    cmd = [sys.executable, script] + list(extra) + ["--dry-run"]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "", "timeout after 600 s"
    return p.returncode, p.stdout, p.stderr


def check_script(script, extra, expected, lenient):
    code, out, err = run_script(script, extra)
    if code == 2 and extra and ("unrecognized arguments" in err or "invalid choice" in err
                                or "error: argument" in err):
        warn(f"{script}: does not accept {extra} ({err.strip().splitlines()[-1]}); retrying with --dry-run only")
        extra, expected = [], None
        code, out, err = run_script(script, extra)
    shown = " ".join([script] + list(extra) + ["--dry-run"])
    if code != 0:
        tail = "\n".join(err.strip().splitlines()[-8:])
        check(False, f"{shown}: exit code {code}\n{tail}")
        return
    check("Traceback" not in err, f"{shown}: traceback on stderr:\n{err[-800:]}")
    cells = [int(m.group(1)) for m in CELLS_RE.finditer(out)]
    if len(cells) != 1:
        check(False, f"{shown}: expected exactly one 'CELLS: <n>' line, found {cells}")
        return
    n = cells[0]
    check(n > 0, f"{shown}: CELLS must be positive")
    dry = []
    for m in DRY_RE.finditer(out):
        kv = dict(re.findall(r"(\w+)=(\S+)", m.group(1)))
        dry.append(kv)
    if dry:
        pairs = [(d.get("ds"), d.get("name")) for d in dry]
        dupes = sorted({p for p in pairs if pairs.count(p) > 1})
        check(not dupes, f"{shown}: duplicate cells {dupes}")
        check(len(dry) == n, f"{shown}: CELLS {n} but {len(dry)} [dry-run] lines")
        shared = os.path.normpath(rc.SEQFT_CKPT_DIR)
        bad = sorted({d.get("name") for d in dry if os.path.normpath(d.get("ckpt", "")) == shared})
        check(not bad, f"{shown}: cells writing into the shared cache {rc.SEQFT_CKPT_DIR}: {bad[:5]}")
        missing_tag = [d.get("name") for d in dry if not d.get("tag") or not d.get("log")]
        check(not missing_tag, f"{shown}: [dry-run] lines without tag/log: {missing_tag[:5]}")
    else:
        warn(f"{shown}: no [dry-run] lines (does it go through rebuttal_common.launch?)")
    if expected is not None and n != expected:
        msg = f"{shown}: CELLS {n}, plan expects {expected}"
        if lenient:
            warn(msg)
        else:
            check(False, msg + " (update EXPECTED in tests/test_dryruns.py if the grid changed on purpose)")
    print(f"PASS {shown}: CELLS {n}" + (f" (expected {expected})" if expected is not None else "")
          + (f", {len(dry)} unique cells" if dry else ""))


def discover():
    names = set()
    for pat in SCRIPT_GLOBS:
        names.update(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, pat)))
    return sorted(names)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", default=None, help="run only these scripts")
    ap.add_argument("--skip-scripts", action="store_true", help="section A only")
    ap.add_argument("--lenient", action="store_true", help="CELLS mismatches are warnings")
    args = ap.parse_args(argv)
    lenient = args.lenient or os.environ.get("DRYRUN_LENIENT") == "1"

    tmp_root = os.environ.get("REBUTTAL_TMP") or tempfile.gettempdir()
    os.makedirs(tmp_root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="dryruns_", dir=tmp_root)
    print(f"== Section A: infrastructure (temp {tmp})")
    try:
        for fn in (test_task_increments, test_merge_tag, test_cfg, test_smokify, test_check_cache_parse,
                   test_memory_table):
            fn()
        for fn in (test_log_checks, test_isolated_dir, test_launch_dry_run):
            fn(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"section A: {len(failures)} failure(s)")

    if not args.skip_scripts:
        print("\n== Section B: driver dry-runs")
        scripts = args.only or discover()
        for script in scripts:
            if not os.path.isfile(os.path.join(ROOT, script)):
                print(f"SKIP {script}: not present")
                continue
            extra, expected = EXPECTED.get(os.path.basename(script), ([], None))
            check_script(script, extra, expected, lenient)

    print(f"\n{len(failures)} failure(s), {len(warnings)} warning(s)")
    for f in failures:
        print("  FAIL:", f.splitlines()[0])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
