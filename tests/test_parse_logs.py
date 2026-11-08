"""Tests for parse_logs.py on the synthetic logs of tests/make_fake_log.py (numpy only).

Covers the behaviours listed under 'Verification' in the rebuttal plan: last block wins
for appended re-runs, n_tasks from the Configuration dump (10/20/50), recomputed
FA/AA/FFM equal to the logged '[Evaluation] FAA' line, crashed seeds flagged and never
averaged, std with ddof=0 equal to the SUMMARY line, val metrics only when present,
utf-8 reading under the Windows locale, unquoted config strings, Configuration-only
sota logs, raw capture of the rebuttal tags, the e1/e2 subcommands and the CLI; plus,
on the second fixture tree (make_fake_log.build_rebuttal_tree), the gated-line parsers
and the e3 e5 e7 e8 e9 e10 e11 e12 e13 e14 e15 subcommands with their CSV outputs.

Run:   python tests/test_parse_logs.py [--out DIR] [-k substr]   (DIR defaults to
       $PARSE_LOGS_TEST_DIR or a fresh temporary directory)
or:    python -m pytest tests/test_parse_logs.py
"""
import contextlib
import csv
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np          # noqa: E402
import parse_logs as pl     # noqa: E402
import make_fake_log as mfl  # noqa: E402

_FX = None
_FX2 = None


def fixture():
    """Build the fixture tree once; returns {'root', 'out', 'fx'}."""
    global _FX
    if _FX is None:
        out = os.environ.get("PARSE_LOGS_TEST_DIR") or tempfile.mkdtemp(prefix="parse_logs_test_")
        root = os.path.join(out, "logs_fake")
        if os.path.isdir(root):
            shutil.rmtree(root)
        _FX = {"root": root, "out": out, "fx": mfl.build_fixture_tree(root)}
    return _FX


def fixture2():
    """The rebuttal fixture tree (new subcommands); returns {'root', 'fig', 'out', 'fx'}."""
    global _FX2
    if _FX2 is None:
        out = fixture()["out"]
        root = os.path.join(out, "logs_fake_rebuttal")
        fig = os.path.join(out, "figures_fake_rebuttal")
        for d in (root, fig):
            if os.path.isdir(d):
                shutil.rmtree(d)
        _FX2 = {"root": root, "fig": fig, "out": out, "fx": mfl.build_rebuttal_tree(root, fig)}
    return _FX2


def _runs(key):
    return pl.parse_file(fixture()["fx"][key]["path"])


def _runs2(key, capture_tags=None):
    return pl.parse_file(fixture2()["fx"][key]["path"], capture_tags=capture_tags)


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _run_sub(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = pl.main(argv)
    return rc, buf.getvalue()


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


# ---------------------------------------------------------------------------
def test_task_increments():
    assert pl.task_increments(100, 10, 10) == [10] * 10
    assert pl.task_increments(196, 16, 20) == [16] + [20] * 9
    assert pl.task_increments(100, 2, 2) == [2] * 50
    assert pl.task_increments(200, 4, 4) == [4] * 50
    assert pl.task_increments(200, 10, 10) == [10] * 20
    assert pl.task_increments(100, 5, 5) == [5] * 20
    assert pl.task_increments(260, 65, 65) == [65] * 4
    assert pl.expected_tasks({"dataset_name": "cars", "dataset_init_cls": 16, "dataset_increment": 20}) == 10
    assert pl.expected_tasks({"dataset_name": "imagenetr", "dataset_init_cls": 4, "dataset_increment": 4,
                              "train_stop_at_task": 1}) == 2
    assert pl.expected_tasks({"dataset_name": "unknown_ds", "dataset_num_task": 7}) == 7


def test_metrics_port_against_reference():
    rng = np.random.RandomState(0)
    for T in (1, 2, 3, 10, 50):
        rows = mfl.synth_rows(T, rng)
        ref = mfl.compute_metrics_ref(rows)
        got = pl.metrics_from_matrix(pl.matrix_from_rows(rows))
        assert np.allclose(ref, got, atol=1e-9), (T, ref, got)
    assert pl.metrics_from_matrix(np.array([[90.0]])) == (90.0, 0.0, 0.0, 90.0)


def test_experiment_blocks_last_wins():
    runs = _runs("alpha_a0.3")
    info = fixture()["fx"]["alpha_a0.3"]
    assert len(runs) == 3 and all(r["mode"] == "experiment" for r in runs)
    r93 = next(r for r in runs if r["seed"] == 1993)
    assert r93["n_blocks"] == 2 and r93["block_statuses"] == ["ok", "ok"]
    assert abs(r93["faa"] - info["repeat"][1993]["faa"]) < 1e-9
    assert abs(r93["faa"] - info["seeds"][1993]["faa"]) > 1e-6
    assert r93["first_seen_ts"] < r93["first_ts"]
    # the last SUMMARY block covers one seed only -> flagged, not silently trusted
    assert r93["summary_check"] == "mismatch"
    with open(info["path"], encoding="utf-8") as fh:
        text = fh.read()
    assert len(pl.split_seed_blocks(text)) == 3
    assert len(pl.split_seed_blocks(text, dedup=False)) == 4
    others = [r for r in runs if r["seed"] != 1993]
    assert all(r["n_blocks"] == 1 for r in others)


def test_n_tasks_from_config():
    for key, T in (("alpha_a0.1", 10), ("long_t20", 20), ("long_t50", 50), ("inr_long_t50", 50)):
        for r in _runs(key):
            assert r["n_tasks"] == T == r["n_tasks_expected"], (key, r["n_tasks"])
            assert r["split"] == f"t{T}" and r["status"] == "ok"
            assert len(r["rows"][-1]) == T and len(r["rows"][0]) == 1
            assert r["config"]["dataset_num_task"] == T
    cars = _runs("best_cars")[0]
    assert cars["split_desc"] == "16+9x20" and cars["n_tasks_expected"] == 10
    assert _runs("long_t50")[0]["config"]["train_epochs"] == 5
    assert _runs("long_t20")[0]["split_desc"] == "20x5"


def test_recomputed_metrics_match_logged():
    for key in ("alpha_a0.1", "long_t20", "long_t50", "best_cars", "e10", "e12_avg", "mc3d"):
        info = fixture()["fx"][key]
        for r in _runs(key):
            assert r["status"] == "ok", (key, r["seed"], r["fail_reason"])
            assert r["metrics_match"] is True, (key, r["seed"])
            ref = info["seeds"][r["seed"]]
            for k in ("faa", "ffm", "ffd", "asa"):
                assert abs(r[k] - ref[k]) < 1e-9, (key, k)
                assert abs(r[k] - r[k + "_logged"]) <= 0.0101, (key, k)
            assert r["old_mean"] is not None and r["new_acc"] == r["rows"][-1][-1]
            assert abs(r["first5_mean"] - np.mean(r["rows"][-1][:5])) < 1e-9


def test_crashed_seed_flagged_and_excluded():
    runs = _runs("alpha_a0.15")
    info = fixture()["fx"]["alpha_a0.15"]
    bad = next(r for r in runs if r["seed"] == 1994)
    assert bad["status"] == "crashed" and "RuntimeError" in bad["fail_reason"]
    assert bad["n_tasks"] == 1 and bad["errors"] and bad["time_s"] is not None
    agg = pl.aggregate(runs, ["dataset", "config_name"])
    assert len(agg) == 1
    assert agg[0]["n_seeds"] == 2 and agg[0]["n_failed"] == 1 and agg[0]["failed_seeds"] == [1994]
    assert agg[0]["seeds"] == [1993, 1995] and agg[0]["status"] == "partial"
    ok_fa = [info["seeds"][s]["faa"] for s in (1993, 1995)]
    assert abs(agg[0]["faa"] - np.mean(ok_fa)) < 1e-9
    # the real SUMMARY averaged the failed seed as 0.0 (main.py:1968): reproduced, so 'match'
    assert runs[0]["summary_check"] == "match"
    assert abs(runs[0]["summary"]["faa"] - info["summary"]["faa"]) <= 0.0101
    assert runs[0]["summary"]["faa"] < 70.0


def test_std_ddof0_matches_summary():
    runs = _runs("alpha_a0.1")
    info = fixture()["fx"]["alpha_a0.1"]
    agg = pl.aggregate(runs, ["config_name"])[0]
    assert abs(agg["faa_sd"] - info["summary"]["faa_sd"]) <= 0.0101
    assert abs(agg["ffm_sd"] - info["summary"]["ffm_sd"]) <= 0.0101
    assert abs(agg["asa_sd"] - info["summary"]["asa_sd"]) <= 0.0101
    assert abs(agg["asa"] - info["summary"]["asa"]) <= 0.0101
    assert runs[0]["summary_check"] == "match"
    assert runs[0]["summary"]["dataset"] == "CIFAR224" and runs[0]["summary"]["config_name"] == "ALPHA_A0.1"
    agg1 = pl.aggregate(runs, ["config_name"], ddof=1)[0]
    assert agg1["faa_sd"] > agg["faa_sd"]
    assert "faa_val" not in agg and agg["status"] == "complete"


def test_val_metrics_only_when_present():
    for r in _runs("alpha_a0.1"):
        assert r["faa_val"] is None and r["rows_val"] == [] and r["metrics_match_val"] is None
        assert r["last_total_val"] is None and "val" not in r["summary"]
    runs = _runs("e14")
    info = fixture()["fx"]["e14"]
    for r in runs:
        assert r["metrics_match_val"] is True
        assert abs(r["faa_val"] - info["seeds"][r["seed"]]["faa_val"]) < 1e-9
        assert r["last_total_val"] is not None and r["config"]["data_val_frac"] == 0.2
        assert len(r["rows_val"]) == 10
    val_mean = np.mean([info["seeds"][s]["faa_val"] for s in pl.SEEDS])
    assert abs(runs[0]["summary"]["val"]["faa"] - val_mean) <= 0.0101
    agg = pl.aggregate(runs, ["config_name"])[0]
    assert abs(agg["faa_val"] - val_mean) < 1e-9


def test_utf8_lines_parsed_under_any_locale():
    r = _runs("alpha_a0.1")[0]
    assert r["drift_lambda"] == 100.0 and r["drift_ridge"] == "identity" and len(r["drift"]) == 9
    assert r["drift"][0]["n"] > 0 and r["drift_cos"][0] > 0.9 and r["drift"][0]["d"] == 768
    assert r["drift_w_minus_i"][0] > 1.0
    assert r["summary"]["faa_sd"] is not None
    with open(fixture()["fx"]["alpha_a0.1"]["path"], "rb") as fh:
        raw = fh.read()
    for ch in ("λ", "±", "—", "·"):
        assert ch.encode("utf-8") in raw


def test_unquoted_config_strings():
    r = _runs("alpha_a0.1")[0]
    c = r["config"]
    assert c["train_merge"] == "pspectral" and c["spectral_variant"] == "only_A"
    assert c["train_prefix"] == "main_cifar224_r64_qkvfc" and c["train_method"] == "seqft"
    assert c["model_lora_target_modules"] == ["qkv", "fc1", "fc2"]
    assert c["train_weight_decay"] == 0.0005 and c["final_model_dir"] is None
    assert c["cache_backbone"] is True and c["train_merge_incremental"] is False
    assert c["train_drift_lambda"] == 100.0 and c["seed"] == r["seed"]
    assert c["dataset_name"] == "cifar224" and c["model_backbone"] == "vit_base_patch16_224"
    assert pl.literal("vit_base_patch16_224") == "vit_base_patch16_224"
    assert pl.literal("5e-05") == 5e-05 and pl.literal("['qkv', 'fc1']") == ["qkv", "fc1"]
    assert pl.literal("checkpoints_main/rebuttal/e14") == "checkpoints_main/rebuttal/e14"
    assert pl.literal("None") is None and pl.literal("nan") == "nan"


def test_configuration_only_sota_log():
    fx = fixture()["fx"]["sota"]
    sel, allruns, summaries, sweeps = pl.parse_file_full(fx["path"])
    assert len(sel) == 9 and len(allruns) == 9
    assert all(r["mode"] == "configuration" for r in sel)
    assert summaries == {}
    names = sorted({(r["dataset"], r["config_name"]) for r in sel})
    assert names == sorted(fx["cells"].keys())
    for r in sel:
        assert not r["name_inferred"] and r["seed"] in pl.SEEDS and r["status"] == "ok"
        assert r["merge_rule"] == "sota" and r["merge_sota_method"] == "dare" and r["merge_coef"] == 1.0
        assert r["ffm"] is not None and r["time_s"] is None
        cell = fx["cells"][(r["dataset"], r["config_name"])]
        assert abs(r["faa"] - cell["seeds"][r["seed"]]["faa"]) < 1e-9
        assert abs(r["sweep"]["FA"][0] - cell["fa"]) <= 0.0101
        assert abs(r["sweep"]["AA"][1] - cell["aa_sd"]) <= 0.0101
    kinds = [s["kind"] for s in sweeps]
    assert kinds.count("cell") == 3 and kinds.count("best") == 1
    best = next(s for s in sweeps if s["kind"] == "best")
    assert best["method"] == "dare" and best["best"] == "q0.5_c1.0"
    agg = pl.aggregate(sel, ["dataset", "config_name"])
    assert len(agg) == 3 and all(a["n_seeds"] == 3 for a in agg)
    # a block with no SWEEP line after it keeps an inferred name
    orphan = pl.split_seed_blocks("\n".join(l for l in open(fx["path"], encoding="utf-8")
                                            .read().splitlines() if "SWEEP SUMMARY" not in l))
    assert all(r["name_inferred"] and r["config_name"].startswith("cfg_dare") for r in orphan)


def test_tags_captured_raw():
    """The gated lines (real main.py wording) land verbatim in run['tags'][tag]."""
    r = _runs("e10")[0]
    info = fixture()["fx"]["e10"]["seeds"][r["seed"]]
    T = r["n_tasks"]
    expected = (("MergeStat", T - 1), ("FeatShift", T - 1), ("Timing", info["n_timing_lines"]),
                ("DriftDiag", T - 1), ("Data", 1), ("Evaluation-til", T), ("Freeze", 1))
    for tag, n in expected:
        assert len(r["tags"].get(tag, [])) == n, (tag, len(r["tags"].get(tag, [])))
    assert r["tags"]["MergeStat"][0].startswith("[MergeStat] task 1 alpha=0.3 op=trunc k_mean=")
    assert "| merge_time=" in r["tags"]["MergeStat"][0]
    assert r["tags"]["Timing"][0].startswith("[Timing] task 0 train=") and "cached=1" in r["tags"]["Timing"][0]
    assert r["tags"]["Data"][0].startswith("[Data] dataset=imagenetr | train: n=")
    assert r["tags"]["Evaluation-til"][0].startswith("[Evaluation-til] Total Acc: ")
    for tag in ("Evaluation", "Training", "Alignment", "Merging", "Drift", "Task 0", "Optimizer", "L2M", "SpecNorm"):
        assert tag not in r["tags"], tag              # cached run: no training-time lines
    assert r["trainable_params"] == 8257536 and r["frozen_params"] == 85798656 and r["peft_key"] == "lora_"
    assert r["config"]["merge_update_op"] == "trunc" and r["config"]["merge_log_stats"] is True
    assert any(lbl == "[Drift] W shape" for _ts, lbl in r["events"])
    assert any(lbl == "[Evaluation] Task 9" for _ts, lbl in r["events"])
    assert not any(lbl == "[Alignment] Task 0" for _ts, lbl in r["events"])      # task 0 has no alignment
    sr = _runs("specreg")[0]
    assert len(sr["tags"]["StepNorm"]) == sr["n_tasks"] - 1 and sr["merge_rule"] is None
    assert len(sr["tags"]["SpecNorm"]) == sr["n_tasks"]                          # uncached: one per trained task
    assert sr["tags"]["StepNorm"][0].startswith("[StepNorm] task 1 k_mean=")
    assert sr["config"]["specreg_weight"] == 0.1 and sr["config"]["train_merge"] == "none"
    only = pl.parse_file(fixture()["fx"]["e10"]["path"], capture_tags=("MergeStat",))[0]
    assert set(only["tags"]) == {"MergeStat"}


def test_merge_lines():
    r = _runs("e10")[0]
    assert r["merge_rule"] == "pspectral" and r["merge_variant"] == "only_A" and r["merge_p"] == 0.9
    assert r["merge_alpha"] == 0.3 and len(r["merge_alphas"]) == r["n_tasks"] - 1
    assert r["merge_incremental_observed"] is None
    avg = _runs("e12_avg")[0]
    assert avg["merge_rule"] == "classical" and avg["merge_incremental_observed"] is True
    assert avg["merge_n_ckpts"] == [2] * 9 and avg["config"]["train_merge_incremental"] is True
    ties = _runs("e12_ties")[0]
    assert ties["merge_incremental_observed"] is False and ties["merge_n_ckpts"] == list(range(2, 11))


def test_time_and_training_counts():
    fx = fixture()["fx"]["best_cifar224"]
    for r in _runs("best_cifar224"):
        assert r["n_cached_tasks"] == 0 and r["n_trained_tasks"] == 10 and r["n_epoch_lines"] == 100
        assert r["time_s"] == fx["seeds"][r["seed"]]["time_s"] and r["time_s"] > 300
        assert r["config"]["train_prefix"] == "best_cifar224_r64"
    for r in _runs("alpha_a0.1"):
        assert r["n_cached_tasks"] == 10 and r["n_trained_tasks"] == 0 and r["time_s"] < 300


def test_stop_at_task_and_smoke_skip():
    fx = fixture()
    smoke = pl.parse_file(fx["fx"]["smoke"]["path"])[0]
    assert smoke["n_tasks"] == 2 and smoke["n_tasks_expected"] == 2 and smoke["status"] == "ok"
    runs = pl.load_runs([fx["root"]], skip_smoke=True)
    assert not any(r["config_name"] == "e8_smoke" for r in runs)
    runs_all = pl.load_runs([fx["root"]], skip_smoke=False)
    assert any(r["config_name"] == "e8_smoke" for r in runs_all)
    assert len(runs_all) == len(runs) + 1
    assert pl.load_runs(globs=[os.path.join(fx["root"], "cifar224", "alpha_*.log")]) and \
        len(pl.parse_glob(os.path.join(fx["root"], "cifar224", "alpha_*.log"))) == 9


def test_incomplete_block():
    fx = fixture()
    with open(fx["fx"]["alpha_a0.1"]["path"], encoding="utf-8") as fh:
        text = fh.read()
    cut = text.split("[Evaluation] Task 6")[0]          # first seed truncated after 6 tasks
    runs = pl.split_seed_blocks(cut)
    assert len(runs) == 1 and runs[0]["status"] == "incomplete" and runs[0]["n_tasks"] == 6
    assert "6/10" in runs[0]["fail_reason"]
    assert pl.aggregate(runs, ["config_name"])[0]["n_seeds"] == 0


def test_dil_and_totals():
    d = _runs("dil")[0]
    assert d["dil_last"] is not None and d["dil_inc"] is not None
    assert d["n_tasks"] == 4 and d["n_tasks_expected"] == 4 and d["dataset"] == "officehome_dil"
    cars = _runs("best_cars")[0]
    assert abs(cars["last_total"] - cars["faa"]) > 1e-6
    assert abs(cars["inc_total"] - np.mean(cars["totals"])) < 1e-9 and len(cars["totals"]) == 10
    cifar = _runs("alpha_a0.1")[0]
    assert abs(cifar["last_total"] - cifar["faa"]) <= 0.0101


def test_e1_subcommand():
    fx = fixture()
    out = os.path.join(fx["out"], "figures_fake_e1")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "e1_long_baselines.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "dataset", "split", "last", "last_sd", "inc", "inc_sd", "source"])
        w.writerow(["E2-LoRA", "cifar224", "t20", "91.42", "0.10", "94.55", "0.07", "RESULTS.md s4"])
        w.writerow(["E2-LoRA", "cifar224", "t50", "90.70", "0.24", "93.86", "0.09", "RESULTS.md s4"])
    rc, text = _run_sub(["e1", "--roots", fx["root"], "--out-dir", out])
    assert rc == 0
    assert "E2-LoRA" in text and "ISM (ours)" in text and "cifar224 -- 20x5 (t20)" in text
    assert "91.42 +- 0.10" in text
    agg = _read_csv(os.path.join(out, "e1_long_agg.csv"))
    keys = {(r["dataset"], r["split"], r["config_name"]) for r in agg}
    assert ("cifar224", "t20", "long_t20") in keys and ("cifar224", "t50", "long_t50") in keys
    assert ("imagenetr", "t50", "long_t50") in keys and ("cifar224", "t10", "best_cifar224") in keys
    assert ("imagenetr", "t10", "best_imagenetr") in keys
    assert not any(k[2].startswith("alpha_") for k in keys)
    row = next(r for r in agg if r["config_name"] == "long_t50" and r["dataset"] == "cifar224")
    assert row["n_seeds"] == "3" and row["complete"] == "True" and row["epochs"] == "5"
    assert row["last_total"] and row["inc_total"] and row["summary_check"] == "match"
    assert abs(float(row["faa"]) - fx["fx"]["long_t50"]["summary"]["faa"]) <= 0.0101
    assert abs(float(row["faa_sd"]) - fx["fx"]["long_t50"]["summary"]["faa_sd"]) <= 0.0101
    traj = _read_csv(os.path.join(out, "e1_long_traj_cifar224_t50.csv"))
    assert len(traj) == 50 and traj[0]["n_seeds"] == "3" and traj[-1]["task"] == "49"
    assert traj[0]["config_name"] == "long_t50" and float(traj[0]["acc_mean"]) > 0
    assert len(_read_csv(os.path.join(out, "e1_long_traj_cifar224_t20.csv"))) == 20
    assert len(_read_csv(os.path.join(out, "e1_long_traj_imagenetr_t50.csv"))) == 50
    per = _read_csv(os.path.join(out, "e1_long_per_seed.csv"))
    assert len(per) == 3 * 5
    assert {r["status"] for r in per} == {"ok"}


def test_e2_subcommand():
    fx = fixture()
    out = os.path.join(fx["out"], "figures_fake_e2")
    os.makedirs(out, exist_ok=True)
    cube_fa = fx["fx"]["mc3d"]["summary"]["faa"]
    with open(os.path.join(out, "mc3d_imagenetr_full.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "p", "alpha", "n_seeds", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd",
                    "old_mean", "new_acc", "first5_mean"])
        w.writerow(["only_A", "0.90", "0.30", "3", f"{cube_fa:.2f}", "0.10", "87.00", "0.10", "3.00",
                    "0.10", "83.00", "85.00", "83.00"])
        w.writerow(["combine", "0.10", "0.10", "3", "81.66", "0.17", "86.19", "0.27", "4.60", "0.20",
                    "81.49", "83.11", "81.76"])
    rc, text = _run_sub(["e2", "--roots", fx["root"], "--out-dir", out, "--validate",
                         "--cubes-dir", out, "--cube-datasets", "imagenetr"])
    assert rc == 0
    rec = _read_csv(os.path.join(out, "e2_tuning_record.csv"))
    cif = [r for r in rec if r["dataset"] == "cifar224" and r["rule"] == "pspectral"]
    by = {(r["train_merge_alpha"], r["train_prefix"], r["split"]): r for r in cif}
    a1 = by[("0.1", "main_cifar224_r64_qkvfc", "t10")]
    a15 = by[("0.15", "main_cifar224_r64_qkvfc", "t10")]
    a3 = by[("0.3", "main_cifar224_r64_qkvfc", "t10")]
    assert a1["n_seeds"] == "3" and a1["n_runs"] == "3" and a1["flags"] == "" and a1["headline"] == "False"
    assert a15["n_seeds"] == "2" and a15["n_failed"] == "1" and "failed seeds" in a15["flags"]
    assert a15["headline"] == "True"
    assert a3["n_seeds"] == "3" and a3["n_runs"] == "4" and "duplicate" in a3["flags"]
    assert by[("0.15", "best_cifar224_r64", "t10")]["headline"] == "True"
    assert ("0.15", "main_cifar224_r64_qkvfc", "t20") in by and ("0.15", "main_cifar224_r64_qkvfc", "t50") in by
    assert by[("0.15", "main_cifar224_r64_qkvfc", "t50")]["train_epochs"] == "5"
    assert len(cif) == 6
    ts = [r["first_ts"] for r in rec if r["dataset"] == "cifar224"]
    assert ts == sorted(ts) and all(ts)
    sota = [r for r in rec if r["rule"] == "dare"]
    assert len(sota) == 3 and all(r["n_seeds"] == "3" for r in sota)
    assert all("merge_dare_q" in r["extras"] for r in sota)
    assert all(r["config_names"].startswith("sota_") for r in sota)
    spec_rows = [r for r in rec if r["rule"] == "none" and "specreg_weight=0.1" in r["extras"]]
    assert len(spec_rows) == 1
    assert len(_read_csv(os.path.join(out, "e2_specreg_runs.csv"))) == 1
    bud = _read_csv(os.path.join(out, "e2_tuning_budget.csv"))
    b = next(r for r in bud if r["dataset"] == "cifar224" and r["rule"] == "pspectral")
    assert b["n_cells"] == "6" and "train_merge_alpha" in b["varied_keys"] and b["headline_present"] == "True"
    assert b["n_seed_runs"] == "19"            # 3+3+4 (dup) +3 +3 +3
    tt = _read_csv(os.path.join(out, "e2_time_per_seed.csv"))
    unc = [r for r in tt if r["kind"] == "uncached" and r["dataset"] == "cifar224"]
    assert len(unc) == 3 and all(r["config_name"] == "best_cifar224" for r in unc)
    assert not any(r["config_name"].startswith("sota_") for r in tt)
    assert "| cifar224 | uncached | 3 |" in text and "| cifar224 | cached |" in text
    assert "2 cube cells: 1 matched, 0 mismatched, 1 missing" in text
    assert "cub: headline cell" in text and "cannot be reconstructed" in text
    assert "cifar224: headline cell present" in text and "argmax alpha of that sweep" in text
    assert "Existing specreg / speccap cells (E10): 1" in text


def test_subcommands_registered():
    for name in ("e1", "e2", "e3", "e5", "e7", "e8", "e9", "e10", "e11", "e12", "e13", "e14", "e15"):
        assert name in pl.SUBCOMMANDS, name
        assert "NOT IMPLEMENTED" not in pl.SUBCOMMANDS[name][1]
    assert pl.main(["e99"]) == 2
    assert pl.main(["e4"]) == 2            # E4 is analyze_spectrum.py, not a parser subcommand


def test_cli_generic_subprocess():
    fx = fixture()
    out = os.path.join(fx["out"], "figures_fake_cli")
    os.makedirs(out, exist_ok=True)
    per, agg = os.path.join(out, "per_seed.csv"), os.path.join(out, "agg.csv")
    res = subprocess.run([sys.executable, os.path.join(ROOT, "parse_logs.py"), "--roots", fx["root"],
                          "--per-seed", per, "--agg", agg],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, res.stderr
    assert "| dataset | config_name |" in res.stdout
    rows = _read_csv(agg)
    r = next(x for x in rows if x["config_name"] == "alpha_a0.15")
    assert r["n_seeds"] == "2" and r["n_failed"] == "1" and r["failed_seeds"] == "1994"
    per_rows = _read_csv(per)
    assert len(per_rows) > 30 and {"faa", "last_total", "merge_alpha", "path"} <= set(per_rows[0])
    for sub in ("e1", "e3", "e5", "e7", "e8", "e9", "e10", "e11", "e12", "e13", "e14", "e15"):
        res = subprocess.run([sys.executable, os.path.join(ROOT, "parse_logs.py"), sub, "--help"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert res.returncode == 0 and "--out-dir" in res.stdout, (sub, res.stderr)


# ---------------------------------------------------------------------------
# Gated-line parsers (real main.py wording) and the two new parse_block keys
# ---------------------------------------------------------------------------
def test_kv_line_parsers():
    ms = pl.parse_norm_line("[MergeStat] task 3 alpha=0.3 op=clip k_mean=12.30 k_min=4 k_max=40 | A_sigma_max_mean=0.4578 "
                            "A_sigma_max_max=0.59514 | eff_sigma_max_mean=0.54 eff_sigma_max_max=0.675 eff_fro_mean=0.63 | "
                            "raw_sigma_max_mean=1.53 raw_eff_sigma_max_mean=1.8 | merge_time=0.874s")
    assert ms["task"] == 3 and ms["alpha"] == 0.3 and ms["op"] == "clip" and ms["k_mean"] == 12.3
    assert ms["k_min"] == 4 and ms["k_max"] == 40 and ms["merge_time"] == 0.874 and ms["degenerate"] is False
    assert ms["raw_eff_sigma_max_mean"] == 1.8 and ms["A_sigma_max_max"] == 0.59514
    red = pl.parse_norm_line("[MergeStat] task 2 | A_sigma_max_mean=0.1 A_sigma_max_max=0.2 | eff_sigma_max_mean=0.3 "
                             "eff_sigma_max_max=0.4 eff_fro_mean=0.5 | raw_sigma_max_mean=1.5 raw_eff_sigma_max_mean=1.8 | "
                             "merge_time=1.250s")
    assert "alpha" not in red and red["task"] == 2 and red["eff_fro_mean"] == 0.5 and red["merge_time"] == 1.25
    deg = pl.parse_norm_line("[MergeStat] task 1 no LoRA factor pairs | merge_time=0.010s")
    assert deg["degenerate"] is True and deg["merge_time"] == 0.01
    sn = pl.parse_norm_line("[StepNorm] task 4 k_mean=14.00 k_min=5 k_max=48 | A_sigma_max_mean=1.5 A_sigma_max_max=1.9 | "
                            "eff_sigma_max_mean=0.611 eff_sigma_max_max=0.76 eff_fro_mean=2.9")
    assert sn["task"] == 4 and sn["eff_sigma_max_mean"] == 0.611 and "merge_time" not in sn
    tm = pl.parse_timing("[Timing] task 3 drift_solve=0.031s n=2400 d=768 solver=dense")
    assert tm["task"] == 3 and tm["stage"] == "drift_solve" and tm["seconds"] == 0.031
    assert tm["extra"] == {"n": 2400, "d": 768, "solver": "dense"}
    assert pl.parse_timing("[Timing] task 0 drift=1.000s")["extra"] == {}
    assert pl.parse_timing("[Timing] task 0 train=0.300s cached=1")["extra"]["cached"] == 1
    fs = pl.parse_featshift("[FeatShift] task 2 shift=0.0752 n=2400")
    assert fs == {"task": 2, "shift": 0.0752, "n": 2400}
    dd = pl.parse_driftdiag("[DriftDiag] task 3 K=60 n_img=6000 src=train | mean_rel raw=0.1300 before=0.0950 after=0.0330 | "
                            "cov_relF raw=0.2300 before=0.1650 after=0.0530 | onestep mean_rel I=0.0930 W=0.0365 oracle=0.0250 "
                            "cov_relF I=0.1800 W=0.0600 oracle=0.0450")
    assert dd["task"] == 3 and dd["K"] == 60 and dd["n_img"] == 6000 and dd["src"] == "train"
    assert dd["mean_rel_raw"] == 0.13 and dd["mean_rel_after"] == 0.033 and dd["cov_relF_before"] == 0.165
    assert dd["os_mean_rel_I"] == 0.093 and dd["os_mean_rel_oracle"] == 0.025 and dd["os_cov_relF_W"] == 0.06
    assert pl.parse_driftdiag("[DriftDiag] raw statistics missing; skipping") is None
    short = pl.parse_driftdiag("[DriftDiag] task 1 K=20 n_img=2000 src=test | mean_rel raw=0.1 before=0.08 after=0.03 | "
                               "cov_relF raw=0.2 before=0.15 after=0.05")
    assert "os_mean_rel_I" not in short and short["src"] == "test"
    dl = pl.parse_data_line("[Data] dataset=imagenetr | train: n=2360 C=200 per-class min/median/max=12/20/20 | val: none | "
                            "test: n=6000 C=200 per-class min/median/max=30/30/30 | fewshot: k=20 (subsample_seed=1993)")
    assert dl["dataset"] == "imagenetr" and dl["train"]["median"] == 20.0 and dl["train"]["min"] == 12
    assert dl["val"] is None and dl["fewshot_k"] == 20 and dl["subsample_seed"] == 1993 and dl["test"]["n"] == 6000
    dl2 = pl.parse_data_line("[Data] dataset=cifar224 | train: n=45000 C=100 per-class min/median/max=450/450/450 | "
                             "val: n=5000 C=100 per-class min/median/max=50/50/50 (val_frac=0.1, val_seed=1993) | "
                             "test: n=10000 C=100 per-class min/median/max=100/100/100 | fewshot: none")
    assert dl2["val"]["n"] == 5000 and dl2["val_frac"] == 0.1 and dl2["val_seed"] == 1993 and dl2["fewshot_k"] is None
    assert pl.TIL_RE.match("[Evaluation-til] Total Acc: 95.12, Grouped: [96.0, 94.5]").group(2) == "96.0, 94.5"
    assert pl.L2M_DIST_RE.match("[L2M] end-of-task ||W_t-anchor||_F=1.2345").group(1) == "1.2345"
    m = pl.L2M_ANCHOR_RE.match("[L2M] anchor=1993_imagenetr_vit_base_patch16_224_lora_main_inr_r64_qkvfc_l2m1.0_backbone_0.pt "
                               "weight=1.0 tensors=72")
    assert m.group(2) == "1.0" and m.group(3) == "72"
    assert pl.kv_pairs("||W_t-anchor||_F=0.5 weight=1.0") == {"weight": 1.0}
    assert pl.kv_pairs("rule=specnorm stat=eff s_t=1.82") == {"rule": "specnorm", "stat": "eff", "s_t": 1.82}


def test_core_merge_state_and_gcv_keys():
    fx = fixture2()["fx"]
    r = _runs2("e15_opcm_0.5_adaptive")[0]
    assert r["merge_rule"] == "sota" and r["merge_sota_method"] == "opcm"
    assert len(r["merge_state"]) == r["n_tasks"] - 1 and r["merge_state"][0][0] == "opcm"
    kv = r["merge_state"][1][1]
    ref = fx["e15_opcm_0.5_adaptive"]["seeds"][r["seed"]]["merge_state"][2]
    assert kv["alpha"] == 0.5 and kv["lam_rule"] == "adaptive" and abs(kv["lam_t"] - ref["lam_t"]) < 1e-4
    assert abs(kv["kept_frac"] - ref["kept_frac"]) < 1e-4 and kv["r_alpha_mean"] == ref["r_alpha_mean"]
    nf = _runs2("e15_nufilt_32_1")[0]
    assert nf["merge_state"][0][0] == "nufilt" and nf["merge_state"][0][1]["rp"] == 32
    assert nf["merge_state"][0][1]["adapt"] == 1 and nf["merge_state"][0][1]["n_adapted"] == 36
    assert nf["merge_state"][0][1]["adapt_loss_last"] == 0.401
    g = _runs2("e7_gcv")[0]
    info = fx["e7_gcv"]["seeds"][g["seed"]]
    assert len(g["drift_gcv"]) == g["n_tasks"] - 1
    for i, rec in enumerate(g["drift_gcv"]):
        lam, scores = info["gcv"][i + 1]
        assert rec["lambda"] == lam and g["drift"][i]["lambda"] == lam
        assert set(rec["scores"]) == set(mfl.GCV_GRID)
        assert abs(rec["scores"][lam] - scores[lam]) < 1e-5
    assert g["drift"][0]["extra"] == "solver=woodbury"
    plain = _runs2("e10_trunc")[0]
    assert plain["merge_state"] == [] and plain["drift_gcv"] == []


# ---------------------------------------------------------------------------
# e3 ... e15 on the rebuttal tree
# ---------------------------------------------------------------------------
def test_e3_stage_times():
    fx = fixture2()
    out = os.path.join(fx["out"], "figures_fake_e3")
    rc, text = _run_sub(["e3", "--roots", fx["root"], "--out-dir", out, "--configs", "cov_class", "e10_trunc_a0.3",
                         "e10_l2m1.0", "--timing-csv", fx["fx"]["timing_csv_imagenetr"]["path"]])
    assert rc == 0
    rows = _read_csv(os.path.join(out, "e3_stage_times_per_task.csv"))
    info = fx["fx"]["cov_imagenetr_class"]["seeds"][1993]
    ts = {int(r["task"]): r for r in rows if r["config_name"] == "cov_class" and r["dataset"] == "imagenetr"
          and r["seed"] == "1993" and r["source"] == "timestamps"}
    tm = {int(r["task"]): r for r in rows if r["config_name"] == "cov_class" and r["dataset"] == "imagenetr"
          and r["seed"] == "1993" and r["source"] == "timing"}
    cs = {int(r["task"]): r for r in rows if r["config_name"] == "cov_class" and r["dataset"] == "imagenetr"
          and r["seed"] == "1993" and r["source"] == "csv"}
    assert set(ts) == set(tm) == set(cs) == set(range(10))
    for t in range(1, 10):
        exp = info["timing"][t]
        for stage in ("train", "merge", "drift", "drift_features", "drift_solve", "transport", "stats", "align", "eval"):
            assert abs(float(tm[t][stage + "_s"]) - exp[stage]) < 1e-6, (t, stage)
            assert abs(float(cs[t][stage + "_s"]) - exp[stage]) < 1e-6, (t, stage)
        # timestamps reproduce the stage lengths the clock spaced the lines with
        assert abs(float(ts[t]["merge_s"]) - exp["merge"]) < 0.01, ts[t]["merge_s"]
        assert abs(float(ts[t]["drift_s"]) - exp["drift"]) < 0.01
        assert abs(float(ts[t]["align_s"]) - exp["align"]) < 0.01
        assert abs(float(ts[t]["eval_s"]) - exp["eval"]) < 0.01
        assert abs(float(ts[t]["train_s"]) - exp["train"]) < 0.01
        assert abs(float(ts[t]["transport_stats_s"]) - (exp["transport"] + exp["stats"])) < 0.01
        assert ts[t]["stats_s"] == "" and ts[t]["drift_solve_s"] == ""      # not separable from timestamps
        assert ts[t]["cached"] == "1" and tm[t]["cached"] == "1"
    assert ts[0]["merge_s"] == "" and ts[0]["drift_s"] == "" and ts[0]["align_s"] == ""
    l2m = [r for r in rows if r["config_name"] == "e10_l2m1.0" and r["source"] == "timestamps" and r["task"] == "3"]
    assert len(l2m) == 3 and all(float(r["train_s"]) > 25 and r["merge_s"] == "" and r["cached"] == "0" for r in l2m)
    share = _read_csv(os.path.join(out, "e3_stage_share.csv"))
    keyed = {(r["dataset"], r["config_name"], r["source"]): r for r in share}
    s = keyed[("imagenetr", "cov_class", "timing")]
    assert s["n_seeds"] == "3" and s["n_tasks"] == "10"
    total = float(s["time_total_s"])
    assert abs(float(s["share_drift"]) - 9 * 4.0 / total) < 0.01
    assert float(s["share_task"]) > 0.9 and float(s["drift_s"]) == 4.0 and float(s["drift_solve_s"]) == 1.0
    assert keyed[("imagenetr", "e10_l2m1.0", "timestamps")]["drift_over_train"] != ""
    assert keyed[("imagenetr", "cov_class", "csv")]["drift_features_s"] == "3"
    assert "| imagenetr | cov_class | timing |" in text and "pct_drift" in text


def test_e5_storage_accuracy():
    fx = fixture2()
    out = os.path.join(fx["out"], "figures_fake_e5")
    rc, text = _run_sub(["e5", "--roots", fx["root"], "--out-dir", out])
    assert rc == 0
    rows = _read_csv(os.path.join(out, "e5_storage_accuracy.csv"))
    by = {(r["dataset"], r["mode"], r["rank"]): r for r in rows}
    assert len(rows) == 7 and set(r["dataset"] for r in rows) == {"imagenetr", "cifar224"}
    full = by[("imagenetr", "class", "")]
    assert full["stats_per_class"] == "590592" and full["stats_floats"] == str(200 * 590592)
    assert full["lora_floats"] == "8259072" and full["head_floats"] == str(200 * 768) and full["config"] == "cov_class"
    assert abs(float(full["stats_MB"]) - 200 * 590592 * 4 / 1e6) < 0.01
    assert by[("imagenetr", "diag", "")]["stats_per_class"] == "1536"
    assert by[("imagenetr", "lowrank", "16")]["stats_per_class"] == str(768 * 18)
    assert by[("imagenetr", "lowrank", "64")]["stats_per_class"] == str(768 * 66)
    assert by[("imagenetr", "shared", "")]["stats_floats"] == str(200 * 768 + 768 * 768)
    assert float(full["dfa_vs_class"]) == 0.0
    d = by[("imagenetr", "diag", "")]
    assert abs(float(d["dfa_vs_class"]) - (float(d["fa"]) - float(full["fa"]))) < 0.011
    assert abs(float(d["fa"]) - fx["fx"]["cov_imagenetr_diag"]["summary"]["faa"]) <= 0.0101
    assert all(r["n_seeds"] == "3" for r in rows)
    # memory_table.join_results (pure Python) accepts the CSV: dataset + config columns
    import memory_table as mt
    joined = mt.join_results(mt.build_rows(datasets=["imagenetr", "cifar224"], ranks=[16, 64]),
                             os.path.join(out, "e5_storage_accuracy.csv"))
    assert len(joined) == 7 and all(j["fa"] for j in joined)
    assert "| imagenetr | cov_shared | shared |" in text and "(no cov_* runs for ['cub', 'cars'])" in text


def test_e7_adaptive_alpha_and_gcv():
    fx = fixture2()
    out = os.path.join(fx["out"], "figures_fake_e7")
    rc, text = _run_sub(["e7", "--roots", fx["root"], "--out-dir", out])
    assert rc == 0
    rows = _read_csv(os.path.join(out, "e7_adaptive_alpha.csv"))
    spec = [r for r in rows if r["config_name"] == "e7_specnorm_0.5" and r["dataset"] == "imagenetr"]
    assert len(spec) == 27 and all(r["rule"] == "specnorm" and r["stat"] == "eff" for r in spec)
    info = fx["fx"]["e7_specnorm"]["seeds"][1993]["adaptive"]
    got = {int(r["task"]): float(r["alpha_t"]) for r in spec if r["seed"] == "1993"}
    for t, s_t, a in info:
        assert abs(got[t] - a) < 1e-4 and abs(float(next(r["s_t"] for r in spec if r["seed"] == "1993"
                                                              and int(r["task"]) == t)) - s_t) < 1e-5
    fsh = [r for r in rows if r["config_name"] == "e7_featshift_0.03"]
    assert len(fsh) == 27 and all(r["rule"] == "featshift" and r["stat"] == "shift" for r in fsh)
    calib = [r for r in rows if r["config_name"] == "e7_calib_featshift"]
    assert len(calib) == 9 and all(r["rule"] == "fixed" and r["stat"] == "shift" and float(r["alpha_t"]) == 0.3 for r in calib)
    assert abs(float(calib[0]["s_t"]) - fx["fx"]["e7_calib"]["seeds"][1993]["featshift"][1]) < 1e-5
    summ = {(r["dataset"], r["config_name"]): r for r in _read_csv(os.path.join(out, "e7_summary.csv"))}
    s = summ[("imagenetr", "e7_specnorm_0.5")]
    assert s["ref_config"] == "e10_trunc_a0.3" and s["headline_alpha"] == "0.3" and s["n_seeds"] == "3"
    assert abs(float(s["alpha_mean"]) - np.mean([a for _t, _s, a in info])) < 1e-3
    assert abs(float(s["fa_gap"]) - (float(s["fa"]) - float(s["ref_fa"]))) < 0.011
    assert abs(float(s["ref_fa"]) - fx["fx"]["e10_trunc"]["summary"]["faa"]) <= 0.0101
    assert summ[("cifar224", "e7_specnorm_0.5")]["ref_config"] == "mc3d_only_A_p0.9_a0.15"
    assert summ[("imagenetr", "e7_featshift_0.03")]["stat"] == "shift"
    gcv = _read_csv(os.path.join(out, "e7_gcv_lambda.csv"))
    assert len(gcv) == 27 and {r["lambda"] for r in gcv} == {"100", "30", "300"}
    assert all(r["scores"].count(":") == 7 for r in gcv)
    gs = _read_csv(os.path.join(out, "e7_gcv_summary.csv"))[0]
    assert gs["config_name"] == "e7_gcv" and gs["lambda_mode"] == "100" and gs["headline_lambda"] == "100"
    assert gs["lambda_per_task"].split(";")[1] == "30" and gs["ref_config"] == "e10_trunc_a0.3"
    assert "GCV-chosen ridge lambda" in text and "| imagenetr | e7_specnorm_0.5 | specnorm | eff | 0.5 | 3 |" in text


def test_e8_mc3d_csvs_and_protection():
    fx = fixture2()
    fig = fx["fig"]
    before = _md5(os.path.join(fig, "mc3d_cars_full.csv"))
    before_inr = _md5(os.path.join(fig, "mc3d_imagenetr_full.csv"))
    rc, text = _run_sub(["e8", "--roots", fx["root"], "--out-dir", fig, "--datasets", "cifar224", "cub", "cars", "imagenetr"])
    assert rc == 0
    assert _md5(os.path.join(fig, "mc3d_cars_full.csv")) == before            # committed cube untouched
    assert _md5(os.path.join(fig, "mc3d_imagenetr_full.csv")) == before_inr
    cif = _read_csv(os.path.join(fig, "mc3d_cifar224_full.csv"))
    assert list(cif[0].keys()) == pl.MC3D_COLS and len(cif) == 10
    assert {(r["p"], r["alpha"]) for r in cif} >= {("0.90", "0.15"), ("1.00", "0.15"), ("0.10", "1.00"), ("0.50", "1.00")}
    row = next(r for r in cif if r["p"] == "0.90" and r["alpha"] == "0.15")
    summ = fx["fx"]["e8_cifar224_p0.9_a0.15"]["summary"]
    assert abs(float(row["fa"]) - summ["faa"]) <= 0.0101 and abs(float(row["fa_sd"]) - summ["faa_sd"]) <= 0.0101
    assert abs(float(row["ffm"]) - summ["ffm"]) <= 0.0101 and row["n_seeds"] == "3"
    seeds = fx["fx"]["e8_cifar224_p0.9_a0.15"]["seeds"]
    assert abs(float(row["new_acc"]) - np.mean([seeds[s]["rows"][-1][-1] for s in pl.SEEDS])) <= 0.0101
    assert abs(float(row["first5_mean"]) - np.mean([np.mean(seeds[s]["rows"][-1][:5]) for s in pl.SEEDS])) <= 0.0101
    assert len(_read_csv(os.path.join(fig, "mc3d_cub_full.csv"))) == 4
    extra = _read_csv(os.path.join(fig, "mc3d_cars_extra.csv"))
    assert len(extra) == 2 and {(r["p"], r["alpha"]) for r in extra} == {("0.90", "0.20"), ("1.00", "0.20")}
    extra_inr = _read_csv(os.path.join(fig, "mc3d_imagenetr_extra.csv"))
    assert len(extra_inr) == 0                                                  # p1.0 a0.3 is already in the cube
    assert "cifar224: paired p=0.9 vs p=1.0" in text and "| 0.15 |" in text and "<-" in text
    assert "large-step claim" in text and "cars: paired" in text and "| cube |" in text
    try:
        pl.write_mc3d(os.path.join(fig, "mc3d_cars_full.csv"), [])
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_e9_driftdiag():
    fx = fixture2()
    out = os.path.join(fx["out"], "figures_fake_e9")
    rc, text = _run_sub(["e9", "--roots", fx["root"], "--out-dir", out])
    assert rc == 0
    inr = _read_csv(os.path.join(out, "e9_driftdiag_imagenetr_t10.csv"))
    assert len(inr) == sum(1 + t for t in range(1, 10))                       # 'all' + one row per old task
    # the schema scripts/fig_e9_driftdiag.py reads: bare metric name = seed mean, <metric>_sd = std
    assert inr[0].keys() >= {"dataset", "split", "task", "group", "n_seeds", "seeds", "source",
                             "mean_rel_raw", "mean_rel_raw_sd", "os_mean_rel_oracle", "cov_relF_after_sd"}
    assert not any(k.endswith("_mean") for k in inr[0])
    allg = [r for r in inr if r["group"] == "all"]
    assert len(allg) == 9 and all(r["n_seeds"] == "3" and r["source"] == "csv" for r in allg)
    assert all(r["dataset"] == "imagenetr" and r["split"] == "t10" and r["seeds"] == "1993;1994;1995" for r in allg)
    dd = fx["fx"]["driftdiag_inr"]["seeds"][1993]["driftdiag"]
    r3 = next(r for r in allg if r["task"] == "3")
    assert abs(float(r3["mean_rel_after"]) - dd[3]["mean_rel_after"]) < 1e-5
    assert abs(float(r3["os_mean_rel_oracle"]) - dd[3]["os_mean_rel_oracle"]) < 1e-5
    assert float(r3["mean_rel_raw_sd"]) == 0.0 and r3["W_minus_I_F"] != ""
    g0 = next(r for r in inr if r["task"] == "3" and r["group"] == "0")
    assert float(g0["mean_rel_raw"]) > float(r3["mean_rel_raw"])
    cif = _read_csv(os.path.join(out, "e9_driftdiag_cifar224_t10.csv"))
    assert len(cif) == 9 and all(r["group"] == "all" and r["source"] == "log" and r["n_seeds"] == "2" for r in cif)
    assert cif[0]["symkl_raw"] == "" and cif[0]["os_mean_rel_W"] != ""
    assert "| imagenetr | t10 | 3 | 9 |" in text and "| cifar224 | t10 | 2 | 9 |" in text and "slope after/task" in text
    # an appended re-run of one seed counts once, with its last row; validate CSVs are skipped
    src = fx["fx"]["driftdiag_csv_inr"]["path"]
    tmp_root = os.path.join(fx["out"], "logs_fake_e9_rerun")
    shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(os.path.join(tmp_root, "imagenetr"))
    dst = os.path.join(tmp_root, "imagenetr", "driftdiag_t10.csv")
    with open(src, encoding="utf-8", newline="") as fh:
        lines = fh.read().splitlines()
    hdr = lines[0].split(",")
    first = dict(zip(hdr, lines[1].split(",")))          # seed 1993, task 1, group 'all'
    first["mean_rel_after"] = "0.5"
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines + [",".join(first[h] for h in hdr)]) + "\n")
    with open(os.path.join(tmp_root, "imagenetr", "driftdiag_t10_validate.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(hdr) + "\n")
    out2 = os.path.join(fx["out"], "figures_fake_e9_rerun")
    rc, text2 = _run_sub(["e9", "--roots", tmp_root, "--out-dir", out2])
    assert rc == 0 and "skip" in text2 and "validate" in text2
    inr2 = _read_csv(os.path.join(out2, "e9_driftdiag_imagenetr_t10.csv"))
    assert not os.path.exists(os.path.join(out2, "e9_driftdiag_imagenetr_t10_validate.csv"))
    r1 = next(r for r in inr2 if r["task"] == str(int(float(first["task"]))) and r["group"] == "all")
    others = [v for s, v in fx["fx"]["driftdiag_inr"]["seeds"].items() if s != 1993]
    expect = (0.5 + sum(v["driftdiag"][int(float(first["task"]))]["mean_rel_after"] for v in others)) / 3
    assert r1["n_seeds"] == "3" and abs(float(r1["mean_rel_after"]) - expect) < 1e-5, (r1, expect)


def test_e10_step_control():
    fx = fixture2()
    out = os.path.join(fx["out"], "figures_fake_e10")
    rc, text = _run_sub(["e10", "--roots", fx["root"], "--out-dir", out, "--cubes-dir", fx["fig"]])
    assert rc == 0
    cells = _read_csv(os.path.join(out, "e10_step_control.csv"))
    by = {(r["rule"], r["alpha_or_weight"]): r for r in cells}
    assert set(by) == {("trunc", "0.3"), ("clip", "0.3"), ("clip", "0.5"), ("scale_fro", "0.3"), ("l2m", "1"),
                       ("scale_p1", "0.3"), ("scale_p1", "0.5")}
    tr = by[("trunc", "0.3")]
    ms = fx["fx"]["e10_trunc"]["seeds"][1993]["mergestat"]
    assert abs(float(tr["eff_sigma_max_mean"]) - np.mean([m["eff_sigma_max_mean"] for m in ms.values()])) < 1e-4
    assert abs(float(tr["k_mean"]) - np.mean([m["k_mean"] for m in ms.values()])) < 1e-4
    assert abs(float(tr["merge_time_s"]) - ms[1]["merge_time"]) < 1e-4 and tr["norm_source"] == "MergeStat"
    assert tr["p"] == "0.9" and tr["source"] == "logs" and tr["n_seeds"] == "3"
    clip = by[("clip", "0.3")]
    assert abs(float(clip["eff_sigma_max_mean"]) - float(tr["eff_sigma_max_mean"])) < 1e-4     # matched spectral norm
    assert float(clip["eff_fro_mean"]) > float(tr["eff_fro_mean"])                               # tail kept
    assert float(by[("scale_fro", "0.3")]["eff_sigma_max_mean"]) < float(tr["eff_sigma_max_mean"])
    l2m = by[("l2m", "1")]
    sn = fx["fx"]["e10_l2m"]["seeds"][1993]["stepnorm"]
    assert l2m["norm_source"] == "StepNorm" and l2m["p"] == ""
    assert abs(float(l2m["eff_sigma_max_mean"]) - np.mean([m["eff_sigma_max_mean"] for m in sn.values()])) < 1e-4
    assert abs(float(l2m["l2m_dist_mean"]) - np.mean(list(fx["fx"]["e10_l2m"]["seeds"][1993]["l2m"].values()))) < 1e-3
    p1 = by[("scale_p1", "0.3")]
    assert p1["source"] == "logs" and abs(float(p1["fa"]) - fx["fx"]["e8_imagenetr_p1.0_a0.3"]["summary"]["faa"]) <= 0.0101
    assert by[("scale_p1", "0.5")]["source"] == "cube" and by[("scale_p1", "0.5")]["fa"] == "83.40"
    per = _read_csv(os.path.join(out, "e10_step_norms_per_task.csv"))
    assert len(per) == 5 * 3 * 9 and {r["source"] for r in per} == {"MergeStat", "StepNorm"}
    assert "| imagenetr | trunc | 0.9 | 0.3 | 3 |" in text and "| imagenetr | l2m |" in text


def test_e11_rank_sweep():
    fx = fixture2()
    out = os.path.join(fx["out"], "figures_fake_e11")
    rc, text = _run_sub(["e11", "--roots", fx["root"], "--out-dir", out, "--spectrum-dir", fx["fig"]])
    assert rc == 0
    rows = {int(r["r"]): r for r in _read_csv(os.path.join(out, "e11_rank_imagenetr.csv"))}
    assert set(rows) == {8, 16, 64}
    for r_, row in rows.items():
        assert row["trainable_params"] == str(129024 * r_) and row["params_match"] == "True"
        assert row["lora_alpha"] == str(2 * r_) and row["train_prefix"] == f"main_inr_r{r_}_qkvfc"
        assert float(row["k_mean"]) <= r_ + 1e-9 and float(row["e4_k_mean"]) <= r_ + 1e-9
    assert rows[16]["n_seeds"] == "2" and rows[8]["n_seeds"] == "3"
    assert rows[8]["e4_frac_layers_k_eq_r"] == "1" and rows[64]["e4_frac_layers_k_eq_r"] == "0"
    assert rows[8]["frac_tasks_kmax_eq_r"] == "1" and rows[64]["frac_tasks_kmax_eq_r"] == "0"
    assert abs(float(rows[8]["svd_ms_per_merge_cpu"]) - 3 * 12 * 0.9 * (1 + 8 / 64)) < 1e-3
    assert abs(float(rows[64]["svd_ms_per_merge_cuda"]) - 3 * 12 * 0.25 * 2) < 1e-3
    ms = fx["fx"]["e11_rank8"]["seeds"][1993]["mergestat"]
    assert abs(float(rows[8]["merge_time_s"]) - ms[1]["merge_time"]) < 1e-4
    assert abs(float(rows[8]["fa"]) - fx["fx"]["e11_rank8"]["summary"]["faa"]) <= 0.0101
    assert "| 8 | 16 | 3 | 1,032,192 | yes |" in text


def test_e12_merge_grid():
    fx = fixture2()
    out = os.path.join(fx["out"], "figures_fake_e12")
    rc, _ = _run_sub(["e2", "--roots", fx["root"], "--out-dir", out])          # ISM budget source
    assert rc == 0
    rc, text = _run_sub(["e12", "--roots", fx["root"], "--out-dir", out, "--cubes-dir", fx["fig"]])
    assert rc == 0
    cells = _read_csv(os.path.join(out, "e12_merge_grid.csv"))
    assert len(cells) == 25
    by = {(r["dataset"], r["rule"], r["coef"]): r for r in cells}
    c = by[("imagenetr", "avg", "0.5")]
    assert c["n_seeds"] == "3" and c["incremental_observed"] == "True" and c["flags"] == ""
    assert abs(float(c["fa"]) - fx["fx"]["e12_imagenetr_avg_c0.5"]["summary"]["faa"]) <= 0.0101
    assert "train_merge_incremental=True" in c["extras"]
    bad = by[("cub", "max_abs", "1")]
    assert "NOT the incremental" in bad["flags"] and bad["incremental_observed"] == "False" and bad["n_seeds"] == "1"
    assert "merge_dare_q=0.5" in by[("imagenetr", "dare", "1")]["extras"]
    assert "(a) Per-dataset best coefficient" in text and "ISM budget from the E2 record" in text
    assert "| ISM (ours), headline cell on the same chain |" in text and "[e12_ism]" in text
    assert "alpha cells at p=0.9" in text and "Task Arithmetic, interpolation form" in text
    assert "(b) Coefficient selected on imagenetr" in text and "| Task Arithmetic (avg) | 0.5 |" in text
    rec = _read_csv(os.path.join(out, "e2_tuning_record.csv"))
    ism_cub = [r for r in rec if r["dataset"] == "cub" and r["rule"] == "pspectral"
               and not any(k + "=" in r["extras"] for k in pl._E12_NOT_TUNING)]
    assert f"{len([r for r in ism_cub if r['pspectral_p'] == '0.9'])} alpha cells at p=0.9" in text
    assert "WARNING cub max_abs c1.0" in text


def test_e13_fewshot():
    fx = fixture2()
    out = os.path.join(fx["out"], "figures_fake_e13")
    rc, text = _run_sub(["e13", "--roots", fx["root"], "--out-dir", out, "--refs", fx["fx"]["refs"]["path"]])
    assert rc == 0
    cif = {(r["method"], r["shots"]): r for r in _read_csv(os.path.join(out, "e13_fewshot_cifar224.csv"))}
    # K rows plus one shots='full' row per method with a reference (avg has none on CIFAR)
    assert set(cif) == {("ism", "5"), ("ism", "20"), ("nomerge", "5"), ("avg", "5"), ("e2lora", "5"),
                        ("ism", "full"), ("nomerge", "full"), ("e2lora", "full")}
    full = cif[("ism", "full")]
    assert full["scenario"] == "full" and full["n_seeds"] == "3" and full["config_names"] == "mc3d_only_A_p0.9_a0.15"
    # fa is a metric column (:.2f), ref_fa a plain one (:g): compare the numbers
    assert abs(float(full["fa"]) - float(cif[("ism", "5")]["ref_fa"])) < 0.006
    assert full["fa_sd"] != "" and full["degradation_fa"] == ""
    assert full["train_prefix"] == "main_cifar224_r64_qkvfc" and full["ref_source"] == "logs:mc3d_only_A_p0.9_a0.15"
    e2full = cif[("e2lora", "full")]
    assert e2full["fa"] == "90.86" and e2full["n_seeds"] == "3" and e2full["config_names"] == "RESULTS.md s2 re-run"
    assert cif[("nomerge", "full")]["config_names"] == "nomerge_full" and cif[("nomerge", "full")]["n_seeds"] == "3"
    ism5 = cif[("ism", "5")]
    assert ism5["n_seeds"] == "3" and ism5["scenario"] == "fewshot" and ism5["train_prefix"] == "main_cifar224_r64_qkvfc_fs5"
    assert abs(float(ism5["fa"]) - fx["fx"]["e13_cifar224_fs5_ism"]["summary"]["faa"]) <= 0.0101
    assert ism5["ref_source"] == "logs:mc3d_only_A_p0.9_a0.15" and float(ism5["degradation_fa"]) > 0
    assert ism5["train_median"] == "5" and ism5["train_min"] == "5" and ism5["til_fa"] != ""
    assert abs(float(ism5["til_fa"]) - np.mean(fx["fx"]["e13_cifar224_fs5_ism"]["seeds"][1993]["til_rows"][-1])) < 0.6
    assert cif[("nomerge", "5")]["ref_source"] == "logs:nomerge_full"
    assert cif[("e2lora", "5")]["ref_source"] == "RESULTS.md s2 re-run" and cif[("e2lora", "5")]["ref_fa"] == "90.86"
    assert cif[("avg", "5")]["ref_fa"] == ""
    inr = {(r["method"], r["shots"]): r for r in _read_csv(os.path.join(out, "e13_fewshot_imagenetr.csv"))}
    assert inr[("ism", "20")]["train_min"] == "12" and inr[("ism", "20")]["train_median"] == "20"
    assert inr[("avg", "5")]["ref_source"] == "logs:e12_avg_c1.0"
    assert inr[("avg", "full")]["config_names"] == "e12_avg_c1.0"
    assert abs(float(inr[("avg", "full")]["fa"]) - float(inr[("avg", "5")]["ref_fa"])) < 0.006
    assert inr[("e2lora", "full")]["fa"] == "80.40" and inr[("e2lora", "full")]["n_seeds"] == ""   # no n_seeds typed
    per = _read_csv(os.path.join(out, "e13_fewshot_cifar224_per_seed.csv"))
    assert len(per) == 15 and {r["status"] for r in per} == {"ok"}
    assert "Few-shot per task" in text and "Realised training images per class" in text and "imagenetr K=20: 12/20/20" in text
    assert "Task-incremental FA" in text and "| ism |" in text and "K=full" in text and "K=full:" not in text


def test_e14_valsplit():
    fx = fixture2()
    out = os.path.join(fx["out"], "figures_fake_e14")
    rc, text = _run_sub(["e14", "--roots", fx["root"], "--out-dir", out])
    assert rc == 0
    cells = _read_csv(os.path.join(out, "e14_valsplit_cub.csv"))
    by = {(r["alpha"], r["lambda"]): r for r in cells}
    assert set(by) == {("0.1", "10"), ("0.1", "100"), ("0.3", "10"), ("0.3", "100")}
    assert by[("0.1", "10")]["is_headline"] == "True" and by[("0.3", "10")]["is_headline"] == "False"
    crashed = by[("0.3", "100")]
    assert crashed["n_seeds"] == "2" and crashed["n_failed"] == "1"
    info = fx["fx"]["e14_cub_a0.1_l10.0"]
    assert abs(float(by[("0.1", "10")]["faa_val"]) - np.mean([info["seeds"][s]["faa_val"] for s in pl.SEEDS])) <= 0.0101
    assert abs(float(by[("0.1", "10")]["fa"]) - info["summary"]["faa"]) <= 0.0101
    sel = _read_csv(os.path.join(out, "e14_valsplit_selected.csv"))[0]
    assert sel["dataset"] == "cub" and sel["n_cells"] == "4" and sel["n_seeds"] == "3" and sel["val_frac"] == "0.1"
    assert sel["sel_alpha_per_seed"].count(";") == 2 and sel["headline_alpha"] == "0.1" and sel["headline_lambda"] == "10"
    assert sel["headline_full_data_source"] == "logs:best_cub"
    assert abs(float(sel["headline_full_data_fa"]) - fx["fx"]["best_cub"]["summary"]["faa"]) <= 0.0101
    assert abs(float(sel["test_fa_headline_on_val_chain"]) - info["summary"]["faa"]) <= 0.0101
    assert sel["oracle_alpha"] and sel["pooled_alpha"] and sel["spearman_val_test"] != ""
    assert float(sel["test_fa_oracle"]) >= float(sel["test_fa_pooled"]) - 1e-9
    assert sel["per_seed_hits_oracle"].endswith("/3")
    # runs of the crashed seed never enter the per-seed selection
    runs = pl.parse_file(fx["fx"]["e14_cub_a0.3_l100.0"]["path"])
    assert sum(r["status"] == "crashed" for r in runs) == 1 and runs[0]["summary"]["val"] is not None
    assert "Validation-split selection" in text and "| cub | 4 | 3 |" in text and "Spearman" in text


def test_e15_opcm_nufilt_inflora():
    fx = fixture2()
    out = os.path.join(fx["out"], "figures_fake_e15")
    rc, text = _run_sub(["e15", "--roots", fx["root"], "--out-dir", out])
    assert rc == 0
    cells = _read_csv(os.path.join(out, "e15_opcm_nufilt.csv"))
    by = {(r["rule"], r["setting"], r["dataset"]): r for r in cells}
    assert set(by) == {("opcm", "a0.5_adaptive", "imagenetr"), ("opcm", "a0.7_sqrt", "imagenetr"),
                       ("nufilt", "rp128_ad0", "imagenetr"), ("nufilt", "rp32_ad1", "imagenetr"),
                       ("opcm", "a0.5_adaptive", "cifar224")}
    o = by[("opcm", "a0.5_adaptive", "imagenetr")]
    st = fx["fx"]["e15_opcm_0.5_adaptive"]["seeds"][1993]["merge_state"]
    assert abs(float(o["lam_t_last"]) - st[9]["lam_t"]) < 1e-3 and o["n_seeds"] == "3" and o["flags"] == ""
    assert abs(float(o["kept_frac_mean"]) - np.mean([v["kept_frac"] for v in st.values()])) < 1e-3
    n = by[("nufilt", "rp32_ad1", "imagenetr")]
    assert n["n_adapted_mean"] == "36" and n["adapt_loss_last_mean"] == "0.401" and n["lam_t_last"] == ""
    assert float(by[("nufilt", "rp128_ad0", "imagenetr")]["filtered_frac_mean"]) > 0
    assert "Stage-1 grid on imagenetr" in text and "OPCM (a0.5_adaptive) [best stage-1]" in text
    assert "NUFILT (rp128_ad0) [best stage-1]" in text and "[e12_ism]" in text and "[mc3d_only_A_p0.9_a0.15]" in text
    infl = {r["dataset"]: r for r in _read_csv(os.path.join(out, "e15_inflora.csv"))}
    assert set(infl) == {"cifar224", "imagenetr"}
    c = infl["cifar224"]
    ci = fx["fx"]["inflora_cifar"]
    assert c["n_seeds"] == "2" and c["n_failed"] == "1" and "failed seeds [1994]" in c["flags"]
    assert abs(float(c["fa"]) - ci["summary"]["faa"]) <= 0.0101 and abs(float(c["sweep_fa"]) - ci["summary"]["faa"]) <= 0.0101
    assert abs(float(c["micro_acc"]) - ci["summary"]["micro_acc"]) <= 0.0101 and abs(float(c["avg"]) - ci["summary"]["avg"]) <= 0.0101
    assert c["resolved_backbone"] == "vit_base_patch16_224.augreg_in21k" and c["published_acc"] == "86.51"
    assert c["basis_cols_monotone"] == "True" and int(c["basis_cols_last_max"]) == max(ci["seeds"][1993]["gpm"][-1][2])
    assert float(c["leak_max"]) < 1e-5 and c["lora_r"] == "10" and c["train_epochs"] == "20"
    i = infl["imagenetr"]
    assert i["n_seeds"] == "3" and i["inflora_lamb"] == "0.98" and i["flags"] == "" and i["published_avg"] == "80.82"
    r = pl.parse_file(ci["path"])
    assert sum(x["status"] == "crashed" for x in r) == 1 and all(x["config"]["train_method"] == "inflora" for x in r)
    okr = next(x for x in r if x["status"] == "ok")
    assert okr["sweep"]["micro ACC"][0] == float(f"{ci['summary']['micro_acc']:.2f}") and "FF" in okr["sweep"]
    pil = pl.parse_inflora_lines(okr)
    assert len(pil["gpm"]) == 10 and pil["gpm"][0]["types"] == ["remove"] * 12 and pil["seed_line"]["seed"] == okr["seed"]
    assert "InfLoRA port" in text and "| cifar224 | inflora_in21k_cifar224 |" in text


# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="directory for the fixture tree and CSV outputs")
    p.add_argument("-k", default=None, help="run only tests whose name contains this text")
    args = p.parse_args(argv)
    if args.out:
        os.environ["PARSE_LOGS_TEST_DIR"] = os.path.abspath(args.out)
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    if args.k:
        tests = [(n, f) for n, f in tests if args.k in n]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed; fixtures under {fixture()['out']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
