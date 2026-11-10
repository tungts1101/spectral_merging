"""Synthetic log fixtures in the exact format of main.run_config_sweep / run_sota_merge.py /
run_inflora.py.

Every parser in the repo (parse_logs.py and the e* subcommands built on it) is tested on
this Windows PC without torch, so the fixtures must look like the real logs down to the
characters that break naive parsers: the '%(asctime)s [main.py] => ' prefix
(main.py:1988-1996), the em dash in 'Experiment: ds -- cfg -- seed N' and in the
SUMMARY header (main.py:2000, 2011), the plus-minus sign of the summary line
(main.py:2014), the Greek lambda and the middle dot of the '[Drift]' line
(main.py:1216-1220), unquoted strings in the Configuration dump (main.py:1915), the
empty-message ruler lines produced by logging.info('\\n' + '=' * 80), the traceback of a
failed seed (main.py:1966-1967; its FA enters the SUMMARY as 0.0), appended duplicate
blocks (FileHandler append mode) and run_sota_merge logs that carry no 'Experiment:'
header at all (run_sota_merge.py:51-60).

Options cover 10/20/50-task splits (DataManager rule, utils/data_manager.py:15-20),
validation-split lines ([Evaluation-val], the '[val]' summary line), crashed seeds,
appended re-runs, cached vs uncached training (epoch lines), classical / SOTA merge
lines, DIL lines, and the gated rebuttal lines in the exact wording of the B1-B4 main.py
edits (``features`` below): [MergeStat] [StepNorm] [FeatShift] [Timing] [DriftDiag] [Data]
[L2M] [SpecNorm] [Evaluation-til] [Alignment] cov_mode, the '[Merging] adaptive alpha' and
'[Drift] lambda_rule=gcv' lines, the '[Merging] opcm|nufilt' state lines, the
'[Training] small task' line, and run_inflora.py logs with the [InfLoRA] lines, the
', lr:' epoch field, '[FAILED]' seeds and the 'SWEEP SUMMARY [inflora-ds]' line.

The accuracy matrix is synthetic but internally consistent: the FAA/FFM/FFD/ASA line is
computed here with a plain-loop transcription of helper.compute_metrics (helper.py:22-43)
that does NOT import parse_logs, so tests compare two independent implementations.

Line order follows main.py: the [Data] line precedes [Freeze]; [L2M] / [SpecNorm] sit
inside the training block (train_backbone); the adaptive-alpha / [FeatShift] lines precede
'[Merging] Task t'; every [Timing] line follows its stage; task 0 has no [Merging],
[Drift] or [Alignment] lines (compute_multivariate_normal logs nothing in the default
mode); the '[Timing] eval' line follows every [Evaluation*] line of the task.

CLI
    python tests/make_fake_log.py --out logs_fake --dataset cifar224 --config long_t20 \
        --init 5 --inc 5 --seeds 1993 1994 1995 [--epochs 5] [--val] [--crash-seed 1994]
        [--repeat-seed 1993] [--uncached] [--tags] [--alpha 0.15] [--lam 100]
    python tests/make_fake_log.py --out logs_fake --sota            (Configuration-only log)
    python tests/make_fake_log.py --out logs_fake --tree            (the whole test fixture tree)
    python tests/make_fake_log.py --out logs_fake --rebuttal-tree --fig-dir figures_fake
"""
import argparse
import csv
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np

# main.py:1729-1743 (default splits and class counts)
DATA_TABLE = {"cifar224": (10, 10, 10), "imagenetr": (10, 20, 20), "cub": (10, 20, 20),
              "cars": (10, 16, 20), "officehome_dil": (4, 65, 65)}
TOTAL_CLASSES = {"cifar224": 100, "imagenetr": 200, "cub": 200, "cars": 196, "officehome_dil": 260}
TRAIN_PER_CLASS = {"cifar224": 500, "imagenetr": 120, "cub": 30, "cars": 41, "officehome_dil": 60}
SEEDS = (1993, 1994, 1995)
TEST_PER_CLASS = {"cifar224": 100, "imagenetr": 30, "cub": 29, "cars": 41, "officehome_dil": 60}

PREFIX = "[main.py] => "
EM = "—"       # em dash
PM = "±"       # plus-minus
LAM = "λ"      # lambda
DOT = "·"      # middle dot

TRAINABLE_PER_RANK = 129_024          # [Freeze] PEFT 'lora_' params per unit rank (12 x 10,752)
FROZEN_PARAMS = 85_798_656
GCV_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)       # driftsolve.GCV_GRID
GCV_SCHEDULE = (100.0, 30.0, 100.0, 300.0, 100.0, 100.0, 30.0, 100.0, 100.0, 300.0)
D_FEAT = 768

# The gated lines switched on by ``tags=True`` (all default-off keys of the rebuttal edits
# that can appear in one cache-served pspectral run).
DEFAULT_TAG_FEATURES = {"timing": True, "mergestat": True, "stepnorm": True, "featshift": True,
                        "driftdiag": True, "data": True, "l2m": True, "specnorm": True, "til": True}
# main.DRIFTDIAG_CSV_HEADER
DRIFTDIAG_CSV_HEADER = ("dataset", "seed", "task", "group", "n_classes", "n_img", "mean_rel_raw", "mean_rel_before",
                        "mean_rel_after", "mean_cos_raw", "mean_cos_before", "mean_cos_after", "cov_relF_raw",
                        "cov_relF_before", "cov_relF_after", "symkl_raw", "symkl_before", "symkl_after",
                        "os_mean_rel_I", "os_mean_rel_W", "os_mean_rel_oracle", "os_cov_relF_I", "os_cov_relF_W",
                        "os_cov_relF_oracle", "W_minus_I_F", "Wor_minus_I_F", "W_minus_Wor_F", "n_fit_new", "n_fit_old")
TIMING_CSV_HEADER = ("dataset", "seed", "task", "stage", "seconds", "extra")          # main.TIMING_CSV_HEADER


def task_increments(total, init_cls, increment):
    """DataManager.__init__ rule (utils/data_manager.py:15-20), copied for the fixtures."""
    incs = [init_cls]
    while sum(incs) + increment < total:
        incs.append(increment)
    offset = total - sum(incs)
    if offset > 0:
        incs.append(offset)
    return incs


def compute_metrics_ref(rows):
    """Plain-loop transcription of helper.compute_metrics (helper.py:22-43) on the
    Grouped rows (row i has i+1 entries).  Independent of parse_logs on purpose."""
    T = len(rows)
    faa = sum(rows[-1]) / len(rows[-1])
    session = [sum(rows[i][: i + 1]) / (i + 1) for i in range(T)]
    asa = sum(session) / T
    if T == 1:
        return faa, 0.0, 0.0, asa
    forget = []
    for j in range(T - 1):
        best = max(rows[i][j] for i in range(j, T))
        forget.append(best - rows[-1][j])
    ffm = sum(forget) / len(forget)
    ffd = max(forget) - min(forget)
    return faa, ffm, ffd, asa


class Clock:
    """Monotone fake asctime generator ('YYYY-MM-DD HH:MM:SS,mmm')."""

    def __init__(self, start="2026-11-08 10:00:00"):
        self.t = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")

    def tick(self, seconds=0.5):
        self.t += timedelta(seconds=seconds)
        return self.t.strftime("%Y-%m-%d %H:%M:%S,") + f"{self.t.microsecond // 1000:03d}"


def base_config(ds, seed, init, inc, n_tasks, **over):
    """Configuration dump in the key order of main.py:1874-1898 (BASE_CONFIG + split keys)."""
    cfg = OrderedDict([
        ("seed", seed), ("train_method", "seqft"), ("reset_train", False), ("reset_merge", True),
        ("train_stop_at_task", -1), ("model_backbone", "vit_base_patch16_224"), ("model_lora_r", 64),
        ("model_lora_alpha", 128), ("model_lora_target_modules", ["qkv", "fc1", "fc2"]),
        ("model_use_norm", True), ("train_epochs", 10), ("train_batch_size", 48), ("train_lr", 0.01),
        ("train_weight_decay", 0.0005), ("train_optimizer", "sgd"), ("train_merge", "pspectral"),
        ("spectral_variant", "only_A"), ("pspectral_p", 0.9), ("train_merge_alpha", 0.3),
        ("train_merge_k", 16), ("train_merge_coef", 1.0), ("train_merge_topk", 100),
        ("train_merge_incremental", False), ("cleanup_merged", True), ("train_ca", True),
        ("train_ca_epochs", 10), ("train_ca_cov_mode", "class"), ("train_ca_cov_shrinkage", 0.0),
        ("train_ca_robust_weight", 0.1), ("train_ca_n_classes", 40), ("train_ca_k_samples", 64),
        ("train_drift", True), ("train_drift_ridge", "identity"), ("train_drift_lambda", 100.0),
        ("train_drift_transport", "full"), ("train_prefix", f"main_{ds}_r64_qkvfc"),
        ("cache_backbone", True), ("final_model_dir", None), ("specreg_weight", 0.0),
        ("speccap", None), ("speccap_mode", "rescale"), ("speccap_anchor", "base"),
        ("merge_dare_q", 0.9), ("merge_della_q", 0.5), ("merge_bc_beta", 0.85),
        ("merge_bc_gamma", 0.01), ("merge_knots_topk", 50.0), ("dataset_name", ds),
        ("dataset_num_task", n_tasks), ("dataset_init_cls", init), ("dataset_increment", inc),
    ])
    cfg.update(over)
    return cfg


def inflora_config(ds, seed, init, inc, n_tasks, **over):
    """Configuration dump of a run_inflora.py cell (RECIPE_PRESETS['inflora'] + run_inflora.RECIPE)."""
    cfg = OrderedDict([
        ("seed", seed), ("train_method", "inflora"), ("reset_train", False), ("reset_merge", True),
        ("train_stop_at_task", -1), ("model_backbone", "vit_base_patch16_224"), ("train_optimizer", "adam"),
        ("train_base_lr", 0.0005), ("train_head_lr", None), ("train_weight_decay", 0.0), ("train_epochs", 20),
        ("train_batch_size", 128), ("train_scheduler", "inflora_cosine"), ("model_use_norm", False),
        ("model_classifier_use_norm", False), ("model_classifier_bias", True), ("train_merge", "none"),
        ("train_ca", False), ("train_drift", False), ("lora_r", 10), ("inflora_lamb", 0.95), ("inflora_lame", 1.0),
        ("inflora_backbone", "vit_base_patch16_224_in21k"), ("inflora_gram_mode", "train"), ("inflora_dual", True),
        ("inflora_layers", None), ("cache_backbone", True), ("checkpoint_dir", "checkpoints_main/inflora"),
        ("train_prefix", f"inflora_in21k_{ds}_r10_l0.95_e20_lr0.0005_gtrain_d1_Lall_b128"),
        ("dataset_name", ds), ("dataset_num_task", n_tasks), ("dataset_init_cls", init), ("dataset_increment", inc),
    ])
    cfg.update(over)
    return cfg


def synth_rows(n_tasks, rng, base=92.0, drop=0.35, noise=0.4):
    """Accuracy matrix rows: a[i][j] for j <= i, 2-dp floats like helper.accuracy."""
    level = [base + rng.uniform(-3.0, 3.0) for _ in range(n_tasks)]
    rows = []
    for i in range(n_tasks):
        row = [float(np.around(level[j] - drop * (i - j) + rng.uniform(-noise, noise), 2))
               for j in range(i + 1)]
        rows.append(row)
    return rows


def totals_from_rows(rows, incs, per_class):
    """Sample-weighted 'Total Acc' per session (2 dp) with per_class test images/class."""
    out = []
    for i, row in enumerate(rows):
        sizes = [incs[j] * per_class for j in range(i + 1)]
        out.append(float(np.around(sum(a * s for a, s in zip(row, sizes)) / sum(sizes), 2)))
    return out


def _fmt_list(vals):
    return "[" + ", ".join(str(float(v)) for v in vals) + "]"


def _ckpt_name(seed, ds, prefix, kind, t, tag="pspec_only_A_p0.9_a0.3"):
    stem = f"{seed}_{ds}_vit_base_patch16_224_lora_{prefix}"
    if kind == "backbone":
        return f"{stem}_backbone_{t}.pt"
    return f"{stem}_merged_{tag}_{t}.pt"


def merge_tag(cfg):
    """rebuttal_common.merge_tag / main.merge_tag rule for the merged file names."""
    rule = cfg.get("train_merge", "pspectral")
    variant = cfg.get("spectral_variant", "only_A")
    if rule == "pspectral":
        arule = cfg.get("merge_alpha_rule", "fixed")
        a = f"a{cfg.get('train_merge_alpha', 0.3)}" if arule == "fixed" else f"a{arule}{cfg.get('merge_alpha_param', 1.0)}"
        op = cfg.get("merge_update_op", "trunc")
        op_part = "" if op == "trunc" else f"{op}_"       # op token before alpha (main.merge_tag)
        return f"pspec_{variant}_p{cfg.get('pspectral_p', 0.9)}_{op_part}{a}"
    if rule == "kspectral":
        return f"kspec_{variant}_k{cfg.get('train_merge_k', 16)}_a{cfg.get('train_merge_alpha', 0.3)}"
    if rule == "opcm":
        return f"opcm_a{cfg.get('merge_opcm_alpha', 0.5)}_{cfg.get('merge_opcm_lam', 'adaptive')}"
    if rule == "nufilt":
        return (f"nufilt_rp{cfg.get('merge_nufilt_rp', 128)}_rv{cfg.get('merge_nufilt_rv', 8)}"
                f"_rl{cfg.get('merge_nufilt_rl', 64)}_ad{int(bool(cfg.get('merge_nufilt_adapt', False)))}")
    return f"{rule}_c{cfg.get('train_merge_coef', 1.0)}"


def data_line(ds, cfg, incs):
    """DataManager.describe() (utils/data_manager.py) for the realised splits."""
    C = TOTAL_CLASSES.get(ds, 100)
    per = TRAIN_PER_CLASS.get(ds, 500)
    k = cfg.get("data_fewshot_k")
    vf = cfg.get("data_val_frac") or 0.0
    seed = cfg.get("seed")
    n_val = int(round(vf * per)) if vf else 0
    n_tr = per - n_val
    if k:
        n_tr = min(int(k), n_tr)
    train = f"train: n={n_tr * C} C={C} per-class min/median/max={n_tr}/{n_tr:.10g}/{n_tr}"
    if ds == "imagenetr" and k and int(k) > 15:
        # some ImageNet-R classes have fewer images than K: the realised min is lower
        train = f"train: n={n_tr * C - 40} C={C} per-class min/median/max={int(k) - 8}/{float(k):.10g}/{int(k)}"
    val = (f"val: n={n_val * C} C={C} per-class min/median/max={n_val}/{n_val:.10g}/{n_val} "
           f"(val_frac={vf:g}, val_seed={cfg.get('data_val_seed', seed)})" if vf else "val: none")
    tpc = TEST_PER_CLASS.get(ds, 100)
    test = f"test: n={tpc * C} C={C} per-class min/median/max={tpc}/{tpc:.10g}/{tpc}"
    few = f"fewshot: k={int(k)} (subsample_seed={cfg.get('data_subsample_seed', seed)})" if k else "fewshot: none"
    return f"[Data] dataset={ds} | {train} | {val} | {test} | {few}", {"train_per_class": n_tr, "val_per_class": n_val}


def _norm_fields(a_mean, a_max, e_mean, e_max, e_fro):
    return (f"A_sigma_max_mean={a_mean:.5g} A_sigma_max_max={a_max:.5g} | "
            f"eff_sigma_max_mean={e_mean:.5g} eff_sigma_max_max={e_max:.5g} eff_fro_mean={e_fro:.5g}")


def seed_block(clock, ds, cfg_name, seed, cfg, rows, totals, rows_val=None, totals_val=None,
               crash_at_task=None, cached=True, tags=False, header=True, src="main.py",
               dil_classes=None, durations=None, features=None):
    """All lines of one seed (list of str, no newlines).  Returns (lines, info).

    ``features`` switches the gated lines on (keys: timing, mergestat, stepnorm, featshift,
    adaptive, gcv, driftdiag, data, til, l2m, specnorm, sota_state, cov_mode, small_task);
    ``tags=True`` is the DEFAULT_TAG_FEATURES bundle.  Their values come from the config
    (merge_update_op, merge_alpha_rule/stat/param, l2m_weight, train_ca_cov_mode, ...) so the
    fixture is consistent with its own Configuration dump.
    """
    d = {"epoch": 3.0, "merge": 2.0, "drift": 4.0, "transport": 1.0, "stats": 0.8, "align": 5.0, "eval": 2.5,
         "driftdiag": 1.5}
    d.update(durations or {})
    feat = dict(DEFAULT_TAG_FEATURES) if tags else {}
    feat.update(features or {})
    L = []
    pre = f"[{src}] => "

    def emit(msg, dt=0.2):
        L.append(f"{clock.tick(dt)} {pre}{msg}")

    t_start = clock.t
    n_tasks = len(rows)
    incs = task_increments(TOTAL_CLASSES.get(ds, 100), cfg["dataset_init_cls"], cfg["dataset_increment"])
    rule = cfg.get("train_merge", "pspectral")
    method = cfg.get("train_method", "seqft")
    prefix = cfg.get("train_prefix", f"main_{ds}_r64_qkvfc")
    variant, p, alpha = cfg.get("spectral_variant", "only_A"), cfg.get("pspectral_p", 0.9), cfg.get("train_merge_alpha", 0.3)
    op = cfg.get("merge_update_op", "trunc")
    r = int(cfg.get("model_lora_r", 64))
    tag = merge_tag(cfg)
    cov_mode = cfg.get("train_ca_cov_mode", "class")
    cov_rank = int(cfg.get("train_ca_cov_rank", 64))
    solver = cfg.get("train_drift_solver", "dense")
    lam_fixed = cfg.get("train_drift_lambda", 100.0)
    arule = cfg.get("merge_alpha_rule", "fixed")
    astat = cfg.get("merge_alpha_stat", "eff")
    aparam = float(cfg.get("merge_alpha_param", 1.0))
    amin = float(cfg.get("merge_alpha_min", 0.05))
    l2m_w = float(cfg.get("l2m_weight", 0.0) or 0.0)
    epochs = int(cfg.get("train_epochs", 10))
    info = {"status": "ok", "seed": seed, "mergestat": OrderedDict(), "stepnorm": OrderedDict(), "alphas": [],
            "adaptive": [], "featshift": OrderedDict(), "gcv": OrderedDict(), "timing": OrderedDict(),
            "driftdiag": OrderedDict(), "l2m": OrderedDict(), "til_rows": [], "merge_state": OrderedDict(),
            "data": None, "n_timing_lines": 0}

    def timing(t, stage, sec, dt=0.0, **extra):
        if not feat.get("timing"):
            return
        ex = " ".join(f"{k}={v}" for k, v in extra.items())
        emit(f"[Timing] task {t} {stage}={sec:.3f}s" + (f" {ex}" if ex else ""), dt)
        info["timing"].setdefault(t, OrderedDict())[stage] = round(sec, 3)
        info["n_timing_lines"] += 1

    if header:
        emit("")                                   # logging.info('\n' + '=' * 80)
        L.append("=" * 80)
        emit(f"Experiment: {ds} {EM} {cfg_name} {EM} seed {seed}")
    emit("Configuration:")
    for k, v in cfg.items():
        emit(f"  {k}: {v}", 0.0)
    if feat.get("data"):
        line, counts = data_line(ds, cfg, incs)
        emit(line)
        info["data"] = counts
    emit(f"[Freeze] Non-PEFT frozen ({FROZEN_PARAMS:,} params), PEFT 'lora_' trainable ({TRAINABLE_PER_RANK * r:,} params)")
    status = "ok"
    for t in range(n_tasks):
        if crash_at_task is not None and t == crash_at_task:
            emit(f"[Experiment {ds}_{cfg_name}] RuntimeError: boom at task {t}")
            emit("Traceback (most recent call last):")
            L.append('  File "main.py", line 1930, in run_single_experiment')
            L.append("    learner.learn(data_manager)")
            L.append(f"RuntimeError: boom at task {t}")
            L.append("")
            status = "crashed"
            break
        total_cls = sum(incs[: t + 1])
        # ---- training (train_backbone) -----------------------------------------------
        emit(f"[Training] Task {t} ({method})")
        ep_here = epochs
        if feat.get("small_task") and cfg.get("train_epochs_small_task") is not None and \
                incs[t] <= int(cfg.get("train_small_task_max_classes", 4)):
            ep_here = int(cfg["train_epochs_small_task"])
            emit(f"[Training] small task ({incs[t]} classes): epochs={ep_here}")
        if cached:
            emit(f"[Training] Reusing cached trained checkpoint for task {t}", 0.3)
            train_s = 0.3
        else:
            emit(f"[Optimizer] {cfg.get('train_optimizer', 'sgd')} (lr={cfg.get('train_lr', 0.01)}, "
                 f"head_lr={cfg.get('train_lr', 0.01)}, weight_decay={cfg.get('train_weight_decay', 0.0005)})")
            if feat.get("l2m") and l2m_w > 0 and t > 0:
                anchor = (_ckpt_name(seed, ds, prefix, "backbone", t - 1) if (t == 1 or rule == "none")
                          else _ckpt_name(seed, ds, prefix, "merged", t - 1, tag))
                emit(f"[L2M] anchor={anchor} weight={l2m_w} tensors=72")
            for e in range(ep_here):
                emit(f"[Training] Epoch {e + 1}/{ep_here}, Loss: {0.9 / (e + 1):.4f}, Acc: {0.80 + 0.019 * e:.4f}", d["epoch"])
            if feat.get("l2m") and l2m_w > 0 and t > 0:
                dist = 2.5 / (1.0 + l2m_w) + 0.01 * t
                emit(f"[L2M] end-of-task ||W_t-anchor||_F={dist:.4f}")
                info["l2m"][t] = round(dist, 4)
            if feat.get("specnorm"):
                emit("[SpecNorm] end-of-task sigma_max(total update): mean 0.512, max 0.903")
            train_s = ep_here * d["epoch"] + 0.2
        timing(t, "train", train_s, cached=int(cached))
        if t == 0:
            emit("[Task 0] Base and task-0 checkpoints updated with LoRA-only state", 0.1)
        # ---- merge --------------------------------------------------------------------
        raw_A = 1.52 + 0.01 * t                      # sigma_max of the raw lora_A update
        raw_eff = 1.80 + 0.02 * t                    # sigma_max of the raw effective update
        if t > 0 and rule != "none":
            alpha_t = alpha
            if feat.get("adaptive") and arule != "fixed":
                if arule == "specnorm":
                    s_t = raw_eff if astat == "eff" else raw_A
                    stat = astat
                else:
                    s_t = 0.0712 + 0.002 * t
                    stat = "shift"
                    emit(f"[FeatShift] task {t} shift={s_t:.6g} n={incs[t] * TRAIN_PER_CLASS.get(ds, 500)}", 1.0)
                    info["featshift"][t] = s_t
                alpha_t = float(min(1.0, max(amin, aparam / s_t)))
                emit(f"[Merging] adaptive alpha rule={arule} stat={stat} s_t={s_t:.6g} -> alpha={alpha_t:.4f}")
                info["adaptive"].append((t, s_t, alpha_t))
            elif feat.get("featshift"):
                s_t = 0.0712 + 0.002 * t
                emit(f"[FeatShift] task {t} shift={s_t:.6g} n={incs[t] * TRAIN_PER_CLASS.get(ds, 500)}", 1.0)
                info["featshift"][t] = s_t
            info["alphas"].append(alpha_t)
            emit(f"[Merging] Task {t}", 0.1)
            prev = (_ckpt_name(seed, ds, prefix, "backbone", 0) if t == 1
                    else _ckpt_name(seed, ds, prefix, "merged", t - 1, tag))
            k_mean = min(float(r), 12.3 + 0.5 * t)
            k_min, k_max = min(r, 4), min(r, 40)
            fac = {"trunc": 1.0, "clip": 1.0, "scale_fro": 0.8}.get(op, 1.0)
            ms = OrderedDict(k_mean=k_mean, k_min=k_min, k_max=k_max, A_sigma_max_mean=alpha_t * raw_A * fac,
                             A_sigma_max_max=alpha_t * raw_A * fac * 1.3, eff_sigma_max_mean=alpha_t * raw_eff * fac,
                             eff_sigma_max_max=alpha_t * raw_eff * fac * 1.25, eff_fro_mean=alpha_t * 2.1 * (1.15 if op == "clip" else 1.0),
                             raw_sigma_max_mean=raw_A, raw_eff_sigma_max_mean=raw_eff, merge_time=0.81 + 0.001 * r)
            nf = _norm_fields(ms["A_sigma_max_mean"], ms["A_sigma_max_max"], ms["eff_sigma_max_mean"],
                              ms["eff_sigma_max_max"], ms["eff_fro_mean"])
            tail = (f"raw_sigma_max_mean={raw_A:.5g} raw_eff_sigma_max_mean={raw_eff:.5g} | "
                    f"merge_time={ms['merge_time']:.3f}s")
            if rule == "pspectral":
                emit(f"[Merging] pspectral {variant} (p={p}, alpha={alpha_t}) of {prev} + task {t}", 0.4)
                if feat.get("mergestat"):
                    emit(f"[MergeStat] task {t} alpha={alpha_t} op={op} k_mean={k_mean:.2f} k_min={k_min} k_max={k_max} | "
                         f"{nf} | {tail}", d["merge"])
            elif rule == "kspectral":
                emit(f"[Merging] kspectral {variant} (k={cfg.get('train_merge_k', 16)}, alpha={alpha}) "
                     f"of {prev} + task {t}", 0.4)
                if feat.get("mergestat"):
                    emit(f"[MergeStat] task {t} alpha={alpha} k_mean={k_mean:.2f} k_min={k_min} k_max={k_max} | "
                         f"{nf} | {tail}", d["merge"])
            elif rule in ("dare", "della", "breadcrumbs", "knots", "modelstock", "opcm", "nufilt"):
                emit(f"[Merging] SOTA {rule} (lamb={cfg.get('train_merge_coef', 1.0)}) of {prev} + task {t}", 0.4)
                if feat.get("mergestat"):
                    emit(f"[MergeStat] task {t} | {nf} | {tail}", d["merge"])
                if feat.get("sota_state") and rule == "opcm":
                    st = OrderedDict(lam_t=math.sqrt(t) * 0.97, lam_prev=(1 if t == 1 else math.sqrt(t - 1) * 0.97),
                                     kept_frac=(1.0 if t == 1 else 0.87 - 0.01 * t), r_alpha_mean=(0 if t == 1 else 12.5),
                                     norm_mean=0.512 + 0.01 * t)
                    emit(f"[Merging] opcm alpha={cfg.get('merge_opcm_alpha', 0.5)} lam_t={st['lam_t']:.4f} "
                         f"lam_rule={cfg.get('merge_opcm_lam', 'adaptive')} lam_prev={st['lam_prev']:.4g} "
                         f"kept_frac={st['kept_frac']:.4g} r_alpha_mean={st['r_alpha_mean']} norm_mean={st['norm_mean']:.4g}")
                    info["merge_state"][t] = st
                elif feat.get("sota_state") and rule == "nufilt":
                    adapt = int(bool(cfg.get("merge_nufilt_adapt", False)))
                    st = OrderedDict(filtered_frac=(0.0 if t == 1 else 0.21 + 0.005 * t), rp_eff_mean=min(64, int(cfg.get("merge_nufilt_rp", 128))),
                                     n_adapted=(36 if adapt else 0))
                    line = (f"[Merging] nufilt rp={cfg.get('merge_nufilt_rp', 128)} adapt={adapt} "
                            f"filtered_frac={st['filtered_frac']:.4g} rp_eff_mean={st['rp_eff_mean']} n_adapted={st['n_adapted']}")
                    if adapt:
                        st["adapt_loss_first"], st["adapt_loss_last"] = 0.912, 0.401
                        line += f" adapt_loss_first={st['adapt_loss_first']:.4g} adapt_loss_last={st['adapt_loss_last']:.4g}"
                    emit(line)
                    info["merge_state"][t] = st
            else:                                        # helper.merge: avg / ties / max_abs ...
                emit("[Merging] 8,257,536 total parameters", 0.4)
                n_ck = 2 if cfg.get("train_merge_incremental", False) else t + 1
                emit(f"[Merging] Merging {n_ck} task checkpoints", 0.0)
                if feat.get("mergestat"):
                    emit(f"[MergeStat] task {t} | {nf} | {tail}", d["merge"])
            if feat.get("mergestat"):
                info["mergestat"][t] = ms
            emit(f"[Merging] Saving merged checkpoint for task {t}", 0.0 if feat.get("mergestat") else d["merge"])
            timing(t, "merge", 0.4 + d["merge"] + 0.3, dt=0.3, rule=rule)
        if t > 0 and rule == "none" and feat.get("stepnorm"):
            e_mean = 0.611 / (1.0 + l2m_w) + 0.01 * t
            sn = OrderedDict(k_mean=min(float(r), 14.0 + 0.4 * t), k_min=min(r, 5), k_max=min(r, 48),
                             A_sigma_max_mean=raw_A, A_sigma_max_max=raw_A * 1.3, eff_sigma_max_mean=e_mean,
                             eff_sigma_max_max=e_mean * 1.25, eff_fro_mean=2.9 / (1.0 + l2m_w))
            emit(f"[StepNorm] task {t} k_mean={sn['k_mean']:.2f} k_min={sn['k_min']} k_max={sn['k_max']} | "
                 + _norm_fields(sn["A_sigma_max_mean"], sn["A_sigma_max_max"], sn["eff_sigma_max_mean"],
                                sn["eff_sigma_max_max"], sn["eff_fro_mean"]))
            info["stepnorm"][t] = sn
        # ---- drift ---------------------------------------------------------------------
        K = sum(incs[:t])
        n_samp = incs[t] * TRAIN_PER_CLASS.get(ds, 500)
        if t > 0 and cfg.get("train_drift", True):
            emit("[Drift] Fitting ridge drift map for task %d" % t, 0.0)
            F, S = 0.75 * d["drift"], 0.25 * d["drift"]
            timing(t, "drift_features", F, dt=F, n=n_samp, d=D_FEAT)
            lam = lam_fixed
            gcv_dt = 0.0
            if feat.get("gcv"):
                lam = GCV_SCHEDULE[(t - 1) % len(GCV_SCHEDULE)]
                scores = OrderedDict((g, 0.95 - 0.12 * math.exp(-(math.log10(g) - math.log10(lam)) ** 2)) for g in GCV_GRID)
                emit(f"[Drift] lambda_rule=gcv chosen lambda={lam} | gcv " + " ".join(f"{g:g}:{s:.6g}" for g, s in scores.items()), 0.05)
                info["gcv"][t] = (lam, scores)
                gcv_dt = 0.05
            timing(t, "drift_solve", S, dt=S, n=n_samp, d=D_FEAT, solver=solver)
            solver_note = "" if solver == "dense" else f", solver={solver}"
            emit(f"[Drift] W shape ({D_FEAT}, {D_FEAT}), ||W||={27.8 + 0.01 * t:.3f}, ||W-I||={1.2 + 0.02 * t:.3f}, "
                 f"mean cos(W{DOT}f_(t-1), f_t)={0.99 - 0.001 * t:.4f} over {n_samp} samples "
                 f"({LAM}={lam}, ridge={cfg.get('train_drift_ridge', 'identity')}{solver_note})",
                 0.0 if feat.get("timing") else d["drift"])
            timing(t, "drift", F + S + gcv_dt)
            emit(f"[Drift] Transporting {K} old-class prototypes (mode={cfg.get('train_drift_transport', 'full')})", 0.0)
            timing(t, "transport", d["transport"], dt=d["transport"], K=K, mode=cfg.get("train_drift_transport", "full"),
                   cov_mode=cov_mode)
            if feat.get("driftdiag"):
                dd = OrderedDict(K=K, n_img=min(K * 100, K * TRAIN_PER_CLASS.get(ds, 500)), src="train",
                                 mean_rel_raw=0.10 + 0.01 * t, mean_rel_before=0.08 + 0.005 * t, mean_rel_after=0.03 + 0.001 * t,
                                 cov_relF_raw=0.20 + 0.01 * t, cov_relF_before=0.15 + 0.005 * t, cov_relF_after=0.05 + 0.001 * t,
                                 os_mean_rel_I=0.09 + 0.001 * t, os_mean_rel_W=0.035 + 0.0005 * t, os_mean_rel_oracle=0.025,
                                 os_cov_relF_I=0.18, os_cov_relF_W=0.06, os_cov_relF_oracle=0.045)
                emit(f"[DriftDiag] task {t} K={K} n_img={dd['n_img']} src=train | "
                     f"mean_rel raw={dd['mean_rel_raw']:.4f} before={dd['mean_rel_before']:.4f} after={dd['mean_rel_after']:.4f} | "
                     f"cov_relF raw={dd['cov_relF_raw']:.4f} before={dd['cov_relF_before']:.4f} after={dd['cov_relF_after']:.4f} | "
                     f"onestep mean_rel I={dd['os_mean_rel_I']:.4f} W={dd['os_mean_rel_W']:.4f} oracle={dd['os_mean_rel_oracle']:.4f} "
                     f"cov_relF I={dd['os_cov_relF_I']:.4f} W={dd['os_cov_relF_W']:.4f} oracle={dd['os_cov_relF_oracle']:.4f}",
                     d["driftdiag"])
                timing(t, "driftdiag", d["driftdiag"])
                info["driftdiag"][t] = dd
        # ---- class statistics ------------------------------------------------------------
        if cfg.get("train_ca", True) or cfg.get("train_drift", True):
            if feat.get("cov_mode") and cov_mode in ("diag", "lowrank", "shared"):
                per = {"diag": 2 * D_FEAT, "lowrank": D_FEAT * (cov_rank + 2), "shared": D_FEAT}[cov_mode]
                once = D_FEAT * D_FEAT if cov_mode == "shared" else 0
                emit(f"[Alignment] cov_mode={cov_mode} rank={cov_rank if cov_mode == 'lowrank' else '-'} "
                     f"stats_floats={total_cls * per + once}", 0.0)
            gap = 0.0 if feat.get("timing") else (d["stats"] + (d["transport"] if (t > 0 and cfg.get("train_drift", True)) else 0.0))
            timing(t, "stats", d["stats"], dt=d["stats"], classes=incs[t])
            # ---- alignment (t > 0 only; task 0 skips align, main.py train()) ----------------
            if t > 0 and cfg.get("train_ca", True):
                emit(f"[Alignment] Task {t}", gap)
                emit(f"[Alignment] Trainable: {total_cls * D_FEAT:,}", 0.0)
                emit("[Alignment] Epoch 5/10, Base Loss: 0.1234, Robust Term: 0.0100, Total Loss: 0.1334, Accuracy: 0.9700", d["align"] / 2)
                emit("[Alignment] Epoch 10/10, Base Loss: 0.0901, Robust Term: 0.0090, Total Loss: 0.0991, Accuracy: 0.9800", d["align"] / 2)
                timing(t, "align", d["align"], classes=total_cls, epochs=cfg.get("train_ca_epochs", 10))
            elif t == 0 and not feat.get("timing"):
                clock.tick(d["stats"])            # the task-0 statistics pass has no line of its own
        # ---- evaluation (learn()) ----------------------------------------------------------
        emit(f"[Evaluation] Task {t}", d["eval"])
        emit(f"[Evaluation] Total Acc: {totals[t]:.2f}, Grouped: {_fmt_list(rows[t])}", 0.0)
        if dil_classes:
            emit(f"[Evaluation] DIL Last-Acc: {totals[t]:.2f}, Inc-Acc: {np.mean(totals[:t + 1]):.2f} "
                 f"(sample-weighted, mod-{dil_classes} scoring)", 0.0)
        faa, ffm, ffd, asa = compute_metrics_ref(rows[: t + 1])
        emit(f"[Evaluation] FAA: {faa:.2f}, FFM: {ffm:.2f}, FFD: {ffd:.2f}, ASA: {asa:.2f}", 0.0)
        eval_s = d["eval"]
        if feat.get("til"):
            til_row = [min(99.5, a + 5.0) for a in rows[t]]
            til_total = min(99.5, totals[t] + 5.0)
            emit(f"[Evaluation-til] Total Acc: {til_total:.2f}, Grouped: {_fmt_list(til_row)}", 0.0)
            info["til_rows"].append(til_row)
        if rows_val is not None:
            emit(f"[Evaluation-val] Total Acc: {totals_val[t]:.2f}, Grouped: {_fmt_list(rows_val[t])}", 0.5)
            fv = compute_metrics_ref(rows_val[: t + 1])
            emit(f"[Evaluation-val] FAA: {fv[0]:.2f}, FFM: {fv[1]:.2f}, FFD: {fv[2]:.2f}, ASA: {fv[3]:.2f}", 0.0)
            eval_s += 0.5
        timing(t, "eval", eval_s, classes=total_cls)
    info["status"] = status
    info["rows"] = rows if status == "ok" else rows[:crash_at_task]
    info["totals"] = totals if status == "ok" else totals[:crash_at_task]
    if status == "ok":
        info["faa"], info["ffm"], info["ffd"], info["asa"] = compute_metrics_ref(rows)
        if rows_val is not None:
            info["faa_val"], info["ffm_val"], info["ffd_val"], info["asa_val"] = compute_metrics_ref(rows_val)
    else:
        info["faa"] = info["ffm"] = info["asa"] = 0.0            # main.py:1968
    if header:
        time_s = (clock.t - t_start).total_seconds()
        emit(f"Time: {time_s:.1f}s")
        info["time_s"] = round(time_s, 1)
    return L, info


def summary_block(clock, ds, cfg_name, infos, with_val=False, src="main.py", n_seeds=None):
    """SUMMARY block of run_config_sweep (main.py:2010-2016) over the given seeds."""
    pre = f"[{src}] => "
    L = [f"{clock.tick(0.1)} {pre}", "=" * 80,
         f"{clock.tick(0.1)} {pre}SUMMARY  {ds.upper()} {EM} {cfg_name.upper()}",
         f"{clock.tick(0.1)} {pre}" + "=" * 80]
    asa = [i["asa"] for i in infos]
    faa = [i["faa"] for i in infos]
    ffm = [i["ffm"] for i in infos]
    L.append(f"{clock.tick(0.1)} {pre}  ASA: {np.mean(asa):.2f} {PM} {np.std(asa):.2f} | "
             f"FAA: {np.mean(faa):.2f} {PM} {np.std(faa):.2f} | FFM: {np.mean(ffm):.2f} {PM} {np.std(ffm):.2f}")
    summ = {"asa": float(np.mean(asa)), "asa_sd": float(np.std(asa)), "faa": float(np.mean(faa)),
            "faa_sd": float(np.std(faa)), "ffm": float(np.mean(ffm)), "ffm_sd": float(np.std(ffm))}
    if with_val:
        ok = [i for i in infos if "faa_val" in i]
        v = {k: [i[k] for i in ok] for k in ("asa_val", "faa_val", "ffm_val")}
        n_all = n_seeds if n_seeds is not None else len(infos)
        L.append(f"{clock.tick(0.1)} {pre}  [val] ASA: {np.mean(v['asa_val']):.2f} {PM} {np.std(v['asa_val']):.2f} | "
                 f"FAA: {np.mean(v['faa_val']):.2f} {PM} {np.std(v['faa_val']):.2f} | "
                 f"FFM: {np.mean(v['ffm_val']):.2f} {PM} {np.std(v['ffm_val']):.2f}"
                 + (f" | n={len(ok)}/{n_all}" if len(ok) != n_all else ""))
        summ["val"] = {"faa": float(np.mean(v["faa_val"])), "faa_sd": float(np.std(v["faa_val"])), "n": len(ok)}
    L.append(f"{clock.tick(0.1)} {pre}[Cleanup] Removed {9 * len(infos)} merged checkpoints for {cfg_name}")
    L.append(f"{clock.tick(0.1)} {pre}" + "=" * 80)
    L.append("")
    return L, summ


def make_sweep_log(ds, cfg_name, seeds=SEEDS, init=None, inc=None, epochs=10, alpha=0.15, lam=100.0,
                   p=0.9, variant="only_A", prefix=None, with_val=False, crash_seed=None,
                   crash_at_task=1, repeat_seeds=(), cached=True, tags=False, rule="pspectral",
                   incremental=False, coef=1.0, stop_at_task=None, extra_config=None,
                   start="2026-11-08 10:00:00", dil_classes=None, matrix_seed=0, base_acc=92.0,
                   clock=None, features=None, val_frac=0.2, durations=None):
    """One logs_main/<ds>/<cfg_name>.log in run_config_sweep format.  Returns (text, info).

    info: {'seeds': {seed: per-seed info}, 'summary': {...}, 'n_tasks', 'incs', 'repeat': {...}}.
    ``features`` (dict) switches the gated lines on, see seed_block; ``extra_config`` keys
    land in the Configuration dump (and drive the feature values).
    """
    n_default, init_d, inc_d = DATA_TABLE.get(ds, (10, 10, 10))
    init = init_d if init is None else init
    inc = inc_d if inc is None else inc
    incs = task_increments(TOTAL_CLASSES.get(ds, 100), init, inc)
    n_tasks = len(incs)
    n_eval = n_tasks if stop_at_task is None else min(n_tasks, stop_at_task + 1)
    clock = clock or Clock(start)
    over = dict(train_epochs=epochs, train_merge_alpha=alpha, train_drift_lambda=lam, pspectral_p=p,
                spectral_variant=variant, train_merge=rule, train_merge_coef=coef,
                train_merge_incremental=incremental)
    if prefix:
        over["train_prefix"] = prefix
    if stop_at_task is not None:
        over["train_stop_at_task"] = stop_at_task
    if dil_classes:
        over["dil_classes_per_domain"] = dil_classes
    if with_val:
        over["data_val_frac"] = val_frac
    over.update(extra_config or {})
    lines, infos = [], OrderedDict()
    for seed in seeds:
        rng = np.random.RandomState(matrix_seed * 1000 + seed)
        rows = synth_rows(n_eval, rng, base=base_acc)
        totals = totals_from_rows(rows, incs, TEST_PER_CLASS.get(ds, 100))
        rows_val = totals_val = None
        if with_val:
            rows_val = [[min(99.5, a + 0.8) for a in r] for r in rows]
            totals_val = totals_from_rows(rows_val, incs, 20)
        cfg = base_config(ds, seed, init, inc, n_tasks, **over)
        L, info = seed_block(clock, ds, cfg_name, seed, cfg, rows, totals, rows_val, totals_val,
                             crash_at_task=(crash_at_task if seed == crash_seed else None),
                             cached=cached, tags=tags, dil_classes=dil_classes, features=features,
                             durations=durations)
        lines += L
        infos[seed] = info
    S, summ = summary_block(clock, ds, cfg_name, list(infos.values()), with_val=with_val, n_seeds=len(seeds))
    lines += S
    out = {"seeds": infos, "summary": summ, "n_tasks": n_tasks, "n_eval": n_eval, "incs": incs,
           "repeat": OrderedDict()}
    if repeat_seeds:
        rep = OrderedDict()
        for seed in repeat_seeds:
            rng = np.random.RandomState(matrix_seed * 1000 + seed + 7)
            rows = synth_rows(n_eval, rng, base=base_acc + 1.0)
            totals = totals_from_rows(rows, incs, TEST_PER_CLASS.get(ds, 100))
            cfg = base_config(ds, seed, init, inc, n_tasks, **over)
            L, info = seed_block(clock, ds, cfg_name, seed, cfg, rows, totals, cached=cached, tags=tags,
                                 features=features, durations=durations)
            lines += L
            rep[seed] = info
        S, summ2 = summary_block(clock, ds, cfg_name, list(rep.values()))
        lines += S
        out["repeat"] = rep
        out["summary_repeat"] = summ2
    return "\n".join(lines) + "\n", out


def make_sota_log(cells, start="2026-09-20 08:00:00", stage1_best=None):
    """run_sota_merge-style log: Configuration-only blocks, no Experiment/Time lines, one
    'SWEEP SUMMARY [sota-{ds}] {name}: FA m +- s | AA m +- s' line per cell
    (run_sota_merge.py:55-57).  ``cells``: list of dicts with ds, name, seeds, overrides."""
    clock = Clock(start)
    lines, info = [], OrderedDict()
    for cell in cells:
        ds, name = cell["ds"], cell["name"]
        n_default, init, inc = DATA_TABLE.get(ds, (10, 10, 10))
        incs = task_increments(TOTAL_CLASSES.get(ds, 100), init, inc)
        infos = OrderedDict()
        for seed in cell.get("seeds", SEEDS):
            rng = np.random.RandomState(cell.get("matrix_seed", 3) * 1000 + seed)
            rows = synth_rows(len(incs), rng, base=cell.get("base_acc", 90.0))
            totals = totals_from_rows(rows, incs, TEST_PER_CLASS.get(ds, 100))
            cfg = base_config(ds, seed, init, inc, len(incs), **cell.get("overrides", {}))
            L, i = seed_block(clock, ds, name, seed, cfg, rows, totals, header=False)
            lines += L
            infos[seed] = i
        fa = [i["faa"] for i in infos.values()]
        aa = [i["asa"] for i in infos.values()]
        lines.append(f"{clock.tick(0.1)} [run_sota_merge.py] => SWEEP SUMMARY [sota-{ds}] {name}: "
                     f"FA {np.mean(fa):.2f} {PM} {np.std(fa):.2f} | AA {np.mean(aa):.2f} {PM} {np.std(aa):.2f}")
        info[(ds, name)] = {"seeds": infos, "fa": float(np.mean(fa)), "fa_sd": float(np.std(fa)),
                            "aa": float(np.mean(aa)), "aa_sd": float(np.std(aa))}
    if stage1_best:
        for method, (tag, fa) in stage1_best.items():
            lines.append(f"{clock.tick(0.1)} [run_sota_merge.py] => SWEEP SUMMARY [sota-stage1] {method} "
                         f"best={tag} FA {fa:.2f}")
    return "\n".join(lines) + "\n", info


def _pm(values):
    return f"{np.mean(values):.2f} {PM} {np.std(values):.2f}"        # run_inflora._pm (std ddof 0)


def keeplora_config(ds, seed, init, inc, n_tasks, **over):
    """Configuration dump of a run_keeplora.py cell (RECIPE_PRESETS['keeplora'] + run_keeplora.RECIPE)."""
    cfg = OrderedDict([
        ("seed", seed), ("train_method", "keeplora"), ("reset_train", False), ("reset_merge", True),
        ("train_stop_at_task", -1), ("model_backbone", "vit_base_patch16_224"), ("train_optimizer", "adamw"),
        ("train_base_lr", 0.001), ("train_head_lr", None), ("train_weight_decay", 0.01), ("train_epochs", 10),
        ("train_batch_size", 64), ("train_scheduler", "onecycle"), ("model_use_norm", False),
        ("model_classifier_use_norm", False), ("model_classifier_bias", True), ("train_merge", "none"),
        ("train_ca", False), ("train_drift", False), ("lora_r", 16), ("keeplora_alpha", 1.0),
        ("keeplora_eps_w", 0.85), ("keeplora_eps_f", 0.99), ("keeplora_wp_side", "output"),
        ("keeplora_feat_mode", "code"), ("keeplora_feat_transform", "train"), ("keeplora_b_init", "svd"),
        ("keeplora_head_init", "ncm"), ("keeplora_backbone", "vit_base_patch16_224"), ("keeplora_layers", None),
        ("keeplora_proj", True), ("cache_backbone", True), ("checkpoint_dir", "checkpoints_main/keeplora"),
        ("train_prefix", f"keeplora_matched_{ds}_r16_w0.85_f0.99_e10_lr0.001_sout_mcode_bsvd_hncm_p1_Lall_bs64"),
        ("dataset_name", ds), ("dataset_num_task", n_tasks), ("dataset_init_cls", init), ("dataset_increment", inc),
    ])
    cfg.update(over)
    return cfg


def make_keeplora_log(ds, cfg_name, seeds=SEEDS, failed_seed=None, fail_at_task=3, lora_r=16, eps_w=0.85,
                      eps_f=0.99, epochs=10, lr=0.001, batch=64, backbone="vit_base_patch16_224",
                      start="2026-11-22 08:00:00", matrix_seed=32, base_acc=90.0, extra_config=None,
                      saturate=False):
    """run_keeplora.py cell log: run_config_sweep-style per-seed blocks (Experiment header,
    Configuration dump, the [KeepLoRA] lines of main._keeplora_before_train / _after_train,
    the ', lr:' epoch field, no merge / drift / alignment), '[FAILED]' for a failed seed, the
    SUMMARY block (' | n=k/n' when a seed failed) and the 'SWEEP SUMMARY [keeplora-{ds}]' line.
    `saturate` makes the last task fill two bases (the parser must flag it).  Returns (text, info)."""
    clock = Clock(start)
    n_default, init, inc = DATA_TABLE.get(ds, (10, 10, 10))
    incs = task_increments(TOTAL_CLASSES.get(ds, 100), init, inc)
    n_tasks = len(incs)
    pre_r = "[run_keeplora.py] => "
    pre_m = "[main.py] => "
    lines, infos, ok, failed = [], OrderedDict(), [], []
    fa_l, aa_l, ff_l, acc_l, avg_l = [], [], [], [], []
    bb = "in21k" if "in21k" in backbone else "matched"
    tag = "vit_base_patch16_224." + ("augreg_in21k" if bb == "in21k" else "augreg2_in21k_ft_in1k")
    n_slices, d = 48, 768
    for seed in seeds:
        rng = np.random.RandomState(matrix_seed * 1000 + seed)
        rows = synth_rows(n_tasks, rng, base=base_acc, drop=0.5)
        totals = totals_from_rows(rows, incs, TEST_PER_CLASS.get(ds, 100))
        cfg = keeplora_config(ds, seed, init, inc, n_tasks, lora_r=lora_r, keeplora_eps_w=eps_w, keeplora_eps_f=eps_f,
                              train_epochs=epochs, train_base_lr=lr, train_batch_size=batch, keeplora_backbone=backbone,
                              train_prefix=f"keeplora_{bb}_{ds}_r{lora_r}_w{eps_w:g}_f{eps_f:g}_e{epochs}_lr{lr:g}"
                                           f"_sout_mcode_bsvd_hncm_p1_Lall_bs{batch}",
                              **(extra_config or {}))
        lines.append(f"{clock.tick(0.1)} {pre_r}")
        lines.append("=" * 80)
        lines.append(f"{clock.tick(0.1)} {pre_r}Experiment: {ds} {EM} {cfg_name} {EM} seed {seed}")
        t_start = clock.t
        lines.append(f"{clock.tick(0.1)} {pre_m}Configuration:")
        for k, v in cfg.items():
            lines.append(f"{clock.tick(0.0)} {pre_m}  {k}: {v}")
        lines.append(f"{clock.tick(0.5)} {pre_m}[KeepLoRA] backbone {backbone} -> {tag}")
        info = {"seed": seed, "status": "ok", "subspace": [], "design": []}
        cols = [300 + 2 * s for s in range(n_slices)]
        for t in range(n_tasks):
            if failed_seed == seed and t == fail_at_task:
                lines.append(f"{clock.tick(0.2)} {pre_m}[Experiment {ds}_{cfg_name}] RuntimeError: CUDA out of memory")
                lines.append(f"{clock.tick(0.0)} {pre_m}Traceback (most recent call last):")
                lines.append('  File "main.py", line 2930, in run_single_experiment')
                lines.append("RuntimeError: CUDA out of memory")
                info["status"] = "crashed"
                break
            lines.append(f"{clock.tick(0.2)} {pre_m}[Training] Task {t} (keeplora)")
            if t == 0:
                lines.append(f"{clock.tick(1.0)} {pre_m}[KeepLoRA] Task 0: weight subspace eps_w={eps_w} side=output | "
                             f"cols min {min(cols)} max {max(cols)} over {n_slices} slices")
            lines.append(f"{clock.tick(0.5)} {pre_m}[KeepLoRA] Task {t}: head {t} set to class means")
            leak = 2.0e-7 + 1e-8 * t
            n_batches = 78 if ds == "cifar224" else 37
            lines.append(f"{clock.tick(1.0)} {pre_m}[KeepLoRA] Task {t}: A/B designed for {n_slices} slices from "
                         f"{n_batches} batches | basis cols min {min(cols)} max {max(cols)} | max|U^T A| {leak:.3e} | B init svd")
            info["design"].append((t, leak, max(cols)))
            lines.append(f"{clock.tick(0.1)} {pre_m}[Training] Model21(trainable_params={4 * d * lora_r * 12 + incs[t] * 769:,}, ...)")
            for e in range(epochs):
                lr_e = lr / 25 if e == 0 else lr * max(0.02, 1.0 - e / epochs)
                lines.append(f"{clock.tick(2.0)} {pre_m}[Training] Epoch {e + 1}/{epochs}, Loss: {0.8 / (e + 1):.4f}, "
                             f"Acc: {0.85 + 0.007 * e:.4f}, lr: {lr_e:.3e}")
            added = [1 + (s + t) % 4 for s in range(n_slices)]
            if saturate and t == n_tasks - 1:
                added[0], added[1] = d - cols[0], d - cols[1]
            cols = [c + a for c, a in zip(cols, added)]
            full = sum(c >= d for c in cols)
            lines.append(f"{clock.tick(1.0)} {pre_m}[KeepLoRA] subspace task {t} eps_f={eps_f} mode=code | "
                         f"cols min {min(cols)} max {max(cols)} mean {np.mean(cols):.1f} | "
                         f"added min {min(added)} max {max(added)} | full {full}/{n_slices} | cols {cols} | added {added}")
            info["subspace"].append((t, list(cols), full))
            lines.append(f"{clock.tick(2.0)} {pre_m}[Evaluation] Task {t}")
            lines.append(f"{clock.tick(0.0)} {pre_m}[Evaluation] Total Acc: {totals[t]:.2f}, Grouped: {_fmt_list(rows[t])}")
            faa, ffm, ffd, asa = compute_metrics_ref(rows[: t + 1])
            lines.append(f"{clock.tick(0.0)} {pre_m}[Evaluation] FAA: {faa:.2f}, FFM: {ffm:.2f}, FFD: {ffd:.2f}, ASA: {asa:.2f}")
        time_s = (clock.t - t_start).total_seconds()
        lines.append(f"{clock.tick(0.1)} {pre_r}Time: {time_s:.1f}s")
        if info["status"] == "crashed":
            failed.append(seed)
            lines.append(f"{clock.tick(0.0)} {pre_r}[FAILED] {ds} {EM} {cfg_name} {EM} seed {seed}: no result "
                         "(see the [Experiment ...] error above); excluded from the summary")
            info["faa"] = info["ffm"] = info["asa"] = 0.0
        else:
            ok.append(seed)
            faa, ffm, ffd, asa = compute_metrics_ref(rows)
            info.update(faa=faa, ffm=ffm, ffd=ffd, asa=asa, micro_acc=totals[-1], avg=float(np.mean(totals)),
                        rows=rows, totals=totals)
            fa_l.append(faa)
            aa_l.append(asa)
            ff_l.append(ffm)
            acc_l.append(totals[-1])
            avg_l.append(float(np.mean(totals)))
            lines.append(f"{clock.tick(0.0)} {pre_r}[KeepLoRA] seed {seed}: FA {faa:.2f} | AA {asa:.2f} | FF {ffm:.2f} | "
                         f"micro ACC {totals[-1]:.2f} | AVG {np.mean(totals):.2f} | curve {[round(v, 2) for v in totals]}")
        infos[seed] = info
    lines.append(f"{clock.tick(0.1)} {pre_r}")
    lines.append("=" * 80)
    lines.append(f"{clock.tick(0.1)} {pre_r}SUMMARY  {ds.upper()} {EM} {cfg_name.upper()}")
    lines.append(f"{clock.tick(0.1)} {pre_r}" + "=" * 80)
    summ = None
    if ok:
        lines.append(f"{clock.tick(0.1)} {pre_r}  ASA: {_pm(aa_l)} | FAA: {_pm(fa_l)} | FFM: {_pm(ff_l)}"
                     + (f" | n={len(ok)}/{len(seeds)}" if failed else ""))
        lines.append(f"{clock.tick(0.1)} {pre_r}SWEEP SUMMARY [keeplora-{ds}] {cfg_name}: FA {_pm(fa_l)} | AA {_pm(aa_l)} | "
                     f"FF {_pm(ff_l)} | micro ACC {_pm(acc_l)} | AVG {_pm(avg_l)} | seeds {ok} | failed {failed}")
        summ = {"faa": float(np.mean(fa_l)), "faa_sd": float(np.std(fa_l)), "asa": float(np.mean(aa_l)),
                "ffm": float(np.mean(ff_l)), "micro_acc": float(np.mean(acc_l)), "avg": float(np.mean(avg_l))}
    else:
        lines.append(f"{clock.tick(0.1)} {pre_r}[FAILED] [keeplora-{ds}] {cfg_name}: every seed failed {failed}; no summary")
    lines.append(f"{clock.tick(0.1)} {pre_r}" + "=" * 80 + "\n")
    return "\n".join(lines) + "\n", {"seeds": infos, "ok": ok, "failed": failed, "summary": summ, "n_tasks": n_tasks}


def make_inflora_log(ds, cfg_name, seeds=SEEDS, failed_seed=None, fail_at_task=3, lora_r=10, lamb=0.95,
                     epochs=20, lr=0.0005, batch=128, backbone="vit_base_patch16_224_in21k",
                     start="2026-11-20 08:00:00", matrix_seed=30, base_acc=90.0, extra_config=None):
    """run_inflora.py cell log: run_config_sweep-style per-seed blocks (Experiment header,
    Configuration dump, [InfLoRA] lines, ', lr:' epoch field, no merge / drift / alignment),
    '[FAILED]' for a failed seed, the SUMMARY block (' | n=k/n' when a seed failed) and the
    'SWEEP SUMMARY [inflora-{ds}]' line.  Returns (text, info)."""
    clock = Clock(start)
    n_default, init, inc = DATA_TABLE.get(ds, (10, 10, 10))
    incs = task_increments(TOTAL_CLASSES.get(ds, 100), init, inc)
    n_tasks = len(incs)
    pre_r = "[run_inflora.py] => "
    pre_m = "[main.py] => "
    lines, infos, ok, failed = [], OrderedDict(), [], []
    fa_l, aa_l, ff_l, acc_l, avg_l = [], [], [], [], []
    bb = "in21k" if "in21k" in backbone else "matched"
    tag = f"{backbone.split('_')[0]}_base_patch16_224." + ("augreg_in21k" if bb == "in21k" else "augreg2_in21k_ft_in1k")
    for seed in seeds:
        rng = np.random.RandomState(matrix_seed * 1000 + seed)
        rows = synth_rows(n_tasks, rng, base=base_acc, drop=0.5)
        totals = totals_from_rows(rows, incs, TEST_PER_CLASS.get(ds, 100))
        cfg = inflora_config(ds, seed, init, inc, n_tasks, lora_r=lora_r, inflora_lamb=lamb, train_epochs=epochs,
                             train_base_lr=lr, train_batch_size=batch, inflora_backbone=backbone,
                             train_prefix=f"inflora_{bb}_{ds}_r{lora_r}_l{lamb}_e{epochs}_lr{lr:g}_gtrain_d1_Lall_b{batch}",
                             **(extra_config or {}))
        lines.append(f"{clock.tick(0.1)} {pre_r}")
        lines.append("=" * 80)
        lines.append(f"{clock.tick(0.1)} {pre_r}Experiment: {ds} {EM} {cfg_name} {EM} seed {seed}")
        t_start = clock.t
        lines.append(f"{clock.tick(0.1)} {pre_m}Configuration:")
        for k, v in cfg.items():
            lines.append(f"{clock.tick(0.0)} {pre_m}  {k}: {v}")
        lines.append(f"{clock.tick(0.5)} {pre_m}[InfLoRA] backbone {backbone} -> {tag}")
        info = {"seed": seed, "status": "ok", "gpm": [], "design": []}
        cols = [21 + 3 * b for b in range(12)]
        for t in range(n_tasks):
            if failed_seed == seed and t == fail_at_task:
                lines.append(f"{clock.tick(0.2)} {pre_m}[Experiment {ds}_{cfg_name}] RuntimeError: CUDA out of memory")
                lines.append(f"{clock.tick(0.0)} {pre_m}Traceback (most recent call last):")
                lines.append('  File "main.py", line 2930, in run_single_experiment')
                lines.append("RuntimeError: CUDA out of memory")
                info["status"] = "crashed"
                break
            lines.append(f"{clock.tick(0.2)} {pre_m}[Training] Task {t} (inflora)")
            leak = 0.0 if t == 0 else 3.1e-6 + 1e-7 * t
            lines.append(f"{clock.tick(1.0)} {pre_m}[InfLoRA] Task {t}: A_t designed for 12 blocks (remove 12 / retain 0) "
                         f"| max|A M| {leak:.3e}")
            info["design"].append((t, leak))
            lines.append(f"{clock.tick(0.1)} {pre_m}[Training] Model21(trainable_params={2 * 768 * lora_r * 12 + incs[t] * 769:,}, ...)")
            for e in range(epochs):
                lr_e = lr * (math.cos(99 * math.pi * e / (200 * (epochs - 1))) if epochs > 1 else 1.0)
                lines.append(f"{clock.tick(2.0)} {pre_m}[Training] Epoch {e + 1}/{epochs}, Loss: {0.8 / (e + 1):.4f}, "
                             f"Acc: {0.85 + 0.007 * e:.4f}, lr: {lr_e:.3e}")
            thr = lamb + (1.0 - lamb) * t / n_tasks
            added = [1 + (b + t) % 3 for b in range(12)]
            cols = [c + a for c, a in zip(cols, added)]
            types = ["remove"] * 12
            lines.append(f"{clock.tick(1.0)} {pre_m}[InfLoRA] DualGPM task {t} threshold {thr:.4f} | cols {cols} | "
                         f"types {types} | changed {added}")
            info["gpm"].append((t, thr, list(cols)))
            lines.append(f"{clock.tick(2.0)} {pre_m}[Evaluation] Task {t}")
            lines.append(f"{clock.tick(0.0)} {pre_m}[Evaluation] Total Acc: {totals[t]:.2f}, Grouped: {_fmt_list(rows[t])}")
            faa, ffm, ffd, asa = compute_metrics_ref(rows[: t + 1])
            lines.append(f"{clock.tick(0.0)} {pre_m}[Evaluation] FAA: {faa:.2f}, FFM: {ffm:.2f}, FFD: {ffd:.2f}, ASA: {asa:.2f}")
        time_s = (clock.t - t_start).total_seconds()
        lines.append(f"{clock.tick(0.1)} {pre_r}Time: {time_s:.1f}s")
        if info["status"] == "crashed":
            failed.append(seed)
            lines.append(f"{clock.tick(0.0)} {pre_r}[FAILED] {ds} {EM} {cfg_name} {EM} seed {seed}: no result "
                         "(see the [Experiment ...] error above); excluded from the summary")
            info["faa"] = info["ffm"] = info["asa"] = 0.0
        else:
            ok.append(seed)
            faa, ffm, ffd, asa = compute_metrics_ref(rows)
            info.update(faa=faa, ffm=ffm, ffd=ffd, asa=asa, micro_acc=totals[-1], avg=float(np.mean(totals)),
                        rows=rows, totals=totals)
            fa_l.append(faa)
            aa_l.append(asa)
            ff_l.append(ffm)
            acc_l.append(totals[-1])
            avg_l.append(float(np.mean(totals)))
            lines.append(f"{clock.tick(0.0)} {pre_r}[InfLoRA] seed {seed}: FA {faa:.2f} | AA {asa:.2f} | FF {ffm:.2f} | "
                         f"micro ACC {totals[-1]:.2f} | AVG {np.mean(totals):.2f} | curve {[round(v, 2) for v in totals]}")
        infos[seed] = info
    lines.append(f"{clock.tick(0.1)} {pre_r}")
    lines.append("=" * 80)
    lines.append(f"{clock.tick(0.1)} {pre_r}SUMMARY  {ds.upper()} {EM} {cfg_name.upper()}")
    lines.append(f"{clock.tick(0.1)} {pre_r}" + "=" * 80)
    summ = None
    if ok:
        lines.append(f"{clock.tick(0.1)} {pre_r}  ASA: {_pm(aa_l)} | FAA: {_pm(fa_l)} | FFM: {_pm(ff_l)}"
                     + (f" | n={len(ok)}/{len(seeds)}" if failed else ""))
        lines.append(f"{clock.tick(0.1)} {pre_r}SWEEP SUMMARY [inflora-{ds}] {cfg_name}: FA {_pm(fa_l)} | AA {_pm(aa_l)} | "
                     f"FF {_pm(ff_l)} | micro ACC {_pm(acc_l)} | AVG {_pm(avg_l)} | seeds {ok} | failed {failed}")
        summ = {"faa": float(np.mean(fa_l)), "faa_sd": float(np.std(fa_l)), "asa": float(np.mean(aa_l)),
                "ffm": float(np.mean(ff_l)), "micro_acc": float(np.mean(acc_l)), "avg": float(np.mean(avg_l))}
    else:
        lines.append(f"{clock.tick(0.1)} {pre_r}[FAILED] [inflora-{ds}] {cfg_name}: every seed failed {failed}; no summary")
    lines.append(f"{clock.tick(0.1)} {pre_r}" + "=" * 80 + "\n")
    return "\n".join(lines) + "\n", {"seeds": infos, "ok": ok, "failed": failed, "summary": summ, "n_tasks": n_tasks}


def make_driftdiag_csv(path, ds, per_seed_diag, n_groups_per_task=True):
    """logs_main/<ds>/driftdiag_<split>.csv (main.DRIFTDIAG_CSV_HEADER) from the per-task
    [DriftDiag] values of seed_block (info['driftdiag']): one 'all' row per task plus one row
    per old task group with a small offset.  Returns the rows written."""
    rows = []
    for seed, diag in per_seed_diag.items():
        for t, dd in diag.items():
            groups = [("all", 0.0)] + ([(str(g), 0.002 * (g + 1)) for g in range(t)] if n_groups_per_task else [])
            for g, off in groups:
                row = OrderedDict((k, "") for k in DRIFTDIAG_CSV_HEADER)
                row.update(dataset=ds, seed=seed, task=t, group=g, n_classes=dd["K"] if g == "all" else dd["K"] // max(t, 1),
                           n_img=dd["n_img"], mean_rel_raw=dd["mean_rel_raw"] + off, mean_rel_before=dd["mean_rel_before"] + off,
                           mean_rel_after=dd["mean_rel_after"] + off, mean_cos_raw=0.95, mean_cos_before=0.97, mean_cos_after=0.995,
                           cov_relF_raw=dd["cov_relF_raw"], cov_relF_before=dd["cov_relF_before"], cov_relF_after=dd["cov_relF_after"],
                           os_mean_rel_I=dd["os_mean_rel_I"], os_mean_rel_W=dd["os_mean_rel_W"], os_mean_rel_oracle=dd["os_mean_rel_oracle"],
                           os_cov_relF_I=dd["os_cov_relF_I"], os_cov_relF_W=dd["os_cov_relF_W"], os_cov_relF_oracle=dd["os_cov_relF_oracle"],
                           W_minus_I_F=1.2 + 0.02 * t, Wor_minus_I_F=1.1 + 0.02 * t, W_minus_Wor_F=0.3, n_fit_new=2400, n_fit_old=dd["n_img"])
                rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(DRIFTDIAG_CSV_HEADER))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


def make_timing_csv(path, ds, per_seed_timing):
    """logs_main/<ds>/timing_<config>.csv (main.TIMING_CSV_HEADER) from info['timing']."""
    extras = {"train": "cached=1", "merge": "rule=pspectral", "drift": "", "drift_features": "n=2400 d=768",
              "drift_solve": "n=2400 d=768 solver=dense", "transport": "K=20 mode=full cov_mode=class",
              "stats": "classes=20", "align": "classes=40 epochs=10", "eval": "classes=40", "driftdiag": ""}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(TIMING_CSV_HEADER)
        for seed, per_task in per_seed_timing.items():
            for t, stages in per_task.items():
                for stage, sec in stages.items():
                    w.writerow([ds, seed, t, stage, f"{sec:.6f}", extras.get(stage, "")])
                    n += 1
    return n


def write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def write_csv_rows(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


def build_fixture_tree(root):
    """The fixture tree used by tests/test_parse_logs.py.  Returns {key: info} with the
    written path in info['path']."""
    fx = OrderedDict()

    def put(key, ds, name, text, info):
        info["path"] = write_text(os.path.join(root, ds, f"{name}.log"), text) if ds else \
            write_text(os.path.join(root, f"{name}.log"), text)
        info["dataset"], info["config_name"] = ds, name
        fx[key] = info

    # Three alpha cells on CIFAR (10x10, cached): clean / one crashed seed / appended re-run.
    t, i = make_sweep_log("cifar224", "alpha_a0.1", alpha=0.1, matrix_seed=1, start="2026-10-01 09:00:00")
    put("alpha_a0.1", "cifar224", "alpha_a0.1", t, i)
    t, i = make_sweep_log("cifar224", "alpha_a0.15", alpha=0.15, crash_seed=1994, matrix_seed=2,
                          start="2026-10-01 11:00:00")
    put("alpha_a0.15", "cifar224", "alpha_a0.15", t, i)
    t, i = make_sweep_log("cifar224", "alpha_a0.3", alpha=0.3, repeat_seeds=(1993,), matrix_seed=3,
                          start="2026-10-01 13:00:00")
    put("alpha_a0.3", "cifar224", "alpha_a0.3", t, i)
    # Headline reproduction (run_best prefix family), uncached: epoch lines -> the time table.
    t, i = make_sweep_log("cifar224", "best_cifar224", alpha=0.15, prefix="best_cifar224_r64", cached=False,
                          matrix_seed=4, start="2026-09-10 08:00:00")
    put("best_cifar224", "cifar224", "best_cifar224", t, i)
    t, i = make_sweep_log("imagenetr", "best_imagenetr", alpha=0.3, prefix="best_imagenetr_r64",
                          matrix_seed=5, start="2026-09-11 08:00:00", base_acc=84.0)
    put("best_imagenetr", "imagenetr", "best_imagenetr", t, i)
    t, i = make_sweep_log("cars", "best_cars", alpha=0.2, prefix="best_cars_r64", matrix_seed=6,
                          start="2026-09-12 08:00:00", base_acc=84.0)
    put("best_cars", "cars", "best_cars", t, i)
    # Long splits (E1): CIFAR 20x5 (10 epochs), CIFAR 50x2 and IN-R 50x4 (5 epochs).
    t, i = make_sweep_log("cifar224", "long_t20", init=5, inc=5, alpha=0.15, matrix_seed=7,
                          start="2026-10-05 08:00:00", base_acc=91.0)
    put("long_t20", "cifar224", "long_t20", t, i)
    t, i = make_sweep_log("cifar224", "long_t50", init=2, inc=2, epochs=5, alpha=0.15, matrix_seed=8,
                          start="2026-10-06 08:00:00", base_acc=89.0)
    put("long_t50", "cifar224", "long_t50", t, i)
    t, i = make_sweep_log("imagenetr", "long_t50", init=4, inc=4, epochs=5, alpha=0.3, matrix_seed=9,
                          start="2026-10-07 08:00:00", base_acc=80.0)
    put("inr_long_t50", "imagenetr", "long_t50", t, i)
    # One cube cell in the export_mc3d_full naming (E2 --validate joins it with the CSV).
    t, i = make_sweep_log("imagenetr", "mc3d_only_A_p0.9_a0.3", alpha=0.3, matrix_seed=17,
                          start="2026-09-15 08:00:00", base_acc=84.0)
    put("mc3d", "imagenetr", "mc3d_only_A_p0.9_a0.3", t, i)
    # Validation-split run (E14) on CUB with [Evaluation-val] lines and the '[val]' summary.
    t, i = make_sweep_log("cub", "e14_val20_a0.15_l100", alpha=0.15, lam=100.0, with_val=True,
                          prefix="main_cub_r64_qkvfc_val20", matrix_seed=10, start="2026-11-09 08:00:00",
                          base_acc=88.0)
    put("e14", "cub", "e14_val20_a0.15_l100", t, i)
    # Rebuttal tags (E10 merge-time run) with every gated line of a cached pspectral run.
    t, i = make_sweep_log("imagenetr", "e10_trunc_a0.3", alpha=0.3, tags=True, matrix_seed=11,
                          extra_config={"merge_update_op": "trunc", "merge_log_stats": True, "stage_timing": True,
                                        "drift_diag": True, "eval_til": True, "data_val_seed": 1993},
                          start="2026-11-10 08:00:00", base_acc=84.0)
    put("e10", "imagenetr", "e10_trunc_a0.3", t, i)
    # Classical rules (E12): incremental avg (2 checkpoints) and non-incremental ties.
    t, i = make_sweep_log("imagenetr", "e12_avg_c1.0", rule="avg", incremental=True, coef=1.0,
                          matrix_seed=12, start="2026-11-11 08:00:00", base_acc=83.0)
    put("e12_avg", "imagenetr", "e12_avg_c1.0", t, i)
    t, i = make_sweep_log("imagenetr", "e12_ties_all", rule="ties", incremental=False, coef=1.0,
                          seeds=(1993,), matrix_seed=13, start="2026-11-11 10:00:00", base_acc=82.0)
    put("e12_ties", "imagenetr", "e12_ties_all", t, i)
    # Training-time spectral control run (E2 must list it): specreg, no merge, StepNorm lines.
    t, i = make_sweep_log("imagenetr", "specreg_0.1", rule="none", tags=True, seeds=(1993,),
                          extra_config={"specreg_weight": 0.1, "train_prefix": "main_inr_r64_qkvfc_sr0.1",
                                        "merge_log_stats": True},
                          cached=False, matrix_seed=14, start="2026-10-20 08:00:00", base_acc=82.0)
    put("specreg", "imagenetr", "specreg_0.1", t, i)
    # Smoke run (train_stop_at_task 1) under the smoke log root.
    t, i = make_sweep_log("imagenetr", "e8_smoke", alpha=0.3, stop_at_task=1, seeds=(1993,),
                          prefix="main_inr_r64_qkvfc_smoke", matrix_seed=15, start="2026-11-12 08:00:00")
    i["path"] = write_text(os.path.join(root, "smoke", "imagenetr", "e8_smoke.log"), t)
    i["dataset"], i["config_name"] = "imagenetr", "e8_smoke"
    fx["smoke"] = i
    # DIL run (Office-Home, 4 domains) with the DIL line.
    t, i = make_sweep_log("officehome_dil", "dil_best", alpha=0.15, dil_classes=65, seeds=(1993,),
                          matrix_seed=16, start="2026-10-25 08:00:00", base_acc=88.0)
    put("dil", "officehome_dil", "dil_best", t, i)
    # run_sota_merge-style shared log: Configuration-only blocks + SWEEP SUMMARY lines.
    cells = [{"ds": "imagenetr", "name": "sota_dare_q0.5_c1.0",
              "overrides": {"train_merge": "dare", "merge_dare_q": 0.5, "train_merge_coef": 1.0}, "matrix_seed": 20},
             {"ds": "imagenetr", "name": "sota_dare_q0.9_c1.0",
              "overrides": {"train_merge": "dare", "merge_dare_q": 0.9, "train_merge_coef": 1.0},
              "matrix_seed": 21, "base_acc": 60.0},
             {"ds": "cifar224", "name": "sota_dare_best",
              "overrides": {"train_merge": "dare", "merge_dare_q": 0.5, "train_merge_coef": 1.0,
                            "train_weight_decay": 0.0005, "train_ca_robust_weight": 0.2}, "matrix_seed": 22}]
    t, i = make_sota_log(cells, stage1_best={"dare": ("q0.5_c1.0", 82.48)})
    info = {"cells": i}
    info["path"] = write_text(os.path.join(root, "sota_merge.log"), t)
    fx["sota"] = info
    return fx


# Rebuttal fixture tree: one or two cells per experiment in the naming of the run_e*.py
# drivers, plus the side files the subcommands join (cube CSV, E4 spectra, SVD bench, refs).
E8_CELLS = {"cifar224": [(0.9, 0.1), (0.9, 0.15), (0.9, 0.3), (0.9, 1.0), (1.0, 0.1), (1.0, 0.15), (1.0, 0.3), (1.0, 1.0),
                         (0.5, 1.0), (0.1, 1.0)],
            "cub": [(0.9, 0.1), (0.9, 0.5), (1.0, 0.1), (1.0, 0.5)],
            "cars": [(0.9, 0.2), (1.0, 0.2)]}
E12_GRID = {"avg": {"rule": "avg", "incremental": True}, "ties": {"rule": "ties", "incremental": True},
            "dare": {"rule": "dare", "extra": {"merge_dare_q": 0.5}}, "knots": {"rule": "knots", "extra": {"merge_knots_topk": 100.0}}}
E12_COEFS = (0.5, 1.0, 1.2)


def build_rebuttal_tree(root, fig_dir):
    """Fixtures for the e3/e5/e7/e8/e9/e10/e11/e12/e13/e14/e15 subcommands.  Returns
    {key: info} (info['path'] = log path); side CSVs are written under ``fig_dir``."""
    fx = OrderedDict()
    HEAD = {"cifar224": (0.15, 100.0), "cub": (0.10, 10.0), "imagenetr": (0.30, 100.0), "cars": (0.20, 100.0)}
    BASE = {"cifar224": 92.0, "cub": 88.0, "imagenetr": 84.0, "cars": 84.0}
    ms = [40]

    def put(key, ds, name, text, info, **extra):
        info["path"] = write_text(os.path.join(root, ds, f"{name}.log"), text)
        info["dataset"], info["config_name"] = ds, name
        info.update(extra)
        fx[key] = info

    def sweep(ds, name, **kw):
        ms[0] += 1
        kw.setdefault("matrix_seed", ms[0])
        kw.setdefault("base_acc", BASE[ds])
        kw.setdefault("alpha", HEAD[ds][0])
        kw.setdefault("lam", HEAD[ds][1])
        kw.setdefault("start", f"2026-11-{10 + ms[0] % 18:02d} {8 + ms[0] % 12:02d}:00:00")
        return make_sweep_log(ds, name, **kw)

    # ---- E8: mc3d cells on CIFAR / CUB / Cars (headline pair on Cars) ------------------
    for ds, cells in E8_CELLS.items():
        for p_, a_ in cells:
            name = f"mc3d_only_A_p{p_}_a{a_}"
            t, i = sweep(ds, name, p=p_, alpha=a_, base_acc=BASE[ds] - (0.8 if p_ == 1.0 else 0.0) - 2.0 * abs(a_ - HEAD[ds][0]),
                         extra_config={"merge_log_stats": True})
            put(f"e8_{ds}_p{p_}_a{a_}", ds, name, t, i, p=p_, alpha=a_)
    # the committed Cars cube (fixture copy with the same schema; the real file is never touched)
    cube_rows = []
    for p_ in (0.9, 1.0):
        for a_ in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            cube_rows.append(["only_A", f"{p_:.2f}", f"{a_:.2f}", 3, f"{80.0 - 1.5 * abs(a_ - 0.3) - (0.3 if p_ == 1.0 else 0):.2f}",
                              "0.30", "85.00", "0.40", "5.00", "0.20", "79.00", "84.00", "78.00"])
    fx["cube_cars"] = {"path": write_csv_rows(os.path.join(fig_dir, "mc3d_cars_full.csv"),
                                               ["variant", "p", "alpha", "n_seeds", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd",
                                                "old_mean", "new_acc", "first5_mean"], cube_rows), "n": len(cube_rows)}
    cube_inr = [["only_A", "0.90", "0.30", 3, "83.65", "0.12", "87.50", "0.20", "3.10", "0.10", "83.00", "88.00", "82.00"],
                ["only_A", "1.00", "0.30", 3, "83.72", "0.15", "87.60", "0.20", "3.20", "0.10", "83.10", "88.10", "82.10"],
                ["only_A", "1.00", "0.50", 3, "83.40", "0.15", "87.30", "0.20", "3.60", "0.10", "82.80", "88.40", "81.70"]]
    fx["cube_imagenetr"] = {"path": write_csv_rows(os.path.join(fig_dir, "mc3d_imagenetr_full.csv"),
                                                    ["variant", "p", "alpha", "n_seeds", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd",
                                                     "old_mean", "new_acc", "first5_mean"], cube_inr), "n": len(cube_inr)}

    # ---- E10: merge-time ops on ImageNet-R + the L2M training arm ------------------------
    t, i = sweep("imagenetr", "e10_trunc_a0.3", tags=True,
                 features={"driftdiag": False, "til": False, "data": False, "l2m": False, "featshift": False},
                 extra_config={"merge_update_op": "trunc", "merge_log_stats": True, "stage_timing": True})
    put("e10_trunc", "imagenetr", "e10_trunc_a0.3", t, i)
    for a_ in (0.3, 0.5):
        t, i = sweep("imagenetr", f"e10_clip_a{a_}", alpha=a_, features={"mergestat": True}, base_acc=83.4,
                     extra_config={"merge_update_op": "clip", "merge_log_stats": True})
        put(f"e10_clip_a{a_}", "imagenetr", f"e10_clip_a{a_}", t, i)
    t, i = sweep("imagenetr", "e10_scalefro_a0.3", features={"mergestat": True}, base_acc=83.2,
                 extra_config={"merge_update_op": "scale_fro", "merge_log_stats": True})
    put("e10_scalefro", "imagenetr", "e10_scalefro_a0.3", t, i)
    t, i = sweep("imagenetr", "e10_l2m1.0", rule="none", cached=False, base_acc=82.0,
                 features={"stepnorm": True, "l2m": True, "timing": False},
                 extra_config={"l2m_weight": 1.0, "merge_log_stats": True, "train_prefix": "main_inr_r64_qkvfc_l2m1.0"})
    put("e10_l2m", "imagenetr", "e10_l2m1.0", t, i)
    t, i = sweep("imagenetr", "mc3d_only_A_p1.0_a0.3", p=1.0, base_acc=83.3)
    put("e8_imagenetr_p1.0_a0.3", "imagenetr", "mc3d_only_A_p1.0_a0.3", t, i)

    # ---- E7: adaptive alpha, feat-shift calibration, GCV lambda ---------------------------
    t, i = sweep("imagenetr", "e7_specnorm_0.5", features={"adaptive": True, "mergestat": True},
                 extra_config={"merge_alpha_rule": "specnorm", "merge_alpha_param": 0.5, "merge_alpha_stat": "eff",
                               "merge_alpha_min": 0.05, "merge_log_stats": True})
    put("e7_specnorm", "imagenetr", "e7_specnorm_0.5", t, i)
    t, i = sweep("imagenetr", "e7_featshift_0.03", features={"adaptive": True},
                 extra_config={"merge_alpha_rule": "featshift", "merge_alpha_param": 0.03, "merge_alpha_min": 0.05})
    put("e7_featshift", "imagenetr", "e7_featshift_0.03", t, i)
    t, i = sweep("imagenetr", "e7_calib_featshift", seeds=(1993,), features={"featshift": True},
                 extra_config={"merge_log_featshift": True})
    put("e7_calib", "imagenetr", "e7_calib_featshift", t, i)
    t, i = sweep("imagenetr", "e7_gcv", features={"gcv": True},
                 extra_config={"train_drift_lambda_rule": "gcv", "train_drift_lambda_grid": list(GCV_GRID),
                               "train_drift_solver": "woodbury"})
    put("e7_gcv", "imagenetr", "e7_gcv", t, i)
    t, i = sweep("cifar224", "e7_specnorm_0.5", features={"adaptive": True, "mergestat": True}, base_acc=91.0,
                 extra_config={"merge_alpha_rule": "specnorm", "merge_alpha_param": 0.5, "merge_alpha_stat": "eff",
                               "merge_alpha_min": 0.05, "merge_log_stats": True})
    put("e7_specnorm_cifar", "cifar224", "e7_specnorm_0.5", t, i)

    # ---- E5 (+ E3 timers on the reference): cov modes on ImageNet-R, two on CIFAR ---------
    for ds, modes in (("imagenetr", ("class", "diag", "lowrank_k16", "lowrank_k64", "shared")), ("cifar224", ("class", "diag"))):
        for m in modes:
            mode = m.split("_")[0]
            extra = {"train_ca_cov_mode": mode}
            if mode == "lowrank":
                extra["train_ca_cov_rank"] = int(m.split("_k")[1])
            feats = {"cov_mode": True}
            if m == "class":
                extra.update({"stage_timing": True, "stage_timing_csv": f"logs_main/{ds}/timing_cov_class.csv"})
                feats["timing"] = True
            drop = {"class": 0.0, "diag": 0.6, "lowrank": 0.2 if m.endswith("64") else 0.4, "shared": 1.1}[mode]
            t, i = sweep(ds, f"cov_{m}", features=feats, extra_config=extra, base_acc=BASE[ds] - drop)
            put(f"cov_{ds}_{m}", ds, f"cov_{m}", t, i, mode=mode)
            if m == "class":
                n = make_timing_csv(os.path.join(root, ds, "timing_cov_class.csv"), ds,
                                    {s: inf["timing"] for s, inf in i["seeds"].items()})
                fx[f"timing_csv_{ds}"] = {"path": os.path.join(root, ds, "timing_cov_class.csv"), "n": n}

    # ---- E9: drift diagnostics (log lines + per-seed CSV) -----------------------------------
    t, i = sweep("imagenetr", "driftdiag_t10", features={"driftdiag": True}, extra_config={"drift_diag": True,
                 "drift_diag_csv": "logs_main/imagenetr/driftdiag_t10.csv"})
    put("driftdiag_inr", "imagenetr", "driftdiag_t10", t, i)
    rows = make_driftdiag_csv(os.path.join(root, "imagenetr", "driftdiag_t10.csv"), "imagenetr",
                              {s: inf["driftdiag"] for s, inf in i["seeds"].items()})
    fx["driftdiag_csv_inr"] = {"path": os.path.join(root, "imagenetr", "driftdiag_t10.csv"), "rows": rows}
    t, i = sweep("cifar224", "driftdiag_t10", seeds=(1993, 1994), features={"driftdiag": True},
                 extra_config={"drift_diag": True})
    put("driftdiag_cifar", "cifar224", "driftdiag_t10", t, i)          # log lines only, no CSV

    # ---- E11: rank sweep on ImageNet-R + E4 spectrum CSVs + SVD bench ------------------------
    for r_, seeds in ((8, SEEDS), (16, (1993, 1994)), (64, SEEDS)):
        t, i = sweep("imagenetr", f"e11_rank{r_}", seeds=seeds, features={"mergestat": True}, base_acc=84.0 - 0.4 * (6 - math.log2(r_)),
                     extra_config={"model_lora_r": r_, "model_lora_alpha": 2 * r_, "train_prefix": f"main_inr_r{r_}_qkvfc",
                                   "merge_log_stats": True})
        put(f"e11_rank{r_}", "imagenetr", f"e11_rank{r_}", t, i, r=r_)
        spec = []
        for seed in seeds:
            for task in range(1, 10):
                for block in range(12):
                    for module in ("qkv", "fc1", "fc2"):
                        k9 = min(r_, 10 + block % 4 + (task % 2))
                        spec.append([seed, task, block, module, "A", r_, 1.5, 4.0, max(1, k9 - 6), max(1, k9 - 3), k9,
                                     min(r_, k9 + 2), min(r_, k9 + 5), 0.31, 0.78])
        fx[f"e4_spectrum_r{r_}"] = {"path": write_csv_rows(os.path.join(fig_dir, f"e4_spectrum_imagenetr_r{r_}.csv"),
                                                            ["seed", "task", "block", "module", "space", "n_sv", "sigma1", "fro",
                                                             "k_p0.5", "k_p0.7", "k_p0.9", "k_p0.95", "k_p0.99", "energy_top1",
                                                             "energy_top8"], spec), "rows": spec}
    svd = []
    for r_ in (8, 16, 32, 64, 128):
        for dev in ("cpu", "cuda"):
            for module in ("qkv", "fc1", "fc2"):
                ms_layer = (0.9 if dev == "cpu" else 0.25) * (1 + r_ / 64)
                svd.append([r_, dev, module, f"{ms_layer:.3f}", f"{12 * ms_layer:.3f}"])
    fx["svd_time"] = {"path": write_csv_rows(os.path.join(fig_dir, "e11_svd_time.csv"),
                                              ["r", "device", "module", "ms_per_layer", "ms_per_merge"], svd)}

    # ---- E12: coefficient grid on ImageNet-R and CUB + ISM cells -----------------------------
    for ds in ("imagenetr", "cub"):
        for rule, spec_ in E12_GRID.items():
            for c_ in E12_COEFS:
                t, i = sweep(ds, f"e12_{rule}_c{c_}", rule=spec_["rule"], incremental=spec_.get("incremental", False), coef=c_,
                             extra_config=spec_.get("extra"), base_acc=BASE[ds] - 1.5 - 2.0 * abs(c_ - (1.0 if rule != "avg" else 0.5)))
                put(f"e12_{ds}_{rule}_c{c_}", ds, f"e12_{rule}_c{c_}", t, i, rule=rule, coef=c_)
        t, i = sweep(ds, "e12_ism", extra_config={"merge_log_stats": True})
        put(f"e12_{ds}_ism", ds, "e12_ism", t, i)
    t, i = sweep("cub", "e12_maxabs_c1.0", rule="max_abs", incremental=False, coef=1.0, seeds=(1993,), base_acc=85.0)
    put("e12_cub_maxabs_nonincr", "cub", "e12_maxabs_c1.0", t, i)

    # ---- E13: few-shot per task -----------------------------------------------------------------
    for ds, ks in (("cifar224", (5, 20)), ("imagenetr", (5, 20))):
        for k in ks:
            common = {"data_fewshot_k": k, "train_prefix": f"main_{ds}_r64_qkvfc_fs{k}", "checkpoint_dir": "checkpoints_main/rebuttal/e13",
                      "eval_til": True}
            drop = 6.0 if k == 5 else 2.5
            t, i = sweep(ds, f"e13_fs{k}_ism", cached=False, epochs=10, features={"data": True, "til": True},
                         extra_config=common, base_acc=BASE[ds] - drop)
            put(f"e13_{ds}_fs{k}_ism", ds, f"e13_fs{k}_ism", t, i, k=k, method="ism")
            if k == 5:
                t, i = sweep(ds, "e13_fs5_nomerge", rule="none", features={"data": True, "til": True}, extra_config=common,
                             base_acc=BASE[ds] - drop - 3.0)
                put(f"e13_{ds}_fs5_nomerge", ds, "e13_fs5_nomerge", t, i, k=k, method="nomerge")
                t, i = sweep(ds, "e13_fs5_avg", rule="avg", incremental=True, coef=1.0, features={"data": True, "til": True},
                             extra_config=common, base_acc=BASE[ds] - drop - 1.0)
                put(f"e13_{ds}_fs5_avg", ds, "e13_fs5_avg", t, i, k=k, method="avg")
                e2 = dict(common, train_method="e2lora", e2lora_backbone="vit_base_patch16_224.augreg2_in21k_ft_in1k",
                          train_merge="none", train_drift=False, train_prefix=f"e13_fs{k}")
                t, i = sweep(ds, "e13_fs5_e2lora", rule="none", cached=False, features={"data": True}, extra_config=e2,
                             base_acc=BASE[ds] - drop - 2.0)
                put(f"e13_{ds}_fs5_e2lora", ds, "e13_fs5_e2lora", t, i, k=k, method="e2lora")
    t, i = sweep("cifar224", "nomerge_full", rule="none", base_acc=BASE["cifar224"] - 3.0)
    put("nomerge_full_cifar", "cifar224", "nomerge_full", t, i)
    # typed references; n_seeds (optional column) is given for the first row only, so the
    # parser's shots='full' row has n_seeds 3 there and an empty n_seeds for the second
    refs = [["e2lora", "cifar224", "90.86", "93.90", "4.10", "RESULTS.md s2 re-run", "3"],
            ["e2lora", "imagenetr", "80.40", "85.10", "4.90", "RESULTS.md s2 re-run", ""],
            ["sgplora", "cifar224", "89.90", "93.10", "4.50", "RESULTS.md s2 ported recipe", ""]]
    fx["refs"] = {"path": write_csv_rows(os.path.join(fig_dir, "e13_refs.csv"),
                                         ["method", "dataset", "fa", "aa", "ffm", "source", "n_seeds"], refs)}

    # ---- E14: validation grid on CUB (alpha x lambda, val_frac 0.1) + full-data headline --------
    for a_ in (0.1, 0.3):
        for l_ in (10.0, 100.0):
            name = f"e14_val10_a{a_:g}_l{l_:g}"
            crash = 1994 if (a_ == 0.3 and l_ == 100.0) else None
            t, i = sweep("cub", name, alpha=a_, lam=l_, with_val=True, val_frac=0.1, crash_seed=crash, crash_at_task=4,
                         features={"data": True}, extra_config={"data_val_seed": None, "train_prefix": "main_cub_r64_qkvfc_val10",
                                       "checkpoint_dir": "checkpoints_main/rebuttal/e14"},
                         base_acc=BASE["cub"] - 0.4 - 2.0 * abs(a_ - 0.1) - (0.3 if l_ == 100.0 else 0.0))
            put(f"e14_cub_a{a_}_l{l_}", "cub", name, t, i, alpha=a_, lam=l_)
    t, i = sweep("cub", "best_cub", prefix="best_cub_r64")
    put("best_cub", "cub", "best_cub", t, i)

    # ---- E15: OPCM / NUFILT stage 1 on ImageNet-R, stage 2 on CIFAR; InfLoRA logs ---------------
    for a_, lam_ in ((0.5, "adaptive"), (0.7, "sqrt")):
        t, i = sweep("imagenetr", f"e15_opcm_a{a_}_{lam_}", rule="opcm", features={"sota_state": True, "mergestat": True},
                     extra_config={"merge_opcm_alpha": a_, "merge_opcm_lam": lam_, "merge_log_stats": True},
                     base_acc=80.0 - (0.5 if lam_ == "sqrt" else 0.0))
        put(f"e15_opcm_{a_}_{lam_}", "imagenetr", f"e15_opcm_a{a_}_{lam_}", t, i)
    for rp, ad in ((128, False), (32, True)):
        t, i = sweep("imagenetr", f"e15_nufilt_rp{rp}_ad{int(ad)}", rule="nufilt", features={"sota_state": True},
                     extra_config={"merge_nufilt_rp": rp, "merge_nufilt_adapt": ad}, base_acc=79.0 + (0.4 if ad else 0.0))
        put(f"e15_nufilt_{rp}_{int(ad)}", "imagenetr", f"e15_nufilt_rp{rp}_ad{int(ad)}", t, i)
    t, i = sweep("cifar224", "e15_opcm_a0.5_adaptive", rule="opcm", features={"sota_state": True},
                 extra_config={"merge_opcm_alpha": 0.5, "merge_opcm_lam": "adaptive"}, base_acc=89.0)
    put("e15_opcm_cifar", "cifar224", "e15_opcm_a0.5_adaptive", t, i)
    t, i = make_inflora_log("cifar224", "inflora_in21k_cifar224", failed_seed=1994, lamb=0.95, epochs=20, base_acc=90.0)
    put("inflora_cifar", "cifar224", "inflora_in21k_cifar224", t, i)
    t, i = make_inflora_log("imagenetr", "inflora_in21k_imagenetr", lamb=0.98, epochs=50, base_acc=79.0, matrix_seed=31,
                            start="2026-11-21 08:00:00")
    put("inflora_inr", "imagenetr", "inflora_in21k_imagenetr", t, i)
    t, i = make_keeplora_log("cifar224", "keeplora_matched_cifar224", failed_seed=1994, base_acc=89.0)
    put("keeplora_cifar", "cifar224", "keeplora_matched_cifar224", t, i)
    t, i = make_keeplora_log("imagenetr", "keeplora_matched_imagenetr", base_acc=78.0, matrix_seed=33,
                             start="2026-11-23 08:00:00", saturate=True)
    put("keeplora_inr", "imagenetr", "keeplora_matched_imagenetr", t, i)
    return fx


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="log root to write into (logs_fake/...)")
    p.add_argument("--dataset", default="cifar224")
    p.add_argument("--config", default="fake_cfg")
    p.add_argument("--init", type=int, default=None)
    p.add_argument("--inc", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--lam", type=float, default=100.0)
    p.add_argument("--p", type=float, default=0.9)
    p.add_argument("--rule", default="pspectral")
    p.add_argument("--val", action="store_true")
    p.add_argument("--crash-seed", type=int, default=None)
    p.add_argument("--repeat-seed", type=int, nargs="*", default=[])
    p.add_argument("--uncached", action="store_true")
    p.add_argument("--tags", action="store_true")
    p.add_argument("--stop-at-task", type=int, default=None)
    p.add_argument("--sota", action="store_true", help="write a Configuration-only sota_merge.log")
    p.add_argument("--tree", action="store_true", help="write the whole test fixture tree")
    p.add_argument("--rebuttal-tree", action="store_true", help="write the rebuttal fixture tree (needs --fig-dir)")
    p.add_argument("--fig-dir", default=None, help="side-CSV directory for --rebuttal-tree")
    args = p.parse_args(argv)
    if args.tree:
        fx = build_fixture_tree(args.out)
        for k, v in fx.items():
            print(f"{k:14s} -> {v['path']}")
        return 0
    if args.rebuttal_tree:
        fx = build_rebuttal_tree(args.out, args.fig_dir or os.path.join(args.out, "figures_fake"))
        for k, v in fx.items():
            print(f"{k:28s} -> {v['path']}")
        return 0
    if args.sota:
        text, _ = make_sota_log([{"ds": "imagenetr", "name": "sota_dare_q0.5_c1.0",
                                  "overrides": {"train_merge": "dare", "merge_dare_q": 0.5}}])
        print(write_text(os.path.join(args.out, "sota_merge.log"), text))
        return 0
    text, _ = make_sweep_log(args.dataset, args.config, seeds=args.seeds, init=args.init, inc=args.inc,
                             epochs=args.epochs, alpha=args.alpha, lam=args.lam, p=args.p, rule=args.rule,
                             with_val=args.val, crash_seed=args.crash_seed, repeat_seeds=args.repeat_seed,
                             cached=not args.uncached, tags=args.tags, stop_at_task=args.stop_at_task)
    print(write_text(os.path.join(args.out, args.dataset, f"{args.config}.log"), text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
