"""Few-shot per task (rebuttal E13): ISM against its in-framework controls and
two ported baselines when every task offers only K training images per class.

Scenario. Class-incremental CIFAR-100 (10 x 10) and ImageNet-R (10 x 20) with
K in {5, 20} shots per class and task. The DataManager hook `data_fewshot_k`
keeps min(K, n_c) training images per class, drawn with a private
np.random.RandomState(run seed); the test set, the class order and the task
split are untouched, so FA / AA / FF keep their meaning. ImageNet-R has classes
with fewer than 20 training images, so K = 20 is a cap there: every run logs
one '[Data] ...' line with the realised per-class min/median/max -- quote the
median in the rebuttal. Headline posture per dataset, no retuning.

Rows per (dataset, K). The first three share one LoRA chain, so `ism` is
launched first and `nomerge` / `avg` are served from its cache:
  ism      rebuttal_common.cfg(ds): pspectral only_A, p 0.9, alpha 0.15 / 0.30,
           lambda 100, local-term weight 0.2 / 0.1, weight decay 5e-4 / 2e-4
  nomerge  same chain, train_merge none (the deployed extractor is W_t)
  avg      same chain, Task Arithmetic / averaging: helper.merge 'avg' on the
           incremental chain, coefficient 1.0 (Tab. 6 and E12 setting)
  e2lora   E2-LoRA: train_method e2lora with RECIPE_PRESETS['e2lora'] (lr 5e-4,
           head 1e-2, 5 epochs, batch 64, constant schedule, biased heads, no
           feature norm), backbone pinned to ours as in the RESULTS.md section-2
           re-run (vit_base_patch16_224.augreg2_in21k_ft_in1k), its native
           classifier stage (train_ca_impl 'e2'), no drift map, no merged-A.
           Confirm the keys against the server's earlier E2-LoRA log with
           --diff-against <log> before launching (torch-free).
  sgplora  SGP-LoRA (ported recipe): train_method sgp_lora (DoRA-projected LoRA
           r = 4 on qkv / proj / fc1 / fc2, soft projection) under our training
           loop (SGD 1e-2, 10 epochs, batch 64, the dataset's weight decay) plus
           the same Stage 2 as ISM (drift map, LCA with the dataset's lambda and
           local-term weight). Label the row "SGP-LoRA (ported recipe) + Stage 2".
Every row goes through the same DataManager, so all of them see the identical
reduced training set.  --full-ref adds the baselines' full-data reference cells
(e13_full_e2lora, e13_full_sgplora) so their degradation is like-for-like; the
ISM-family references are the headline chain (Tab. 1 / E8 same-chain cell).
eval_til (default on, --no-eval-til) adds the zero-cost task-incremental line
'[Evaluation-til]' to every row (q6ZZ-W4).

Chains and files. ISM rows: train_prefix <prefix>_fs<K> (main_cifar224_r64_qkvfc_fs5,
...); baseline rows: train_prefix e13_fs<K> (their file stems carry the recipe
name, e.g. 1993_cifar224_e2lora_e13_fs5_e2state_3.pt). Everything lives in the
fresh directory checkpoints_main/rebuttal/e13 (plan, ground rule 3): no link to
the shared cache, so the merged-checkpoint cleanup and the base-file rewrite
cannot touch another job. One process per directory: run this driver once over
both datasets (cells run sequentially); for two parallel processes give each
its own --ckpt-dir. The SGP-LoRA row does not cache its states by default
(--sgp-cache): its covariance EMA cache costs ~0.5 GB per task and nothing
reuses the chain (few-shot training is seconds per task).

Logs: logs_main/<dataset>/e13_fs<K>_<row>.log in run_config_sweep's format
(parse_logs.py e13 -> figures_data/e13_fewshot_<ds>.csv; figure
scripts/fig_e13_fewshot.py). Smoke: --smoke (2 tasks x 1 epoch, first seed,
`_smoke` suffix on name, prefix and directory, logs under logs_main/smoke).
Resume: a finished cell is skipped; a partial log needs --redo or --append.

Cost (plan estimate, re-cost from the E2 time table): training is seconds per
task, so every run costs about the cached Stage-2 figure (~10 min per seed):
60 runs ~ 10 GPU-h; K = 5 only ~ 5 GPU-h; --full-ref adds 12 full-data baseline
trainings.

Usage:
    python run_e13_fewshot.py --dry-run                 # 20 cells; no torch needed
    python run_e13_fewshot.py --smoke                   # server, first GPU minutes
    python run_e13_fewshot.py --diff-against logs_main/cifar224/<e2lora log>.log
    python run_e13_fewshot.py --k 5                     # minimum viable regime first
    python run_e13_fewshot.py --k 20
    python run_e13_fewshot.py --full-ref --rows e2lora sgplora
"""
import argparse
import os
import sys
import time
from collections import OrderedDict

import rebuttal_common as rc   # torch-free: posture, launcher, resume rule

E13_DATASETS = ("cifar224", "imagenetr")
KS = (5, 20)
ROWS = ("ism", "nomerge", "avg", "e2lora", "sgplora")
CACHED_ROWS = ("ism", "nomerge", "avg")        # one seqft chain per (dataset, K)
BASELINE_ROWS = ("e2lora", "sgplora")          # merge-family recipes, own chains
CKPT_DIR = rc.isolated_ckpt_path("e13")        # checkpoints_main/rebuttal/e13

# E2-LoRA backbone pinned to ours (RESULTS.md section 2); the preset default
# 'vit_base_patch16_224_in21k' is the published pure-IN21K tag.
E2_BACKBONE = "vit_base_patch16_224.augreg2_in21k_ft_in1k"

# Task Arithmetic / averaging exactly as the Tab. 6 row and the E12 grid cells:
# helper.merge 'avg' on the incremental chain, coefficient 1.0.
AVG_KEYS = {"train_merge": "avg", "train_merge_incremental": True,
            "train_merge_coef": 1.0, "train_merge_topk": 100}

# Stage-2 keys copied from the ISM posture onto the SGP-LoRA row.
STAGE2_KEYS = ("train_ca", "train_drift", "train_drift_ridge", "train_drift_lambda",
               "train_drift_transport", "train_ca_cov_shrinkage", "train_ca_robust_weight",
               "train_ca_n_classes", "train_ca_k_samples")

# Merge-family cache files per task (Learner.e2_checkpoint / state_checkpoint).
STATE_KIND = {"e2lora": "e2state", "sgp_lora": "state"}


# ----------------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------------

def shots_token(k):
    """'fs5' for K = 5, 'full' for the full-data reference (k None)."""
    return "full" if k is None else f"fs{int(k)}"


def cell_name(k, row):
    return f"e13_{shots_token(k)}_{row}"


def build_cell(ds, k, row, ckpt_dir=CKPT_DIR, eval_til=True, sgp_cache=False):
    """Experiment config of one row (merged over BASE_CONFIG and the recipe
    preset by main.run_single_experiment)."""
    common = {"checkpoint_dir": ckpt_dir}
    if k is not None:
        common["data_fewshot_k"] = int(k)     # DataManager(fewshot_k=...), run-seed draw
    if eval_til:
        common["eval_til"] = True             # '[Evaluation-til]' line, zero cost
    if row in CACHED_ROWS:
        over = dict(common, train_prefix=f"{rc.prefix(ds)}_{shots_token(k)}")
        if row == "nomerge":
            over["train_merge"] = "none"
        elif row == "avg":
            over.update(AVG_KEYS)
        return rc.cfg(ds, **over)
    if row == "e2lora":
        return dict(common,
                    train_method="e2lora",
                    e2lora_backbone=E2_BACKBONE,
                    train_prefix=f"e13_{shots_token(k)}",
                    cache_backbone=True, reset_train=False,
                    train_merge="none",            # preset value, stated for the tag
                    train_ca=True, train_ca_impl="e2",   # their native classifier stage
                    train_drift=False,
                    e2_merge=False)                # plain E2-LoRA, no merged-A
    if row == "sgplora":
        posture = rc.cfg(ds)                       # Stage-2 source: the ISM posture
        c = dict(common,
                 train_method="sgp_lora",
                 train_prefix=f"e13_{shots_token(k)}",
                 cache_backbone=bool(sgp_cache), reset_train=False,
                 train_merge="none",               # merge family folds its adapter
                 train_weight_decay=posture["train_weight_decay"])
        c.update({key: posture[key] for key in STAGE2_KEYS})
        return c
    raise ValueError(f"unknown row {row!r}; rows: {ROWS}")


def build_cells(datasets, ks, rows, full_ref=False, **kw):
    """[(ds, config_name, cfg)] in launch order: per dataset and K the ISM row
    first (it trains the shared chain), then its cached controls, then the
    baselines; the optional full-data baseline references close the dataset."""
    rows = [r for r in ROWS if r in rows]           # canonical order
    cells = []
    for ds in datasets:
        for k in ks:
            for row in rows:
                cells.append((ds, cell_name(k, row), build_cell(ds, k, row, **kw)))
        if full_ref:
            for row in rows:
                if row in BASELINE_ROWS:
                    cells.append((ds, cell_name(None, row), build_cell(ds, None, row, **kw)))
    names = [(ds, name) for ds, name, _ in cells]
    assert len(set(names)) == len(names), f"duplicate cells: {names}"
    return cells


# ----------------------------------------------------------------------------
# Dry-run extras (the launcher's cache field assumes the seqft file stem)
# ----------------------------------------------------------------------------

def baseline_cache(ds, cfg, seeds):
    """'k/n' seeds whose merge-family chain is complete under the cell's
    checkpoint_dir: {seed}_{ds}_{method}_{prefix}_{e2state|state}_{t}.pt and
    _classifier_{t}.pt for every task (Learner.prefix for non-seqft recipes)."""
    method = cfg["train_method"]
    kind = STATE_KIND[method]
    T = rc.n_tasks(ds, cfg)
    d = cfg["checkpoint_dir"]
    done = 0
    for seed in seeds:
        stem = f"{seed}_{ds}_{method}_{cfg['train_prefix']}"
        if all(os.path.isfile(os.path.join(d, f"{stem}_{k}_{t}.pt"))
               for t in range(T) for k in (kind, "classifier")):
            done += 1
    return f"{done}/{len(seeds)}"


def print_dry_extras(ds, configs, seeds):
    for name, cfg in configs.items():
        if cfg.get("train_method", "seqft") in STATE_KIND:
            print(f"[e13-cache] ds={ds} name={name} chain={baseline_cache(ds, cfg, seeds)} "
                  f"(merge-family stem {'{seed}'}_{ds}_{cfg['train_method']}_{cfg['train_prefix']})")


# ----------------------------------------------------------------------------
# Preset check against a server log (torch-free)
# ----------------------------------------------------------------------------

_DIFF_IGNORE = {"seed", "dataset_name", "dataset_num_task", "dataset_init_cls", "dataset_increment",
                "train_prefix", "checkpoint_dir", "eval_til", "cache_backbone", "reset_train",
                "data_fewshot_k", "data_val_frac", "data_val_seed", "data_subsample_seed"}
_DIFF_INTEREST = ("train_base_lr", "train_head_lr", "train_epochs", "train_batch_size",
                  "train_weight_decay", "train_scheduler", "train_optimizer", "model_use_norm",
                  "model_classifier_use_norm", "model_classifier_bias", "lora_r", "lora_scale",
                  "sgp_module", "weight_temp", "weight_kind", "weight_p", "cov_ema_decay")


def diff_against(log, cells):
    """Compare each baseline row with the 'Configuration:' dump of a server log
    of the same train_method (e.g. the RESULTS.md section-2 E2-LoRA re-run), so
    the preset can be confirmed before any GPU time. Prints the keys where the
    row differs from the dump plus the dump's recipe keys the row leaves to the
    preset. Returns 0 when nothing differs, 1 otherwise."""
    import parse_logs   # torch-free (numpy only)
    runs = parse_logs.parse_file(log)
    if not runs:
        raise SystemExit(f"{log}: no 'Experiment:' / 'Configuration:' block found")
    dump = runs[0]["config"]
    method = dump.get("train_method", "seqft")
    print(f"[diff] {log}: {runs[0].get('dataset')} / {runs[0].get('config_name')} "
          f"seed {runs[0].get('seed')} (train_method {method})")
    n_diff = 0
    seen = set()
    for ds, name, cfg in cells:
        if cfg.get("train_method", "seqft") != method or method in seen:
            continue
        seen.add(method)
        print(f"[diff] row {name} ({ds}) vs the dump:")
        for key in sorted(cfg):
            if key in _DIFF_IGNORE:
                continue
            if key not in dump:
                print(f"  {key}: ours {cfg[key]!r} | dump: absent")
                n_diff += 1
            elif dump[key] != cfg[key]:
                print(f"  {key}: ours {cfg[key]!r} | dump {dump[key]!r}")
                n_diff += 1
        extra = {k: dump[k] for k in dump
                 if k not in cfg and (k.startswith("e2") or k in _DIFF_INTEREST)}
        print(f"  recipe keys taken from the preset / BASE_CONFIG (dump values): {extra}")
    if not seen:
        print(f"[diff] no row with train_method {method!r} among the cells (--rows?)")
        return 1
    print(f"[diff] {n_diff} differing key(s)")
    return 0 if n_diff == 0 else 1


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Few-shot-per-task campaign (rebuttal E13); see the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, datasets=E13_DATASETS)
    ap.add_argument("--k", nargs="+", type=int, default=list(KS),
                    help="shots per class and task (default 5 20)")
    ap.add_argument("--rows", nargs="+", choices=ROWS, default=list(ROWS),
                    help="rows to run (default all; ism first per (dataset, K))")
    ap.add_argument("--full-ref", action="store_true",
                    help="add the baselines' full-data reference cells e13_full_<row>")
    ap.add_argument("--no-eval-til", action="store_true",
                    help="do not add eval_til=True (the '[Evaluation-til]' line)")
    ap.add_argument("--sgp-cache", action="store_true",
                    help="cache the SGP-LoRA states and covariance EMAs (~0.5 GB per task)")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR,
                    help=f"fresh checkpoint directory of this job (default {CKPT_DIR})")
    ap.add_argument("--diff-against", metavar="LOG", default=None,
                    help="compare the baseline rows with a server log's Configuration dump and exit")
    args = rc.apply_common_args(ap.parse_args(argv))
    if any(k <= 0 for k in args.k):
        raise SystemExit(f"--k must be positive integers, got {args.k}")
    if os.path.normpath(args.ckpt_dir) == os.path.normpath(rc.SEQFT_CKPT_DIR):
        raise SystemExit("--ckpt-dir must not be the shared cache checkpoints_main/seqft")
    if ("nomerge" in args.rows or "avg" in args.rows) and "ism" not in args.rows:
        print("[note] nomerge / avg without ism: the first of them trains the shared chain "
              "(same result; ism is served from that cache later)")

    cells = build_cells(args.datasets, args.k, args.rows, full_ref=args.full_ref,
                        ckpt_dir=args.ckpt_dir, eval_til=not args.no_eval_til,
                        sgp_cache=args.sgp_cache)
    if args.diff_against:
        return diff_against(args.diff_against, cells)

    kw = rc.launch_kwargs(args)
    if os.path.normpath(kw["ckpt_src"]) == os.path.normpath(rc.SEQFT_CKPT_DIR):
        kw["ckpt_src"] = args.ckpt_dir        # the few-shot chains live in the E13 directory
    if len(args.datasets) < len(E13_DATASETS) and not args.dry_run:
        print(f"[note] one process per directory: another process on {args.ckpt_dir} "
              "would share the merged-checkpoint cleanup; use --ckpt-dir for parallel runs")

    t0 = time.time()
    total = 0
    outcome = {}
    for ds in args.datasets:
        configs = OrderedDict((name, cfg) for d, name, cfg in cells if d == ds)
        if not configs:
            continue
        outcome[ds] = rc.launch(ds, configs, args.seeds, **kw)
        total += len(configs)
        if args.dry_run and not args.smoke:
            print_dry_extras(ds, configs, args.seeds)
    if args.dry_run:
        rc.print_cells(total)
        return 0
    for ds, res in outcome.items():
        for name, status in res.items():
            print(f"{ds} | {name}: {status}")
    print(f"Total: {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
