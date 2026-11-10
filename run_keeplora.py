"""KeepLoRA baseline rows for the rebuttal (E15b): the KeepLoRA port under our
protocol, summarised in task-equal FA / AA / FF and in micro ACC / AVG.

KeepLoRA (Luo et al., "KeepLoRA: Continual Learning with Residual Gradient
Adaptation", ICLR 2026, arXiv:2601.19659; code github.com/MaolinLuo/KeepLoRA,
branch v1, folder MTIL/) is ported in backbones.KeepLoRAViT + keeplora_subspace.py
and driven by main.py's train_method "keeplora" (RECIPE_PRESETS["keeplora"]:
AdamW 1e-3, weight decay 0.01, batch 64, 10 epochs, OneCycleLR per batch, rank
16 on q/k/v + proj of all 12 blocks, eps_w 0.85, eps_f 0.99, gradient-SVD
initialisation of A (frozen) and B (trained), class-mean head init, native
biased per-task heads with local CE, no Stage 2). Our protocol: class order
shuffled by the DataManager with seeds 1993/1994/1995, 10-task splits (CIFAR-100
10x10, ImageNet-R 10x20, CUB 10x20, Cars 16+9x20), the fixed Zhou ImageNet-R
split.

KeepLoRA publishes no class-incremental ViT numbers (its benchmarks are MTIL
with CLIP ViT-B/16 and two LLaVA streams, each task scored inside its own label
set), so there is no parity gate: the main stage runs the released recipe on
OUR matched backbone (vit_base_patch16_224, augreg2 IN21k -> IN1k, the backbone
of every row of Tab. 1 / Tab. 3), and the in21k stage repeats it on the
IN21k-only weights that E2-LoRA and InfLoRA report. Two released-code details
differ from the paper text and are exposed as switches so the grid stage can
show they do not drive the result: the weight subspace W_p is built from the
LEFT singular vectors of the torch-layout weight (keeplora_wp_side 'output';
'input' = paper Eq. 2) and the feature statistic is the element-wise mean of the
per-batch token matrices (keeplora_feat_mode 'code'; 'gram' = all tokens).

Stages (--stage):
    smoke    cifar224, first seed, 2 tasks x 1 epoch, logs under logs_main/smoke;
             run it twice: the second run must reuse both cached tasks and log
             identical 'Total Acc' lines (fold + basis round-trip); never skipped
    main     the four datasets on our matched backbone, 3 seeds          [default]
    in21k    the four datasets on vit_base_patch16_224_in21k, 3 seeds (optional)
    grid     ImageNet-R and CIFAR-100, first seed, one switch changed per cell:
             eps_w 0.5 | eps_w 0 (no weight subspace, upstream ablation iii) |
             eps_f 0.95 (the 0.99 basis reached 614 / 768 columns after one
             ImageNet-R task in the smoke run) | wp_side input | feat_mode gram |
             b_init zero (KeepLoRA++ branch) | lr 5e-4 | head_init default
             (8 variants x 2 datasets = 16 cells; --variants picks a subset)
    all      main, grid, in21k in that order

Every cell logs to logs_main/<dataset>/<config>.log (per-seed 'Experiment:'
headers, 'Time:' and the SUMMARY block in run_config_sweep's format, so
parse_logs.py and rebuttal_common.has_summary read it) and to
logs_main/keeplora_campaign.log, and ends with
    SWEEP SUMMARY [keeplora-<ds>] <config>: FA m ± s | AA m ± s | FF m ± s |
        micro ACC m ± s | AVG m ± s | seeds [...] | failed [...]
Failed seeds (an '[Experiment ...]' error, empty curve) are logged as FAILED and
excluded from every mean. train_prefix encodes backbone, dataset, rank, both
thresholds, epochs, lr, the four switches, proj flag, layers and batch size, so
no variant can hit another variant's cache under checkpoints_main/keeplora;
main.g is re-seeded before every run, so a (dataset, seed) cell does not depend
on what ran before it in the same process. A finished cell is skipped on
relaunch; a partial log needs --redo (backup + rerun) or --append (missing
seeds only).

Cost per seed (estimate, one RTX 4090; 10 epochs plus two extra no-backward /
one-backward passes per task): CIFAR-100 ~0.8 h, ImageNet-R ~0.5 h, CUB ~0.3 h,
Cars ~0.3 h; main 6 GPU-h, grid 10 GPU-h, in21k 6 GPU-h. About 115 MB of cached
state per task (folded attention weights + 48 basis buffers of 768 x 768).

Usage:
    python run_keeplora.py --dry-run                     # main cells; no torch needed
    python run_keeplora.py --stage all --dry-run
    python run_keeplora.py --stage smoke                 # run twice (cache round-trip)
    python run_keeplora.py                               # main stage
    python run_keeplora.py --ds imagenetr --seeds 1993
    python run_keeplora.py --stage grid --variants epsw0 featgram
    python run_keeplora.py --stage in21k
"""
import argparse
import logging
import os
import pprint
import sys
import time

import numpy as np

import rebuttal_common as rc   # torch-free: paths, log inspection, CELLS line

SEEDS = tuple(rc.SEEDS)                                  # (1993, 1994, 1995)

# timm names: our matched backbone (augreg2_in21k_ft_in1k, the default of every
# other recipe) and the IN21k-only weights (timm 1.0.x alias of
# vit_base_patch16_224.augreg_in21k) that E2-LoRA / InfLoRA report on.
BACKBONES = {"matched": "vit_base_patch16_224", "in21k": "vit_base_patch16_224_in21k"}

# The released recipe (MTIL/configs/keeplora_order1.yaml + trainer.py).
# RECIPE_PRESETS["keeplora"] in main.py holds the same values; they are
# repeated here so the Configuration dump and the cache prefix show every
# setting explicitly.
RECIPE = {
    "lora_r": 16,
    "train_base_lr": 1e-3,
    "train_batch_size": 64,
    "train_epochs": 10,
    "keeplora_eps_w": 0.85,
    "keeplora_eps_f": 0.99,
    "keeplora_wp_side": "output",
    "keeplora_feat_mode": "code",
    "keeplora_b_init": "svd",
    "keeplora_head_init": "ncm",
    "keeplora_proj": True,
    "keeplora_layers": None,
    "train_merge": "none",
}
DATASETS = ("cifar224", "imagenetr", "cub", "cars")

STAGES = ("smoke", "main", "in21k", "grid")
ALL_ORDER = ("main", "grid", "in21k")
STAGE_DATASETS = {
    "smoke": ["cifar224"],
    "main": list(DATASETS),
    "in21k": list(DATASETS),
    "grid": ["imagenetr", "cifar224"],
}
STAGE_BACKBONE = {"smoke": "matched", "main": "matched", "in21k": "in21k", "grid": "matched"}
FIRST_SEED_ONLY = ("smoke", "grid")
# grid: (tag, overrides); one switch per cell relative to the main recipe
VARIANTS = [
    ("epsw0.5", {"keeplora_eps_w": 0.5}),
    ("epsw0", {"keeplora_eps_w": 0.0}),
    ("epsf0.95", {"keeplora_eps_f": 0.95}),
    ("wpinput", {"keeplora_wp_side": "input"}),
    ("featgram", {"keeplora_feat_mode": "gram"}),
    ("bzero", {"keeplora_b_init": "zero"}),
    ("lr5e-4", {"train_base_lr": 5e-4}),
    ("headdefault", {"keeplora_head_init": "default"}),
]
VARIANT_TAGS = [t for t, _ in VARIANTS]

# Learner._ckpt_dir default for a merge-family method is checkpoints_main/<method>;
# passed explicitly so every cell (and the dry-run) names its directory. Smoke
# states get their own directory for the day-0 cleanup.
CKPT_DIR = os.path.join(rc.CHECKPOINT_ROOT, "keeplora")
SMOKE_CKPT_DIR = os.path.join(CKPT_DIR, "smoke")
CAMPAIGN_LOG = "keeplora_campaign.log"
LOG_FORMAT = "%(asctime)s [%(filename)s] => %(message)s"   # main.run_config_sweep


# ----------------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------------

def train_prefix(bb, ds, c):
    """Cache key of a variant: backbone, dataset, rank, thresholds, epochs, lr,
    the four switches, proj flag, layers and batch size."""
    layers = c.get("keeplora_layers")
    lay = "all" if layers is None else "-".join(str(i) for i in layers)
    return (f"keeplora_{bb}_{ds}_r{c['lora_r']}_w{c['keeplora_eps_w']:g}_f{c['keeplora_eps_f']:g}"
            f"_e{c['train_epochs']}_lr{c['train_base_lr']:g}_s{c['keeplora_wp_side'][:3]}"
            f"_m{c['keeplora_feat_mode']}_b{c['keeplora_b_init']}_h{c['keeplora_head_init']}"
            f"_p{int(bool(c['keeplora_proj']))}_L{lay}_bs{c['train_batch_size']}")


def build_cfg(ds, bb, smoke=False, **over):
    """Experiment config of one cell (merged over RECIPE_PRESETS['keeplora'] by
    main.run_single_experiment). `over` holds the grid / triage overrides."""
    c = {"train_method": "keeplora", "keeplora_backbone": BACKBONES[bb]}
    c.update(RECIPE)
    c.update(over)
    if smoke:
        c.update({"train_stop_at_task": 1, "train_epochs": 1})
    c["train_prefix"] = train_prefix(bb, ds, c) + ("_smoke" if smoke else "")
    c.update({"cache_backbone": True, "reset_train": False,
              "checkpoint_dir": SMOKE_CKPT_DIR if smoke else CKPT_DIR})
    return c


def variant_suffix(over):
    """Config-name suffix of the triage overrides (new name = new log file)."""
    s = ""
    if "train_batch_size" in over:
        s += f"_bs{over['train_batch_size']}"
    return s


def stage_cells(stage, datasets, bb, over, variants):
    """[(ds, config_name, cfg)] of one stage."""
    bb = bb or STAGE_BACKBONE[stage]
    cells = []
    for ds in datasets:
        if stage == "smoke":
            name = f"keeplora_{bb}_{ds}{variant_suffix(over)}_smoke"
            cells.append((ds, name, build_cfg(ds, bb, smoke=True, **over)))
        elif stage == "grid":
            for tag, vo in VARIANTS:
                if variants and tag not in variants:
                    continue
                name = f"keeplora_{bb}_{ds}_grid_{tag}{variant_suffix(over)}"
                cells.append((ds, name, build_cfg(ds, bb, **vo, **over)))
        else:
            name = f"keeplora_{bb}_{ds}{variant_suffix(over)}"
            cells.append((ds, name, build_cfg(ds, bb, **over)))
    return cells


def chain_files(ds, cfg, seed):
    """Cached files of one seed: Learner.prefix() is {seed}_{ds}_keeplora_{train_prefix}
    and a merge-family task leaves _state_{t}.pt and _classifier_{t}.pt."""
    stem = f"{seed}_{ds}_keeplora_{cfg['train_prefix']}"
    T = rc.n_tasks(ds, cfg)
    stop = cfg.get("train_stop_at_task", -1)
    if stop is not None and stop >= 0:
        T = min(T, stop + 1)
    return [os.path.join(cfg["checkpoint_dir"], f"{stem}_{kind}_{t}.pt")
            for t in range(T) for kind in ("state", "classifier")]


def cache_summary(ds, cfg, seeds):
    done = sum(1 for s in seeds if all(os.path.isfile(f) for f in chain_files(ds, cfg, s)))
    return f"{done}/{len(seeds)}"


# ----------------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------------

def dry_run_cell(ds, name, cfg, seeds, log_root):
    """One '[dry-run]' line in rebuttal_common.launch's format plus the config."""
    lp = rc.log_path(ds, name, log_root)
    status = "done" if rc.has_summary(lp, name, len(seeds)) else "todo"
    print(f"[dry-run] ds={ds} name={name} tag={rc.merge_tag(cfg)} prefix={cfg['train_prefix']} "
          f"ckpt={cfg['checkpoint_dir']} log={lp} status={status} "
          f"cache={cache_summary(ds, cfg, seeds)} seeds={','.join(str(s) for s in seeds)} "
          "(train_merge=none: no merged checkpoints)")
    for line in pprint.pformat(cfg, width=100, sort_dicts=True).splitlines():
        print("    " + line)


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def _pm(values):
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"     # std ddof 0 like SUMMARY


def _setup_logging(ds, name, log_root):
    os.makedirs(os.path.join(log_root, ds), exist_ok=True)     # run_config_sweep is bypassed
    logging.basicConfig(
        level=logging.INFO, format=LOG_FORMAT, force=True,
        handlers=[logging.FileHandler(rc.log_path(ds, name, log_root), encoding="utf-8"),
                  logging.FileHandler(os.path.join(log_root, CAMPAIGN_LOG), encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def run_cell(ds, name, cfg, seeds, log_root):
    """Run every seed of one cell through main.run_single_experiment and write
    the SUMMARY block plus the SWEEP SUMMARY line. Returns (ok_seeds, failed)."""
    import main   # lazy: torch, timm, peft
    _setup_logging(ds, name, log_root)
    ok, failed = [], []
    fa, aa, ff, acc, avg = [], [], [], [], []
    for seed in seeds:
        logging.info("\n" + "=" * 80)
        logging.info(f"Experiment: {ds} — {name} — seed {seed}")
        # Every DataLoader draws from main.g (seeded once at import): re-seed
        # it so the cell does not depend on the experiments run before it.
        main.g.manual_seed(0)
        t0 = time.time()
        r = main.run_single_experiment(ds, name, cfg, seed)
        logging.info(f"Time: {time.time() - t0:.1f}s")
        curve = list(r.get("total_curve") or [])
        if not curve or r.get("faa", 0.0) == 0.0:
            # run_single_experiment caught an exception (its '[Experiment ...]'
            # error line is above) and returned zeros: exclude, never average.
            failed.append(seed)
            logging.error(f"[FAILED] {ds} — {name} — seed {seed}: no result "
                          "(see the [Experiment ...] error above); excluded from the summary")
            continue
        ok.append(seed)
        fa.append(r["faa"])
        aa.append(r["asa"])
        ff.append(r["ffm"])
        acc.append(curve[-1])                 # last micro Total Acc
        avg.append(float(np.mean(curve)))     # mean of the curve
        logging.info(f"[KeepLoRA] seed {seed}: FA {r['faa']:.2f} | AA {r['asa']:.2f} | "
                     f"FF {r['ffm']:.2f} | micro ACC {curve[-1]:.2f} | AVG {np.mean(curve):.2f} | "
                     f"curve {[round(v, 2) for v in curve]}")

    # SUMMARY block in run_config_sweep's format (parse_logs.py and
    # rebuttal_common.has_summary read it); means over the seeds that finished.
    logging.info("\n" + "=" * 80)
    logging.info(f"SUMMARY  {ds.upper()} — {name.upper()}")
    logging.info("=" * 80)
    if ok:
        logging.info(f"  ASA: {_pm(aa)} | FAA: {_pm(fa)} | FFM: {_pm(ff)}"
                     + (f" | n={len(ok)}/{len(seeds)}" if failed else ""))
        logging.info(f"SWEEP SUMMARY [keeplora-{ds}] {name}: FA {_pm(fa)} | AA {_pm(aa)} | "
                     f"FF {_pm(ff)} | micro ACC {_pm(acc)} | AVG {_pm(avg)} | "
                     f"seeds {ok} | failed {failed}")
    else:
        # No 'SWEEP SUMMARY' prefix here: parse_logs.SWEEP_RE must not read an empty cell.
        logging.error(f"[FAILED] [keeplora-{ds}] {name}: every seed failed {failed}; no summary")
    logging.info("=" * 80 + "\n")
    return ok, failed


def _backup_log(path):
    """Rename a partial log to <path>.bak<N> (never deleted)."""
    n = 1
    while os.path.exists(f"{path}.bak{n}"):
        n += 1
    os.replace(path, f"{path}.bak{n}")
    return f"{path}.bak{n}"


def resolve_seeds(stage, ds, name, seeds, log_root, redo, append):
    """Resume rule of rebuttal_common.launch: a finished clean cell is skipped,
    a partial log needs --redo or --append. Smoke cells always run (the
    cache round-trip is checked by running the same cell twice)."""
    lp = rc.log_path(ds, name, log_root)
    if stage == "smoke":
        return list(seeds)
    if rc.has_summary(lp, name, len(seeds)) and not redo:
        print(f"[skip] {ds} | {name}: complete for {len(seeds)} seed(s) in {lp}")
        return []
    if os.path.isfile(lp) and os.path.getsize(lp) > 0:
        if redo:
            print(f"[redo] {ds} | {name}: moved the previous log to {_backup_log(lp)}")
        elif append:
            status = rc.seed_status(lp, name)
            missing = [s for s in seeds if status.get(s) != "ok"]
            if not missing:
                print(f"[skip] {ds} | {name}: every seed is ok in {lp}")
                return []
            print(f"[append] {ds} | {name}: seeds {missing} appended to {lp}; status was {status}")
            return missing
        else:
            raise SystemExit(
                f"{lp} exists but is not a complete clean run of {name} "
                f"(seed status {rc.seed_status(lp, name)}); pass --redo to back it up and "
                "rerun, or --append to run only the missing seeds")
    return list(seeds)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="KeepLoRA baseline campaign (rebuttal E15b); see the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ("all",), default="main",
                    help="smoke | main (default) | in21k | grid | all")
    ap.add_argument("--smoke", action="store_true", help="alias for --stage smoke")
    ap.add_argument("--datasets", "--ds", nargs="+", default=None,
                    help="restrict the stage to these datasets")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                    help="seeds (default 1993 1994 1995; smoke and grid use the first one)")
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default=None,
                    help="override the stage's backbone (matched = ours, in21k = IN21k-only)")
    ap.add_argument("--variants", nargs="+", choices=VARIANT_TAGS, default=None,
                    help="grid stage: run only these variants")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="train_batch_size override (OOM fallback 32; own name and prefix)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every cell and its config; never imports main (no torch)")
    ap.add_argument("--redo", action="store_true",
                    help="back up an existing log (.bakN) and rerun every seed")
    ap.add_argument("--append", action="store_true",
                    help="append only the seeds that are not ok to a partial log")
    ap.add_argument("--log-root", default=None,
                    help="log root (default logs_main; smoke runs use logs_main/smoke)")
    args = ap.parse_args(argv)
    if args.smoke:
        args.stage = "smoke"
    stages = list(ALL_ORDER) if args.stage == "all" else [args.stage]

    over = {}
    if args.batch_size is not None:
        over["train_batch_size"] = args.batch_size

    if args.datasets:
        unknown = [d for d in args.datasets if d not in DATASETS]
        if unknown:
            raise SystemExit(f"unknown dataset(s) {unknown}; known: {sorted(DATASETS)}")

    cells = []   # (stage, ds, name, cfg, seeds, log_root)
    for stage in stages:
        datasets = STAGE_DATASETS[stage]
        if args.datasets:
            datasets = [d for d in args.datasets if d in datasets] if args.stage == "all" \
                else list(args.datasets)
        seeds = args.seeds[:1] if stage in FIRST_SEED_ONLY else list(args.seeds)
        log_root = args.log_root or (rc.SMOKE_LOG_ROOT if stage == "smoke" else rc.LOG_DIR)
        for ds, name, cfg in stage_cells(stage, datasets, args.backbone, over, args.variants):
            cells.append((stage, ds, name, cfg, seeds, log_root))
    names = [(ds, name) for _, ds, name, _, _, _ in cells]
    assert len(set(names)) == len(names), f"duplicate cells: {names}"

    if args.dry_run:
        for _, ds, name, cfg, seeds, log_root in cells:
            dry_run_cell(ds, name, cfg, seeds, log_root)
        rc.print_cells(len(cells))
        return 0

    t0 = time.time()
    outcome = {}
    for stage, ds, name, cfg, seeds, log_root in cells:
        run_seeds = resolve_seeds(stage, ds, name, seeds, log_root, args.redo, args.append)
        if not run_seeds:
            outcome[(ds, name)] = "done"
            continue
        print(f"[launch] {stage} | {ds} | {name} | seeds {run_seeds} | ckpt {cfg['checkpoint_dir']} "
              f"| log {rc.log_path(ds, name, log_root)}")
        ok, failed = run_cell(ds, name, cfg, run_seeds, log_root)
        outcome[(ds, name)] = f"ok {ok}" + (f" FAILED {failed}" if failed else "")
    for key, val in outcome.items():
        print(f"{key[0]} | {key[1]}: {val}")
    print(f"Total: {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
