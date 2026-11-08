"""InfLoRA baseline rows for the rebuttal (E15b): the InfLoRA port under our
protocol, summarised in InfLoRA's own metrics next to ours.

InfLoRA (Liang & Li, "InfLoRA: Interference-Free Low-Rank Adaptation for
Continual Learning", CVPR 2024, arXiv:2404.00228; code
github.com/liangyanshuo/InfLoRA, MIT) is ported in backbones.InfLoRAViT +
inflora_gpm.py and driven by main.py's train_method "inflora"
(RECIPE_PRESETS["inflora"]: Adam 5e-4, weight decay 0, batch 128, per-epoch
quarter cosine, rank 10 on the K/V rows of all 12 blocks, DualGPM threshold
eps -> 1.0 over the 10 sessions, native biased per-task heads with local CE,
no Stage 2). Our protocol: class order shuffled by the DataManager with seeds
1993/1994/1995, 10-task splits (CIFAR-100 10x10, ImageNet-R 10x20, CUB 10x20,
Cars 16+9x20), the fixed Zhou ImageNet-R split, no input normalisation (as
upstream).

Published numbers (ACC_T = final micro top-1 over all seen classes = our last
'Total Acc'; AVG_T = mean of that curve; 5 trials, natural class order,
ViT-B/16 augreg_in21k):
    CIFAR-100  10x10   ACC 86.51 +- 0.73 | AVG 91.70 +- 0.32   (paper Table 2)
    ImageNet-R 10x20   ACC 75.65 +- 0.14 | AVG 80.82 +- 0.24   (paper Table 1)
Authors' repo logs with the PyCIL shuffled order at seed 1993 (the permutation
our DataManager produces): CIFAR-100 86.95 / 92.07, curve starting at 98.9;
ImageNet-R 75.97 / 82.35 (their test split is a random 80/20, ours the fixed
24k/6k split). CUB and Cars have no published InfLoRA setting and run the
ImageNet-R recipe (RESULTS.md carries a third-party CUB value 70.82 / 81.39).

Parity gate before CUB/Cars (plan E15b): Model21 repr trainable_params 192,010
(CIFAR) / 199,700 (ImageNet-R); '[InfLoRA] DualGPM' column counts
non-decreasing per 'remove' layer and <= 384; CIFAR micro ACC in [85.3, 87.7]
and AVG in [90.7, 92.7]; ImageNet-R ACC in [74.2, 77.2] and AVG in
[79.3, 82.4]; the per-epoch 'lr:' field starts at 5.000e-04 and ends at
7.854e-06. Triage if a band is missed: the '[InfLoRA] backbone ... ->' tag
line, --gram-mode test, --batch-size 64 (recorded as a deviation), head init.

Stages (--stage):
    smoke    cifar224, first seed, 2 tasks x 1 epoch, logs under logs_main/smoke;
             run it twice: the second run must reuse both cached tasks and log
             identical 'Total Acc' lines (fold + basis round-trip); never skipped
    parity   CIFAR-100 (20 epochs, eps 0.95) and ImageNet-R (50 epochs, eps 0.98),
             published backbone, 3 seeds                            [default]
    cubcars  CUB and Cars with the ImageNet-R recipe, 3 seeds
    matched  the four datasets on our backbone (vit_base_patch16_224, augreg2
             IN21k -> IN1k), optional
    grid     CUB / Cars sensitivity, first seed: lr {5e-4, 1e-3} x eps {0.95, 0.98}
             (documents the tuning budget; the lr 5e-4 / eps 0.98 cell shares the
             cubcars chain and is served from its cache -- run it after cubcars)
    all      parity, cubcars, matched, grid in that order

Every cell logs to logs_main/<dataset>/<config>.log (per-seed 'Experiment:'
headers, 'Time:' and the SUMMARY block in run_config_sweep's format, so
parse_logs.py and rebuttal_common.has_summary read it) and to
logs_main/inflora_campaign.log, and ends with
    SWEEP SUMMARY [inflora-<ds>] <config>: FA m ± s | AA m ± s | FF m ± s |
        micro ACC m ± s | AVG m ± s | seeds [...] | failed [...]
Failed seeds (an '[Experiment ...]' error, empty curve) are logged as FAILED and
excluded from every mean. train_prefix encodes backbone, dataset, rank, eps,
epochs, lr, Gram mode, dual flag, layers and batch size, so no variant can hit
another variant's cache under checkpoints_main/inflora; main.g is re-seeded
before every run, so a (dataset, seed) cell does not depend on what ran before
it in the same process. A finished cell is skipped on relaunch; a partial log
needs --redo (backup + rerun) or --append (missing seeds only).

Cost per seed (plan estimate, one RTX 4090): CIFAR-100 ~1.3 h, ImageNet-R
~1.5 h, CUB ~0.45 h, Cars ~0.55 h; parity 8.5 GPU-h, CUB + Cars 3 GPU-h,
matched 11.5 GPU-h (optional). About 115 MB of cached state per task.

Usage:
    python run_inflora.py --dry-run                      # parity cells; no torch needed
    python run_inflora.py --stage all --dry-run
    python run_inflora.py --stage smoke                  # run twice (cache round-trip)
    python run_inflora.py --stage parity
    python run_inflora.py --stage parity --ds cifar224 --seeds 1993
    python run_inflora.py --stage cubcars
    python run_inflora.py --stage matched
    python run_inflora.py --stage grid
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

# timm names: the published checkpoint (AugReg ViT-B/16 pre-trained on
# ImageNet-21k only; timm 1.0.x alias of vit_base_patch16_224.augreg_in21k) and
# our matched backbone (augreg2_in21k_ft_in1k, the default of every other recipe).
BACKBONES = {"in21k": "vit_base_patch16_224_in21k", "matched": "vit_base_patch16_224"}

# Published per-dataset settings (configs/cifar100_inflora.json and
# mimg10_inflora.json). CUB (10x20) and Cars (16+9x20, 10 sessions) have no
# published InfLoRA setting and take the ImageNet-R recipe.
DS_RECIPE = {
    "cifar224":  {"train_epochs": 20, "inflora_lamb": 0.95},
    "imagenetr": {"train_epochs": 50, "inflora_lamb": 0.98},
    "cub":       {"train_epochs": 50, "inflora_lamb": 0.98},
    "cars":      {"train_epochs": 50, "inflora_lamb": 0.98},
}

# Dataset-independent part of the recipe. RECIPE_PRESETS["inflora"] in main.py
# holds the same values; they are repeated here so the Configuration dump and
# the cache prefix show every setting explicitly.
RECIPE = {
    "lora_r": 10,
    "train_base_lr": 5e-4,
    "train_batch_size": 128,
    "inflora_lame": 1.0,
    "inflora_gram_mode": "train",
    "inflora_dual": True,
    "inflora_layers": None,
    "train_merge": "none",
}

STAGES = ("smoke", "parity", "cubcars", "matched", "grid")
ALL_ORDER = ("parity", "cubcars", "matched", "grid")
STAGE_DATASETS = {
    "smoke": ["cifar224"],
    "parity": ["cifar224", "imagenetr"],
    "cubcars": ["cub", "cars"],
    "matched": ["cifar224", "imagenetr", "cub", "cars"],
    "grid": ["cub", "cars"],
}
STAGE_BACKBONE = {"smoke": "in21k", "parity": "in21k", "cubcars": "in21k",
                  "matched": "matched", "grid": "in21k"}
FIRST_SEED_ONLY = ("smoke", "grid")
GRID_LR = (5e-4, 1e-3)
GRID_LAMB = (0.95, 0.98)

# Learner._ckpt_dir default for a merge-family method is checkpoints_main/<method>;
# passed explicitly so every cell (and the dry-run) names its directory. Smoke
# states get their own directory for the day-0 cleanup.
CKPT_DIR = os.path.join(rc.CHECKPOINT_ROOT, "inflora")
SMOKE_CKPT_DIR = os.path.join(CKPT_DIR, "smoke")
CAMPAIGN_LOG = "inflora_campaign.log"
LOG_FORMAT = "%(asctime)s [%(filename)s] => %(message)s"   # main.run_config_sweep


# ----------------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------------

def train_prefix(bb, ds, c):
    """Cache key of a variant: backbone, dataset, rank, eps, epochs, lr, Gram
    mode, dual flag, layers and batch size (critic fix: a triage rerun with
    another Gram mode or batch size must not hit the main run's cache)."""
    layers = c.get("inflora_layers")
    lay = "all" if layers is None else "-".join(str(i) for i in layers)
    return (f"inflora_{bb}_{ds}_r{c['lora_r']}_l{c['inflora_lamb']}_e{c['train_epochs']}"
            f"_lr{c['train_base_lr']:g}_g{c['inflora_gram_mode']}"
            f"_d{int(bool(c['inflora_dual']))}_L{lay}_b{c['train_batch_size']}")


def build_cfg(ds, bb, smoke=False, **over):
    """Experiment config of one cell (merged over RECIPE_PRESETS['inflora'] by
    main.run_single_experiment). `over` holds the grid / triage overrides."""
    c = {"train_method": "inflora", "inflora_backbone": BACKBONES[bb]}
    c.update(RECIPE)
    c.update(DS_RECIPE[ds])
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
        s += f"_b{over['train_batch_size']}"
    if "inflora_gram_mode" in over:
        s += f"_g{over['inflora_gram_mode']}"
    return s


def stage_cells(stage, datasets, bb, over):
    """[(ds, config_name, cfg)] of one stage."""
    bb = bb or STAGE_BACKBONE[stage]
    cells = []
    for ds in datasets:
        if stage == "smoke":
            name = f"inflora_{bb}_{ds}{variant_suffix(over)}_smoke"
            cells.append((ds, name, build_cfg(ds, bb, smoke=True, **over)))
        elif stage == "grid":
            for lr in GRID_LR:
                for lamb in GRID_LAMB:
                    name = f"inflora_{bb}_{ds}_grid_lr{lr:g}_l{lamb}{variant_suffix(over)}"
                    cells.append((ds, name, build_cfg(ds, bb, train_base_lr=lr,
                                                      inflora_lamb=lamb, **over)))
        else:
            name = f"inflora_{bb}_{ds}{variant_suffix(over)}"
            cells.append((ds, name, build_cfg(ds, bb, **over)))
    return cells


def chain_files(ds, cfg, seed):
    """Cached files of one seed: Learner.prefix() is {seed}_{ds}_inflora_{train_prefix}
    and a merge-family task leaves _state_{t}.pt and _classifier_{t}.pt."""
    stem = f"{seed}_{ds}_inflora_{cfg['train_prefix']}"
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
        acc.append(curve[-1])                 # InfLoRA ACC_T: last micro Total Acc
        avg.append(float(np.mean(curve)))     # InfLoRA AVG_T: mean of the curve
        logging.info(f"[InfLoRA] seed {seed}: FA {r['faa']:.2f} | AA {r['asa']:.2f} | "
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
        logging.info(f"SWEEP SUMMARY [inflora-{ds}] {name}: FA {_pm(fa)} | AA {_pm(aa)} | "
                     f"FF {_pm(ff)} | micro ACC {_pm(acc)} | AVG {_pm(avg)} | "
                     f"seeds {ok} | failed {failed}")
    else:
        # No 'SWEEP SUMMARY' prefix here: parse_logs.SWEEP_RE must not read an empty cell.
        logging.error(f"[FAILED] [inflora-{ds}] {name}: every seed failed {failed}; no summary")
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
        description="InfLoRA baseline campaign (rebuttal E15b); see the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ("all",), default="parity",
                    help="smoke | parity (default) | cubcars | matched | grid | all")
    ap.add_argument("--smoke", action="store_true", help="alias for --stage smoke")
    ap.add_argument("--datasets", "--ds", nargs="+", default=None,
                    help="restrict the stage to these datasets")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                    help="seeds (default 1993 1994 1995; smoke and grid use the first one)")
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default=None,
                    help="override the stage's backbone (in21k = published, matched = ours)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="train_batch_size override (OOM fallback 64; own name and prefix)")
    ap.add_argument("--gram-mode", choices=("train", "test"), default=None,
                    help="inflora_gram_mode override for the triage rerun (own name and prefix)")
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
    if args.gram_mode is not None:
        over["inflora_gram_mode"] = args.gram_mode

    if args.datasets:
        unknown = [d for d in args.datasets if d not in DS_RECIPE]
        if unknown:
            raise SystemExit(f"unknown dataset(s) {unknown}; known: {sorted(DS_RECIPE)}")

    cells = []   # (stage, ds, name, cfg, seeds, log_root)
    for stage in stages:
        datasets = STAGE_DATASETS[stage]
        if args.datasets:
            datasets = [d for d in args.datasets if d in datasets] if args.stage == "all" \
                else list(args.datasets)
        seeds = args.seeds[:1] if stage in FIRST_SEED_ONLY else list(args.seeds)
        log_root = args.log_root or (rc.SMOKE_LOG_ROOT if stage == "smoke" else rc.LOG_DIR)
        for ds, name, cfg in stage_cells(stage, datasets, args.backbone, over):
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
