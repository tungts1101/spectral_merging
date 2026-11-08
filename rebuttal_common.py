"""Shared, torch-free posture and launcher for the rebuttal run scripts.

Every rebuttal driver (run_e*.py, run_long.py, run_covmode.py, run_driftdiag.py)
builds its configs from this module and launches them through `launch()`, so
the headline posture, the resume rule, the smoke rule and the checkpoint
isolation exist exactly once.

What it provides
  SEEDS, DATASETS            the three paper seeds and the four CIL benchmarks
  POSTURE[ds]                prefix, weight decay, drift lambda, local-term
                             (robust) weight, headline alpha, p -- copied once
                             from run_sota_merge.DS and run_best.CONFIGS
  BEST_PREFIX[ds]            run_best's `best_<ds>_r64` family (--prefix-family best)
  COMMON                     the dataset-independent keys of the cache-served
                             Stage-2 config (run_sota_merge.base_cfg + run_best.BASE)
  cfg(ds, **over)            the full cache-served config for one dataset
  merge_tag(cfg)             the merged-checkpoint tag main.merged_checkpoint builds
  task_increments(...)       the DataManager split rule (utils/data_manager.py:14-20)
  has_summary(...)           resume check on logs_main/<ds>/<config>.log
  seed_status(...)           per-seed ok / failed / incomplete from the same log
  smoke_overrides()          2-task, 1-epoch smoke posture
  isolated_ckpt_dir(...)     private checkpoint directory per job (copies + symlinks)
  launch(...)                lazy `import main` + main.run_config_sweep, one config
                             at a time, with --dry-run / --smoke / --redo / --append
  add_common_args(parser)    --datasets --seeds --dry-run --smoke --redo --append
                             --prefix-family main|best --log-root --ckpt-src
  check_posture()            server-only: imports run_best and run_sota_merge and
                             asserts that the copied posture equals theirs

Nothing here imports torch, timm, peft or main at module level, so every
driver's `--dry-run` runs on a machine without torch.  `python rebuttal_common.py
--check` (server) runs check_posture(); without flags it prints the posture.

Dry-run contract used by tests/test_dryruns.py: launch(dry_run=True) prints one
line per cell of the form
    [dry-run] ds=<ds> name=<config> tag=<merged tag> prefix=<train_prefix>
              ckpt=<checkpoint_dir> log=<log path> status=<done|todo> cache=<k/n|->
and every driver prints exactly one `CELLS: <n>` line (use print_cells()) with
the total number of unique cells it would run.
"""
import argparse
import copy
import glob
import os
import pprint
import re
import shutil
import sys

# ----------------------------------------------------------------------------
# Constants mirrored from the code (sources in the comments)
# ----------------------------------------------------------------------------

SEEDS = (1993, 1994, 1995)                     # run_best.SEEDS, run_sota_merge.SEEDS
DATASETS = ("cifar224", "cub", "imagenetr", "cars")

MODEL_BACKBONE = "vit_base_patch16_224_lora"   # BASE_CONFIG["model_backbone"] (main.py:1767)
CHECKPOINT_ROOT = "checkpoints_main"           # main.CHECKPOINT_ROOT (main.py:67)
SEQFT_CKPT_DIR = os.path.join(CHECKPOINT_ROOT, "seqft")       # Learner._ckpt_dir default (main.py:1654-1659)
REBUTTAL_CKPT_ROOT = os.path.join(CHECKPOINT_ROOT, "rebuttal")  # plan, ground rule 3
LOG_DIR = "logs_main"                          # main.LOG_DIR (main.py:68)
SMOKE_LOG_ROOT = os.path.join(LOG_DIR, "smoke")  # main.run_smoke_tests (main.py:2081-2082)

# main.DATA_TABLE (main.py:1729-1739): (num_task, init_cls, increment)
DATA_TABLE = {
    "cifar224": (10, 10, 10),
    "imageneta": (10, 20, 20),
    "imagenetr": (10, 20, 20),
    "cub": (10, 20, 20),
    "cars": (10, 16, 20),
    "officehome_dil": (4, 65, 65),
    "dndil": (6, 345, 345),
}
# main.TOTAL_CLASSES (main.py:1741-1743)
TOTAL_CLASSES = {"cifar224": 100, "imageneta": 200, "imagenetr": 200,
                 "cub": 200, "cars": 196,
                 "officehome_dil": 260, "dndil": 2070}

# BASE_CONFIG merge keys (main.py:1787-1793) that merged_checkpoint reads when a
# config does not set them.
BASE_MERGE_DEFAULTS = {
    "train_merge": "pspectral",
    "spectral_variant": "only_A",
    "pspectral_p": 0.9,
    "train_merge_alpha": 0.3,
    "train_merge_k": 16,
    "train_merge_coef": 1.0,
}

# ----------------------------------------------------------------------------
# Posture: copied ONCE from run_sota_merge.DS (prefix, weight decay, drift
# lambda, robust weight) and run_best.CONFIGS (alpha, p).  check_posture()
# asserts equality on the server, so this table cannot drift silently.
# ----------------------------------------------------------------------------

POSTURE = {
    "imagenetr": {"train_prefix": "main_inr_r64_qkvfc", "train_weight_decay": 2e-4,
                  "train_drift_lambda": 100.0, "train_ca_robust_weight": 0.1,
                  "train_merge_alpha": 0.30, "pspectral_p": 0.9},
    "cifar224":  {"train_prefix": "main_cifar224_r64_qkvfc", "train_weight_decay": 5e-4,
                  "train_drift_lambda": 100.0, "train_ca_robust_weight": 0.2,
                  "train_merge_alpha": 0.15, "pspectral_p": 0.9},
    "cub":       {"train_prefix": "main_cub_r64_qkvfc", "train_weight_decay": 1e-3,
                  "train_drift_lambda": 10.0, "train_ca_robust_weight": 0.1,
                  "train_merge_alpha": 0.10, "pspectral_p": 0.9},
    "cars":      {"train_prefix": "main_cars_r64_qkvfc", "train_weight_decay": 1e-3,
                  "train_drift_lambda": 100.0, "train_ca_robust_weight": 0.2,
                  "train_merge_alpha": 0.20, "pspectral_p": 0.9},
}

# run_best.py: train_prefix = f"best_{ds}_r64" (the Tab. 1 chains)
BEST_PREFIX = {ds: f"best_{ds}_r64" for ds in DATASETS}
PREFIX_FAMILIES = ("main", "best")
_PREFIX_FAMILY = "main"      # module default; set_prefix_family() / --prefix-family

# Dataset-independent keys: run_sota_merge.base_cfg minus the per-dataset
# values, plus run_best.BASE's merge keys.
COMMON = {
    "train_method": "seqft",
    "model_lora_r": 64, "model_lora_alpha": 128,
    "model_lora_target_modules": ["qkv", "fc1", "fc2"],
    "train_merge": "pspectral", "spectral_variant": "only_A",
    "train_ca": True, "train_drift": True,
    "train_drift_ridge": "identity", "train_drift_transport": "full",
    "train_ca_cov_shrinkage": 0.0,
    "train_ca_n_classes": 40, "train_ca_k_samples": 64,
    "cache_backbone": True, "reset_train": False,
    "reset_merge": True, "cleanup_merged": True,
}

HEADLINE_ALPHA = {ds: POSTURE[ds]["train_merge_alpha"] for ds in POSTURE}
HEADLINE_LAMBDA = {ds: POSTURE[ds]["train_drift_lambda"] for ds in POSTURE}


def set_prefix_family(family):
    """Select the chain family every later cfg() call uses ('main' or 'best')."""
    global _PREFIX_FAMILY
    if family not in PREFIX_FAMILIES:
        raise ValueError(f"prefix family must be one of {PREFIX_FAMILIES}, got {family!r}")
    _PREFIX_FAMILY = family


def get_prefix_family():
    return _PREFIX_FAMILY


def prefix(ds, family=None):
    """Cached train_prefix of `ds` for the family (`main` = main_*_r64_qkvfc,
    `best` = best_<ds>_r64)."""
    family = family or _PREFIX_FAMILY
    if family == "main":
        return POSTURE[ds]["train_prefix"]
    if family == "best":
        return BEST_PREFIX[ds]
    raise ValueError(f"unknown prefix family {family!r}")


def cfg(ds, prefix_family=None, **over):
    """Full cache-served Stage-2 config for `ds` (headline posture) with overrides.

    Equals run_sota_merge.base_cfg(ds) + run_best's merge keys (pspectral,
    only_A, p 0.9, headline alpha), i.e. the ISM headline cell on the cached
    main_*_r64_qkvfc chain.  Overrides are applied last; lists are copied so
    callers can mutate the result freely.
    """
    if ds not in POSTURE:
        raise KeyError(f"no posture for dataset {ds!r}; known: {sorted(POSTURE)}")
    out = copy.deepcopy(COMMON)
    out.update(copy.deepcopy(POSTURE[ds]))
    out["train_prefix"] = prefix(ds, prefix_family)
    out.update(copy.deepcopy(over))
    return out


# ----------------------------------------------------------------------------
# Merged-checkpoint tag and file names (main.py:1661-1710, 2026-2033)
# ----------------------------------------------------------------------------

def merge_tag(config):
    """Tag of the merged checkpoints a config produces, torch-free.

    Legacy rule (main.merged_checkpoint, main.py:1694-1710, duplicated in the
    cleanup block 2026-2033):
        pspectral -> pspec_{variant}_p{p}_a{alpha}
        kspectral -> kspec_{variant}_k{k}_a{alpha}
        otherwise -> {method}_c{coef}
    New tokens from main.merge_tag (integration item 1): `{op}_` BEFORE the alpha
    token for merge_update_op != 'trunc' (pspec_only_A_p0.9_clip_a0.3, so the plain
    cell's cleanup glob '*_merged_pspec_only_A_p0.9_a0.3_*.pt' cannot match a clip
    cell's files), `a{rule}{param}` for merge_alpha_rule != 'fixed',
    `opcm_a{alpha}_{lam}` and `nufilt_rp{rp}_rv{rv}_rl{rl}_ad{0|1}`.  main.merge_tag
    is the authority; `--check` compares the two when main is importable.
    Values are formatted with the same f-strings as main, so floats must be
    passed as the same Python objects (0.3 -> 'a0.3', 1.0 -> 'p1.0').
    """
    c = dict(BASE_MERGE_DEFAULTS)
    c.update(config)
    mm = c["train_merge"]
    variant = c.get("spectral_variant", c.get("pspectral_variant", "only_A"))
    if mm == "pspectral":
        rule = c.get("merge_alpha_rule", "fixed")
        if rule == "fixed":
            alpha_part = f"a{c['train_merge_alpha']}"
        else:
            alpha_part = f"a{rule}{c.get('merge_alpha_param', 1.0)}"
        op = c.get("merge_update_op", "trunc")
        op_part = "" if op == "trunc" else f"{op}_"
        return f"pspec_{variant}_p{c['pspectral_p']}_{op_part}{alpha_part}"
    if mm == "kspectral":
        return f"kspec_{variant}_k{c['train_merge_k']}_a{c['train_merge_alpha']}"
    if mm == "opcm":
        return f"opcm_a{c.get('merge_opcm_alpha', 0.5)}_{c.get('merge_opcm_lam', 'adaptive')}"
    if mm == "nufilt":
        return (f"nufilt_rp{c.get('merge_nufilt_rp', 128)}_rv{c.get('merge_nufilt_rv', 8)}"
                f"_rl{c.get('merge_nufilt_rl', 64)}_ad{int(bool(c.get('merge_nufilt_adapt', False)))}")
    return f"{mm}_c{c['train_merge_coef']}"


def chain_stem(seed, ds, train_prefix, model_backbone=MODEL_BACKBONE):
    """Learner.prefix() for the seqft recipe: {seed}_{ds}_{model_backbone}_{train_prefix}."""
    parts = [str(seed), ds, model_backbone]
    if train_prefix:
        parts.append(train_prefix)
    return "_".join(parts)


def merged_filename(seed, ds, config, task):
    """File name of M_task for this config (main.merged_checkpoint)."""
    stem = chain_stem(seed, ds, config.get("train_prefix", ""),
                      config.get("model_backbone", MODEL_BACKBONE))
    return f"{stem}_merged_{merge_tag(config)}_{task}.pt"


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------

def task_increments(total, init, inc):
    """Per-task class counts, replicating DataManager.__init__ (utils/data_manager.py:14-20)."""
    incs = [init]
    while sum(incs) + inc < total:
        incs.append(inc)
    offset = total - sum(incs)
    if offset > 0:
        incs.append(offset)
    return incs


def n_tasks(ds, config=None):
    """Number of tasks a config runs on `ds` (split overrides honoured as in
    run_single_experiment, main.py:1883-1897)."""
    config = config or {}
    _, init, inc = DATA_TABLE[ds]
    init = config.get("dataset_init_cls", init)
    inc = config.get("dataset_increment", inc)
    if config.get("joint", False):
        return 1
    return len(task_increments(TOTAL_CLASSES[ds], init, inc))


# ----------------------------------------------------------------------------
# Log inspection (resume).  Formats from run_config_sweep (main.py:1999-2018)
# and run_single_experiment (main.py:1966); the dash is U+2014.
# ----------------------------------------------------------------------------

_HEADER_RE = re.compile(r"Experiment: (?P<ds>\S+) [—-] (?P<cfg>.+?) [—-] seed (?P<seed>\d+)\s*$", re.M)
_SUMMARY_RE = re.compile(r"SUMMARY  (?P<ds>\S+) [—-] (?P<cfg>.+?)\s*$", re.M)
_ERROR_MARK = "[Experiment "          # logging.error(f"[Experiment {name}] {type(e).__name__}: {e}")
_FAA_RE = re.compile(r"\[Evaluation\] FAA: ")
_TIME_RE = re.compile(r"^.*Time: [\d.]+s\s*$", re.M)


def read_log(path):
    """Read a log as utf-8 (they contain em dashes, plus-minus and lambda)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def log_path(ds, config_name, log_root=None):
    return os.path.join(log_root or LOG_DIR, ds, config_name + ".log")


def _blocks(text, config_name):
    """Yield (seed, block_text) for every 'Experiment:' block of `config_name`,
    in file order; a block runs to the next header of any config."""
    heads = list(_HEADER_RE.finditer(text))
    for i, m in enumerate(heads):
        if m.group("cfg").strip() != config_name:
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        yield int(m.group("seed")), text[m.start():end]


def has_summary(log_path, config_name, n_seeds):
    """True when the log is a finished, clean run of `config_name`:
    a SUMMARY block for it is present, at least `n_seeds` 'Experiment:' headers
    carry its name, and no '[Experiment ...]' error line exists in the file."""
    if not os.path.isfile(log_path):
        return False
    text = read_log(log_path)
    if _ERROR_MARK in text:
        return False
    headers = sum(1 for _ in _blocks(text, config_name))
    if headers < n_seeds:
        return False
    want = config_name.upper()
    return any(m.group("cfg").strip() == want for m in _SUMMARY_RE.finditer(text))


def seed_status(log_path, config_name):
    """{seed: 'ok' | 'failed' | 'incomplete'} from the LAST block of each seed
    (logs are appended across re-runs).  'ok' = has a '[Evaluation] FAA:' line and
    the 'Time:' line and no error line; 'failed' = error line; else incomplete."""
    out = {}
    if not os.path.isfile(log_path):
        return out
    for seed, block in _blocks(read_log(log_path), config_name):
        if _ERROR_MARK in block:
            out[seed] = "failed"
        elif _FAA_RE.search(block) and _TIME_RE.search(block):
            out[seed] = "ok"
        else:
            out[seed] = "incomplete"
    return out


def _backup_log(path):
    """Rename a partial log to <path>.bak<N> (never deleted)."""
    n = 1
    while os.path.exists(f"{path}.bak{n}"):
        n += 1
    dst = f"{path}.bak{n}"
    os.replace(path, dst)
    return dst


# ----------------------------------------------------------------------------
# Smoke posture
# ----------------------------------------------------------------------------

def smoke_overrides():
    """Config overrides of a smoke run: two tasks, one epoch, two LCA epochs
    (main.run_smoke_tests posture, main.py:2065-2082).  launch(smoke=True) adds
    the `_smoke` suffix to the config name and train_prefix, its own checkpoint
    directory and log_root = logs_main/smoke, and keeps the first seed only."""
    return {"train_stop_at_task": 1, "train_epochs": 1, "train_ca_epochs": 2}


def smokify(name, config):
    """(name, config) of the smoke version of a cell; the originals are untouched."""
    c = copy.deepcopy(config)
    c.update(smoke_overrides())
    if c.get("train_prefix"):
        c["train_prefix"] = c["train_prefix"] + "_smoke"
    else:
        c["train_prefix"] = "smoke"
    d = c.get("checkpoint_dir")
    c["checkpoint_dir"] = (d.rstrip("/\\") + "_smoke") if d else os.path.join(REBUTTAL_CKPT_ROOT, "smoke")
    return name + "_smoke", c


# ----------------------------------------------------------------------------
# Checkpoint isolation (plan, ground rule 3)
# ----------------------------------------------------------------------------

def isolated_ckpt_path(name, root=REBUTTAL_CKPT_ROOT):
    """Path of the isolated directory for job `name` (nothing is created)."""
    return os.path.join(root, name)


_TASK_FILE_RE = re.compile(r"_(backbone|classifier)_(\d+)\.pt$")


def isolated_ckpt_dir(ds, prefix, seeds, name, src_dir=SEQFT_CKPT_DIR,
                      model_backbone=MODEL_BACKBONE, root=REBUTTAL_CKPT_ROOT,
                      dry_run=False, verbose=True):
    """Create checkpoints_main/rebuttal/<name>/ holding, per seed of the chain
    (`ds`, `prefix`), private COPIES of `_backbone_base.pt`, `_backbone_0.pt`
    and `_classifier_0.pt` (Learner.__init__ and task 0 rewrite these files,
    main.py:274 and 479-491) and SYMLINKS to `_backbone_{t}.pt` /
    `_classifier_{t}.pt` for t >= 1 (read-only on the cached chain).  Symlinks
    fall back to copies when the OS refuses them.  Idempotent: existing files
    are kept, dangling links are replaced.  Returns the directory path.

    Pass the result as `checkpoint_dir` so the merged-checkpoint cleanup glob and
    the base-file rewrite stay inside the job's own directory.  With dry_run the
    directory is not created; only the plan is printed.
    """
    dst_dir = isolated_ckpt_path(name, root)
    if not dry_run:
        os.makedirs(dst_dir, exist_ok=True)
    report = {}
    for seed in seeds:
        stem = chain_stem(seed, ds, prefix, model_backbone)
        copied = linked = kept = fallback = 0
        missing = []
        # 1) private copies of the files the job rewrites
        for suffix in ("_backbone_base.pt", "_backbone_0.pt", "_classifier_0.pt"):
            src = os.path.join(src_dir, stem + suffix)
            dst = os.path.join(dst_dir, stem + suffix)
            if not os.path.isfile(src):
                missing.append(stem + suffix)
                continue
            if os.path.lexists(dst):
                if os.path.islink(dst) and not os.path.exists(dst):
                    if not dry_run:
                        os.remove(dst)
                else:
                    kept += 1
                    continue
            if not dry_run:
                shutil.copy2(src, dst)
            copied += 1
        # 2) symlinks to the read-only task states t >= 1
        pattern = os.path.join(src_dir, stem + "_*.pt")
        for src in sorted(glob.glob(pattern)):
            m = _TASK_FILE_RE.search(os.path.basename(src))
            if not m or int(m.group(2)) < 1:
                continue
            if not os.path.basename(src).startswith(stem + "_"):
                continue
            dst = os.path.join(dst_dir, os.path.basename(src))
            if os.path.lexists(dst):
                if os.path.islink(dst) and not os.path.exists(dst):
                    if not dry_run:
                        os.remove(dst)
                else:
                    kept += 1
                    continue
            if dry_run:
                linked += 1
                continue
            try:
                os.symlink(os.path.relpath(os.path.abspath(src), os.path.abspath(dst_dir)), dst)
                linked += 1
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)
                fallback += 1
        report[seed] = {"copied": copied, "linked": linked, "copied_fallback": fallback,
                        "kept": kept, "missing": missing}
        if verbose:
            tag = "[ckpt dry-run]" if dry_run else "[ckpt]"
            print(f"{tag} {dst_dir}: seed {seed} {ds}/{prefix}: copied {copied}, "
                  f"linked {linked}, copied-instead-of-linked {fallback}, kept {kept}"
                  + (f", MISSING {missing}" if missing else ""))
            if any(f.endswith("_backbone_0.pt") for f in missing):
                print(f"{tag} WARNING: no cached task-0 state for seed {seed} under "
                      f"{src_dir}/{stem}_* -- the job would TRAIN this chain from scratch")
    return dst_dir


# ----------------------------------------------------------------------------
# Launcher
# ----------------------------------------------------------------------------

def print_cells(n):
    """The one line tests/test_dryruns.py reads from every driver's --dry-run."""
    print(f"CELLS: {int(n)}")


def _cache_summary(ds, config, seeds, ckpt_src):
    """'k/n' complete seeds of the config's chain under ckpt_src, or '-' when the
    inventory cannot be computed."""
    try:
        import check_cache
    except ImportError:
        return "-"
    try:
        st = check_cache.chain_status(ds, config.get("train_prefix", ""), seeds,
                                      n_tasks(ds, config), ckpt_dir=ckpt_src,
                                      model_backbone=config.get("model_backbone", MODEL_BACKBONE))
    except Exception as exc:  # inventory must never break a dry-run
        return f"-({type(exc).__name__})"
    done = sum(1 for s in seeds if st["seeds"][s]["complete"])
    return f"{done}/{len(seeds)}"


def launch(ds, configs, seeds, log_root=None, dry_run=False, smoke=False, *,
           redo=False, append=False, ckpt_src=SEQFT_CKPT_DIR, reseed=False,
           verbose=True):
    """Run `configs` ({name: config}) on dataset `ds`, one config at a time.

    - dry_run: print every config dict, its merged tag, log path, checkpoint
      directory, resume status and cached-chain status; never imports main.
    - smoke: smokify() every cell (suffix `_smoke` on name and prefix, own
      checkpoint directory, 2 tasks x 1 epoch), first seed only, log_root
      logs_main/smoke.
    - resume: a cell whose log passes has_summary() for these seeds is skipped.
      A partial log (crashed or interrupted) is never appended to silently:
      `redo` renames it to .bakN and reruns every seed; `append` reruns only the
      seeds that are not 'ok' and appends (parsers keep the last complete
      block per seed); with neither, a RuntimeError names the cell.
    - reseed: call main.g.manual_seed(0) before each config (process-order
      independence of the DataLoader generator; off by default to keep the
      behaviour of run_best/run_sota_merge).
    Returns {name: 'dry' | 'done' | 'ran' | 'ran-append'}.
    """
    if not isinstance(configs, dict):
        raise TypeError("configs must be an ordered dict {config_name: config}")
    seeds = list(seeds)
    cells = list(configs.items())
    if smoke:
        seeds = seeds[:1]
        log_root = SMOKE_LOG_ROOT
        cells = [smokify(name, c) for name, c in cells]
    log_root = log_root or LOG_DIR
    outcome = {}
    for name, config in cells:
        lp = log_path(ds, name, log_root)
        tag = merge_tag(config)
        ckpt_dir = config.get("checkpoint_dir", SEQFT_CKPT_DIR)
        done = has_summary(lp, name, len(seeds))
        if dry_run:
            cache = _cache_summary(ds, config, seeds, ckpt_src)
            print(f"[dry-run] ds={ds} name={name} tag={tag} prefix={config.get('train_prefix', '')} "
                  f"ckpt={ckpt_dir} log={lp} status={'done' if done else 'todo'} cache={cache}"
                  + (" (train_merge=none: no merged checkpoints)" if config.get("train_merge") == "none" else ""))
            if verbose:
                for line in pprint.pformat(config, width=100, sort_dicts=True).splitlines():
                    print("    " + line)
            outcome[name] = "dry"
            continue
        if done and not redo:
            if verbose:
                print(f"[skip] {ds} | {name}: complete for {len(seeds)} seed(s) in {lp}")
            outcome[name] = "done"
            continue
        run_seeds = seeds
        mode = "ran"
        if os.path.isfile(lp) and os.path.getsize(lp) > 0:
            if redo:
                bak = _backup_log(lp)
                print(f"[redo] {ds} | {name}: moved partial log to {bak}")
            elif append:
                st = seed_status(lp, name)
                run_seeds = [s for s in seeds if st.get(s) != "ok"]
                if not run_seeds:
                    print(f"[skip] {ds} | {name}: every seed is ok in {lp} (no SUMMARY block, "
                          f"parsers recompute from the per-seed blocks)")
                    outcome[name] = "done"
                    continue
                print(f"[append] {ds} | {name}: seeds {run_seeds} appended to {lp}; "
                      f"status was {st}")
                mode = "ran-append"
            else:
                raise RuntimeError(
                    f"{lp} exists but is not a complete clean run of {name} "
                    f"(seed status {seed_status(lp, name)}); pass --redo to back it up "
                    f"and rerun, or --append to run only the missing seeds")
        import main  # lazy: torch, timm, peft
        if reseed and hasattr(main, "g"):
            main.g.manual_seed(0)
        print(f"[launch] {ds} | {name} | seeds {run_seeds} | tag {tag} | ckpt {ckpt_dir} | log {lp}")
        main.run_config_sweep([ds], {name: config}, run_seeds, log_root=log_root)
        outcome[name] = mode
    return outcome


# ----------------------------------------------------------------------------
# CLI helpers shared by the drivers
# ----------------------------------------------------------------------------

def add_common_args(parser, datasets=DATASETS):
    """--datasets --seeds --dry-run --smoke --redo --append --prefix-family
    --log-root --ckpt-src.  Returns the parser."""
    parser.add_argument("--datasets", "--ds", nargs="+", default=list(datasets),
                        help=f"datasets to run (default: {' '.join(datasets)})")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                        help="seeds (default: 1993 1994 1995)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print every cell, its merged tag, log and checkpoint "
                             "directory; never imports main (no torch needed)")
    parser.add_argument("--smoke", action="store_true",
                        help="2-task 1-epoch run of every cell, first seed, "
                             "`_smoke` suffix, logs under logs_main/smoke")
    parser.add_argument("--redo", action="store_true",
                        help="back up partial or finished logs (.bakN) and rerun every seed")
    parser.add_argument("--append", action="store_true",
                        help="append only the seeds that are not ok to a partial log")
    parser.add_argument("--prefix-family", choices=PREFIX_FAMILIES, default="main",
                        help="cached chain family: main (main_*_r64_qkvfc, default) or "
                             "best (best_<ds>_r64 from run_best.py)")
    parser.add_argument("--log-root", default=None,
                        help="log root (default logs_main; smoke runs use logs_main/smoke)")
    parser.add_argument("--ckpt-src", default=SEQFT_CKPT_DIR,
                        help="directory of the cached chains (default checkpoints_main/seqft)")
    return parser


def apply_common_args(args):
    """Validate the common arguments and set the prefix family.  Returns args."""
    set_prefix_family(args.prefix_family)
    unknown = [d for d in args.datasets if d not in POSTURE]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; known: {sorted(POSTURE)}")
    args.seeds = [int(s) for s in args.seeds]
    return args


def launch_kwargs(args):
    """The launch() keyword arguments that follow from the common arguments."""
    return {"log_root": args.log_root, "dry_run": args.dry_run, "smoke": args.smoke,
            "redo": args.redo, "append": args.append, "ckpt_src": args.ckpt_src}


# ----------------------------------------------------------------------------
# Server-side consistency check
# ----------------------------------------------------------------------------

def check_posture(verbose=True):
    """Import run_best and run_sota_merge (they import main, so torch is needed)
    and assert that POSTURE / COMMON / cfg() equal their tables.  Also compares
    merge_tag() with main.merge_tag when that function exists.  Raises
    AssertionError listing every mismatch; returns the list of warnings."""
    import run_best
    import run_sota_merge
    problems, warnings = [], []
    for ds in DATASETS:
        pre, wd, lam, robust = run_sota_merge.DS[ds]
        mine = POSTURE[ds]
        for key, theirs in (("train_prefix", pre), ("train_weight_decay", wd),
                            ("train_drift_lambda", lam), ("train_ca_robust_weight", robust)):
            if mine[key] != theirs:
                problems.append(f"{ds}: POSTURE[{key}]={mine[key]!r} but run_sota_merge.DS has {theirs!r}")
        rb = run_best.CONFIGS[ds]
        for key in ("train_weight_decay", "train_merge_alpha", "pspectral_p",
                    "train_drift_lambda", "train_ca_robust_weight"):
            if mine[key] != rb[key]:
                problems.append(f"{ds}: POSTURE[{key}]={mine[key]!r} but run_best.CONFIGS has {rb[key]!r}")
        # run_sota_merge.base_cfg(ds) must be a sub-dict of cfg(ds)
        mine_cfg = cfg(ds, prefix_family="main")
        for key, val in run_sota_merge.base_cfg(ds).items():
            if key not in mine_cfg or mine_cfg[key] != val:
                problems.append(f"{ds}: cfg()[{key}]={mine_cfg.get(key)!r} but base_cfg has {val!r}")
        # run_best's full config must equal cfg(ds, 'best') (reset_train False is BASE_CONFIG's default)
        theirs_cfg = {**run_best.BASE, **rb, "train_prefix": f"best_{ds}_r64"}
        best_cfg = cfg(ds, prefix_family="best")
        for key, val in theirs_cfg.items():
            if key not in best_cfg or best_cfg[key] != val:
                problems.append(f"{ds}: cfg(best)[{key}]={best_cfg.get(key)!r} but run_best has {val!r}")
        extra = set(best_cfg) - set(theirs_cfg)
        if extra - {"reset_train"}:
            problems.append(f"{ds}: cfg(best) has keys run_best lacks: {sorted(extra - {'reset_train'})}")
    if tuple(run_best.SEEDS) != SEEDS or tuple(run_sota_merge.SEEDS) != SEEDS:
        problems.append(f"SEEDS {SEEDS} differ from run_best/run_sota_merge")
    # merged tags against main.merge_tag (only when the integrated main.py has it)
    try:
        import main
        their_tag = getattr(main, "merge_tag", None)
    except Exception as exc:  # pragma: no cover - server only
        their_tag = None
        warnings.append(f"main not importable for the tag check: {exc}")
    if their_tag is not None:
        samples = [cfg("imagenetr"), cfg("cifar224", pspectral_p=1.0),
                   cfg("cub", train_merge="kspectral", train_merge_k=16),
                   cfg("cars", train_merge="avg", train_merge_coef=1.0),
                   cfg("cars", train_merge="dare", train_merge_coef=0.3, merge_dare_q=0.5),
                   cfg("imagenetr", merge_update_op="clip"),
                   cfg("imagenetr", merge_alpha_rule="specnorm", merge_alpha_param=0.5),
                   cfg("imagenetr", train_merge="opcm", merge_opcm_alpha=0.5, merge_opcm_lam="adaptive"),
                   cfg("imagenetr", train_merge="nufilt", merge_nufilt_rp=128)]
        for sample in samples:
            try:
                theirs = their_tag({**getattr(main, "BASE_CONFIG", {}), **sample})
            except Exception as exc:
                warnings.append(f"main.merge_tag failed on {sample['train_merge']}: {exc}")
                continue
            if theirs != merge_tag(sample):
                warnings.append(f"tag mismatch for train_merge={sample['train_merge']}: "
                                f"rebuttal_common {merge_tag(sample)!r} vs main {theirs!r}")
    if verbose:
        for w in warnings:
            print("WARNING:", w)
        for p in problems:
            print("MISMATCH:", p)
    if problems:
        raise AssertionError(f"{len(problems)} posture mismatch(es); see above")
    if verbose:
        print(f"check_posture: OK ({len(DATASETS)} datasets, {len(warnings)} warning(s))")
    return warnings


def print_posture():
    print("dataset    prefix (main)             prefix (best)       wd      lambda  robust  alpha  p   tag")
    for ds in DATASETS:
        p = POSTURE[ds]
        print(f"{ds:<10} {p['train_prefix']:<25} {BEST_PREFIX[ds]:<19} {p['train_weight_decay']:<7g} "
              f"{p['train_drift_lambda']:<7g} {p['train_ca_robust_weight']:<7g} {p['train_merge_alpha']:<6g} "
              f"{p['pspectral_p']:<3g} {merge_tag(cfg(ds))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="server: import run_best/run_sota_merge and assert posture equality")
    ap.add_argument("--show", metavar="DS", default=None,
                    help="print cfg(DS) as the drivers build it")
    ap.add_argument("--prefix-family", choices=PREFIX_FAMILIES, default="main")
    args = ap.parse_args(argv)
    set_prefix_family(args.prefix_family)
    if args.check:
        try:
            check_posture()
        except ImportError as exc:
            print(f"--check needs run_best/run_sota_merge, which import main (torch): {exc}; run it on the server")
            return 2
        return 0
    print_posture()
    if args.show:
        print()
        pprint.pprint(cfg(args.show), width=100, sort_dicts=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
