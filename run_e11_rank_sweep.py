"""LoRA rank sweep on ImageNet-R (rebuttal E11): r in {8, 16, 64, 128} at the
headline posture, kept directions at p = 0.9, merge time.

Rank changes training, so every r != 64 trains a new 10-task chain under its own
train_prefix (the cache key is seed / dataset / backbone / train_prefix only; a
reused prefix would load states of the wrong shape or silently reuse them):
    prefix  main_inr_r{r}_qkvfc              (rebuttal_common.prefix(ds) with r64 -> r{r})
    cell    e11_rank{r}                      checkpoints_main/rebuttal/e11_rank{r} (fresh)
    r = 64  the cached headline chain, re-run as the paired reference with the same
            [MergeStat] logging, in an isolated copy/symlink directory.
Convention: model_lora_alpha = 2 r keeps PEFT's scaling alpha / r = 2 of the
headline recipe (helper.get_backbone), so the effective product scale, learning
rate and weight decay posture transfer across ranks. --alpha-fixed A keeps
model_lora_alpha = A for every rank instead (scaling A / r changes with r); those
chains get the prefix main_inr_r{r}_a{A}_qkvfc and the cell name
e11_rank{r}_a{A}, except where A == 2 r (identical chain).
Every cell sets merge_log_stats True, so the log carries one
'[MergeStat] task t alpha=.. op=.. k_mean=.. | ... | merge_time=..s' line per merge
(kept k at p and the GPU wall-clock of the pspectral_merging call) and the
'[Freeze] ... trainable (n params)' line (expected 129,024 r).

After each rank's sweep (not in --dry-run / --smoke) analyze_spectrum.py replays
the chain and writes figures_data/e4_spectrum_imagenetr_r{r}.csv (kept directions
per rank from the exact truncate_energy rule); bench_svd.py times the SVD per
merge independently, and parse_logs.py e11 joins the three sources into
figures_data/e11_rank_imagenetr.csv for scripts/fig10_rank.py.

Run
    python run_e11_rank_sweep.py --dry-run                   # 5 cells, no torch needed
    python run_e11_rank_sweep.py --ranks 8 16 128            # training stream (about 2 h per seed)
    python run_e11_rank_sweep.py --ranks 64                  # paired reference, cached
    python run_e11_rank_sweep.py --alpha-fixed 128 --ranks 8 16 128
    python run_e11_rank_sweep.py --smoke                     # r = 8, seed 1993, 2 tasks x 1 epoch
    python bench_svd.py && python parse_logs.py e11 && python scripts/fig10_rank.py
"""
import argparse
import os
import sys
from collections import OrderedDict

import rebuttal_common as rc

RANKS = (8, 16, 64, 128)      # strategy 2026-11-10: r in {8, 16, 128} + the r = 64 reference
DS = "imagenetr"
HEADLINE_R = 64
PARAMS_PER_RANK = 129_024          # PEFT LoRA floats per unit rank (qkv + fc1 + fc2, 12 blocks):
                                   # the '[Freeze] ... trainable (n params)' count; the Model21
                                   # repr adds the 1,536 LayerNorm floats (8,259,072 at r = 64)


def lora_alpha(r, alpha_fixed=None):
    """model_lora_alpha of rank r: 2 r (default) or the fixed value."""
    return int(alpha_fixed) if alpha_fixed is not None else 2 * int(r)


def rank_prefix(ds, r, alpha_fixed=None):
    """train_prefix of the rank-r chain: the family prefix with r64 -> r{r}, plus
    _a{A} when a fixed lora_alpha differs from the 2 r convention."""
    base = rc.prefix(ds)
    if "r64" not in base:
        raise ValueError(f"cannot derive a rank prefix from {base!r} (no 'r64' token)")
    pre = base.replace("r64", f"r{int(r)}")
    if alpha_fixed is not None and int(alpha_fixed) != 2 * int(r):
        pre = pre.replace(f"r{int(r)}", f"r{int(r)}_a{int(alpha_fixed)}")
    return pre


def cell_name(r, alpha_fixed=None):
    name = f"e11_rank{int(r)}"
    if alpha_fixed is not None and int(alpha_fixed) != 2 * int(r):
        name += f"_a{int(alpha_fixed)}"
    return name


def is_cached_chain(ds, r, alpha_fixed=None):
    """True when the cell reuses the headline chain (r = 64 under the 2 r / 128 convention)."""
    return rank_prefix(ds, r, alpha_fixed) == rc.prefix(ds)


def build_cell(ds, r, seeds, args):
    """(name, config) of one rank; the checkpoint directory is isolated per cell:
    copies + symlinks of the cached chain for r = 64, a fresh directory otherwise."""
    name = cell_name(r, args.alpha_fixed)
    prefix = rank_prefix(ds, r, args.alpha_fixed)
    config = rc.cfg(ds, model_lora_r=int(r), model_lora_alpha=lora_alpha(r, args.alpha_fixed),
                    train_prefix=prefix, merge_log_stats=True)
    if is_cached_chain(ds, r, args.alpha_fixed):
        config["checkpoint_dir"] = rc.isolated_ckpt_dir(
            ds, prefix, seeds, name, src_dir=args.ckpt_src, dry_run=args.dry_run)
    else:
        config["checkpoint_dir"] = rc.isolated_ckpt_path(name)   # fresh training directory
        if not args.dry_run and not args.smoke:
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
    return name, config


def analyze_rank(ds, r, name, config, seeds, args):
    """Replay the finished chain with analyze_spectrum (figures_data/e4_spectrum_{ds}_r{r}.csv)."""
    import analyze_spectrum
    suffix = name[len(f"e11_rank{int(r)}"):]          # '' or '_a{A}' (fixed-alpha convention)
    argv = ["--ds", ds, "--prefix", config["train_prefix"], "--ckpt-dir", config["checkpoint_dir"],
            "--seeds", *map(str, seeds), "--alpha", str(config["train_merge_alpha"]),
            "--p", str(config["pspectral_p"]), "--lora-alpha", str(config["model_lora_alpha"]),
            "--tag", f"_r{int(r)}{suffix}", "--prefix-family", args.prefix_family]
    print(f"[E11] analyze_spectrum.py {' '.join(argv)}")
    try:
        analyze_spectrum.main(argv)
    except SystemExit as exc:
        print(f"[E11] analyze_spectrum skipped for {name}: {exc}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, datasets=(DS,))
    ap.add_argument("--ranks", nargs="+", type=int, default=None,
                    help="LoRA ranks (default 8 16 64 128; --smoke defaults to 8)")
    ap.add_argument("--alpha-fixed", type=int, default=None,
                    help="keep model_lora_alpha at this value for every rank (default 2 r)")
    ap.add_argument("--no-analyze", action="store_true",
                    help="skip the analyze_spectrum replay after each rank")
    return ap


def main(argv=None):
    args = rc.apply_common_args(build_parser().parse_args(argv))
    ranks = args.ranks or ([8] if args.smoke else list(RANKS))
    datasets = args.datasets
    if datasets != [DS]:
        print(f"[E11] note: the plan's sweep is ImageNet-R; running on {datasets} with the same rule")
    n = 0
    for ds in datasets:
        for r in ranks:
            name, config = build_cell(ds, r, args.seeds, args)
            if args.dry_run:
                print(f"[E11] {ds} r={r}: alpha={config['model_lora_alpha']} prefix={config['train_prefix']} "
                      f"{'cached headline chain' if is_cached_chain(ds, r, args.alpha_fixed) else 'TRAINS a new chain'} "
                      f"| expected [Freeze] PEFT params {PARAMS_PER_RANK * r:,} "
                      f"(Model21 repr adds the 1,536 norm floats: {PARAMS_PER_RANK * r + 1536:,})")
            outcome = rc.launch(ds, OrderedDict([(name, config)]), args.seeds, reseed=True,
                                **rc.launch_kwargs(args))
            n += 1
            if (not args.dry_run and not args.smoke and not args.no_analyze
                    and outcome.get(name) in ("done", "ran", "ran-append")):
                analyze_rank(ds, r, name, config, args.seeds, args)
    if args.dry_run:
        rc.print_cells(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
