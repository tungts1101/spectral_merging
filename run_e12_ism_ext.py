"""E12 ISM extension (user decision 2026-11-12): LoRA-rank chains and a wider alpha
axis for ISM on Cars and CUB, proxy protocol (cosine-prototype head), 3 seeds.

Stage 'chain'  (r != 64): train the rank-r chain with the FULL-protocol ISM headline
    cell (merge + drift map + alignment; the paper's protocol), so every rank also
    yields the paper-protocol FA / AA for the reviewers' rank question.  E11
    convention: model_lora_alpha = 2 r, train_prefix = main prefix with r64 -> r{r},
    fresh directory checkpoints_main/rebuttal/e12r_<ds>_r{r}, log e12r_ism_r{r}.
Stage 'grid'   (every r): proxy ISM cells alpha x p on that rank's chain,
    alpha in {0.05, 0.1, 0.2, 0.3, 0.5}, p in {0.7, 0.9} (10 cells).  r = 64 uses the
    existing e12p_<ds> directory and names (e12p_ism_p{p}_a{a}; the 4 budget-4 cells
    are already there and are skipped by the resume logic); r != 64 uses
    e12p_<ds>_r{r} (symlinks of the e12r chain) and names e12p_ism_r{r}_p{p}_a{a}.

    python run_e12_ism_ext.py --ds cars --rank 16 --stage both --dry-run
    python run_e12_ism_ext.py --ds cub --rank 64 --stage grid          # the 6 new r64 cells
Driven per machine by scripts/e12r_machine.sh; parse with
    python parse_logs.py e12 --pattern '^e12p_ism'   (rank in the config: model_lora_r)
"""
import argparse
import os
import sys
import time
from collections import OrderedDict

import rebuttal_common as rc
from run_e12_merge_grid import PROTO, PROTO_JOB, JOB
from run_e11_rank_sweep import lora_alpha, rank_prefix

ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.5)
PS = (0.7, 0.9)
RANKS = (16, 32, 64, 128)
CHAIN_JOB = "e12r"


def chain_cell(ds, r):
    """(name, config) of the full-protocol headline cell that trains the rank-r chain."""
    name = f"{CHAIN_JOB}_ism_r{r}"
    cfg = rc.cfg(ds, model_lora_r=int(r), model_lora_alpha=lora_alpha(r), train_prefix=rank_prefix(ds, r),
                 merge_log_stats=True, checkpoint_dir=rc.isolated_ckpt_path(f"{CHAIN_JOB}_{ds}_r{r}"))
    return name, cfg


def grid_cells(ds, r, alphas=ALPHAS, ps=PS):
    """OrderedDict name -> proxy config, p-major, alpha ascending."""
    cells = OrderedDict()
    if int(r) == 64:
        ckpt = rc.isolated_ckpt_path(f"{PROTO_JOB}_{ds}")
        extra = {}
        tag = ""
    else:
        ckpt = rc.isolated_ckpt_path(f"{PROTO_JOB}_{ds}_r{r}")
        extra = {"model_lora_r": int(r), "model_lora_alpha": lora_alpha(r), "train_prefix": rank_prefix(ds, r)}
        tag = f"_r{r}"
    for p in ps:
        for a in alphas:
            cells[f"{PROTO_JOB}_ism{tag}_p{p}_a{a}"] = rc.cfg(
                ds, pspectral_p=p, train_merge_alpha=a, merge_log_stats=True, checkpoint_dir=ckpt,
                **PROTO, **extra)
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, datasets=("cars", "cub"))
    ap.add_argument("--rank", type=int, required=True, choices=RANKS)
    ap.add_argument("--stage", choices=("chain", "grid", "both"), default="both")
    ap.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    ap.add_argument("--ps", nargs="+", type=float, default=list(PS))
    args = rc.apply_common_args(ap.parse_args(argv))
    r = args.rank
    t0 = time.time()
    for ds in args.datasets:
        if args.stage in ("chain", "both") and r != 64:
            name, cfg = chain_cell(ds, r)
            print(f"[{CHAIN_JOB}] {ds} r={r}: {name} prefix={cfg['train_prefix']} lora_alpha={cfg['model_lora_alpha']} "
                  f"dir={cfg['checkpoint_dir']} (full protocol, trains the chain)")
            if not args.dry_run and not args.smoke:
                os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
            rc.launch(ds, OrderedDict([(name, cfg)]), args.seeds, **rc.launch_kwargs(args))
        if args.stage in ("grid", "both"):
            cells = grid_cells(ds, r, args.alphas, args.ps)
            print(f"[{PROTO_JOB}] {ds} r={r}: {len(cells)} proxy cells x {len(args.seeds)} seed(s)")
            if not args.smoke:
                prefix = next(iter(cells.values()))["train_prefix"]
                src = rc.isolated_ckpt_path(f"{JOB}_{ds}") if r == 64 else rc.isolated_ckpt_path(f"{CHAIN_JOB}_{ds}_r{r}")
                dst = f"{PROTO_JOB}_{ds}" if r == 64 else f"{PROTO_JOB}_{ds}_r{r}"
                rc.isolated_ckpt_dir(ds, prefix, args.seeds, dst, src_dir=src, dry_run=args.dry_run)
            rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))
    if args.dry_run:
        return 0
    print(f"[e12 ism ext] done rank {r} on {args.datasets}. Total: {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
