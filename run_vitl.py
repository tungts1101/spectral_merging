"""ISM on a larger backbone (rebuttal follow-up): ViT-L/16 on ImageNet-R.

The paper's ImageNet-R cell (rebuttal_common.cfg: LoRA r = 64 on qkv / fc1 / fc2,
pspectral only_A, p 0.9, alpha 0.3, lambda 100, local-term weight 0.1, weight
decay 2e-4, SGD 1e-2, 10 epochs, batch 64) with the backbone swapped for timm's
vit_large_patch16_224 (AugReg, ImageNet-21K -> ImageNet-1K, like our ViT-B/16):
d = 1024, 24 blocks, 22.0M LoRA parameters. Nothing is retuned.

Learning rate: at the paper's 1e-2 the ViT-L fine-tuning diverged on one seed
(seed 1994, task 8: training loss 0.9 -> 3.5 within three epochs, never
recovered), so every ViT-L cell uses 5e-3 (--lr), the one deviation from the
paper's cell; the cell names, chain prefix and checkpoint directory carry the
value (vitl_ism_lr0.005, ...). The 1e-2 logs stay as vitl_ism.log.

Memory: a batch of 64 at ViT-L does not fit a 24 GB card in fp32, so the cell
sets train_micro_batches 2 (main.Learner: the same batch processed in two
halves with one optimizer step on the summed gradient; the update equals the
whole-batch step). --micro changes the split; 1 = the plain loop.

Cells (logs_main/<ds>/<cell>.log, run_config_sweep format), 3 seeds:
    vitl_ism       ISM at ViT-L with the stage timers on (E3 instrumentation:
                   logs_main/<ds>/timing_vitl_ism.csv, --no-timing to switch off)
    --controls adds, served from the same cached chain (no new training):
    vitl_nomerge   the raw sequential module (train_merge none), same later stages
    vitl_avg       plain averaging (Task Arithmetic, coefficient 1), same later stages
Everything lives in checkpoints_main/rebuttal/vitl_<ds> (own chain, own stem
{seed}_<ds>_vit_large_patch16_224_lora_vitl_<ds>_r64_qkvfc). Default dataset:
ImageNet-R only (the appendix default).

    python run_vitl.py --dry-run
    python run_vitl.py --smoke              # 2 tasks x 1 epoch, first seed; memory check
    python run_vitl.py                      # 3 seeds, ~1.6 h per seed on an RTX 4090
    python run_vitl.py --controls           # afterwards: the two cached controls
"""
import argparse
import sys
from collections import OrderedDict

import rebuttal_common as rc
import run_long

VITL_BACKBONE = "vit_large_patch16_224_lora"   # helper.get_backbone: timm name + '_lora'
VITL_OUTDIM = 1024
VITL_DATASETS = ("imagenetr",)
DEFAULT_MICRO = 2
DEFAULT_LR = 5e-3          # the paper's 1e-2 diverged at ViT-L (seed 1994, task 8)
PAPER_LR = 1e-2

# Task Arithmetic / averaging exactly as the Tab. 6 row (run_e13_fewshot.AVG_KEYS).
AVG_KEYS = {"train_merge": "avg", "train_merge_incremental": True,
            "train_merge_coef": 1.0, "train_merge_topk": 100}


def lr_tag(lr):
    """'' at the paper's learning rate, else '_lr5e-3' and the like."""
    return "" if float(lr) == PAPER_LR else f"_lr{float(lr):g}"


def job_name(ds, lr=DEFAULT_LR):
    """Name of the isolated checkpoint directory of the ViT-L chain of `ds`."""
    return f"vitl_{ds}{lr_tag(lr)}"


def prefix(ds, lr=DEFAULT_LR):
    """train_prefix of the ViT-L chain (the stem also carries the backbone name)."""
    return f"vitl_{ds}_r64_qkvfc{lr_tag(lr)}"


def build_cells(ds, ckpt_dir, micro=DEFAULT_MICRO, lr=DEFAULT_LR, controls=False,
                timing=True, smoke=False, solver="dense"):
    """OrderedDict {cell name: config}: the ISM cell first (it trains the chain).
    solver 'woodbury' names the cell vitl_ism<tag>_woodbury: the same chain
    (cache-served) with the push-through form of the ridge solve, for the
    Table 17 timing row at d = 1024."""
    base = dict(model_backbone=VITL_BACKBONE, model_outdim=VITL_OUTDIM,
                train_prefix=prefix(ds, lr), checkpoint_dir=ckpt_dir,
                train_micro_batches=int(micro), train_base_lr=float(lr))
    tag = lr_tag(lr) + ("_woodbury" if solver == "woodbury" else "")
    cells = OrderedDict()
    ism = rc.cfg(ds, **base)
    if solver == "woodbury":
        ism["train_drift_solver"] = "woodbury"
    if timing:
        ism["stage_timing"] = True
        ism["stage_timing_csv"] = run_long.side_csv_path(ds, f"timing_vitl_ism{tag}", None, smoke)
    cells[f"vitl_ism{tag}"] = ism
    if controls:
        cells[f"vitl_nomerge{tag}"] = rc.cfg(ds, **base, train_merge="none")
        cells[f"vitl_avg{tag}"] = rc.cfg(ds, **base, **AVG_KEYS)
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, datasets=VITL_DATASETS)
    ap.add_argument("--micro", type=int, default=DEFAULT_MICRO,
                    help="train_micro_batches: chunks per batch of 64 (default 2; 1 = plain loop)")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR,
                    help=f"backbone and head learning rate (default {DEFAULT_LR:g}; the paper's {PAPER_LR:g} diverged)")
    ap.add_argument("--solver", choices=("dense", "woodbury"), default="dense",
                    help="ridge solver of the drift map (woodbury = cached timing cell, suffix _woodbury)")
    ap.add_argument("--controls", action="store_true",
                    help="add the cached controls vitl_nomerge and vitl_avg")
    ap.add_argument("--no-timing", action="store_true",
                    help="switch the stage timers of the ISM cell off")
    args = rc.apply_common_args(ap.parse_args(argv))
    if args.micro < 1:
        raise SystemExit(f"--micro must be a positive integer, got {args.micro}")

    total = 0
    for ds in args.datasets:
        ckpt_dir = rc.isolated_ckpt_path(job_name(ds, args.lr))
        print(f"[plan] {ds}: ViT-L/16 ({VITL_BACKBONE}, d = {VITL_OUTDIM}), chain {prefix(ds, args.lr)}, "
              f"lr {args.lr:g}, micro-batches {args.micro}, ckpt {ckpt_dir}"
              + (" + cached controls" if args.controls else ""))
        cells = build_cells(ds, ckpt_dir, micro=args.micro, lr=args.lr, controls=args.controls,
                            timing=not args.no_timing, smoke=args.smoke, solver=args.solver)
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))
        total += len(cells)
    if args.dry_run:
        rc.print_cells(total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
