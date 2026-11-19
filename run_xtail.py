"""ISM on CLIP's image tower in RAIL's X-TAIL stream (rebuttal follow-up; preparation).

Stream (utils.data.xtail): the ten domains of RAIL (Xu et al., NeurIPS 2024) in
its order, aircraft, caltech101, dtd, eurosat, oxford_flowers, food101, mnist,
oxford_pets, stanford_cars, sun397, with the CoOp splits, 16 training images
per class (data_fewshot_k 16, drawn with the run seed), labels offset per
domain so one domain is one task of the class-incremental learner (1,100
classes in total), test sets untouched. Inputs use CLIP's normalisation.

Model: timm's CLIP ViT-B/16 image tower (vit_base_patch16_clip_224.openai, the
OpenAI weights), d = 768 class token before the projection, one LoRA module
r = 64 on qkv / fc1 / fc2, our linear head trained by LCA. Everything else is
the paper's ImageNet-R cell (pspectral only_A, p 0.9, alpha 0.3, lambda 100,
local-term weight 0.1, weight decay 2e-4, SGD 1e-2, 10 epochs, batch 64):
nothing is tuned for this stream. No text encoder is used, so the numbers
comparable with RAIL are the per-domain accuracies after the last domain
(RAIL's "Last" = our FA) and the lower triangle of the accuracy matrix;
RAIL's "Transfer" (zero-shot on unseen domains) has no counterpart here.

Cells (logs_main/xtail/<cell>.log, run_config_sweep format):
    xtail_ism      the cell above, 3 seeds, checkpoints_main/rebuttal/xtail
Options: --alpha, --lam, --shots change the three knobs (the cell name gets a
suffix); --smoke runs the first two domains for one epoch (first seed).

    python run_xtail.py --dry-run
    XTAIL_ROOT=/mnt/data/greenhorn/xtail python run_xtail.py --smoke
    XTAIL_ROOT=/mnt/data/greenhorn/xtail python run_xtail.py
"""
import argparse
import sys
from collections import OrderedDict

import rebuttal_common as rc

DS = "xtail"
CLIP_BACKBONE = "vit_base_patch16_clip_224.openai_lora"   # helper.get_backbone: timm name + '_lora'
CLIP_OUTDIM = 768
DEFAULT_SHOTS = 16
# The paper's ImageNet-R posture (rebuttal_common.POSTURE['imagenetr'] minus its prefix).
POSTURE = {"train_weight_decay": 2e-4, "train_drift_lambda": 100.0,
           "train_ca_robust_weight": 0.1, "train_merge_alpha": 0.30, "pspectral_p": 0.9}


def prefix(shots):
    return f"xtail_clipb16_r64_qkvfc_fs{int(shots)}"


def cell_name(alpha, lam, shots):
    name = "xtail_ism"
    if alpha != POSTURE["train_merge_alpha"]:
        name += f"_a{alpha:g}"
    if lam != POSTURE["train_drift_lambda"]:
        name += f"_l{lam:g}"
    if shots != DEFAULT_SHOTS:
        name += f"_fs{int(shots)}"
    return name


def build_cells(ckpt_dir, alpha, lam, shots):
    cfg = dict(rc.COMMON)
    cfg.update(POSTURE)
    cfg.update(model_backbone=CLIP_BACKBONE, model_outdim=CLIP_OUTDIM,
               train_prefix=prefix(shots), checkpoint_dir=ckpt_dir,
               data_fewshot_k=int(shots), train_merge_alpha=float(alpha),
               train_drift_lambda=float(lam))
    return OrderedDict([(cell_name(alpha, lam, shots), cfg)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(rc.SEEDS))
    ap.add_argument("--alpha", type=float, default=POSTURE["train_merge_alpha"])
    ap.add_argument("--lam", type=float, default=POSTURE["train_drift_lambda"])
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="first two domains, one epoch, first seed")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--log-root", default=None)
    args = ap.parse_args(argv)
    ckpt_dir = rc.isolated_ckpt_path("xtail")
    cells = build_cells(ckpt_dir, args.alpha, args.lam, args.shots)
    print(f"[plan] {DS}: {CLIP_BACKBONE} (d = {CLIP_OUTDIM}), {args.shots} shots per class, "
          f"alpha {args.alpha:g}, lambda {args.lam:g}, ckpt {ckpt_dir}")
    rc.launch(DS, cells, args.seeds, log_root=args.log_root, dry_run=args.dry_run,
              smoke=args.smoke, redo=args.redo, append=args.append)
    if args.dry_run:
        rc.print_cells(len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
