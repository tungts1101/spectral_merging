"""Export the deployed CIFAR-100 models of the robustness study (paper Table 3
and Figure 3; RESULTS.md Section 2): 10 tasks x 10 classes, seeds 1993 / 1994 /
1995. Each export is one small file (LoRA trainables + aligned head + class
Gaussians, ~25 MB) written by main.run_single_experiment through
config["final_model_dir"]; robustness/eval_*.py load it.

Arms (one config each; the log is logs_main/cifar224/rob_<arm>.log):
  ours            ISM, the main-table CIFAR-100 cell of run_best.py
                  (pspectral only_A, p 0.9, alpha 0.15, lambda 100, local-term 0.2)
  merge_avg / merge_ties / merge_max_abs
                  the same chain and later stages with the merge rule swapped
                  (incremental task-arithmetic / TIES / max-abs); the
                  "Merge (...)" rows of RESULTS.md Section 2
  nomerge         the raw sequential module, same later stages
  e2lora          E2-LoRA under its published recipe on our backbone (the
                  "matched backbone" row); its own chain
All seqft arms share ONE cached chain: the chain of run_best.py
(train_prefix best_cifar224_r64, checkpoints_main/seqft/) when it exists, so
only merge + drift + alignment + export run (about 10 min per seed on an RTX
4090); otherwise the chain is trained first (about 2 h).

    python run_robustness_models.py                       # ours, 3 seeds
    python run_robustness_models.py --arms ours merge_avg merge_ties merge_max_abs
    python run_robustness_models.py --prefix main_cifar224_r64_qkvfc \
        --ckpt-dir checkpoints_main/rebuttal/e12_cifar224   # reuse another cached chain
then
    python robustness/eval_corruption.py   final_models/cifar224_rob_ours_s*.pt
    python robustness/eval_perturbation.py final_models/cifar224_rob_ours_s*.pt

The published baselines of Table 3 (SLCA, EASE, MOS, TUNA) were exported from
a LAMDA-PILOT checkout with a save_final_model hook; see robustness/model_zoo.py.
"""
import argparse
import time

import main
import run_best

ARMS = {
    "ours": {},
    "merge_avg": {"train_merge": "avg", "train_merge_incremental": True},
    "merge_ties": {"train_merge": "ties", "train_merge_incremental": True},
    "merge_max_abs": {"train_merge": "max_abs", "train_merge_incremental": True},
    "nomerge": {"train_merge": "none"},
}


def build_config(arm, prefix, ckpt_dir, out_dir):
    if arm == "e2lora":
        cfg = {"train_method": "e2lora", "train_prefix": f"{prefix}_e2",
               "cache_backbone": True, "reset_train": False}
    else:
        cfg = {**run_best.BASE, **run_best.CONFIGS["cifar224"],
               "train_prefix": prefix, "reset_train": False}
        cfg.update(ARMS[arm])
        if ckpt_dir:
            cfg["checkpoint_dir"] = ckpt_dir
    cfg["final_model_dir"] = out_dir
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", nargs="+", default=["ours"],
                    choices=sorted(ARMS) + ["e2lora"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(run_best.SEEDS))
    ap.add_argument("--prefix", default="best_cifar224_r64",
                    help="train_prefix of the cached sequential chain (run_best.py's by default)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="checkpoint_dir holding that chain (default checkpoints_main/seqft)")
    ap.add_argument("--out", default="final_models", help="final_model_dir of the exports")
    ap.add_argument("--dry-run", action="store_true", help="print the configs and exit")
    args = ap.parse_args()

    configs = {f"rob_{arm}": build_config(arm, args.prefix, args.ckpt_dir, args.out)
               for arm in args.arms}
    if args.dry_run:
        for name, cfg in configs.items():
            print(name)
            for k in sorted(cfg):
                print(f"  {k}: {cfg[k]}")
        raise SystemExit(0)
    t0 = time.time()
    main.run_config_sweep(["cifar224"], configs, list(args.seeds))
    print(f"Robustness model export total: {(time.time() - t0) / 3600:.2f}h")
