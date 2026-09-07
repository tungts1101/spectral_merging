"""Reproduce the headline results on the four benchmarks (3 seeds each).

Framework: sequential LoRA fine-tuning (ViT-B/16, r=64, qkv+fc1+fc2)
-> incremental spectral merging (pspectral, only_A) -> identity-ridge drift
compensation -> NxK LCA alignment. See README.md.

Expected results (FA | AA | FF, mean +- std over seeds 1993/1994/1995):
  CIFAR-100  91.87 +- 0.28 | 94.79 +- 0.15 | 3.10 +- 0.28
  CUB        88.90 +- 0.46 | 92.16 +- 0.63 | 4.74 +- 0.39
  ImageNet-R 83.68 +- 0.15 | 87.54 +- 0.20 | 3.74 +- 0.39
  Cars       83.72 +- 0.05 | 87.09 +- 0.47 | 5.71 +- 0.53

Single-run noise is ~+-0.2-0.3; expect per-seed numbers within that band.
Backbone checkpoints are cached under checkpoints_exp21/ after the first run;
Stage-2 re-runs are then cheap (~10 min/seed).
"""
import time
import exp21

SEEDS = (1993, 1994, 1995)

BASE = {"train_method": "seqft",
        "model_lora_r": 64, "model_lora_alpha": 128,
        "model_lora_target_modules": ["qkv", "fc1", "fc2"],
        "train_merge": "pspectral", "spectral_variant": "only_A",
        "pspectral_p": 0.9,
        "train_ca": True, "train_drift": True,
        "train_drift_ridge": "identity", "train_drift_transport": "full",
        "train_ca_cov_shrinkage": 0.0,
        "cache_backbone": True, "reset_merge": True, "cleanup_merged": True}

# dataset -> (weight decay, merge alpha, drift lambda, robust weight)
TUNED = {
    "cifar224": (5e-4, 0.15, 100.0, 0.2),
    "cub": (1e-3, 0.10, 10.0, 0.1),
    "imagenetr": (2e-4, 0.30, 100.0, 0.1),
    "cars": (1e-3, 0.20, 100.0, 0.2),
}

if __name__ == "__main__":
    t0 = time.time()
    for ds, (wd, alpha, lam, robust) in TUNED.items():
        cfg = {**BASE, "train_weight_decay": wd, "train_merge_alpha": alpha,
               "train_drift_lambda": lam, "train_ca_robust_weight": robust,
               "train_prefix": f"best_{ds}_r64_qkvfc"}
        exp21.run_config_sweep([ds], {f"best_{ds}": cfg}, list(SEEDS))
    print(f"Total: {(time.time() - t0) / 3600:.2f}h")
