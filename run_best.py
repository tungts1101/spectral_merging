"""Reproduce the headline results on the four benchmarks (3 seeds each).

Framework: sequential LoRA fine-tuning (ViT-B/16) -> incremental spectral
merging (pspectral) -> identity-ridge drift compensation -> NxK LCA
alignment. See README.md.

Expected results (FA | AA | FF, mean +- std over seeds 1993/1994/1995):
  CIFAR-100  91.97 +- 0.32 | 94.80 +- 0.14 | 2.94 +- 0.32
  CUB        89.09 +- 0.02 | 92.45 +- 0.45 | 4.47 +- 0.24
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
        "train_merge": "pspectral", "spectral_variant": "only_A",
        "train_ca": True, "train_drift": True,
        "train_drift_ridge": "identity", "train_drift_transport": "full",
        "cache_backbone": True, "reset_merge": True, "cleanup_merged": True}

# Full per-dataset configurations (tuned per benchmark).
CONFIGS = {
    "cifar224": {
        "train_weight_decay": 5e-4,
        "model_lora_target_modules": ["qkv", "fc1", "fc2"],
        "train_merge_alpha": 0.134, "pspectral_p": 0.910,
        "train_drift_lambda": 3.92, "train_ca_cov_shrinkage": 0.032,
        "train_ca_robust_weight": 0.279,
        "train_ca_n_classes": 40, "train_ca_k_samples": 64,
    },
    "cub": {
        "train_weight_decay": 1e-3,
        "model_lora_target_modules": ["qkv"],
        "train_merge_alpha": 0.10, "pspectral_p": 0.9,
        "train_drift_lambda": 10.0, "train_ca_cov_shrinkage": 0.05,
        "train_ca_robust_weight": 0.1,
        "train_ca_n_classes": 40, "train_ca_k_samples": 64,
    },
    "imagenetr": {
        "train_weight_decay": 2e-4,
        "model_lora_target_modules": ["qkv", "fc1", "fc2"],
        "train_merge_alpha": 0.30, "pspectral_p": 0.9,
        "train_drift_lambda": 100.0, "train_ca_cov_shrinkage": 0.0,
        "train_ca_robust_weight": 0.1,
        "train_ca_n_classes": 40, "train_ca_k_samples": 64,
    },
    "cars": {
        "train_weight_decay": 1e-3,
        "model_lora_target_modules": ["qkv", "fc1", "fc2"],
        "train_merge_alpha": 0.20, "pspectral_p": 0.9,
        "train_drift_lambda": 100.0, "train_ca_cov_shrinkage": 0.0,
        "train_ca_robust_weight": 0.2,
        "train_ca_n_classes": 40, "train_ca_k_samples": 64,
    },
}

if __name__ == "__main__":
    t0 = time.time()
    for ds, overrides in CONFIGS.items():
        cfg = {**BASE, **overrides,
               "train_prefix": f"best_{ds}_r64"}
        exp21.run_config_sweep([ds], {f"best_{ds}": cfg}, list(SEEDS))
    print(f"Total: {(time.time() - t0) / 3600:.2f}h")
