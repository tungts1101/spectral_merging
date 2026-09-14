"""Joint-training upper bound with OUR OWN recipe (user request, Sep 14).

One task containing all classes (config "joint": True), trained with exactly
the per-task recipe of the method: seqft PEFT LoRA r64 alpha128 on
qkv+fc1+fc2, SGD momentum 0.9, lr 1e-2 tied, batch 64, 10 epochs, cosine,
per-dataset weight decay from the tuned postures. Stage 2 disabled — with a
single task there is no merge chain, no drift, no LCA; FAA = plain accuracy
on all classes. 3 seeds.

This replaces the SLCA-published joint numbers currently cited in RESULTS.md
(CIFAR 93.22, IN-R 79.60, CUB 88.00, Cars 80.31) with bounds measured under
our own backbone/recipe, making the "gap to upper bound" row self-consistent.
"""
import time
import exp21

SEEDS = (1993, 1994, 1995)
WD = {"cifar224": 5e-4, "imagenetr": 2e-4, "cub": 1e-3, "cars": 1e-3}

CONFIGS_BY_DS = {
    ds: {f"joint_ours_{ds}": {
        "train_method": "seqft", "joint": True,
        "model_lora_r": 64, "model_lora_alpha": 128,
        "model_lora_target_modules": ["qkv", "fc1", "fc2"],
        "train_weight_decay": wd,
        "train_ca": False, "train_drift": False,
        "train_prefix": f"exp21_{ds}_joint",
        "cache_backbone": True, "reset_merge": True, "cleanup_merged": True}}
    for ds, wd in WD.items()
}

if __name__ == "__main__":
    t0 = time.time()
    for ds, cfgs in CONFIGS_BY_DS.items():
        exp21.run_config_sweep([ds], cfgs, list(SEEDS))
    print(f"Joint upper bounds total: {(time.time() - t0) / 3600:.2f}h")
