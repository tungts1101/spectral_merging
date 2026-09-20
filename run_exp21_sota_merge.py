"""SOTA merging-method comparison (user-approved strategy, Sep 19).

Stage 1: hyperparameter cells per method on IN-R (full stack, 3 seeds,
cache-served). Stage 2: each method's best stage-1 setting runs on
CIFAR-100, CUB, Cars. Rules from merging_sota.py: dare, della,
breadcrumbs, knots, modelstock. Existing rows for comparison: pspectral
(ours), avg (Task Arithmetic), TIES, max_abs (= MagMax's max-magnitude
selection).
"""
import logging
import sys
import time
import numpy as np
import exp21

SEEDS = (1993, 1994, 1995)
DS = {
    "imagenetr": ("exp21_inr_r64_qkvfc", 2e-4, 100.0, 0.1),
    "cifar224":  ("exp21_cifar224_r64_qkvfc", 5e-4, 100.0, 0.2),
    "cub":       ("exp21_cub_r64_qkvfc", 1e-3, 10.0, 0.1),
    "cars":      ("exp21_cars_r64_qkvfc", 1e-3, 100.0, 0.2),
}

def base_cfg(ds):
    prefix, wd, lam, robust = DS[ds]
    return {"train_method": "seqft", "train_weight_decay": wd,
            "model_lora_r": 64, "model_lora_alpha": 128,
            "model_lora_target_modules": ["qkv", "fc1", "fc2"],
            "train_prefix": prefix,
            "train_ca": True, "train_drift": True,
            "train_drift_ridge": "identity", "train_drift_lambda": lam,
            "train_drift_transport": "full",
            "train_ca_cov_shrinkage": 0.0, "train_ca_robust_weight": robust,
            "cache_backbone": True, "reset_train": False,
            "reset_merge": True, "cleanup_merged": True}

# Stage-1 grids (method -> list of (tag, overrides))
GRIDS = {
    "dare": [(f"q{q}_c{c}", {"merge_dare_q": q, "train_merge_coef": c})
             for q in (0.5, 0.9) for c in (0.3, 1.0)],
    "della": [(f"q{q}_c{c}", {"merge_della_q": q, "train_merge_coef": c})
              for q in (0.5, 0.9) for c in (0.3, 1.0)],
    "breadcrumbs": [(f"b{b}_c{c}", {"merge_bc_beta": b, "merge_bc_gamma": 0.01,
                                    "train_merge_coef": c})
                    for b in (0.5, 0.85) for c in (0.3, 1.0)],
    "knots": [(f"k{k}_c{c}", {"merge_knots_topk": k, "train_merge_coef": c})
              for k in (50.0, 100.0) for c in (0.3, 1.0)],
    "modelstock": [("default", {"train_merge_coef": 1.0})],
}

def run(ds, name, cfg):
    faas, asas = [], []
    for seed in SEEDS:
        r = exp21.run_single_experiment(ds, name, cfg, seed)
        faas.append(r["faa"]); asas.append(r["asa"])
    logging.info(f"SWEEP SUMMARY [sota-{ds}] {name}: "
                 f"FA {np.mean(faas):.2f} ± {np.std(faas):.2f} | "
                 f"AA {np.mean(asas):.2f} ± {np.std(asas):.2f}")
    return float(np.mean(faas))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(filename)s] => %(message)s",
                        handlers=[logging.FileHandler(f"{exp21.LOG_DIR}/sota_merge.log"),
                                  logging.StreamHandler(sys.stdout)], force=True)
    t0 = time.time()
    best = {}
    for method, grid in GRIDS.items():
        scores = {}
        for tag, ov in grid:
            cfg = {**base_cfg("imagenetr"), "train_merge": method, **ov}
            scores[tag] = run("imagenetr", f"sota_{method}_{tag}", cfg)
        best_tag = max(scores, key=scores.get)
        best[method] = dict(grid)[best_tag]
        logging.info(f"SWEEP SUMMARY [sota-stage1] {method} best={best_tag} "
                     f"FA {scores[best_tag]:.2f}")
    for method, ov in best.items():
        for ds in ("cifar224", "cub", "cars"):
            cfg = {**base_cfg(ds), "train_merge": method, **ov}
            run(ds, f"sota_{method}_best", cfg)
    print(f"SOTA merge campaign total: {(time.time() - t0) / 3600:.2f}h")
