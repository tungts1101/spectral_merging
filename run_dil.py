"""Domain-incremental learning (paper Table 2; RESULTS.md Section 3), the
E2-LoRA / DUCT protocol: Office-Home (4 sessions = Art, Clipart, Product, Real
World; 65 classes per domain) and DomainNet (6 sessions = clipart, infograph,
painting, quickdraw, real, sketch; 345 classes per domain; the official cleaned
splits). Labels are offset per domain and scored modulo the classes per domain
(main.py, dil_classes_per_domain). Full ISM stack, LoRA r = 64 on qkv / fc1 /
fc2, 3 seeds; DomainNet trains 5 epochs per session (its sessions hold ~70k
images), Office-Home the default 10.

Reported (mean +- std over seeds 1993 / 1994 / 1995, Last-Acc / Inc-Acc):
  Office-Home 89.96 +- 0.13 / 88.10 +- 0.20
  DomainNet   71.05 +- 0.14 / 69.98 +- 0.07

Datasets are manual downloads (README.md, Datasets): $DATA_ROOT/OfficeHome/<domain>/<class>/
and $DATA_ROOT/DomainNet/<domain>/ with the official <domain>_{train,test}.txt files.

    python run_dil.py                       # both benchmarks
    python run_dil.py --ds officehome_dil   # one of them

This file is the port of the two server drivers that produced Section 3
(run_exp21_dil_officehome.py / run_exp21_dil_domainnet.py, same configuration,
train_prefix renamed); it was not re-run after the port.
"""
import argparse
import time

import main

SEEDS = (1993, 1994, 1995)

COMMON = {"train_method": "seqft",
          "model_lora_r": 64, "model_lora_alpha": 128,
          "model_lora_target_modules": ["qkv", "fc1", "fc2"],
          "train_merge": "pspectral", "spectral_variant": "only_A",
          "pspectral_p": 0.9, "train_merge_alpha": 0.15,
          "train_ca": True, "train_drift": True,
          "train_drift_ridge": "identity", "train_drift_lambda": 100.0,
          "train_drift_transport": "full",
          "train_ca_cov_shrinkage": 0.0, "train_ca_robust_weight": 0.1,
          "cache_backbone": True, "reset_merge": True, "cleanup_merged": True}

CONFIGS = {
    "officehome_dil": {**COMMON, "train_weight_decay": 1e-3,
                       "train_prefix": "dil_ohdil_r64_qkvfc",
                       "dil_classes_per_domain": 65},
    "dndil": {**COMMON, "train_weight_decay": 5e-4, "train_epochs": 5,
              "train_prefix": "dil_dndil_r64_qkvfc",
              "dil_classes_per_domain": 345},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ds", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--dry-run", action="store_true", help="print the configs and exit")
    args = ap.parse_args()
    if args.dry_run:
        for ds in args.ds:
            print(ds)
            for k in sorted(CONFIGS[ds]):
                print(f"  {k}: {CONFIGS[ds][k]}")
        raise SystemExit(0)
    t0 = time.time()
    for ds in args.ds:
        main.run_config_sweep([ds], {f"dil_{ds}": CONFIGS[ds]}, list(args.seeds))
    print(f"DIL total: {(time.time() - t0) / 3600:.2f}h")
