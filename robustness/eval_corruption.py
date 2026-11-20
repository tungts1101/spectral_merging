"""CIFAR-100-C corruption robustness: clean accuracy + per-corruption /
per-severity accuracy of exported final models.

Usage: python robustness/eval_corruption.py <model.pt> [<model.pt> ...]
Appends one CSV row per (model, corruption, severity) to
robustness/results/corruption.csv and prints the summary: clean accuracy,
mean over all 19 corruptions x 5 severities (RESULTS.md, Section 2) and mean
over the 15 standard corruptions (the paper's Table 3 and Figure 3, which
exclude the four held-out corruptions gaussian_blur, saturate, spatter and
speckle_noise).

Data: $DATA_ROOT/CIFAR-100-C/{<corruption>.npy, labels.npy} from Zenodo
record 3555552 (Hendrycks & Dietterich, 2019).
"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robustness.common import DATA_ROOT, accuracy_u8, cifar_test_raw
from robustness.model_zoo import load_model

C_DIR = f"{DATA_ROOT}/CIFAR-100-C"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
HELD_OUT = ("gaussian_blur", "saturate", "spatter", "speckle_noise")


def corruptions():
    return sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(f"{C_DIR}/*.npy")
                  if not p.endswith("labels.npy"))


def evaluate(model_path):
    m = load_model(model_path)
    name = os.path.basename(model_path)[:-3]
    clean_x, clean_y = cifar_test_raw()
    rows = [(name, "clean", 0, accuracy_u8(m, clean_x, clean_y))]
    print(f"{name}  clean {rows[0][3]:.2f}")
    corrs = corruptions()
    if not corrs:
        print(f"{name}  no corruption files under {C_DIR}; clean accuracy only")
        return rows
    labels = np.load(f"{C_DIR}/labels.npy")
    for corr in corrs:
        data = np.load(f"{C_DIR}/{corr}.npy", mmap_mode="r")
        for sev in range(5):
            sl = slice(sev * 10000, (sev + 1) * 10000)
            acc = accuracy_u8(m, np.asarray(data[sl]), labels[sl])
            rows.append((name, corr, sev + 1, acc))
        sev_accs = [r[3] for r in rows if r[1] == corr]
        print(f"{name}  {corr:20s} {np.mean(sev_accs):.2f}")
    corr_accs = [r[3] for r in rows if r[1] != "clean"]
    std_accs = [r[3] for r in rows if r[1] != "clean" and r[1] not in HELD_OUT]
    print(f"{name}  MEAN-CORRUPTED {np.mean(corr_accs):.2f} over {len(corrs)} corruptions "
          f"(clean {rows[0][3]:.2f})")
    if std_accs:
        n_std = len({r[1] for r in rows if r[1] != "clean" and r[1] not in HELD_OUT})
        print(f"{name}  MEAN-CORRUPTED-STANDARD {np.mean(std_accs):.2f} over {n_std} corruptions")
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, "corruption.csv")
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "corruption", "severity", "acc"])
        for path in sys.argv[1:]:
            for row in evaluate(path):
                w.writerow(row)
            f.flush()
    print(f"rows appended to {out_csv}")
