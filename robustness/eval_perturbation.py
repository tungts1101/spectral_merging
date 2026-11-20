"""CIFAR-100-P perturbation stability: flip probability per perturbation
(fraction of consecutive-frame prediction changes along each ~31-frame
sequence) and its mean over perturbations. Lower is better. This is the
"mean flip probability" of RESULTS.md, Section 2, and of the paper's Table 3.

Usage: python robustness/eval_perturbation.py <model.pt> [...]
Appends to robustness/results/perturbation.csv.

Data: $DATA_ROOT/CIFAR-100-P/*.npy from Zenodo record 3555553; the nested
layout CIFAR-100-P/CIFAR-100-P/*.npy that the archive produces is accepted too.
"""
import csv
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robustness.common import DATA_ROOT, normalize
from robustness.model_zoo import load_model

P_DIR = f"{DATA_ROOT}/CIFAR-100-P"
if os.path.isdir(os.path.join(P_DIR, "CIFAR-100-P")):
    P_DIR = os.path.join(P_DIR, "CIFAR-100-P")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def perturbations():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{P_DIR}/*.npy"))


@torch.no_grad()
def flip_probability(model, pert, batch_seqs=8, n_limit=2500):
    """First n_limit sequences (2500 x ~30 transitions = 75k comparisons; the
    flip-probability estimates converge well before the full 10k)."""
    data = np.load(f"{P_DIR}/{pert}.npy", mmap_mode="r")  # [N, T, 32, 32, 3]
    n, t = min(data.shape[0], n_limit), data.shape[1]
    flips = comparisons = 0
    for i in range(0, n, batch_seqs):
        chunk = torch.from_numpy(np.asarray(data[i:i + batch_seqs]))  # [b,T,...]
        b = chunk.shape[0]
        x = chunk.reshape(b * t, 32, 32, 3).cuda().permute(0, 3, 1, 2).float() / 255.0
        preds = model.logits(normalize(model.preprocess(x))).argmax(1).reshape(b, t)
        flips += (preds[:, 1:] != preds[:, :-1]).sum().item()
        comparisons += b * (t - 1)
    return 100.0 * flips / comparisons


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, "perturbation.csv")
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "perturbation", "flip_prob"])
        for path in sys.argv[1:]:
            m = load_model(path)
            name = os.path.basename(path)[:-3]
            fps = []
            for pert in perturbations():
                fp = flip_probability(m, pert)
                fps.append(fp)
                w.writerow([name, pert, f"{fp:.3f}"])
                f.flush()
                print(f"{name}  {pert:18s} FP {fp:.3f}")
            if fps:
                w.writerow([name, "MEAN", f"{np.mean(fps):.3f}"])
                print(f"{name}  MEAN FP {np.mean(fps):.3f}")
            else:
                print(f"{name}  no perturbation files under {P_DIR}")
    print(f"appended to {out_csv}")
