"""FGSM / PGD adversarial robustness on the clean CIFAR-100 test set
(RESULTS.md, "Known weaknesses": PGD-10 at eps = 1/255).

Attacks operate in [0,1] pixel space at 224x224 (the identity normalization
is folded into the forward), eps in {1,2,4,8}/255; PGD-10 with step eps/4 and
a random start; the first 2,000 test images.

Usage: python robustness/eval_adversarial.py <model.pt> [...]
Appends to robustness/results/adversarial.csv.
"""
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robustness.common import batched_uint8_loader, cifar_test_raw, normalize
from robustness.model_zoo import load_model

EPS = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def attack_batch(model, x01, y, eps, steps, alpha):
    # Non-differentiable routed inference (MOS/TUNA) exposes a differentiable
    # surrogate via attack_logits; the final evaluation always uses model.logits.
    grad_logits = getattr(model, "attack_logits", model.logits)
    x_adv = x01.clone()
    if steps > 1:
        x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = grad_logits(normalize(model.preprocess(x_adv)))
        loss = torch.nn.functional.cross_entropy(logits, y)
        (grad,) = torch.autograd.grad(loss, x_adv)
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = x01 + (x_adv - x01).clamp(-eps, eps)
            x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()


def robust_acc(model, eps, steps, n_max=2000, batch=64):
    x_u8, y_np = cifar_test_raw()
    y_np = model.map_labels(np.asarray(y_np))
    correct = total = 0
    for xb, yb in batched_uint8_loader(x_u8[:n_max], y_np[:n_max], batch):
        xb = xb.cuda().permute(0, 3, 1, 2).float() / 255.0
        yb = yb.cuda()
        alpha = eps if steps == 1 else eps / 4
        x_adv = attack_batch(model, xb, yb, eps, steps, alpha)
        with torch.no_grad():
            pred = model.logits(normalize(model.preprocess(x_adv))).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return 100.0 * correct / total


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, "adversarial.csv")
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "attack", "eps_x255", "acc"])
        for path in sys.argv[1:]:
            m = load_model(path)
            name = os.path.basename(path)[:-3]
            # EASE's white-box attack backprops through 10 adapter passes;
            # batch 64 does not fit 24 GB, so the attack batch shrinks for it.
            batch = 8 if "ease" in name else 64
            for eps in EPS:
                for attack, steps in [("fgsm", 1), ("pgd10", 10)]:
                    acc = robust_acc(m, eps, steps, batch=batch)
                    w.writerow([name, attack, round(eps * 255), acc])
                    f.flush()
                    print(f"{name}  {attack} eps={round(eps*255)}/255: {acc:.2f}")
    print(f"appended to {out_csv}")
