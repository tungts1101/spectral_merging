"""Pure-torch error metrics between estimated and reference class Gaussians,
used by main.Learner.drift_diagnostics (E9) and tested on the CPU.

gaussian_errors compares an estimate (mu_est, cov_est) of a class with a
reference (mu_ref, cov_ref) recomputed from images under the deployed
extractor. transport_gaussian applies a drift map W to a Gaussian with the
same arithmetic as main.transport_prototypes (main.py:1229-1237). csv_append
and group_means are small helpers for the per-task CSV side output.
No timm / peft import.
"""
import csv
import math
import os

import torch

#: Jitter of transport_prototypes (main.py:1236).
JITTER = 1e-4


def transport_gaussian(mu, cov, W, jitter=JITTER):
    """(W mu, W cov W^T symmetrised + jitter I), computed in float64 and
    returned in the input dtypes; mirrors transport_prototypes."""
    Wd = W.double()
    mu_t = (mu.double() @ Wd.T).to(mu.dtype)
    S = Wd @ cov.double() @ Wd.T
    S = 0.5 * (S + S.T) + torch.eye(cov.size(-1), dtype=torch.double) * jitter
    return mu_t, S.to(cov.dtype)


def _chol_logdet(L):
    return 2.0 * torch.log(torch.diagonal(L)).sum()


def kl_gaussian(mu0, S0, mu1, S1):
    """KL(N(mu0, S0) || N(mu1, S1)) in float64 through Cholesky factors:
    0.5 [tr(S1^-1 S0) + (mu1-mu0)^T S1^-1 (mu1-mu0) - d + ln det S1 - ln det S0]."""
    mu0, mu1 = mu0.double(), mu1.double()
    S0, S1 = S0.double(), S1.double()
    d = mu0.numel()
    L0 = torch.linalg.cholesky(S0)
    L1 = torch.linalg.cholesky(S1)
    M = torch.linalg.solve_triangular(L1, L0, upper=False)        # L1^-1 L0
    trace_term = (M ** 2).sum()
    diff = (mu1 - mu0).unsqueeze(1)
    v = torch.linalg.solve_triangular(L1, diff, upper=False)      # L1^-1 (mu1-mu0)
    quad = (v ** 2).sum()
    return float(0.5 * (trace_term + quad - d + _chol_logdet(L1) - _chol_logdet(L0)))


def symmetric_kl(mu0, S0, mu1, S1):
    """0.5 [KL(0||1) + KL(1||0)]."""
    return 0.5 * (kl_gaussian(mu0, S0, mu1, S1) + kl_gaussian(mu1, S1, mu0, S0))


def gaussian_errors(mu_est, cov_est, mu_ref, cov_ref, kl=False, eps=1e-12):
    """Error metrics of an estimated Gaussian against a reference:

    mean_rel    ||mu_est - mu_ref|| / ||mu_ref||
    mean_abs    ||mu_est - mu_ref||
    mean_cos    cos(mu_est, mu_ref)
    cov_relF    ||cov_est - cov_ref||_F / ||cov_ref||_F
    trace_ratio tr(cov_est) / tr(cov_ref)
    symkl       symmetric KL (float64 Cholesky), only when kl=True
    Computed in float64.
    """
    mu_e, mu_r = mu_est.double().reshape(-1), mu_ref.double().reshape(-1)
    S_e, S_r = cov_est.double(), cov_ref.double()
    dmu = mu_e - mu_r
    nr = float(mu_r.norm())
    out = {
        "mean_rel": float(dmu.norm()) / max(nr, eps),
        "mean_abs": float(dmu.norm()),
        "mean_cos": float(torch.dot(mu_e, mu_r) / max(float(mu_e.norm()) * nr, eps)),
        "cov_relF": float((S_e - S_r).norm()) / max(float(S_r.norm()), eps),
        "trace_ratio": float(torch.trace(S_e)) / max(float(torch.trace(S_r)), eps),
    }
    if kl:
        out["symkl"] = symmetric_kl(mu_e, S_e, mu_r, S_r)
    return out


def group_means(rows, key=None):
    """Mean of every numeric field of `rows` (list of dicts) per value of
    `key` (None: one group 'all'). Each result dict also carries 'n'."""
    groups = {}
    for r in rows:
        gk = "all" if key is None else r.get(key)
        groups.setdefault(gk, []).append(r)
    out = {}
    for gk, rs in groups.items():
        acc = {}
        for r in rs:
            for f, v in r.items():
                if f == key or isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                if isinstance(v, float) and not math.isfinite(v):
                    continue
                acc.setdefault(f, []).append(float(v))
        out[gk] = {f: sum(vs) / len(vs) for f, vs in acc.items()}
        out[gk]["n"] = len(rs)
    return out


def csv_append(path, header, rows):
    """Append `rows` (dicts keyed by `header`, or sequences) to a utf-8 CSV,
    writing the header only when the file is new or empty."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(list(header))
        for r in rows:
            if isinstance(r, dict):
                w.writerow([r.get(h, "") for h in header])
            else:
                w.writerow(list(r))
