"""Pure-torch class-statistics variants for Stage 2 (E5) and the drift-map
diagnostics (E9): fitting, drift transport, LCA sampling distributions and
storage accounting for the covariance modes

    class    : one full covariance per class (the method; main.py default)
    diag     : one variance vector per class
    lowrank  : rank-k factor U (d, k) plus a residual diagonal per class
    shared   : one pooled within-class covariance for all classes (+ means)

main.py keeps its own 'class' / 'shared_corr' code paths byte-identical and
calls these functions only for the new modes, so this module must not be
imported by anything that cannot afford a torch import. No timm / peft.

Provenance: fit_full mirrors compute_multivariate_normal (main.py:1143-1161
with shrinkage gamma = 0: torch.cov + 1e-4 I, a single sample duplicated);
transport_full mirrors transport_prototypes (main.py:1229-1237: double
precision, W S W^T symmetrised + 1e-4 I, stored as float32); make_dist('class')
mirrors align (main.py:1271-1275: MultivariateNormal(mu, cov) on the device).
"""
import torch
from torch.distributions import (
    Independent,
    LowRankMultivariateNormal,
    MultivariateNormal,
    Normal,
)

#: Covariance modes accepted by train_ca_cov_mode after E5.
COV_MODES = ("class", "shared_corr", "diag", "lowrank", "shared")

#: Jitter added to every covariance before it is stored or sampled (main.py).
JITTER = 1e-4

#: PEFT LoRA shapes of helper.get_backbone (in, out) for ViT-B/16, r on every
#: target module; the LayerNorm of Model21 adds 2 * 768 untrained floats.
LORA_SHAPES = {"qkv": (768, 2304), "fc1": (768, 3072), "fc2": (3072, 768)}
NORM_FLOATS = 2 * 768


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _prep(feats):
    """float32 (n, d) feature matrix; a single sample is duplicated so that
    torch.cov is defined (zero covariance + jitter), as main.py:1143-1145."""
    X = feats.float()
    if X.dim() != 2:
        raise ValueError(f"expected (n, d) features, got {tuple(X.shape)}")
    if X.size(0) == 1:
        X = torch.cat([X, X], dim=0)
    return X


def fit_full(feats, jitter=JITTER):
    """(mean (d,), cov (d, d)) with cov = torch.cov(X^T) + jitter I."""
    X = _prep(feats)
    d = X.size(1)
    return X.mean(dim=0), torch.cov(X.T) + torch.eye(d) * jitter


def fit_diag(feats, jitter=JITTER):
    """Unbiased per-dimension variance + jitter (d,): the diagonal of
    fit_full's covariance, so 'diag' and 'class' agree on the diagonal."""
    X = _prep(feats)
    return X.var(dim=0, unbiased=True) + jitter


def fit_lowrank(feats, rank, jitter=JITTER):
    """(U (d, k), var_res (d,)) with U U^T + diag(var_res) approximating the
    full covariance: U = V_k sqrt(lam_k) from the top-k eigenpairs of
    torch.cov(X^T), var_res = diag(cov) - diag(U U^T) clamped >= 0 + jitter.
    The diagonal of the approximation equals the diagonal of fit_full."""
    X = _prep(feats)
    d = X.size(1)
    k = max(1, min(int(rank), d))
    cov = torch.cov(X.T)
    lam, V = torch.linalg.eigh(cov.double())          # ascending eigenvalues
    lam_k = lam[-k:].clamp_min(0.0)
    U = (V[:, -k:] * lam_k.sqrt()).float()
    var_res = (cov.diagonal() - (U ** 2).sum(dim=1)).clamp_min(0.0) + jitter
    return U, var_res


def scatter(feats):
    """Centred scatter (X - mean)^T (X - mean) in float64 (d, d): the pooled
    within-class accumulator of the 'shared' mode. The matching count is
    n - 1 per class (unbiased pooled covariance)."""
    X = _prep(feats).double()
    Xc = X - X.mean(dim=0, keepdim=True)
    return Xc.T @ Xc


def shared_cov(scatter_sum, count, jitter=JITTER):
    """Pooled covariance scatter / max(1, count) + jitter I as float32."""
    d = scatter_sum.size(0)
    return (scatter_sum.double() / max(1, int(count))).float() + torch.eye(d) * jitter


def shared_scale_tril(scatter_sum, count, jitter=JITTER):
    """Cholesky factor (float32) of shared_cov, computed in float64, for
    make_dist('shared'); build it once per align() call."""
    d = scatter_sum.size(0)
    S = scatter_sum.double() / max(1, int(count)) + torch.eye(d, dtype=torch.double) * jitter
    return torch.linalg.cholesky(S).float()


# ---------------------------------------------------------------------------
# Drift transport (transport_prototypes, main.py:1229-1237)
# ---------------------------------------------------------------------------

def transport_full(S, W, jitter=JITTER):
    """W S W^T, symmetrised, + jitter I, in float64, returned in S.dtype.
    S may be (d, d) or a batch (K, d, d). Pass jitter=0.0 for an
    unnormalised scatter (the 'shared' accumulator)."""
    Wd = W.double()
    out = Wd @ S.double() @ Wd.T
    out = 0.5 * (out + out.transpose(-1, -2))
    if jitter:
        out = out + torch.eye(S.size(-1), dtype=torch.double, device=out.device) * jitter
    return out.to(S.dtype)


def transport_diag(v, W, jitter=0.0):
    """Exact diagonal of W diag(v) W^T: v @ (W * W)^T (+ jitter).
    v may be (d,) or (K, d); W is (d, d)."""
    W2 = (W.double() ** 2)
    out = v.double() @ W2.T
    if jitter:
        out = out + jitter
    return out.to(v.dtype)


def transport_lowrank_U(U, W):
    """W U for one factor (d, k) or a batch (K, d, k). The transported
    covariance is (W U)(W U)^T + W diag(v) W^T; the second term is kept by its
    exact diagonal transport_diag(v, W)."""
    return (W.double() @ U.double()).to(U.dtype)


# ---------------------------------------------------------------------------
# Full covariance reconstruction (diagnostics, E9) and sampling (align)
# ---------------------------------------------------------------------------

def to_full_cov(mode, cov=None, var=None, U=None, shared=None):
    """(d, d) covariance implied by one class's statistics in `mode`."""
    if mode in ("class", "shared_corr"):
        if cov is None:
            raise ValueError("mode 'class' needs cov")
        return cov
    if mode == "diag":
        if var is None:
            raise ValueError("mode 'diag' needs var")
        return torch.diag(var)
    if mode == "lowrank":
        if U is None or var is None:
            raise ValueError("mode 'lowrank' needs U and var")
        return U.float() @ U.float().T + torch.diag(var.float())
    if mode == "shared":
        if shared is None:
            raise ValueError("mode 'shared' needs the shared covariance")
        return shared
    raise ValueError(f"unknown cov mode {mode!r}; expected one of {COV_MODES}")


def make_dist(mode, mu, cov=None, var=None, U=None, shared_L=None, device=None):
    """Sampling distribution of one class for align(); every return value
    exposes .sample((K,)) -> (K, d) on `device` (default: mu's device).

    class / shared_corr : MultivariateNormal(mu, cov)            (as main.py)
    diag                : Independent(Normal(mu, sqrt(var)), 1)
    lowrank             : LowRankMultivariateNormal(mu, U, var)
    shared              : MultivariateNormal(mu, scale_tril=shared_L)
    """
    if device is None:
        device = mu.device
    mu = mu.to(device).float()
    if mode in ("class", "shared_corr"):
        if cov is None:
            raise ValueError("mode 'class' needs cov")
        return MultivariateNormal(mu, cov.to(device).float())
    if mode == "diag":
        if var is None:
            raise ValueError("mode 'diag' needs var")
        return Independent(Normal(mu, var.to(device).float().clamp_min(1e-12).sqrt()), 1)
    if mode == "lowrank":
        if U is None or var is None:
            raise ValueError("mode 'lowrank' needs U and var")
        return LowRankMultivariateNormal(
            mu, U.to(device).float(), var.to(device).float().clamp_min(1e-12)
        )
    if mode == "shared":
        if shared_L is None:
            raise ValueError("mode 'shared' needs shared_L (see shared_scale_tril)")
        return MultivariateNormal(mu, scale_tril=shared_L.to(device).float())
    raise ValueError(f"unknown cov mode {mode!r}; expected one of {COV_MODES}")


# ---------------------------------------------------------------------------
# Storage accounting (E5 memory table)
# ---------------------------------------------------------------------------

def storage_floats(mode, d, C, rank=None):
    """Number of floats the class statistics of C classes occupy in `mode`.

    class / shared_corr / full : C (d + d^2)       (d=768: 590,592 per class)
    full_sym                   : C (d + d(d+1)/2)  (upper triangle: 296,064)
    diag                       : 2 C d             (1,536 per class)
    lowrank                    : C d (rank + 2)    (k=16: 13,824; k=64: 50,688)
    shared                     : C d + d^2
    'shared_corr' is counted as full because main.py materialises every class
    covariance in that mode.
    """
    d, C = int(d), int(C)
    if mode in ("class", "shared_corr", "full"):
        return C * (d + d * d)
    if mode == "full_sym":
        return C * (d + d * (d + 1) // 2)
    if mode == "diag":
        return C * 2 * d
    if mode == "lowrank":
        if rank is None:
            raise ValueError("mode 'lowrank' needs rank")
        return C * d * (int(rank) + 2)
    if mode == "shared":
        return C * d + d * d
    raise ValueError(f"unknown storage mode {mode!r}")


def lora_param_count(r, blocks=12, shapes=None, norm=NORM_FLOATS):
    """Floats of one cached LoRA state: r (in + out) per target module and
    block, plus the Model21 LayerNorm (norm=2*768; pass norm=0 for the
    trainable count 129,024 r). r=64 on qkv+fc1+fc2 gives 8,259,072."""
    if shapes is None:
        shapes = LORA_SHAPES
    per_block = sum(int(r) * (i + o) for i, o in shapes.values())
    return int(blocks) * per_block + int(norm)
