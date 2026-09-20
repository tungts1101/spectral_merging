"""SOTA merging baselines for the incremental two-state setting.

Each rule merges the previous DEPLOYED state w_prev with the newly trained
state w_new (LoRA state dicts), producing the next deployed state — the same
sequential protocol as the spectral and classical branches in exp21:

    C = w_prev + lamb * R(w_new - w_prev)        (update-transform rules)

where R is the rule's sparsification/transform of the update. KnOTS and
Model Stock instead operate on the two task vectors (w - base) jointly, per
their publications, adapted pairwise.

Rules (references):
  dare        - drop fraction q of update entries uniformly at random,
                rescale survivors by 1/(1-q). (Yu et al.; "DARE the
                extreme", ICLR'25)
  della       - magnitude-proportional probabilistic drop with rescaling
                (rank-based keep probability), DARE's structured cousin.
  breadcrumbs - deterministic layer-wise band-pass magnitude mask: drop the
                top-gamma fraction (outliers) AND bottom-beta fraction
                (noise) of |update|, keep the middle band. (ECCV'24)
  knots       - pairwise KnOTS-TIES: joint SVD of the two task vectors
                stacked in a shared row basis, sign-consensus (TIES) merge
                of the aligned right factors, reconstruct. (ICLR'25)
  modelstock  - analytic anchor interpolation: w = t*avg(tv1,tv2) + base,
                t = 2cos(theta)/(1+cos(theta)) per layer, theta the angle
                between the two task vectors. (ECCV'24)

All rules are training-free and produce a single merged state.
"""
import torch


def _flat(t):
    return t.float().reshape(-1)


def dare_update(delta, q):
    if q <= 0:
        return delta
    mask = torch.rand_like(delta.float()) >= q
    return (delta.float() * mask / (1.0 - q)).to(delta.dtype)


def della_update(delta, q):
    """Keep probability proportional to magnitude rank (higher |d| ->
    higher keep prob), mean keep rate = 1-q, rescale kept by 1/p_keep."""
    if q <= 0:
        return delta
    d = delta.float().reshape(-1)
    n = d.numel()
    ranks = torch.empty_like(d)
    ranks[d.abs().argsort()] = torch.arange(n, dtype=d.dtype)
    p_keep = (ranks + 1) / n * 2 * (1.0 - q)
    p_keep = p_keep.clamp(1e-3, 1.0)
    mask = torch.rand_like(d) < p_keep
    out = torch.zeros_like(d)
    out[mask] = d[mask] / p_keep[mask]
    return out.reshape(delta.shape).to(delta.dtype)


def breadcrumbs_update(delta, beta, gamma):
    """Keep entries with |d| in the (beta, 1-gamma) quantile band."""
    d = delta.float()
    a = d.abs().reshape(-1)
    if a.numel() < 10:
        return delta
    lo = torch.quantile(a, beta)
    hi = torch.quantile(a, 1.0 - gamma)
    mask = (d.abs() >= lo) & (d.abs() <= hi)
    return (d * mask).to(delta.dtype)


def modelstock_pair(base, w1, w2):
    """Per-tensor Model Stock for two fine-tuned states and an anchor."""
    out = {}
    for k in w2:
        t1, t2, b = _flat(w1[k] - base[k]), _flat(w2[k] - base[k]), base[k]
        denom = t1.norm() * t2.norm()
        if denom < 1e-12:
            out[k] = ((w1[k].float() + w2[k].float()) / 2).to(w2[k].dtype)
            continue
        cos = float(torch.dot(t1, t2) / denom)
        cos = max(min(cos, 1.0), -1.0)
        t = 2.0 * cos / (1.0 + cos) if cos > -0.999 else 0.0
        avg = (w1[k].float() + w2[k].float()) / 2
        out[k] = (t * avg + (1.0 - t) * b.float()).to(w2[k].dtype)
    return out


def knots_pair(base, w1, w2, topk_pct=50.0, lamb=1.0):
    """Pairwise KnOTS-TIES on each 2-D tensor: stack the two task vectors
    row-wise, joint SVD -> shared right basis; the aligned per-model
    coefficient blocks are TIES-merged (magnitude trim + sign consensus),
    then reconstructed. 1-D tensors fall back to mean of task vectors."""
    out = {}
    for k in w2:
        tv1 = (w1[k] - base[k]).float()
        tv2 = (w2[k] - base[k]).float()
        if tv1.dim() != 2:
            out[k] = (base[k].float() + lamb * (tv1 + tv2) / 2).to(w2[k].dtype)
            continue
        stacked = torch.cat([tv1, tv2], dim=0)           # [2m, n]
        U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
        coef = U * S                                      # [2m, r]
        m = tv1.shape[0]
        c1, c2 = coef[:m], coef[m:]
        merged_c = _ties_pair(c1, c2, topk_pct)
        out[k] = (base[k].float() + lamb * (merged_c @ Vh)).to(w2[k].dtype)
    return out


def _ties_pair(a, b, topk_pct):
    def trim(x):
        f = x.reshape(-1)
        k = max(1, int(f.numel() * topk_pct / 100.0))
        thr = f.abs().topk(k).values.min()
        return torch.where(x.abs() >= thr, x, torch.zeros_like(x))
    a, b = trim(a), trim(b)
    sign = torch.sign(a + b)
    keep_a = (torch.sign(a) == sign) & (sign != 0)
    keep_b = (torch.sign(b) == sign) & (sign != 0)
    num = a * keep_a + b * keep_b
    cnt = keep_a.float() + keep_b.float()
    return num / cnt.clamp(min=1.0)


def sota_merge(base, w_prev, w_new, method, lamb=1.0, dare_q=0.9,
               della_q=0.5, bc_beta=0.85, bc_gamma=0.01, knots_topk=50.0):
    """Dispatch. Returns the merged state dict."""
    if method == "modelstock":
        return modelstock_pair(base, w_prev, w_new)
    if method == "knots":
        return knots_pair(base, w_prev, w_new, topk_pct=knots_topk, lamb=lamb)
    # Sparsification rules follow their publications: sparsify each TASK
    # VECTOR (w - base) independently, then merge by elementwise mean and
    # add back to the base scaled by lamb — the identical protocol to the
    # classical TIES/avg branch (helper.merge with incremental task vectors).
    out = {}
    for k in w_new:
        tvs = [w_prev[k] - base[k], w_new[k] - base[k]]
        if method == "dare":
            rs = [dare_update(t, dare_q) for t in tvs]
        elif method == "della":
            rs = [della_update(t, della_q) for t in tvs]
        elif method == "breadcrumbs":
            rs = [breadcrumbs_update(t, bc_beta, bc_gamma) for t in tvs]
        else:
            raise ValueError(f"unknown sota merge method {method!r}")
        merged_tv = (rs[0].float() + rs[1].float()) / 2
        out[k] = (base[k].float() + lamb * merged_tv).to(w_new[k].dtype)
    return out


SOTA_METHODS = ("dare", "della", "breadcrumbs", "knots", "modelstock")
