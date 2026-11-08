"""Pure-torch spectral utilities for the incremental spectral merge (ISM).

Shared by helper.pspectral_merging (the ``update_op`` step), main.Learner.merge
(the gated ``[MergeStat]`` / ``[StepNorm]`` lines), analyze_spectrum.py (E4,
E11), analyze_subspace.py (E6), run_e7_adaptive_alpha.py (E7) and
bench_svd.py (E11). The module imports torch only (no timm / peft), so it runs
with a CPU-only torch wheel and is unit-tested by tests/smoke_rebuttal.py.

Provenance of the transcribed pieces:
  * FACTOR_SUFFIXES and lora_pairs copy the factor-pair detection of
    helper.pspectral_merging (helper.py:157-167).
  * truncate_energy is a verbatim copy of helper.truncate_energy
    (helper.py:128-139); rank_from_svals / energy_rank reproduce its rule
    k = #(cumsum(S^2)/sum(S^2) < p) + 1, clamped to [1, numel(S)].
  * spectral_step('trunc') is the only_A step of helper.pspectral_merging
    (helper.py:180): alpha * truncate_energy(dA, p).

Conventions: LoRA factors are A = down [r, in] and B = up [out, r]; the
effective weight update of a pair is scale * (B @ A) with
scale = model_lora_alpha / model_lora_r (2 for the headline r=64, alpha=128).
"""
import math
import re

import torch


# ---------------------------------------------------------------------------
# Factor-pair detection (helper.pspectral_merging, helper.py:157-167)
# ---------------------------------------------------------------------------

#: (A suffix, B suffix) pairs: PEFT LoRA and AdaptFormer naming.
FACTOR_SUFFIXES = [
    ("lora_A.default.weight", "lora_B.default.weight"),
    ("down_proj.weight", "up_proj.weight"),
]

#: Update operators accepted by spectral_step / helper.pspectral_merging.
UPDATE_OPS = ("trunc", "clip", "scale_fro")


def lora_pairs(state):
    """Return the complete (A_name, B_name) factor pairs of a LoRA state dict.

    Same detection and the same order as helper.pspectral_merging: names are
    scanned in dict order, grouped by the prefix in front of the factor
    suffix, and a pair is reported once both factors were seen. Keys that
    belong to no complete pair (norm.weight, norm.bias, ...) are ignored.
    """
    pairs = {}
    for name in state:
        for a_suf, b_suf in FACTOR_SUFFIXES:
            if name.endswith(a_suf):
                pairs.setdefault(name[: -len(a_suf)], {})["A"] = name
            elif name.endswith(b_suf):
                pairs.setdefault(name[: -len(b_suf)], {})["B"] = name
    return [(d["A"], d["B"]) for d in pairs.values() if "A" in d and "B" in d]


def pair_prefix(name):
    """Prefix shared by the two factors of a pair (the key without its suffix)."""
    for a_suf, b_suf in FACTOR_SUFFIXES:
        if name.endswith(a_suf):
            return name[: -len(a_suf)]
        if name.endswith(b_suf):
            return name[: -len(b_suf)]
    return name


def pair_label(name):
    """(block index or None, module name) of a factor key, for CSV rows.

    'base_model.model.blocks.3.mlp.fc1.lora_A.default.weight' -> (3, 'fc1').
    """
    m = re.search(r"blocks\.(\d+)\.", name)
    block = int(m.group(1)) if m else None
    stem = pair_prefix(name).rstrip(".")
    module = stem.split(".")[-1] if stem else name
    return block, module


def lora_scale(lora_alpha, lora_r):
    """PEFT scaling of the effective update: alpha / r (helper.get_backbone)."""
    return float(lora_alpha) / float(lora_r)


# ---------------------------------------------------------------------------
# Energy truncation (helper.truncate_energy, helper.py:128-139)
# ---------------------------------------------------------------------------

def truncate_energy(A, p):
    """Rank-k SVD approximation of A keeping the SMALLEST set of top singular
    values whose cumulative energy (sum of squares) reaches fraction p of the
    total. p in (0,1]; p<1 keeps k < full rank.

    Verbatim copy of helper.truncate_energy so that this module needs no
    helper import; spectral_step('trunc') must stay torch.equal to
    alpha * helper.truncate_energy(dA, p)."""
    U, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)
    if S.numel() == 0 or float(S[0]) == 0.0:
        return A.clone()
    e = S ** 2
    cum = torch.cumsum(e, 0) / e.sum()
    k = int((cum < p).sum().item()) + 1          # smallest k with cum >= p
    k = max(1, min(k, S.numel()))
    return ((U[:, :k] * S[:k]) @ Vh[:k, :]).to(A.dtype)


def _svals(d):
    """Singular values of d through the same LAPACK path as truncate_energy
    (full thin SVD, not svdvals), so the kept rank is reproduced bit for bit
    even when the cumulative energy sits exactly at p."""
    return torch.linalg.svd(d.float(), full_matrices=False)[1]


def rank_from_svals(S, p):
    """Kept rank of truncate_energy for singular values S (descending).

    Returns 0 for the degenerate case (no singular values or S[0] == 0), in
    which truncate_energy returns A.clone() and the rank is undefined.
    """
    if S.numel() == 0 or float(S[0]) == 0.0:
        return 0
    e = S ** 2
    cum = torch.cumsum(e, 0) / e.sum()
    k = int((cum < p).sum().item()) + 1
    return max(1, min(k, S.numel()))


def energy_rank(d, p):
    """(k, S, cum) of a 2-D update d at energy fraction p.

    k follows the exact truncate_energy rule, S are the singular values
    (descending, float32) and cum the cumulative energy fraction. k == 0
    flags a degenerate update (see rank_from_svals).
    """
    S = _svals(d)
    if S.numel() == 0:
        return 0, S, S
    e = S ** 2
    tot = e.sum()
    cum = torch.cumsum(e, 0) / tot if float(tot) > 0.0 else torch.zeros_like(e)
    return rank_from_svals(S, p), S, cum


def energy_share(S, k):
    """Fraction of the energy sum(S^2) held by the top-k singular values."""
    if S.numel() == 0:
        return 0.0
    e = S ** 2
    tot = float(e.sum())
    if tot <= 0.0:
        return 0.0
    return float(e[: max(0, int(k))].sum()) / tot


# ---------------------------------------------------------------------------
# Effective update and the three merge-time step operators (E10)
# ---------------------------------------------------------------------------

def effective_update(w_prev, w_new, a_name, b_name, scale):
    """scale * (B_new @ A_new - B_prev @ A_prev) in float32: the change of
    the deployed weight of one adapted layer (paper Eq. 2 is stated on it)."""
    A0, B0 = w_prev[a_name].float(), w_prev[b_name].float()
    A1, B1 = w_new[a_name].float(), w_new[b_name].float()
    return float(scale) * (B1 @ A1 - B0 @ A0)


def spectral_step(d, p, alpha, op="trunc"):
    """Merge-time step applied to a factor update d (the only_A branch).

    op = 'trunc'    : alpha * truncate_energy(d, p); identical expression to
                      helper.pspectral_merging only_A (byte-identical result).
    op = 'clip'     : keep every direction, cap the singular values at
                      c = alpha * sigma_1(d): U min(S, c) Vh. Spectral norm
                      alpha * sigma_1, same as truncation; the low-energy tail
                      passes through untouched. At alpha = 1 it is the identity.
    op = 'scale_fro': scale the full update by beta = alpha * ||trunc_p(d)||_F
                      / ||d||_F, so ||step||_F equals the Frobenius norm of the
                      truncated step. At p = 1 it is damped interpolation.
    The degenerate case (no singular values or sigma_1 == 0) returns
    alpha * d.clone() for every op, as truncate_energy would.
    """
    if op == "trunc":
        return alpha * truncate_energy(d, p)
    if op not in UPDATE_OPS:
        raise ValueError(f"unknown update_op {op!r}; expected one of {UPDATE_OPS}")
    U, S, Vh = torch.linalg.svd(d.float(), full_matrices=False)
    if S.numel() == 0 or float(S[0]) == 0.0:
        return alpha * d.clone()
    if op == "clip":
        cap = float(alpha) * float(S[0])
        S_c = S.clamp(max=cap)
        return ((U * S_c) @ Vh).to(d.dtype)
    # scale_fro
    k = rank_from_svals(S, p)
    e = S ** 2
    fro_trunc = float(e[:k].sum().sqrt())
    fro_full = float(e.sum().sqrt())
    beta = float(alpha) * fro_trunc / fro_full
    return (beta * d.float()).to(d.dtype)


def step_norms(w_prev, w_after, scale, p=0.9, pairs=None):
    """Per-pair and aggregate norms of the step from w_prev to w_after.

    For every factor pair: dA = A_after - A_prev (the matrix only_A truncates)
    and eff = scale * (B_after A_after - B_prev A_prev) (the deployed change).
    Call it with w_after = the merged state for the realised step, or with
    w_after = W_t for the raw update (the '[StepNorm]' line of merge-less runs).

    Returns {'pairs': {prefix: {k, n_sv, sigma1_A, fro_A, sigma1_eff,
    fro_eff}}, 'agg': {n_pairs, k_mean, k_min, k_max, sigma1_A_mean,
    sigma1_A_max, fro_A_mean, sigma1_eff_mean, sigma1_eff_max, fro_eff_mean}}.
    k is the truncate_energy rank of dA at p (0 = degenerate).
    """
    if pairs is None:
        pairs = lora_pairs(w_after)
    per = {}
    for an, bn in pairs:
        A0, B0 = w_prev[an].float(), w_prev[bn].float()
        A1, B1 = w_after[an].float(), w_after[bn].float()
        dA = A1 - A0
        S_A = _svals(dA)
        k = rank_from_svals(S_A, p)
        eff = float(scale) * (B1 @ A1 - B0 @ A0)
        S_e = _svals(eff)
        per[pair_prefix(an)] = {
            "k": k,
            "n_sv": int(S_A.numel()),
            "sigma1_A": float(S_A[0]) if S_A.numel() else 0.0,
            "fro_A": float(dA.norm()),
            "sigma1_eff": float(S_e[0]) if S_e.numel() else 0.0,
            "fro_eff": float(eff.norm()),
        }
    n = len(per)
    if n == 0:
        agg = {"n_pairs": 0}
    else:
        ks = [v["k"] for v in per.values()]
        s1a = [v["sigma1_A"] for v in per.values()]
        s1e = [v["sigma1_eff"] for v in per.values()]
        agg = {
            "n_pairs": n,
            "k_mean": sum(ks) / n,
            "k_min": min(ks),
            "k_max": max(ks),
            "sigma1_A_mean": sum(s1a) / n,
            "sigma1_A_max": max(s1a),
            "fro_A_mean": sum(v["fro_A"] for v in per.values()) / n,
            "sigma1_eff_mean": sum(s1e) / n,
            "sigma1_eff_max": max(s1e),
            "fro_eff_mean": sum(v["fro_eff"] for v in per.values()) / n,
        }
    return {"pairs": per, "agg": agg}


# ---------------------------------------------------------------------------
# Adaptive alpha rule (E7)
# ---------------------------------------------------------------------------

def alpha_from_stat(stat, param, alpha_min=0.05, alpha_max=1.0):
    """alpha_t = clip(param / stat, alpha_min, alpha_max).

    stat is the per-task statistic (mean sigma_max of the update for the
    'specnorm' rule, the relative feature shift for 'featshift'); param is
    tau (kappa). A non-positive or non-finite stat means "no update", which
    maps to alpha_max. The rule is non-increasing in stat.
    """
    stat = float(stat)
    param = float(param)
    lo, hi = float(alpha_min), float(alpha_max)
    if not math.isfinite(stat) or stat <= 0.0:
        return hi
    return float(min(hi, max(lo, param / stat)))
