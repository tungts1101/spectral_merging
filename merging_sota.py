"""SOTA merging baselines for the incremental two-state setting.

Each rule merges the previous DEPLOYED state w_prev with the newly trained
state w_new (LoRA state dicts), producing the next deployed state — the same
sequential protocol as the spectral and classical branches in main.py:

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
  opcm        - OPCM: project the new task vector onto the complement of
                the old merged task vector (zero diagonal and dominant
                r_alpha block in its SVD basis), accumulate with one global
                norm-tracking scalar lam_t. (Tang et al., NeurIPS'25)
  nufilt      - NUFILT: null-space filter of the new task vector on the
                top-r_p input directions of the old merged task vector,
                optional data-free rank-r_l LoRA re-adaptation, additive
                fuse without a coefficient. (Qiu et al., ICLR'26)

The first five rules are training-free and produce a single merged state.
The two continual-merging rules (rebuttal E15a) take tau_t = w_new - base
and dM = w_prev - base per tensor, ignore `lamb`, and are data-free
(NUFILT's optional adaptation fits a LoRA on a data-free objective); see
the section before them.
"""
import math

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


# ---------------------------------------------------------------------------
# Continual-merging rules: OPCM and NUFILT (rebuttal experiment E15a)
# ---------------------------------------------------------------------------
#
# Both rules come from continual model merging, where every task vector
# tau_t = theta_t - theta_0 belongs to an independently fine-tuned expert and
# the running merged model theta_bar_{t-1} is the only memory of the past.
# On the sequential chain of this repo the SOTA branch of Learner.merge
# supplies base = W_0, w_prev = M_{t-1} (deployed state) and w_new = W_t, so
#     tau_t = W_t - W_0          (cumulative, like the DARE/DELLA/KnOTS rows)
#     dM    = M_{t-1} - W_0      (old knowledge, recomputed from the merged
#                                 model as both papers do).
# Both rules act per tensor in LoRA-factor space (lora_A [r, in], lora_B
# [out, r]) like every other rule of this module; 1-D tensors (norm.*) use
# the identity projection.  Internal arithmetic is float64; results are cast
# back to the tensor's dtype.  Neither rule uses the merge coefficient
# `lamb` of the sparsification rules: OPCM carries its own global scalar
# lam_t and NUFILT fuses additively by construction.

_LORA_A = "lora_A.default.weight"      # key suffixes of the PEFT LoRA factors
_LORA_B = "lora_B.default.weight"      # (same test as helper.pspectral_merging)
_ZERO_REL = 1e-7                       # ||dM|| <= _ZERO_REL ||tau||: no old knowledge (t = 1)


def _thin_svd(x, ref=None):
    """Thin SVD of a 2-D float64 tensor, truncated to its numerical rank.

    Returns (U, S, Vh) with k columns, k = number of singular values above
    the numpy.linalg.matrix_rank tolerance s_max * max(m, n) * eps.  k = 0
    for a zero matrix, and also when `ref` is given and ||x||_F <=
    _ZERO_REL * ||ref||_F (the t = 1 case, w_prev == base, where the SVD
    basis of dM would be pure round-off noise)."""
    if ref is not None and float(x.norm()) <= _ZERO_REL * float(ref.norm()):
        return (x.new_zeros((x.shape[0], 0)), x.new_zeros((0,)),
                x.new_zeros((0, x.shape[1])))
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    if S.numel() == 0 or float(S[0]) <= 0.0:
        k = 0
    else:
        tol = float(S[0]) * max(x.shape) * torch.finfo(x.dtype).eps
        k = int((S > tol).sum().item())
    return U[:, :k], S[:k], Vh[:k]


def _energy_rank(s, alpha):
    """OPCM's r_alpha: the smallest r with sum_{i<=r} s_i >= alpha * sum_i s_i
    (Tang et al., NeurIPS 2025, Sec. 3.2; singular values, not their
    squares).  0 for alpha <= 0 or an empty spectrum, all of them for
    alpha >= 1."""
    k = int(s.numel())
    if k == 0 or alpha <= 0.0:
        return 0
    if alpha >= 1.0:
        return k
    return min(k, int((s.cumsum(0) < alpha * s.sum()).sum().item()) + 1)


def _opcm_project(dM, tau, alpha):
    """OPCM projection of the new task vector `tau` against the old merged
    task vector `dM` (Tang et al., NeurIPS 2025, Sec. 3.2, Eq. 7-9; the
    semantics of fusion_bench/method/opcm/opcm.py).

    With dM = U S V^T, the coefficient matrix C = U^T tau V gets its diagonal
    zeroed (so that <P(tau), dM>_F = sum_i s_i C_ii = 0) and its top-left
    r_alpha x r_alpha block zeroed (no interaction with the dominant
    subspace of the old knowledge), then P(tau) = U C V^T in the full
    orthonormal bases: the components of tau outside span(U) x span(V) are
    kept.  Written without the completion vectors:

        P(tau) = tau - sum_{r_alpha < i <= rank} (u_i^T tau v_i) u_i v_i^T
                     - U_a (U_a^T tau V_a) V_a^T,

    U_a, V_a = first r_alpha singular vectors.  Only the numerically
    non-zero singular directions take part (diagonal entries beyond the
    rank would pair arbitrary completion vectors and do not enter the inner
    product with dM).  Returns (P(tau), r_alpha); a zero dM (t = 1) gives
    P = identity and r_alpha = 0.  Both inputs are 2-D float64."""
    U, S, Vh = _thin_svd(dM, ref=tau)
    k = int(S.numel())
    proj = tau.clone()
    if k == 0:
        return proj, 0
    r_a = _energy_rank(S, alpha)
    C = U.T @ tau @ Vh.T                                  # [k, k]
    if r_a > 0:                                           # top-left block
        proj -= U[:, :r_a] @ C[:r_a, :r_a] @ Vh[:r_a]
    if k > r_a:                                           # remaining diagonal
        d = C.diagonal()[r_a:]
        proj -= (U[:, r_a:] * d) @ Vh[r_a:]
    return proj, r_a


def opcm_pair(base, w_prev, w_new, alpha=0.5, lam_rule="adaptive", state=None,
              task=None):
    """One OPCM step (Tang et al., NeurIPS 2025, Alg. 1; fusion_bench
    opcm.py semantics) on the pair (M_{t-1}, W_t) against base = W_0.

    Per 2-D tensor: P = _opcm_project(dM, tau, alpha).  Accumulator
    S_t = lam_{t-1} dM + P(tau_t) and M_t = base + S_t / lam_t; 1-D tensors
    use P = identity (plain weighted averaging, as the paper does for
    biases and norms).  lam_t is one global scalar for the whole state:
      'adaptive' : ||S_t||_F / mean_{i<=t} ||tau_i||_F  (paper default;
                   keeps ||M_t - base|| at the mean task-vector norm),
      'sqrt'     : sqrt(t)                              (closed form).
    At t = 1 (w_prev == base) dM = 0, P = identity, lam_1 = 1 and M_1 = W_1
    exactly, as the paper initialises.
    In the independent-init regime (seqft_independent, base = fresh init) the
    first merge sees w_prev = W_0 with dM != 0: that expert is counted too
    (norms[t-1] = ||dM||_F, lam_{t-1} = 1), so 'sqrt' gives sqrt(#experts).

    `state` is a dict kept by the caller across tasks (main.py passes
    Learner._merge_rule_state; valid because reset_merge is True).  After
    the call it holds
      t, lam (= lam_t), lam_prev (= lam_{t-1}), norms {i: ||tau_i||_F},
      norm_mean, r_alpha_mean (over 2-D tensors) and kept_frac
      (||P(tau)||_F / ||tau||_F over 2-D tensors; 1.0 at t = 1).
    `task` is the step index t >= 1 (Learner._cur_task); when None the
    state's own counter continues.  Re-merging the same t is idempotent.
    Without a usable previous lam in the state (no state, or a gap in the
    sequence) lam_{t-1} falls back to the sqrt(t-1) closed form."""
    if lam_rule not in ("adaptive", "sqrt"):
        raise ValueError(f"unknown opcm lam rule {lam_rule!r}")
    st = {} if state is None else state
    t = int(task) if task is not None else int(st.get("t", 0)) + 1
    if t < 1:
        raise ValueError(f"opcm task index must be >= 1, got {t}")
    if st.get("t") == t and "lam_prev" in st:            # same task again
        lam_prev = float(st["lam_prev"])
    elif st.get("t") == t - 1 and "lam" in st:           # the normal chain
        lam_prev = float(st["lam"])
    else:                                                # no history
        lam_prev = 1.0 if t <= 1 else math.sqrt(t - 1)

    has_history = st.get("t") in (t, t - 1)

    acc, ranks = {}, []
    tau_sq = proj_sq = tau_sq_2d = dM_sq = 0.0
    for k in w_new:
        tau = w_new[k].double() - base[k].double()
        dM = w_prev[k].double() - base[k].double()
        tau_sq += float(tau.pow(2).sum())
        dM_sq += float(dM.pow(2).sum())
        if tau.dim() == 2:
            proj, r_a = _opcm_project(dM, tau, alpha)
            ranks.append(r_a)
            tau_sq_2d += float(tau.pow(2).sum())
            proj_sq += float(proj.pow(2).sum())
        else:
            proj = tau
        acc[k] = lam_prev * dM + proj

    norms = {int(i): n for i, n in dict(st.get("norms", {})).items() if int(i) < t}
    if not has_history and (t - 1) not in norms and dM_sq > (_ZERO_REL ** 2) * tau_sq:
        # First merge of a chain whose previous module is NOT the base (the
        # independent-init regime, seqft_independent=True: the base stays the
        # fresh initial module, so M_{t-1} = W_{t-1} is itself an expert with
        # task vector dM, lam_{t-1} = 1).  Count it, or norm_mean and every
        # later lam_t would be biased.  In the sequential regime dM = 0 here.
        norms[t - 1] = math.sqrt(dM_sq)
    norms[t] = math.sqrt(tau_sq)
    norm_mean = sum(norms.values()) / len(norms)
    if lam_rule == "sqrt":
        lam_t = math.sqrt(len(norms))                    # sqrt(number of experts in M_t)
    else:
        # ||S_t||_F / mean_i ||tau_i||_F; at the paper's first step (dM = 0,
        # one expert) this is ||tau||/||tau|| = 1, so no special case.
        acc_norm = math.sqrt(sum(float(v.pow(2).sum()) for v in acc.values()))
        lam_t = acc_norm / norm_mean if norm_mean > 0.0 else 1.0
    if not lam_t > 0.0:                                  # all-zero update
        lam_t = 1.0

    out = {}
    for k in w_new:
        out[k] = (base[k].double() + acc[k] / lam_t).to(w_new[k].dtype)
    st.update(
        t=t, lam=lam_t, lam_prev=lam_prev, norms=norms, norm_mean=norm_mean,
        r_alpha_mean=(sum(ranks) / len(ranks)) if ranks else 0.0,
        kept_frac=math.sqrt(proj_sq / tau_sq_2d) if tau_sq_2d > 0.0 else 1.0,
    )
    return out


def _nufilt_filter(dM, tau, r_p, side="right"):
    """NUFILT null-space filter (Qiu et al., ICLR 2026, Sec. 3.2
    'Filtering') of the new task vector `tau` against the old merged task
    vector `dM`, both 2-D float64.

    side='right' (the paper's rule; lora_A [r, in] and generic [out, in]
    weights): V_hat = top-min(r_p, rank) right singular vectors of dM,
    P = I - V_hat V_hat^T, tau_f = tau P = tau - (tau V_hat) V_hat^T, so the
    filtered update has zero response on the old input subspace.
    side='left' (our transpose analogue for lora_B [out, r], whose
    r-dimensional input is the LoRA rank space): U_hat = top-min(r_p, rank)
    left singular vectors of dM, tau_f = tau - U_hat (U_hat^T tau).
    r_p clamps to the numerical rank of dM, at most r for a LoRA factor, so
    r_p >= r removes the whole stored subspace.  Returns (tau_f, basis) with
    basis = V_hat [in, k] or U_hat [out, k]; k = 0 and tau_f = tau when dM is
    zero (t = 1)."""
    if side not in ("right", "left"):
        raise ValueError(f"unknown nufilt side {side!r}")
    U, S, Vh = _thin_svd(dM, ref=tau)
    k = max(0, min(int(r_p), int(S.numel())))
    if side == "right":
        V = Vh[:k].T                                      # [in, k]
        return tau - (tau @ V) @ V.T, V
    Ub = U[:, :k]                                         # [out, k]
    return tau - Ub @ (Ub.T @ tau), Ub


def _nufilt_adapt(dM, tau, tau_f, V_old, r_v, r_l, iters, lr, generator):
    """Data-free rank-r_l adaptation of NUFILT (Qiu et al., ICLR 2026,
    Sec. 3.2 'Adapting'; objective as transcribed in
    merging_methods_catalog.md, section NUFILT):

        min_{A,B} || T - (M + tau B A) V_hat ||_F^2,
        M = dM + tau_f,   V_hat = [V_old, V_new],   T = [dM V_old, tau V_new],

    V_new = top-min(r_v, rank) right singular vectors of tau (the new
    task's principal input directions).  Because tau_f V_old = 0 the first
    block asks the LoRA to stay silent on the old subspace, the second asks
    the fused update to match the new expert on V_new.  B [in, r_l] starts
    at zero, A [r_l, in] ~ N(0, 1/in) from `generator` (the paper states
    A ~ N(0, sigma^2) without giving sigma); Adam, `iters` steps at `lr`.
    Returns (tau @ B @ A, losses): the dense correction to fuse and the
    objective before every step plus once after the last one."""
    n = tau.shape[1]
    _, S_new, Vh_new = _thin_svd(tau)
    kv = max(0, min(int(r_v), int(S_new.numel())))
    V_new = Vh_new[:kv].T                                 # [in, kv]
    V_hat = torch.cat([V_old, V_new], dim=1)              # [in, k_p + kv]
    M = dM + tau_f
    T = torch.cat([dM @ V_old, tau @ V_new], dim=1)       # [r, k_p + kv]
    resid0 = M @ V_hat - T                                # LoRA-free residual
    rl = max(1, min(int(r_l), n))
    A = torch.randn(rl, n, generator=generator, dtype=tau.dtype) / math.sqrt(n)
    A = A.to(tau.device).requires_grad_(True)
    B = torch.zeros(n, rl, dtype=tau.dtype, device=tau.device,
                    requires_grad=True)
    losses = []
    with torch.enable_grad():                             # merge() may run under no_grad
        opt = torch.optim.Adam([A, B], lr=lr)
        for _ in range(max(0, int(iters))):
            opt.zero_grad()
            loss = (resid0 + tau @ (B @ (A @ V_hat))).pow(2).sum()
            losses.append(loss.detach().item())
            loss.backward()
            opt.step()
    with torch.no_grad():
        losses.append((resid0 + tau @ (B @ (A @ V_hat))).pow(2).sum().item())
        delta = tau @ (B @ A)
    return delta.detach(), losses


def nufilt_pair(base, w_prev, w_new, r_p=128, r_v=8, r_l=64, adapt=False,
                iters=50, lr=1e-3, state=None, seed=0):
    """One NUFILT step (Qiu et al., ICLR 2026, Alg. 1: filter, adapt, fuse)
    on the pair (M_{t-1}, W_t) against base = W_0.

    tau = w_new - base is the incoming task vector, dM = w_prev - base the
    old knowledge.  lora_A tensors (and any other [out, in] weight): the
    paper's right-side filter on the top-r_p input directions of dM and,
    with adapt=True, the data-free rank-r_l LoRA fitted by Adam (`iters`,
    `lr`) on the projection-aware objective of _nufilt_adapt; lora_B
    tensors: the left-side filter only; 1-D tensors: identity.  Fuse
    without a coefficient, M_t = w_prev + tau (P + B A), as in the paper.
    At t = 1 (w_prev == base) nothing is filtered or adapted and M_1 = W_1
    exactly.  The paper's defaults are r_p = 128, r_v = 8, r_l = 64,
    50 Adam steps at lr 1e-3; r_p clamps to the LoRA rank here.

    `state`, when given, receives the statistics of this step:
      filtered_frac  = 1 - sum ||tau_f||^2 / sum ||tau||^2 over 2-D tensors,
      rp_eff_mean    = mean number of removed directions per 2-D tensor,
      n_adapted, adapt_loss_first, adapt_loss_last (sums over adapted
      tensors; 0 when no adaptation ran).
    `seed` fixes the private generator of the LoRA initialisation, so the
    result does not depend on the global torch RNG."""
    st = {} if state is None else state
    gen = torch.Generator().manual_seed(int(seed))
    out = {}
    tau_sq = filt_sq = 0.0
    ranks, loss_first, loss_last, n_adapted = [], 0.0, 0.0, 0
    for k in w_new:
        tau = w_new[k].double() - base[k].double()
        if tau.dim() != 2:
            out[k] = (w_prev[k].double() + tau).to(w_new[k].dtype)
            continue
        dM = w_prev[k].double() - base[k].double()
        side = "left" if k.endswith(_LORA_B) else "right"
        tau_f, basis = _nufilt_filter(dM, tau, r_p, side)
        ranks.append(basis.shape[1])
        tau_sq += float(tau.pow(2).sum())
        filt_sq += float(tau_f.pow(2).sum())
        update = tau_f
        if adapt and side == "right" and basis.shape[1] > 0:
            delta, losses = _nufilt_adapt(dM, tau, tau_f, basis, r_v, r_l,
                                          iters, lr, gen)
            update = tau_f + delta
            loss_first += losses[0]
            loss_last += losses[-1]
            n_adapted += 1
        out[k] = (w_prev[k].double() + update).to(w_new[k].dtype)
    st.update(
        filtered_frac=(1.0 - filt_sq / tau_sq) if tau_sq > 0.0 else 0.0,
        rp_eff_mean=(sum(ranks) / len(ranks)) if ranks else 0.0,
        n_adapted=n_adapted, adapt_loss_first=loss_first,
        adapt_loss_last=loss_last,
    )
    return out


def sota_merge(base, w_prev, w_new, method, lamb=1.0, dare_q=0.9,
               della_q=0.5, bc_beta=0.85, bc_gamma=0.01, knots_topk=50.0,
               opcm_alpha=0.5, opcm_lam="adaptive", nufilt_rp=128, nufilt_rv=8,
               nufilt_rl=64, nufilt_adapt=False, nufilt_iters=50, nufilt_lr=1e-3,
               state=None, task=None):
    """Dispatch. Returns the merged state dict.

    opcm / nufilt (E15a) take their own keyword arguments and ignore `lamb`;
    `state` is the caller's dict for OPCM's lam_{t-1} and task-vector norms
    and for the per-step statistics of both rules, `task` the step index
    t >= 1 (OPCM only).  The five original rules are unchanged."""
    if method == "modelstock":
        return modelstock_pair(base, w_prev, w_new)
    if method == "knots":
        return knots_pair(base, w_prev, w_new, topk_pct=knots_topk, lamb=lamb)
    if method == "opcm":
        return opcm_pair(base, w_prev, w_new, alpha=opcm_alpha, lam_rule=opcm_lam,
                         state=state, task=task)
    if method == "nufilt":
        return nufilt_pair(base, w_prev, w_new, r_p=nufilt_rp, r_v=nufilt_rv,
                           r_l=nufilt_rl, adapt=nufilt_adapt, iters=nufilt_iters,
                           lr=nufilt_lr, state=state)
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


SOTA_METHODS = ("dare", "della", "breadcrumbs", "knots", "modelstock",
                "opcm", "nufilt")
