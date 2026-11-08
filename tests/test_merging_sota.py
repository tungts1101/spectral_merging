"""Unit tests for merging_sota: the OPCM and NUFILT rules (rebuttal E15a)
and a regression check of the five existing rules against the committed
module.

Needs torch (no timm/peft), so it runs on the server CPU:
    python tests/test_merging_sota.py              # plain asserts, prints PASS
    python -m pytest tests/test_merging_sota.py    # the same functions
Set MERGING_SOTA_REF=<git revision> (or --ref) to choose the revision of
merging_sota.py that is the reference for the regression test.  The default is
PRE_REBUTTAL_REF, the last commit before the rebuttal code (0eab95e), so the
test stays meaningful once the E15a change is committed; a reference that is
identical to the worktree file fails the test instead of passing vacuously.
tests/smoke_rebuttal.py --tier b runs TESTS as one of its checks.
"""
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import merging_sota as ms  # noqa: E402

#: Last commit before the rebuttal code (merging_sota.py without opcm / nufilt).
PRE_REBUTTAL_REF = "0eab95e4f6fe1523647618fa46af66fd8ff9af06"

R = 64
SHAPES = {  # module -> (in, out) of the three adapted ViT-B/16 modules
    "attn.qkv": (768, 2304),
    "mlp.fc1": (768, 3072),
    "mlp.fc2": (3072, 768),
}


def make_states(n_blocks=2, seed=0, zero_prev=False, scale=0.01):
    """Random LoRA states (base, w_prev, w_new) with the real key names and
    shapes (lora_A [r, in], lora_B [out, r], norm.*); w_prev == base when
    zero_prev, which is the t = 1 situation of Learner.merge."""
    g = torch.Generator().manual_seed(seed)
    base, w_prev, w_new = {}, {}, {}

    def add(key, shape):
        base[key] = torch.randn(*shape, generator=g) * 0.02
        if zero_prev:
            w_prev[key] = base[key].clone()
        else:
            w_prev[key] = base[key] + torch.randn(*shape, generator=g) * scale
        w_new[key] = base[key] + torch.randn(*shape, generator=g) * scale

    for i in range(n_blocks):
        for mod, (n_in, n_out) in SHAPES.items():
            pre = f"base_model.model.blocks.{i}.{mod}."
            add(pre + "lora_A.default.weight", (R, n_in))
            add(pre + "lora_B.default.weight", (n_out, R))
    add("norm.weight", (768,))
    add("norm.bias", (768,))
    return base, w_prev, w_new


def _tv(w, base, k):
    return w[k].double() - base[k].double()


def _global_norm(w, base):
    return math.sqrt(sum(float(_tv(w, base, k).pow(2).sum()) for k in w))


# ---------------------------------------------------------------- OPCM ---

def test_opcm_first_step_returns_w1():
    base, w_prev, w_new = make_states(zero_prev=True)
    st = {}
    out = ms.opcm_pair(base, w_prev, w_new, alpha=0.5, lam_rule="adaptive",
                       state=st, task=1)
    for k in w_new:
        assert torch.equal(out[k], w_new[k]), k
    assert st["t"] == 1 and st["lam"] == 1.0 and st["lam_prev"] == 1.0
    assert st["kept_frac"] == 1.0 and st["r_alpha_mean"] == 0.0
    tau_norm = _global_norm(w_new, base)
    assert abs(st["norms"][1] - tau_norm) <= 1e-9 * tau_norm
    assert abs(st["norm_mean"] - tau_norm) <= 1e-9 * tau_norm
    # the sqrt rule and a stateless call give the same first step
    out2 = ms.opcm_pair(base, w_prev, w_new, lam_rule="sqrt", task=1)
    assert all(torch.equal(out2[k], w_new[k]) for k in w_new)
    out3 = ms.opcm_pair(base, w_prev, w_new)
    assert all(torch.equal(out3[k], w_new[k]) for k in w_new)


def test_opcm_projection_orthogonal_and_block_zero():
    base, w_prev, w_new = make_states()
    for k in w_new:
        if w_new[k].dim() != 2:
            continue
        dM, tau = _tv(w_prev, base, k), _tv(w_new, base, k)
        U, S, Vh = torch.linalg.svd(dM, full_matrices=False)
        for alpha in (0.0, 0.5, 1.0):
            proj, r_a = ms._opcm_project(dM, tau, alpha)
            # <P(tau), dM>_F == 0
            ip = float((proj * dM).sum())
            assert abs(ip) <= 1e-10 * float(tau.norm() * dM.norm()), (k, alpha, ip)
            # zero diagonal and zero top-left r_alpha block of U^T P(tau) V
            C = U.T @ proj @ Vh.T
            tol = 1e-10 * float(tau.norm())
            assert float(C.diagonal().abs().max()) <= tol, (k, alpha)
            if r_a > 0:
                assert float(C[:r_a, :r_a].abs().max()) <= tol, (k, alpha)
            # the part of tau outside span(U) x span(V) is kept
            outside = tau - U @ (U.T @ tau @ Vh.T) @ Vh
            outside_p = proj - U @ (U.T @ proj @ Vh.T) @ Vh
            assert torch.allclose(outside, outside_p, atol=1e-12, rtol=0)
            # r_alpha: smallest r with cumulative singular-value sum >= alpha * total
            if alpha == 0.0:
                assert r_a == 0
            elif alpha == 1.0:
                assert r_a == S.numel() == R
            else:
                assert r_a == int((S.cumsum(0) < alpha * S.sum()).sum()) + 1
                assert 0 < r_a < R


def test_opcm_project_matches_full_basis_construction():
    """The fusion_bench construction in the full bases (full_matrices=True):
    zero the diagonal of U^T tau V and its top-left block, reconstruct."""
    g = torch.Generator().manual_seed(1)
    for (m, n) in ((8, 12), (12, 8), (8, 8), (R, 768), (2304, R)):
        dM = torch.randn(m, n, generator=g, dtype=torch.float64)
        tau = torch.randn(m, n, generator=g, dtype=torch.float64)
        for alpha in (0.3, 0.5, 0.8):
            proj, r_a = ms._opcm_project(dM, tau, alpha)
            U, S, Vh = torch.linalg.svd(dM, full_matrices=True)
            C = U.T @ tau @ Vh.T
            C.diagonal().zero_()
            C[:r_a, :r_a] = 0.0
            ref = U @ C @ Vh
            assert torch.allclose(proj, ref, atol=1e-9, rtol=0), (m, n, alpha)


def test_opcm_sequence_lam_rules():
    base, _, _ = make_states()
    g = torch.Generator().manual_seed(7)
    experts = [{k: base[k] + torch.randn(base[k].shape, generator=g) * 0.01
                for k in base} for _ in range(4)]
    for rule in ("sqrt", "adaptive"):
        st, Ms, lam_before = {}, [base], None
        for t, W in enumerate(experts, start=1):
            M = ms.opcm_pair(base, Ms[-1], W, alpha=0.5, lam_rule=rule,
                             state=st, task=t)
            Ms.append(M)
            assert st["t"] == t and sorted(st["norms"]) == list(range(1, t + 1))
            dev = _global_norm(M, base)
            if rule == "sqrt":
                assert abs(st["lam"] - math.sqrt(t)) < 1e-12
            else:
                # ||M_t - base|| = ||S_t|| / lam_t = mean_{i<=t} ||tau_i|| by construction
                assert abs(dev - st["norm_mean"]) <= 1e-5 * st["norm_mean"], (t, dev)
            if t == 1:
                assert st["lam"] == 1.0 and st["kept_frac"] == 1.0
            else:
                assert st["lam_prev"] == lam_before
                assert 0.0 < st["kept_frac"] < 1.0 and 0 < st["r_alpha_mean"] < R
                assert st["lam"] > 1.0
            lam_before = st["lam"]
        # re-merging the last task with the same state is idempotent
        M_again = ms.opcm_pair(base, Ms[-2], experts[-1], alpha=0.5,
                               lam_rule=rule, state=st, task=4)
        assert all(torch.equal(M_again[k], Ms[-1][k]) for k in M_again)
        assert st["t"] == 4 and st["lam"] == lam_before
    # unknown lam rule
    try:
        ms.opcm_pair(base, experts[0], experts[1], lam_rule="bogus", task=2)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown lam rule must raise ValueError")


# -------------------------------------------------------------- NUFILT ---

def test_opcm_independent_regime_counts_first_expert():
    """Independent-init regime (seqft_independent=True): the base stays the
    fresh initial module, so the first merge (task=1) sees w_prev = W_0, an
    expert with task vector dM != 0.  The rule must count it: two experts at
    t = 1 (lam = sqrt(2) under the sqrt rule), norms keyed 0..t, and
    ||M_t - base|| = mean_i ||tau_i|| under the adaptive rule."""
    base, _, _ = make_states()
    g = torch.Generator().manual_seed(11)
    experts = [{k: base[k] + torch.randn(base[k].shape, generator=g) * 0.01
                for k in base} for _ in range(4)]
    for rule in ("sqrt", "adaptive"):
        st, Ms = {}, [experts[0]]                       # M_0 = W_0 (no merge at task 0)
        for t, W in enumerate(experts[1:], start=1):
            M = ms.opcm_pair(base, Ms[-1], W, alpha=0.5, lam_rule=rule, state=st, task=t)
            Ms.append(M)
            assert sorted(st["norms"]) == list(range(0, t + 1)), st["norms"]
            n0 = _global_norm(experts[0], base)
            assert abs(st["norms"][0] - n0) <= 1e-9 * n0
            if rule == "sqrt":
                assert abs(st["lam"] - math.sqrt(t + 1)) < 1e-12, (t, st["lam"])
            else:
                dev = _global_norm(M, base)
                assert abs(dev - st["norm_mean"]) <= 1e-5 * st["norm_mean"], (t, dev)
            if t == 1:
                assert st["lam_prev"] == 1.0            # the single expert W_0 is unscaled
                assert 0.0 < st["kept_frac"] < 1.0      # dM != 0: the projection acted


def test_nufilt_filter_zero_response_and_clamp():
    base, w_prev, w_new = make_states()
    for k in w_new:
        if w_new[k].dim() != 2:
            continue
        dM, tau = _tv(w_prev, base, k), _tv(w_new, base, k)
        tol = 1e-10 * float(tau.norm())
        if k.endswith(ms._LORA_A):
            tau_f, V = ms._nufilt_filter(dM, tau, 128, "right")
            assert V.shape == (tau.shape[1], R)          # r_p = 128 clamps to rank 64
            assert float((tau_f @ V).abs().max()) <= tol
            assert torch.allclose(tau - tau_f, (tau @ V) @ V.T, atol=1e-12, rtol=0)
            tau_f32, V32 = ms._nufilt_filter(dM, tau, 32, "right")
            assert V32.shape == (tau.shape[1], 32)
            assert float((tau_f32 @ V32).abs().max()) <= tol
            assert float(tau_f32.norm()) > float(tau_f.norm())   # fewer directions removed
        else:
            tau_f, U = ms._nufilt_filter(dM, tau, 128, "left")
            assert U.shape == (tau.shape[0], R)
            assert float((U.T @ tau_f).abs().max()) <= tol
            assert torch.allclose(tau - tau_f, U @ (U.T @ tau), atol=1e-12, rtol=0)
    # zero old knowledge: nothing is filtered, empty basis
    tau = torch.randn(R, 768, dtype=torch.float64)
    tau_f, V = ms._nufilt_filter(torch.zeros_like(tau), tau, 128, "right")
    assert V.shape == (768, 0) and torch.equal(tau_f, tau)
    tau_f, U = ms._nufilt_filter(torch.zeros(2304, R, dtype=torch.float64),
                                 torch.randn(2304, R, dtype=torch.float64), 8, "left")
    assert U.shape == (2304, 0)


def test_nufilt_first_step_returns_w1():
    base, w_prev, w_new = make_states(zero_prev=True)
    for adapt in (False, True):
        st = {}
        out = ms.nufilt_pair(base, w_prev, w_new, adapt=adapt, state=st)
        assert all(torch.equal(out[k], w_new[k]) for k in w_new)
        assert st["filtered_frac"] == 0.0 and st["rp_eff_mean"] == 0.0
        assert st["n_adapted"] == 0


def test_nufilt_pair_fuse():
    base, w_prev, w_new = make_states()
    st = {}
    out = ms.nufilt_pair(base, w_prev, w_new, r_p=128, adapt=False, state=st)
    for k in w_new:
        tau = _tv(w_new, base, k)
        if w_new[k].dim() != 2:
            ref = w_prev[k].double() + tau
        else:
            dM = _tv(w_prev, base, k)
            side = "left" if k.endswith(ms._LORA_B) else "right"
            ref = w_prev[k].double() + ms._nufilt_filter(dM, tau, 128, side)[0]
        assert out[k].dtype == w_new[k].dtype
        assert torch.allclose(out[k].double(), ref, atol=1e-6, rtol=1e-6), k
    assert 0.0 < st["filtered_frac"] < 1.0 and st["rp_eff_mean"] == R
    assert st["n_adapted"] == 0
    st32 = {}
    ms.nufilt_pair(base, w_prev, w_new, r_p=32, adapt=False, state=st32)
    assert st32["rp_eff_mean"] == 32 and st32["filtered_frac"] < st["filtered_frac"]


def test_nufilt_adaptation_loss_decreases():
    base, w_prev, w_new = make_states(n_blocks=1)
    k = next(k for k in w_new if k.endswith(ms._LORA_A))
    dM, tau = _tv(w_prev, base, k), _tv(w_new, base, k)
    tau_f, V = ms._nufilt_filter(dM, tau, 128, "right")
    gen = torch.Generator().manual_seed(0)
    delta, losses = ms._nufilt_adapt(dM, tau, tau_f, V, r_v=8, r_l=64,
                                     iters=50, lr=1e-3, generator=gen)
    assert delta.shape == tau.shape and len(losses) == 51
    assert losses[0] > 0.0 and losses[-1] < losses[0], (losses[0], losses[-1])
    # through the pair function: statistics, private RNG, outer no_grad
    st1, st2 = {}, {}
    out1 = ms.nufilt_pair(base, w_prev, w_new, adapt=True, iters=10, state=st1, seed=3)
    torch.manual_seed(999)                               # the global RNG must not matter
    with torch.no_grad():                                # Learner.merge may run under no_grad
        out2 = ms.nufilt_pair(base, w_prev, w_new, adapt=True, iters=10, state=st2, seed=3)
    assert all(torch.equal(out1[k], out2[k]) for k in out1)
    assert st1["n_adapted"] == 3 and st1["adapt_loss_last"] < st1["adapt_loss_first"]
    assert st1 == st2
    # the adapted lora_A tensors differ from the filter-only fuse, lora_B do not
    out0 = ms.nufilt_pair(base, w_prev, w_new, adapt=False)
    for k in out0:
        if k.endswith(ms._LORA_A):
            assert not torch.equal(out0[k], out1[k]), k
        else:
            assert torch.equal(out0[k], out1[k]), k


# ------------------------------------------------------------ dispatch ---

def test_sota_merge_dispatch_and_methods():
    assert ms.SOTA_METHODS[:5] == ("dare", "della", "breadcrumbs", "knots", "modelstock")
    assert "opcm" in ms.SOTA_METHODS and "nufilt" in ms.SOTA_METHODS
    base, w_prev, w_new = make_states(n_blocks=1)
    st_a, st_b = {}, {}
    a = ms.sota_merge(base, w_prev, w_new, "opcm", lamb=0.3, opcm_alpha=0.6,
                      opcm_lam="sqrt", state=st_a, task=3)
    b = ms.opcm_pair(base, w_prev, w_new, alpha=0.6, lam_rule="sqrt", state=st_b, task=3)
    assert all(torch.equal(a[k], b[k]) for k in a) and st_a == st_b
    st_a, st_b = {}, {}
    a = ms.sota_merge(base, w_prev, w_new, "nufilt", lamb=0.3, nufilt_rp=32,
                      nufilt_rv=4, nufilt_rl=16, nufilt_adapt=True, nufilt_iters=5,
                      nufilt_lr=1e-3, state=st_a)
    b = ms.nufilt_pair(base, w_prev, w_new, r_p=32, r_v=4, r_l=16, adapt=True,
                       iters=5, lr=1e-3, state=st_b)
    assert all(torch.equal(a[k], b[k]) for k in a) and st_a == st_b
    assert st_a["rp_eff_mean"] == 32 and st_a["n_adapted"] == 3
    try:
        ms.sota_merge(base, w_prev, w_new, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown method must raise ValueError")


def _load_reference_module():
    """merging_sota.py at the git revision MERGING_SOTA_REF (default
    PRE_REBUTTAL_REF), loaded from a temporary file as module merging_sota_ref;
    the temporary directory is removed once the module is loaded.  Raises
    AssertionError when the reference equals the worktree file (a vacuous
    comparison, e.g. HEAD after the rebuttal commit)."""
    ref = os.environ.get("MERGING_SOTA_REF", PRE_REBUTTAL_REF)
    try:
        src = subprocess.run(["git", "show", f"{ref}:merging_sota.py"], cwd=REPO,
                             capture_output=True, text=True, encoding="utf-8",
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        return None, f"{ref}: {e}"
    with open(os.path.join(REPO, "merging_sota.py"), encoding="utf-8") as f:
        current = f.read()
    if src.replace("\r\n", "\n") == current.replace("\r\n", "\n"):
        raise AssertionError(
            f"{ref}:merging_sota.py is identical to the worktree file; the regression test "
            f"would compare the code with itself. Pass --ref <pre-rebuttal commit> "
            f"(default {PRE_REBUTTAL_REF[:7]}).")
    tmp = tempfile.mkdtemp(prefix="merging_sota_ref_")
    try:
        path = os.path.join(tmp, "merging_sota_ref.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("merging_sota_ref", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return mod, ref


def test_existing_rules_unchanged():
    """The five original rules give bit-identical results to the committed
    module on random states (DARE/DELLA under the same global seed)."""
    ref, name = _load_reference_module()
    if ref is None:
        print(f"SKIP regression test: reference module not available ({name})")
        return
    base, w_prev, w_new = make_states(n_blocks=1)
    kw = dict(lamb=0.8, dare_q=0.7, della_q=0.4, bc_beta=0.6, bc_gamma=0.05,
              knots_topk=40.0)
    for method in ("dare", "della", "breadcrumbs", "knots", "modelstock"):
        torch.manual_seed(2024)
        a = ms.sota_merge(base, w_prev, w_new, method, **kw)
        torch.manual_seed(2024)
        b = ref.sota_merge(base, w_prev, w_new, method, **kw)
        assert a.keys() == b.keys(), method
        for k in a:
            assert torch.equal(a[k], b[k]), (method, k)
    assert tuple(ref.SOTA_METHODS) == ms.SOTA_METHODS[:len(ref.SOTA_METHODS)]
    print(f"    regression reference: merging_sota.py at {name}")


TESTS = [
    test_opcm_first_step_returns_w1,
    test_opcm_projection_orthogonal_and_block_zero,
    test_opcm_project_matches_full_basis_construction,
    test_opcm_sequence_lam_rules,
    test_opcm_independent_regime_counts_first_expert,
    test_nufilt_filter_zero_response_and_clamp,
    test_nufilt_first_step_returns_w1,
    test_nufilt_pair_fuse,
    test_nufilt_adaptation_loss_decreases,
    test_sota_merge_dispatch_and_methods,
    test_existing_rules_unchanged,
]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default=os.environ.get("MERGING_SOTA_REF", PRE_REBUTTAL_REF),
                    help="git revision of the reference merging_sota.py "
                         "(default: the last pre-rebuttal commit %(default).7s)")
    args = ap.parse_args()
    os.environ["MERGING_SOTA_REF"] = args.ref
    for fn in TESTS:
        t0 = time.time()
        fn()
        print(f"ok  {fn.__name__} ({time.time() - t0:.1f}s)")
    print("PASS tests/test_merging_sota.py")


if __name__ == "__main__":
    main()
