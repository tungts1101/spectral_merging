"""Unit tests for keeplora_subspace.py (NumPy only; runs on a torch-less machine).

Run:  python scripts/test_keeplora_subspace.py  ->  prints 'all keeplora_subspace tests passed'

The reference is a verbatim NumPy transcription of the released KeepLoRA code
(github.com/MaolinLuo/KeepLoRA, branch v1, MTIL/): ``PrinSubspace._update_prin_subspace``
(weight step at task 0 and the feature step after every task, both via SVD of the
matrix) and ``Trainer.in_proj_weight_process`` (projection of the gradient and the
rank-r SVD initialisation; exact SVD in place of ``torch.svd_lowrank``).

Checks
  1. energy_count: strict '<' count; weight_subspace reproduces the released weight
     step (left singular vectors, side='output'), the paper's side='input' gives the
     right singular vectors, eps_w <= 1e-8 -> None, non-square 'output' raises.
  2. first-task extension on a Gram equals the released SVD route (count, projector).
  3. later tasks over a 4-task x 3-layer synthetic stream: identical column counts and
     projectors (tol 1e-6), the 'skip when captured' case, the cap at d columns, and
     scale invariance of the Gram.
  4. gradient_init vs the released init: A orthonormal and orthogonal to the basis
     (basis_leakage ~ 0), the increment A B equal to the reference increment, B = S V^T.
  5. BatchMeanGram equals the released running mean (n/(n+m) recursion) up to scale and
     skips a batch of another size.
  6. pack_basis / unpack_basis round trip.

Fixture: d = 48, token features with planted class-mean + low-rank structure per task
(seed 7); every count decision of the reference sits at least 1e-6 away from its
threshold (asserted), so no BLAS build can flip a count.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import keeplora_subspace as ks  # noqa: E402

D, SEED = 48, 7
EPS_W, EPS_F, RANK = 0.85, 0.99, 4
TOL = 1e-6
MARGIN = 1e-6
_decisions = []   # |cumulative share - threshold| at every reference count decision


# ---------------------------------------------------------------------------
# Reference: released MTIL/model/prin_subspace.py and trainer.py, in NumPy
# ---------------------------------------------------------------------------

def ref_update_prin_subspace(mat_list, prin_subspace_layers, threshold):
    """PrinSubspace._update_prin_subspace (mutates prin_subspace_layers)."""
    if len(prin_subspace_layers) == 0:
        for i in range(len(mat_list)):
            activation = np.array(mat_list[i], dtype=np.float64)
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            cs = np.cumsum(sval_ratio)
            _decisions.extend(np.abs(cs - threshold).tolist())
            r = int(np.sum(cs < threshold))
            prin_subspace_layers.append(U[:, 0:max(r, 1)])
    else:
        for i in range(len(mat_list)):
            activation = np.array(mat_list[i], dtype=np.float64)
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()
            feature_tensor = np.array(prin_subspace_layers[i], dtype=np.float64)
            act_hat = activation - (feature_tensor @ feature_tensor.T) @ activation
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total
            r = 0
            for ii in range(sval_ratio.shape[0]):
                _decisions.append(abs(accumulated_sval - threshold))
                if accumulated_sval < threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                continue
            Ui = np.hstack((feature_tensor, U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                prin_subspace_layers[i] = Ui[:, 0:Ui.shape[0]]
            else:
                prin_subspace_layers[i] = Ui
    return prin_subspace_layers


def ref_init(weight_grad, r_dim, prin_subspace):
    """Trainer.in_proj_weight_process for one slice: weight_grad is (out, in) as
    torch stores it; the projection matrix is U U^T of the slice's basis."""
    grad = np.array(weight_grad, dtype=np.float64).T
    if prin_subspace is not None:
        P = prin_subspace @ prin_subspace.T
        grad = grad - P @ grad
    U, S, Vh = np.linalg.svd(grad, full_matrices=False)
    A = U[:, :r_dim]
    B = Vh[:r_dim, :] * S[:r_dim][:, None]
    return A, B


def ref_running_mean(batches):
    """KeepLoRA.accumulate_features: n/(n+m) * acc + 1/(n+m) * X_b^T."""
    acc, n = None, 0
    for xt in batches:
        m = xt.shape[1]
        if acc is None:
            acc = np.zeros_like(xt)
        acc = (n / (n + m)) * acc + (1.0 / (n + m)) * xt
        n += m
    return acc


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def make_stream(rng, n_tasks=4, n_layers=3, tokens=600):
    """Per task and layer a feature matrix X (d, tokens): a class mean plus a
    planted 3-dim subspace plus isotropic noise; later tasks reuse part of the
    earlier subspace so some directions are already captured."""
    stream = []
    shared = rng.standard_normal((D, 2))
    for t in range(n_tasks):
        layers = []
        for l in range(n_layers):
            mean = 3.0 * rng.standard_normal((D, 1))
            Bsub = np.hstack([shared, rng.standard_normal((D, 1 + t))])
            coef = rng.standard_normal((Bsub.shape[1], tokens)) * np.linspace(2.0, 0.5, Bsub.shape[1])[:, None]
            X = mean + Bsub @ coef + 0.3 * rng.standard_normal((D, tokens))
            layers.append(X)
        stream.append(layers)
    return stream


def projector(M):
    return M @ M.T


def same_subspace(M1, M2, tol=TOL):
    return M1.shape == M2.shape and np.max(np.abs(projector(M1) - projector(M2))) < tol


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_energy_count_and_weight_subspace(rng):
    e = np.array([0.5, 0.3, 0.15, 0.05])
    assert ks.energy_count(e, 0.8) == 1, "strict '<': 0.5 < 0.8, 0.8 < 0.8 is False"
    assert ks.energy_count(e, 0.81) == 2
    assert ks.energy_count(e, 1.0) == 3, "the full sum is never < 1.0"
    assert ks.energy_count(np.zeros(3), 0.5) == 0
    W = rng.standard_normal((D, D)) @ np.diag(np.linspace(3.0, 0.1, D))
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    out = ks.weight_subspace(W, EPS_W, side="output")
    inp = ks.weight_subspace(W, EPS_W, side="input")
    layers = ref_update_prin_subspace([W], [], EPS_W)
    assert same_subspace(out, layers[0]), "side='output' must reproduce the released weight step"
    r = out.shape[1]
    assert same_subspace(inp, Vh[:r].T), "side='input' = right singular vectors"
    assert np.allclose(out.T @ out, np.eye(r), atol=1e-10)
    assert ks.weight_subspace(W, 0.0) is None and ks.weight_subspace(W, 1e-9) is None
    try:
        ks.weight_subspace(rng.standard_normal((D, 2 * D)), EPS_W, side="output")
        raise AssertionError("non-square 'output' must raise")
    except ValueError:
        pass
    assert ks.weight_subspace(rng.standard_normal((D, 2 * D)), EPS_W, side="input").shape[0] == 2 * D
    print("ok    test_energy_count_and_weight_subspace")


def test_first_task(rng, stream):
    X = stream[0][0]
    ref = ref_update_prin_subspace([X], [], EPS_F)[0]
    M, added = ks.extend_subspace(None, X @ X.T, EPS_F)
    assert added == ref.shape[1] and same_subspace(M, ref), "first-task branch differs from the released SVD"
    M2, added2 = ks.extend_subspace_from_matrix(None, X, EPS_F)
    assert added2 == added and same_subspace(M2, M)
    M3, _ = ks.extend_subspace([], X @ X.T, EPS_F)
    assert same_subspace(M3, M), "an empty list counts as an empty basis"
    print(f"ok    test_first_task (r = {added})")


def test_stream(rng, stream):
    n_layers = len(stream[0])
    weights = [rng.standard_normal((D, D)) @ np.diag(np.linspace(2.0, 0.2, D)) for _ in range(n_layers)]
    ref_layers = ref_update_prin_subspace([w for w in weights], [], EPS_W)
    mine = [ks.weight_subspace(w, EPS_W, side="output") for w in weights]
    for l in range(n_layers):
        assert same_subspace(mine[l], ref_layers[l])
    cols_hist = []
    for t, layers in enumerate(stream):
        ref_update_prin_subspace(layers, ref_layers, EPS_F)
        new = []
        for l in range(n_layers):
            M, added = ks.extend_subspace(mine[l], layers[l] @ layers[l].T, EPS_F)
            Ms, added_s = ks.extend_subspace(mine[l], 7.0 * layers[l] @ layers[l].T, EPS_F)
            assert added_s == added and same_subspace(Ms, M), "Gram scale must not matter"
            assert M.shape[1] == ref_layers[l].shape[1], \
                f"task {t} layer {l}: {M.shape[1]} columns vs released {ref_layers[l].shape[1]}"
            assert same_subspace(M, ref_layers[l]), f"task {t} layer {l}: projector differs"
            assert np.allclose(M.T @ M, np.eye(M.shape[1]), atol=1e-8), "basis not orthonormal"
            new.append(M)
        mine = new
        cols_hist.append([M.shape[1] for M in mine])
    # captured already: feeding the same Gram again adds nothing
    M, added = ks.extend_subspace(mine[0], stream[-1][0] @ stream[-1][0].T, EPS_F)
    assert added == 0 and M is mine[0] or same_subspace(M, mine[0])
    # cap at d columns: a full-rank isotropic Gram with a tiny threshold gap
    big = np.eye(D)
    M_full, added_full = ks.extend_subspace(mine[0], big, 1.0)
    assert M_full.shape[1] == D, f"cap at d columns, got {M_full.shape[1]}"
    print(f"ok    test_stream (columns per task {cols_hist})")
    return mine


def test_gradient_init(rng, bases):
    grad_t = rng.standard_normal((D, D))            # torch layout (out, in)
    for basis in (None, bases[0]):
        A, B = ks.gradient_init(grad_t.T, basis, RANK)
        A_ref, B_ref = ref_init(grad_t, RANK, basis)
        assert A.shape == (D, RANK) and B.shape == (RANK, D)
        assert np.allclose(A.T @ A, np.eye(RANK), atol=1e-10)
        assert np.max(np.abs(A @ B - A_ref @ B_ref)) < TOL, "increment A B differs from the released init"
        assert same_subspace(A, A_ref)
        if basis is not None:
            assert ks.basis_leakage(A, basis) < 1e-9, "A must be orthogonal to the protected basis"
            G_hat = grad_t.T - basis @ (basis.T @ grad_t.T)
        else:
            assert ks.basis_leakage(A, None) == 0.0
            G_hat = grad_t.T
        U, S, Vh = np.linalg.svd(G_hat, full_matrices=False)
        assert np.allclose(np.abs(np.linalg.norm(B, axis=1)), S[:RANK], atol=1e-8), "rows of B carry the singular values"
    # a rank larger than the gradient allows, or a basis of the wrong size, raise
    for bad in ((grad_t.T, None, D + 1), (grad_t.T, np.eye(D + 1)[:, :3], RANK)):
        try:
            ks.gradient_init(*bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    print("ok    test_gradient_init")


def test_batch_mean(rng):
    batches = [rng.standard_normal((D, 30)) + 2.0 for _ in range(5)]
    acc = ks.BatchMeanGram()
    for xt in batches:
        acc.update(xt)
    acc.update(rng.standard_normal((D, 31)))       # another token count: skipped
    assert acc.n_batches == 5 and acc.n_skipped == 1
    ref = ref_running_mean(batches)
    mean = acc.mean()
    scale = np.sum(ref * mean) / np.sum(mean * mean)
    assert np.max(np.abs(ref - scale * mean)) < 1e-10, "batch mean differs from the released recursion (up to scale)"
    K = acc.gram()
    K2 = ks.BatchMeanGram.gram_from_sum(acc.sum, acc.n_batches)
    assert np.allclose(K, K2) and np.allclose(K, mean @ mean.T)
    M_a, r_a = ks.extend_subspace(None, K, EPS_F)
    M_b = ref_update_prin_subspace([ref], [], EPS_F)[0]
    assert r_a == M_b.shape[1] and same_subspace(M_a, M_b)
    assert ks.BatchMeanGram().gram() is None
    print(f"ok    test_batch_mean (first-task r = {r_a}; the mean direction dominates)")


def test_pack_unpack(rng):
    M = np.linalg.qr(rng.standard_normal((D, 5)))[0]
    buf = ks.pack_basis(M, D)
    assert buf.shape == (D, D) and buf.dtype == np.float32
    back = ks.unpack_basis(buf, 5)
    assert back.shape == (D, 5) and np.max(np.abs(back - M)) < 1e-6
    assert ks.unpack_basis(buf, 0) is None
    assert np.all(ks.pack_basis(None, D) == 0)
    try:
        ks.pack_basis(np.zeros((D + 1, 2)), D)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("ok    test_pack_unpack")


def main():
    rng = np.random.default_rng(SEED)
    stream = make_stream(rng)
    test_energy_count_and_weight_subspace(rng)
    test_first_task(rng, stream)
    bases = test_stream(rng, stream)
    test_gradient_init(rng, bases)
    test_batch_mean(rng)
    test_pack_unpack(rng)
    margin = min(_decisions)
    assert margin > MARGIN, f"fixture too close to a threshold: min decision margin {margin:.2e}"
    print(f"fixture: {len(_decisions)} count decisions, min margin {margin:.2e}")
    print("all keeplora_subspace tests passed")


if __name__ == "__main__":
    main()
