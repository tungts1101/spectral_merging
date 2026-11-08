"""Unit tests for inflora_gpm.py (NumPy only; runs on a torch-less machine).

Run:  python scripts/test_inflora_gpm.py   ->  prints 'all inflora_gpm tests passed'

Checks
  1. threshold_schedule values (0.95 -> 0.995, 0.98 -> 0.998 over 10 tasks).
  2. quarter-cosine lr factors (1.0 at epoch 0, cos(99 pi / 200) at the last epoch,
     strictly decreasing, 1.0 for a single epoch).
  3. first-task basis: orthonormal, column count = independent cumsum(lambda^2) count,
     'remove'; dual=False never gives 'retain', dual=True stores the complement and warns.
  4. down-projection design: shape (r, d), float32, A A^T = I/3, orthogonal to a 'remove'
     basis, inside a 'retain' basis; basis_leakage agrees.
  5. 'remove' expansion: growth with orthonormal new columns, the add-one quirk right after
     the first task, skip on a repeated Gram, cap at d columns under plain GPM.
  6. finalize: a 'remove' basis with > d/2 columns becomes its orthogonal complement
     ('retain'); plain GPM never switches; a 'retain' basis with > d/2 columns asserts.
  7. 'retain' shrink: a Gram concentrated inside M_perp removes the expected number of
     directions; the result is orthonormal, inside the old span, orthogonal to the removed
     directions.
  8. equality with a verbatim list-based transcription of upstream update_DualGPM,
     update_GPM and the _train A-design (methods/inflora.py) over 4 synthetic tasks x 3
     layers: identical column counts and types, projectors within tolerance, both with the
     reference run in float32 (as upstream) and in float64.
  9. pack_basis / unpack_basis round trip.

Fixture: d = 64 Gram matrices G = X^T X / n (n = 5000) with planted orthonormal subspaces and
geometric spectra (seed 14).  The seed was chosen so that every count decision in test 8 sits
at least 2e-3 away from its threshold (asserted), so no BLAS build can flip a count.
"""
import math
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import inflora_gpm as g  # noqa: E402

D, N, SEED = 64, 5000, 14
LAMB, LAME, TOTAL_TASKS, RANK = 0.95, 1.0, 10, 3
TOL_F64 = 1e-8      # memory projectors, reference in float64: only round-off differs
TOL_F32 = 1e-4      # reference in float32 like upstream (design tolerance)
TOL_DESIGN = 1e-6   # design step vs the float64 reference: A_t is returned in float32
MARGIN = 2e-3       # fixture sanity: distance of every count decision from its threshold


# ---------------------------------------------------------------------------
# Reference: verbatim list-based transcription of upstream methods/inflora.py
# (github.com/liangyanshuo/InfLoRA).  Changes are limited to passing the
# ``self.`` state (feature_list, project_type, lamb, lame, _cur_task,
# total_sessions) as arguments, dropping the print statements, and using
# np.linalg.svd where upstream calls torch.linalg.svd in the A design.
# ---------------------------------------------------------------------------

def ref_update_dualgpm(mat_list, feature_list, project_type, lamb, lame, cur_task, total_sessions):
    """InfLoRA.update_DualGPM; mutates feature_list / project_type in place."""
    threshold = (lame - lamb)*cur_task/total_sessions + lamb
    if len(feature_list) == 0:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold) #+1
            if r < (activation.shape[0]/2):
                feature_list.append(U[:,0:max(r,1)])
                project_type.append('remove')
            else:
                feature_list.append(U[:,0:max(r,1)])
                project_type.append('retain')
    else:
        for i in range(len(mat_list)):
            if project_type[i] == 'remove':
                activation = mat_list[i]
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-8)
                act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-9)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                accumulated_sval = (sval_total-sval_hat)/sval_total

                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    continue  # print ('Skip Updating DualGPM for layer: {}'.format(i+1))
                # update GPM
                Ui=np.hstack((feature_list[i],U[:,0:r]))
                if Ui.shape[1] > Ui.shape[0] :
                    feature_list[i]=Ui[:,0:Ui.shape[0]]
                else:
                    feature_list[i]=Ui
            else:
                activation = mat_list[i]
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-8)
                act_hat = np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-9)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                accumulated_sval = sval_hat/sval_total

                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval >= (1-threshold):
                        accumulated_sval -= sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    continue  # print ('Skip Updating DualGPM for layer: {}'.format(i+1))

                # update GPM by Projected Representation (Eq-8)
                act_feature = feature_list[i] - np.dot(np.dot(U[:,0:r],U[:,0:r].transpose()),feature_list[i])
                Ui, Si, Vi = np.linalg.svd(act_feature)
                feature_list[i]=Ui[:,:feature_list[i].shape[1]-r]

    # 'Gradient Constraints Summary'
    for i in range(len(feature_list)):
        if project_type[i]=='remove' and (feature_list[i].shape[1] > (feature_list[i].shape[0]/2)):
            feature = feature_list[i]
            U, S, V = np.linalg.svd(feature)
            new_feature = U[:,feature.shape[1]:]
            feature_list[i] = new_feature
            project_type[i] = 'retain'
        elif project_type[i]=='retain':
            assert feature_list[i].shape[1] <= (feature_list[i].shape[0]/2)
    return threshold


def ref_update_gpm(mat_list, feature_list, lamb, lame, cur_task, total_sessions):
    """InfLoRA.update_GPM (plain GPM, 'remove' only); mutates feature_list in place."""
    threshold = (lame - lamb)*cur_task/total_sessions + lamb
    if len(feature_list) == 0:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold) #+1
            feature_list.append(U[:,0:max(r,1)])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            accumulated_sval = (sval_total-sval_hat)/sval_total

            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                continue  # print ('Skip Updating GPM for layer: {}'.format(i+1))
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    return threshold


def ref_design_A(cur_matrix, feature_mat, project_type, rank):
    """The A-design of InfLoRA._train (lora_A_k[t] = lora_A_v[t] = U[:, :rank].T / sqrt(3)).

    feature_mat None = first task (``self._cur_task == 0``: raw Gram, torch.linalg.svd default
    full_matrices); otherwise feature_mat = F F^T precomputed after update_DualGPM.
    """
    if feature_mat is None:
        U, S, V = np.linalg.svd(cur_matrix)
        return U[:, :rank].T / math.sqrt(3)
    if project_type == 'remove':
        cur_matrix = cur_matrix - np.dot(feature_mat, cur_matrix)
    else:
        assert project_type == 'retain'
        cur_matrix = np.dot(feature_mat, cur_matrix)
    cU, cS, cV = np.linalg.svd(cur_matrix, full_matrices=False)
    return cU[:, :rank].T / math.sqrt(3)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def orthonormal(rng, d):
    return np.linalg.qr(rng.standard_normal((d, d)))[0]


def sample_gram(rng, dirs, eigs, n=N, noise=0.1):
    """G = X^T X / n with X = Z diag(sqrt(eigs)) dirs^T + noise * N(0, I): planted
    eigenvalues ``eigs`` on the orthonormal columns ``dirs`` plus isotropic noise."""
    Z = rng.standard_normal((n, dirs.shape[1]))
    X = (Z * np.sqrt(eigs)) @ dirs.T + noise * rng.standard_normal((n, dirs.shape[0]))
    return X.T @ X / n


def build_fixture(seed=SEED):
    """3 layers x 4 tasks of float64 Grams.

    layer 0: a fresh 16-dim subspace per task (geometric spectrum 100 * 0.9^j) -> the basis
             grows past d/2 after task 2 (switch to 'retain') and shrinks at task 3;
    layer 1: task 1 repeats task 0's 12-dim Gram (adds one direction; 9 of the 12 are
             stored after task 0, so the projected Gram keeps >= RANK strong directions
             and the rank-3 design stays well conditioned), then two 8-dim tasks;
    layer 2: 4 new directions per task with a steep spectrum (slow growth).
    """
    rng = np.random.default_rng(seed)
    Q = orthonormal(rng, D)
    layer0 = [sample_gram(rng, Q[:, 16 * t:16 * (t + 1)], 100 * 0.9 ** np.arange(16)) for t in range(4)]
    Q = orthonormal(rng, D)
    g0 = sample_gram(rng, Q[:, :12], 100 * 0.9 ** np.arange(12))
    layer1 = [g0, g0.copy(),
              sample_gram(rng, Q[:, 12:20], 100 * 0.9 ** np.arange(8)),
              sample_gram(rng, Q[:, 20:28], 100 * 0.9 ** np.arange(8))]
    Q = orthonormal(rng, D)
    layer2 = [sample_gram(rng, Q[:, 4 * t:4 * (t + 1)], 100 * 0.7 ** np.arange(4)) for t in range(4)]
    return [layer0, layer1, layer2]


def count_margin(G, M, retain, thr):
    """Smallest distance between a count decision and its threshold (independent re-count)."""
    S1 = np.linalg.svd(G, compute_uv=False)
    total = (S1 ** 2).sum()
    if M is None:
        return float(np.abs(np.cumsum(S1 ** 2 / total) - thr).min())
    P = M @ M.T
    act = (P @ G) if retain else (G - P @ G)
    S = np.linalg.svd(act, compute_uv=False)
    ratio = S ** 2 / total
    acc = (S ** 2).sum() / total if retain else (total - (S ** 2).sum()) / total
    level = (1 - thr) if retain else thr
    ms = []
    for ii in range(len(ratio)):
        ms.append(abs(acc - level))
        if (acc >= level) if retain else (acc < level):
            acc += -ratio[ii] if retain else ratio[ii]
        else:
            break
    return float(min(ms))


def top_eigvecs(G, k):
    w, V = np.linalg.eigh(G)
    return V[:, ::-1][:, :k]


def assert_orthonormal(M, tol=1e-10):
    err = np.abs(M.T @ M - np.eye(M.shape[1])).max()
    assert err < tol, f"columns not orthonormal: {err:.2e}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_threshold_schedule():
    vals = [g.threshold_schedule(0.95, 1.0, t, 10) for t in range(10)]
    assert np.allclose(vals, [0.95 + 0.005 * t for t in range(10)], atol=1e-12), vals
    assert abs(g.threshold_schedule(0.98, 1.0, 9, 10) - 0.998) < 1e-12
    assert abs(g.threshold_schedule(0.98, 1.0, 0, 10) - 0.98) < 1e-12


def test_cosine_lr_factor():
    lam = [g.cosine_lr_factor(e, 20) for e in range(20)]
    assert lam[0] == 1.0
    assert abs(lam[19] - math.cos(99 * math.pi / 200)) < 1e-12
    assert abs(lam[19] - 0.015708) < 1e-6
    assert all(b < a for a, b in zip(lam[:-1], lam[1:])), "not strictly decreasing"
    assert all(0.0 < v <= 1.0 for v in lam)
    assert abs(g.cosine_lr_factor(49, 50) - lam[19]) < 1e-12   # same end point for K = 50
    assert g.cosine_lr_factor(0, 1) == 1.0                      # single-epoch guard


def test_init(fix):
    G = fix[0][0]
    M, retain = g.dualgpm_init(G, 0.95)
    assert retain is False
    assert_orthonormal(M)
    ev = np.sort(np.linalg.eigvalsh(G))[::-1]
    ratio = ev ** 2 / (ev ** 2).sum()
    expected = int(np.sum(np.cumsum(ratio) < 0.95))
    assert 1 <= expected < D / 2 and M.shape[1] == expected, (M.shape, expected)
    Pe = top_eigvecs(G, expected)
    assert np.abs(M @ M.T - Pe @ Pe.T).max() < 1e-8
    # dualgpm_step with an empty basis is the same as init (+ finalize, no switch here)
    for empty in (None, np.zeros((D, 0))):
        M2, retain2, changed = g.dualgpm_step(G, empty, False, 0.95)
        assert changed == expected and retain2 is False and np.array_equal(M2, M)
    # a flat spectrum needs r >= d/2 directions: plain GPM keeps them as 'remove',
    # DualGPM stores the complement as 'retain' and warns
    rng = np.random.default_rng(1)
    E = rng.standard_normal((D, D)) * 1e-3
    flat = np.eye(D) + E + E.T
    Mg, rg = g.dualgpm_init(flat, 0.95, dual=False)
    assert rg is False and Mg.shape[1] >= D / 2, Mg.shape
    Ms, rs, _ = g.dualgpm_step(flat, None, False, 0.95, dual=False)
    assert rs is False and Ms.shape[1] == Mg.shape[1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Md, rd = g.dualgpm_init(flat, 0.95, dual=True)
    assert len(w) == 1 and "DualGPM init" in str(w[0].message)
    assert rd is True and Md.shape[1] == D - Mg.shape[1]
    assert_orthonormal(Md)
    assert np.abs(Md.T @ Mg).max() < 1e-10                       # complement of the top block
    assert np.abs(Md @ Md.T + Mg @ Mg.T - np.eye(D)).max() < 1e-10
    for bad in (0.0, 1.0, 1.2):
        try:
            g.dualgpm_init(G, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"threshold {bad} accepted")


def test_design(fix):
    G0, G1 = fix[0][0], fix[0][1]
    A = g.design_down_projection(G0, None, False, RANK)
    assert A.shape == (RANK, D) and A.dtype == np.float32
    assert np.abs(A @ A.T - np.eye(RANK) / 3).max() < 1e-6
    Pe = top_eigvecs(G0, RANK)
    assert np.abs(3 * A.T.astype(np.float64) @ A - Pe @ Pe.T).max() < 1e-5
    assert np.array_equal(A, g.design_down_projection(G0, np.zeros((D, 0)), False, RANK))
    assert g.basis_leakage(A, None, False) == 0.0
    # 'remove' basis: rows orthogonal to the stored subspace
    M, _ = g.dualgpm_init(G0, 0.95)
    A2 = g.design_down_projection(G1, M, False, RANK)
    assert np.abs(A2 @ A2.T - np.eye(RANK) / 3).max() < 1e-6
    leak = np.abs(A2 @ M).max()
    assert leak < 1e-6, leak
    assert abs(g.basis_leakage(A2, M, False) - leak) < 1e-12
    # 'retain' basis: rows inside the retained complement
    rng = np.random.default_rng(2)
    Q = orthonormal(rng, D)
    Mperp, retain = g.dualgpm_finalize(Q[:, :40], False)
    assert retain is True and Mperp.shape[1] == 24
    G3 = sample_gram(rng, Mperp[:, :6], 100 * 0.9 ** np.arange(6))
    A3 = g.design_down_projection(G3, Mperp, True, RANK)
    assert np.abs(A3 @ A3.T - np.eye(RANK) / 3).max() < 1e-6
    out = np.abs(A3 - (A3 @ Mperp) @ Mperp.T).max()
    assert out < 1e-6, out
    assert abs(g.basis_leakage(A3, Mperp, True) - out) < 1e-12
    assert np.abs(A3 @ Q[:, :40]).max() < 1e-6                   # nothing in the protected part


def test_remove_update(fix):
    thr = 0.95
    G0, G1 = fix[0][0], fix[0][1]
    M0, _ = g.dualgpm_init(G0, thr)
    # add-one quirk: the first-task count is strict (<), an expansion stops when the
    # covered energy reaches the threshold -> the same Gram appends exactly one direction
    M1, retain, added = g.dualgpm_update(G0, M0, False, thr)
    assert retain is False and added == 1 and M1.shape[1] == M0.shape[1] + 1
    assert np.array_equal(M1[:, :M0.shape[1]], M0)
    assert_orthonormal(M1)
    # the same Gram once more at the same threshold -> skip, basis returned unchanged
    M2, retain, added = g.dualgpm_update(G0, M1, False, thr)
    assert added == 0 and retain is False and np.array_equal(M2, M1)
    M2s, retain_s, added_s = g.dualgpm_step(G0, M1, False, thr)
    assert added_s == 0 and retain_s is False and np.array_equal(M2s, M1)
    # a Gram in a new subspace grows the basis; new columns orthonormal to the old ones
    M3, retain, added = g.dualgpm_update(G1, M1, False, thr)
    assert added > 0 and M3.shape[1] == M1.shape[1] + added and retain is False
    assert np.array_equal(M3[:, :M1.shape[1]], M1)
    assert_orthonormal(M3)
    new = M3[:, M1.shape[1]:]
    assert np.abs(new.T @ M1).max() < 1e-10
    # the new columns are the leading left singular vectors of the residual (I - M M^T) G1
    U = np.linalg.svd(G1 - M1 @ (M1.T @ G1), full_matrices=False)[0]
    assert np.abs(new @ new.T - U[:, :added] @ U[:, :added].T).max() < 1e-8
    # cap at d columns under plain GPM with a high threshold over many tasks
    rng = np.random.default_rng(3)
    M, retain, full_at = None, False, None
    for task in range(8):
        Gt = sample_gram(rng, orthonormal(rng, D)[:, :20], 100 * 0.9 ** np.arange(20))
        cols_before = 0 if M is None else M.shape[1]
        M, retain, added = g.dualgpm_step(Gt, M, retain, 0.9999, dual=False)
        assert retain is False and M.shape[1] <= D
        assert added == M.shape[1] - cols_before
        assert_orthonormal(M)
        if M.shape[1] == D and full_at is None:
            full_at = task
        if full_at is not None and task > full_at:
            assert added == 0, "a full basis must skip"
    assert full_at is not None, "cap never reached"


def test_finalize():
    rng = np.random.default_rng(4)
    Q = orthonormal(rng, D)
    M40 = Q[:, :40]
    Mp, retain = g.dualgpm_finalize(M40, False, dual=True)
    assert retain is True and Mp.shape == (D, 24)
    assert_orthonormal(Mp)
    assert np.abs(Mp.T @ M40).max() < 1e-10
    assert np.abs(Mp @ Mp.T + M40 @ M40.T - np.eye(D)).max() < 1e-10
    Mk, retain = g.dualgpm_finalize(M40, False, dual=False)       # plain GPM never switches
    assert retain is False and np.array_equal(Mk, M40)
    M32, retain = g.dualgpm_finalize(Q[:, :32], False, dual=True)  # exactly d/2 stays 'remove'
    assert retain is False and M32.shape[1] == 32
    Mr, retain = g.dualgpm_finalize(Mp, True, dual=True)           # a legal 'retain' basis
    assert retain is True and np.array_equal(Mr, Mp)
    try:
        g.dualgpm_finalize(M40, True, dual=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("'retain' basis with > d/2 columns accepted")


def test_retain_shrink():
    thr = 0.95
    rng = np.random.default_rng(5)
    Q = orthonormal(rng, D)
    Mperp, _ = g.dualgpm_finalize(Q[:, :40], False)                # 24 retained columns
    dirs = Mperp[:, :6]
    G = sample_gram(rng, dirs, 100 * 0.9 ** np.arange(6))
    # independent expectation: remove leading directions of (M M^T) G while the energy
    # still inside M_perp is >= 1 - thr
    S1 = np.linalg.svd(G, compute_uv=False)
    total = (S1 ** 2).sum()
    U, S, _ = np.linalg.svd(Mperp @ (Mperp.T @ G), full_matrices=False)
    acc, expected = (S ** 2).sum() / total, 0
    while acc >= 1 - thr:
        acc -= S[expected] ** 2 / total
        expected += 1
    assert expected == 6, expected
    M2, retain, removed = g.dualgpm_update(G, Mperp, True, thr)
    assert retain is True and removed == expected and M2.shape[1] == 24 - expected
    assert_orthonormal(M2)
    assert np.abs(M2 - Mperp @ (Mperp.T @ M2)).max() < 1e-10       # inside the old span
    assert np.abs(M2.T @ U[:, :expected]).max() < 1e-10            # removed directions gone
    assert np.abs(M2.T @ dirs).max() < 1e-2                        # ~ the planted directions
    M3, retain3, removed3 = g.dualgpm_step(G, Mperp, True, thr)
    assert retain3 is True and removed3 == expected and np.abs(M3 @ M3.T - M2 @ M2.T).max() < 1e-10
    # a Gram outside M_perp has no energy to remove -> skip
    Gout = sample_gram(rng, Q[:, :6], 100 * 0.9 ** np.arange(6))
    M4, retain4, removed4 = g.dualgpm_update(Gout, Mperp, True, thr)
    assert removed4 == 0 and retain4 is True and np.array_equal(M4, Mperp)


def run_reference(fix, dtype, tol, dual):
    """Port vs the verbatim transcription over 4 tasks x 3 layers.

    Both sides receive the same float32 Gram (upstream accumulates it in float32 on the
    GPU; the port receives it through compute_covariances and casts to float64); the
    reference then works in ``dtype``.  Returns the per-task column counts of the port.
    """
    L, T = len(fix), 4
    feature_list, project_type, feature_mat = [], [], None
    state = [(None, False)] * L
    counts, worst_a, worst_p = [], 0.0, 0.0
    for t in range(T):
        mat_list = [fix[l][t].astype(np.float32).astype(dtype) for l in range(L)]
        grams = [m.astype(np.float64) for m in mat_list]
        # (a) the A-design before training, from the memory left by the previous task
        for l in range(L):
            M, retain = state[l]
            A = g.design_down_projection(grams[l], M, retain, RANK).astype(np.float64)
            A_ref = ref_design_A(mat_list[l], None if feature_mat is None else feature_mat[l],
                                 project_type[l] if project_type else 'remove', RANK).astype(np.float64)
            diff = np.abs(3 * A.T @ A - 3 * A_ref.T @ A_ref).max()   # row-space projectors
            worst_a = max(worst_a, diff)
            assert diff < max(tol, TOL_DESIGN), f"design task {t} layer {l}: {diff:.2e}"
        # (b) the memory update after training
        thr = g.threshold_schedule(LAMB, LAME, t, TOTAL_TASKS)
        if dual:
            thr_ref = ref_update_dualgpm(mat_list, feature_list, project_type, LAMB, LAME, t, TOTAL_TASKS)
        else:
            thr_ref = ref_update_gpm(mat_list, feature_list, LAMB, LAME, t, TOTAL_TASKS)
            project_type = ['remove'] * L
        assert abs(thr - thr_ref) < 1e-12
        row = []
        for l in range(L):
            M, retain = state[l]
            mg = count_margin(grams[l], M, retain, thr)
            assert mg > MARGIN, f"fixture: task {t} layer {l} sits {mg:.1e} from the threshold"
            M2, retain2, _ = g.dualgpm_step(grams[l], M, retain, thr, dual=dual)
            state[l] = (M2, retain2)
            F = feature_list[l].astype(np.float64)
            assert M2.shape[1] == F.shape[1], f"task {t} layer {l}: {M2.shape[1]} vs {F.shape[1]}"
            assert retain2 == (project_type[l] == 'retain'), f"task {t} layer {l}: type"
            diff = np.abs(M2 @ M2.T - F @ F.T).max()
            worst_p = max(worst_p, diff)
            assert diff < tol, f"memory task {t} layer {l}: {diff:.2e}"
            row.append((M2.shape[1], project_type[l]))
        counts.append(row)
        feature_mat = [np.dot(f, f.transpose()) for f in feature_list]   # upstream precomputation
    print(f"    {'DualGPM' if dual else 'GPM':7s} reference {np.dtype(dtype).name}: "
          f"max projector diff design {worst_a:.1e} (tol {max(tol, TOL_DESIGN):g}), "
          f"memory {worst_p:.1e} (tol {tol:g})")
    return counts


def test_reference_equality(fix):
    for dtype, tol in ((np.float64, TOL_F64), (np.float32, TOL_F32)):
        counts = run_reference(fix, dtype, tol, dual=True)
        # structural expectations of the fixture (layer 0 switches after task 2 and shrinks
        # at task 3; layer 1 adds exactly one direction on the repeated Gram; layer 2 grows)
        c = [[row[l][0] for row in counts] for l in range(3)]
        ty = [[row[l][1] for row in counts] for l in range(3)]
        assert ty[0] == ['remove', 'remove', 'retain', 'retain'], ty[0]
        assert c[0][0] < c[0][1] and c[0][2] < D / 2 and c[0][3] < c[0][2], c[0]
        assert ty[1] == ['remove'] * 4 and c[1][1] == c[1][0] + 1 and c[1][2] > c[1][1] < c[1][3], c[1]
        assert ty[2] == ['remove'] * 4 and all(b > a for a, b in zip(c[2][:-1], c[2][1:])), c[2]
        counts_gpm = run_reference(fix, dtype, tol, dual=False)
        cg = [[row[l][0] for row in counts_gpm] for l in range(3)]
        assert cg[0][2] > D / 2 and cg[0][3] > cg[0][2], cg[0]           # no switch, keeps growing
        assert all(row[l][1] == 'remove' for row in counts_gpm for l in range(3))
        assert cg[1] == c[1] and cg[2] == c[2]                            # identical where no switch happened


def test_pack_unpack(fix):
    M, _ = g.dualgpm_init(fix[0][0], 0.95)
    k = M.shape[1]
    buf = g.pack_basis(M, D)
    assert buf.shape == (D, D) and buf.dtype == np.float64
    assert np.array_equal(buf[:, :k], M) and not buf[:, k:].any()
    back = g.unpack_basis(buf, k)
    assert back.dtype == np.float64 and np.array_equal(back, M)
    assert g.unpack_basis(buf, 0) is None
    assert not g.pack_basis(None, D).any() and not g.pack_basis(np.zeros((D, 0)), D).any()
    buf32 = g.pack_basis(M, D, dtype=np.float32)
    assert buf32.dtype == np.float32 and np.abs(g.unpack_basis(buf32, k) - M).max() < 1e-6
    # the stored basis drives the next steps exactly like the live one
    M2, r2, a2 = g.dualgpm_step(fix[0][1], g.unpack_basis(buf, k), False, 0.955)
    M3, r3, a3 = g.dualgpm_step(fix[0][1], M, False, 0.955)
    assert a2 == a3 and r2 == r3 and np.array_equal(M2, M3)
    try:
        g.pack_basis(M, D - 1)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong buffer size accepted")


def main():
    fix = build_fixture()
    steps = [("threshold schedule", lambda: test_threshold_schedule()),
             ("quarter-cosine lr factor", lambda: test_cosine_lr_factor()),
             ("first-task basis", lambda: test_init(fix)),
             ("down-projection design", lambda: test_design(fix)),
             ("'remove' expansion, skip and cap", lambda: test_remove_update(fix)),
             ("finalize switch and assert", lambda: test_finalize()),
             ("'retain' shrink", lambda: test_retain_shrink()),
             ("equality with the upstream transcription", lambda: test_reference_equality(fix)),
             ("pack/unpack round trip", lambda: test_pack_unpack(fix))]
    for name, fn in steps:
        fn()
        print(f"ok  {name}")
    print("all inflora_gpm tests passed")


if __name__ == "__main__":
    main()
