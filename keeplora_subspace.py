"""Subspace arithmetic of the KeepLoRA port (rebuttal E15b; NumPy only).

KeepLoRA (Luo et al., "KeepLoRA: Continual Learning with Residual Gradient
Adaptation", ICLR 2026, arXiv:2601.19659; official code
github.com/MaolinLuo/KeepLoRA, branch ``v1``, folder ``MTIL/``) keeps one
orthonormal basis U per adapted weight that spans (a) the principal subspace of
the pre-trained weight (paper Eq. 2, energy fraction eps_w) and (b) the
dominant directions of the inputs the weight saw on every previous task (paper
Eq. 3-4, energy fraction eps_f).  Before a task trains, the gradient of the
frozen weight on the new task is projected away from U (Eq. 5) and its top-r
singular pairs initialise a LoRA whose A is frozen and whose B is trained
(Eq. 6); after the task the LoRA is folded and U grows (Alg. 1).

This module holds the three computations as pure NumPy functions so they can be
unit-tested without torch (scripts/test_keeplora_subspace.py) and so the SVDs
run in float64 on the CPU, outside torch's deterministic mode.  The torch side
(backbones.KeepLoRALinear / KeepLoRAViT, main.Learner._keeplora_* hooks) only
moves tensors in and out.

Conventions.  Weights are kept in torch layout W (out, in); the paper's W is its
transpose (y = x W, W in R^{in x out}).  The basis U is (in, cols) and acts on
the INPUT side -- the side on which the paper's unified subspace [W_p, M_t] is
defined (Sec. 3.2 bounds its size by d_in).  The released code builds W_p from
the LEFT singular vectors of the torch-layout weight (its output side) and
applies them on the input side, which only type-checks because every adapted
matrix is square; ``weight_subspace(..., side="output")`` reproduces that
choice and is the port's default (it is what the published numbers come from),
``side="input"`` is the paper's Eq. 2 (right singular vectors of the torch
weight).  Feature statistics enter as Grams K = X X^T (in x in, any positive
scale): ``extend_subspace`` needs only the spectrum of the residual, and a Gram
is 768 x 768 whatever the number of tokens.  The released code SVDs the
element-wise MEAN of the per-batch token matrices (``accumulate_features``,
batches of equal size); ``BatchMeanGram`` reproduces that statistic as a Gram,
and the alternative Gram over all tokens (plain GPM, what Eq. 3-4 literally
describe) is one ``mode`` switch away in the torch collector.
"""
import numpy as np

__all__ = [
    "weight_subspace", "extend_subspace", "extend_subspace_from_matrix", "gradient_init",
    "basis_leakage", "BatchMeanGram", "pack_basis", "unpack_basis", "energy_count",
]


def _check_eps(eps, name):
    eps = float(eps)
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {eps}")
    return eps


def _basis_array(basis):
    """float64 (d, cols) copy of ``basis`` or None when it is None / empty."""
    if basis is None:
        return None
    M = np.asarray(basis, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] == 0:
        return None
    return M


def energy_count(energies, threshold):
    """Released ``_update_prin_subspace`` (first task) / weight step: the number
    of leading ``energies`` whose cumulative share is STRICTLY below
    ``threshold`` (``torch.sum(torch.cumsum(ratio) < threshold)``), i.e. the
    kept rank before the ``max(r, 1)`` floor.  ``energies`` are squared singular
    values (or Gram eigenvalues) in descending order."""
    e = np.asarray(energies, dtype=np.float64)
    total = float(e.sum())
    if total <= 0.0:
        return 0
    return int(np.sum(np.cumsum(e / total) < threshold))


def weight_subspace(weight, eps_w, side="output"):
    """Principal subspace of one pre-trained weight (paper Eq. 2; released
    ``PrinSubspace.update_prin_subspace`` on ``get_accumulated_weight_matrix_list``).

    ``weight``: torch layout (out, in).  Returns an orthonormal (in, r) basis
    with r = max(energy_count(S^2, eps_w), 1), or None when eps_w <= 1e-8 (the
    released code skips thresholds below 1e-8).  ``side='output'``: the left
    singular vectors of ``weight`` (released code; square matrices only);
    ``side='input'``: its right singular vectors (the paper's W_p for y = x W)."""
    eps_w = _check_eps(eps_w, "eps_w")
    if eps_w <= 1e-8:
        return None
    W = np.asarray(weight, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {W.shape}")
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    r = max(energy_count(S ** 2, eps_w), 1)
    if side == "output":
        if W.shape[0] != W.shape[1]:
            raise ValueError("side='output' needs a square weight (the released code "
                             f"applies output-side vectors on the input side); got {W.shape}")
        return np.ascontiguousarray(U[:, :r])
    if side == "input":
        return np.ascontiguousarray(Vh[:r].T)
    raise ValueError(f"side must be 'output' or 'input', got {side!r}")


def extend_subspace(basis, gram, eps_f):
    """One KeepLoRA memory update (paper Eq. 3-4; released
    ``PrinSubspace._update_prin_subspace`` with a feature matrix), on the Gram
    K = X X^T (in x in) of the features X the weight saw on the task.

    Empty ``basis``: U = leading eigenvectors of K with r = max(energy_count, 1)
    (the released first-task branch).  Otherwise the residual K_hat =
    (I - U U^T) K (I - U U^T) is eigen-decomposed, the already captured share
    a = (tr K - tr K_hat) / tr K is topped up with residual eigen-shares one at
    a time while a < eps_f (the released loop, which stops at the FIRST
    eigenvalue that lifts a to >= eps_f and therefore includes it), and the
    corresponding eigenvectors are appended; nothing is added when a >= eps_f
    already.  The result is cut to its first d columns (released cap).

    Returns (new_basis or None, number of columns added)."""
    eps_f = _check_eps(eps_f, "eps_f")
    K = np.asarray(gram, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"gram must be square, got shape {K.shape}")
    d = K.shape[0]
    K = 0.5 * (K + K.T)
    M = _basis_array(basis)
    if M is None:
        lam, Q = np.linalg.eigh(K)
        order = np.argsort(lam)[::-1]
        lam, Q = np.clip(lam[order], 0.0, None), Q[:, order]
        r = max(energy_count(lam, eps_f), 1)
        return np.ascontiguousarray(Q[:, :r]), r
    if M.shape[0] != d:
        raise ValueError(f"basis has {M.shape[0]} rows but the Gram is {d} x {d}")
    total = float(np.trace(K))
    if total <= 0.0:
        return M, 0
    P = np.eye(d) - M @ M.T
    K_hat = P @ K @ P
    K_hat = 0.5 * (K_hat + K_hat.T)
    lam, Q = np.linalg.eigh(K_hat)
    order = np.argsort(lam)[::-1]
    lam, Q = np.clip(lam[order], 0.0, None), Q[:, order]
    hat = float(lam.sum())
    ratio = lam / total
    accumulated = (total - hat) / total
    r = 0
    for i in range(ratio.shape[0]):
        if accumulated < eps_f:
            accumulated += ratio[i]
            r += 1
        else:
            break
    if r == 0:
        return M, 0
    Ui = np.hstack((M, Q[:, :r]))
    if Ui.shape[1] > d:
        Ui = Ui[:, :d]
    return np.ascontiguousarray(Ui), int(Ui.shape[1] - M.shape[1])


def extend_subspace_from_matrix(basis, X, eps_f):
    """``extend_subspace`` on an explicit feature matrix X (in, n): identical to
    the released SVD route because X X^T shares X's left singular vectors and
    its squared singular values."""
    X = np.asarray(X, dtype=np.float64)
    return extend_subspace(basis, X @ X.T, eps_f)


def gradient_init(grad, basis, rank):
    """KeepLoRA initialisation (paper Eq. 5-6; released ``in_proj_weight_process``
    / ``out_proj_weight_process`` with an exact SVD in place of
    ``torch.svd_lowrank(q=4r, niter=4)``).

    ``grad``: the task gradient of the frozen weight in PAPER layout (in, out),
    i.e. the transpose of ``weight.grad``.  G_hat = G - U U^T G; A = top-``rank``
    left singular vectors of G_hat (in, rank); B = S_r V_r^T (rank, out).  The
    LoRA increment of the released module is x A B * scaling, and
    ``KeepLoRALinear`` subtracts scaling * (A B)^T from the weight once at
    initialisation so the network starts unchanged (Alg. 1).  Returns (A, B)."""
    G = np.asarray(grad, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"grad must be 2-D, got shape {G.shape}")
    rank = int(rank)
    if not 1 <= rank <= min(G.shape):
        raise ValueError(f"rank {rank} outside [1, {min(G.shape)}] for a {G.shape} gradient")
    M = _basis_array(basis)
    if M is not None:
        if M.shape[0] != G.shape[0]:
            raise ValueError(f"basis has {M.shape[0]} rows, the gradient {G.shape[0]} (paper layout (in, out))")
        G = G - M @ (M.T @ G)
    U, S, Vh = np.linalg.svd(G, full_matrices=False)
    A = np.ascontiguousarray(U[:, :rank])
    B = np.ascontiguousarray(S[:rank, None] * Vh[:rank])
    return A, B


def basis_leakage(A, basis):
    """max |U^T A|: how much of the designed A lies inside the protected
    subspace (0 up to round-off when every kept singular value is non-zero)."""
    M = _basis_array(basis)
    if M is None:
        return 0.0
    return float(np.max(np.abs(M.T @ np.asarray(A, dtype=np.float64))))


class BatchMeanGram:
    """The released feature statistic as a Gram.

    ``KeepLoRA.accumulate_features`` keeps, per module, the element-wise
    running mean over batches of the token matrix X_b^T (in, tokens_per_batch)
    -- a (in, B*L) matrix whose SVD then feeds ``_update_prin_subspace``.  (Its
    update ``n/(n+m) * acc + 1/(n+m) * X_b^T`` with m tokens per batch equals
    the plain mean of the batch matrices divided by m; the scale is irrelevant
    to every energy ratio.)  Batches of a different token count cannot be
    averaged element-wise: the released loader uses ``drop_last=True``, and
    this class counts and skips such a batch instead of raising.

    ``update(xt)`` takes X_b^T as (in, m); ``gram()`` returns Xbar Xbar^T with
    Xbar the batch mean, so ``extend_subspace`` sees exactly the spectrum the
    released SVD would see.  Pure NumPy; the torch collector in backbones.py
    keeps the running sum on the GPU and calls ``gram_from_sum``."""

    def __init__(self):
        self.sum = None
        self.n_batches = 0
        self.n_skipped = 0

    def update(self, xt):
        xt = np.asarray(xt, dtype=np.float64)
        if xt.ndim != 2:
            raise ValueError(f"expected (in, tokens), got shape {xt.shape}")
        if self.sum is None:
            self.sum = xt.copy()
            self.n_batches = 1
        elif xt.shape != self.sum.shape:
            self.n_skipped += 1
        else:
            self.sum += xt
            self.n_batches += 1

    def mean(self):
        if self.sum is None:
            return None
        return self.sum / self.n_batches

    def gram(self):
        Xbar = self.mean()
        if Xbar is None:
            return None
        return Xbar @ Xbar.T

    @staticmethod
    def gram_from_sum(total, n_batches):
        """Gram of the batch mean from a running sum (in, m) and its count."""
        Xbar = np.asarray(total, dtype=np.float64) / float(n_batches)
        return Xbar @ Xbar.T


def pack_basis(basis, d, dtype=np.float32):
    """Zero-padded (d, d) array holding ``basis`` in its first columns (the
    fixed-shape buffer of KeepLoRALinear, so trainable_state round-trips it)."""
    buf = np.zeros((d, d), dtype=dtype)
    M = _basis_array(basis)
    if M is not None:
        if M.shape[0] != d or M.shape[1] > d:
            raise ValueError(f"basis shape {M.shape} does not fit a {d} x {d} buffer")
        buf[:, :M.shape[1]] = M
    return buf


def unpack_basis(buf, cols):
    """Inverse of ``pack_basis``: float64 (d, cols) copy, or None when ``cols`` is 0."""
    cols = int(cols)
    if cols <= 0:
        return None
    return np.array(np.asarray(buf)[:, :cols], dtype=np.float64)
