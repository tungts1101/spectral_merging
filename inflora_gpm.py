"""NumPy port of InfLoRA's projection math (Liang & Li, CVPR 2024).

InfLoRA keeps one frozen LoRA branch per task on the K and V rows of every
qkv projection.  Its fixed down-projection A_t is designed from the current
task's input statistics before training, and after training a DualGPM memory
of the old tasks' input subspace is grown from the same statistics.  Both steps
are pure linear algebra on the uncentered Gram matrix of the module input,
G = X^T X / n (upstream ``Attention_LoRA.cur_matrix``; here the output of
``backbones.compute_covariances``).  This module transcribes that math from the
official code so it is deterministic on the CPU (NumPy SVD, as upstream), is
importable by ``backbones.py`` without torch, and is unit-tested on a torch-less
machine by ``scripts/test_inflora_gpm.py``.

Provenance (github.com/liangyanshuo/InfLoRA, MIT licence):
  threshold_schedule      methods/inflora.py: update_DualGPM / update_GPM,
                          ``threshold = (lame - lamb) * cur_task / total_sessions + lamb``
  cosine_lr_factor        utils/schedulers.py: CosineSchedule.cosine,
                          ``base_lr * cos(99 pi last_epoch / (200 (K - 1)))``
  design_down_projection  methods/inflora.py: InfLoRA._train (the design of
                          lora_A_k[t] / lora_A_v[t]): one-sided removal
                          (``cur - feature_mat @ cur``) or retention
                          (``feature_mat @ cur``) of the stored subspace from
                          the current task's Gram, then ``U[:, :r].T / sqrt(3)``
  dualgpm_init            first-task branch of update_DualGPM (update_GPM when
                          ``dual`` is False)
  dualgpm_update          the 'remove' (GPM expansion) and 'retain' (complement
                          shrinking) branches of update_DualGPM
  dualgpm_finalize        the summary loop of update_DualGPM: a 'remove' layer
                          with more than d/2 columns switches to 'retain' and
                          stores the orthogonal complement; a 'retain' layer
                          must keep at most d/2 columns
  dualgpm_step            one update_DualGPM call for one layer
  pack_basis/unpack_basis fixed-size (d x d, zero-padded) storage of the basis
                          for the torch buffers of the InfLoRAQKV wrapper

Conventions and caveats reproduced on purpose:
  * The energy criterion is applied to the squared singular values of the Gram
    matrix, i.e. to the fourth powers of the activation singular values.  That
    is far more peaked than the Frobenius criterion of the paper's Eqs. (14)
    and (16); the port follows the code so the published numbers are matched.
  * The first-task count is strict: ``r = sum(cumsum(ratio) < threshold)`` (the
    upstream ``#+1`` is commented out), with at least one column kept.  An
    expansion instead appends directions until the accumulated energy of the
    new Gram covered by the basis *reaches* the threshold.  So presenting the
    same Gram again right after the first task still appends one direction.
  * A 'remove' basis is capped at d columns after an expansion.
  * Upstream's first-task 'retain' branch stores the leading singular vectors
    under the label 'retain' and then trips its own ``cols <= d/2`` assertion.
    It is unreachable at the published thresholds (<= 0.998 on ViT Grams); the
    port stores the orthogonal complement instead and warns.

Shapes: ``gram`` is (d, d); a basis is (d, cols) with orthonormal columns
(``retain`` False: the protected subspace M; True: its complement M_perp).
All computations run in float64; ``design_down_projection`` returns float32
because it is copied into the LoRA parameters.
"""
import math
import warnings

import numpy as np

SQRT3 = math.sqrt(3.0)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def threshold_schedule(lamb, lame, task, total_tasks):
    """DualGPM energy threshold after finishing task ``task`` (0-based).

    Upstream: ``(lame - lamb) * cur_task / total_sessions + lamb``.
    CIFAR-100: lamb 0.95 -> 0.95, 0.955, ..., 0.995; ImageNet-R: lamb 0.98 ->
    0.98, ..., 0.998 (``total_tasks`` = 10, ``task`` in 0..9).
    """
    return (lame - lamb) * task / total_tasks + lamb


def cosine_lr_factor(epoch, total_epochs):
    """Per-epoch learning-rate factor of upstream ``CosineSchedule(K)``.

    ``cos(99 pi e / (200 (K - 1)))``: 1.0 at epoch 0, cos(99 pi / 200) = 0.0157
    at the last epoch.  Returns 1.0 when ``total_epochs`` <= 1 (the smoke
    configs train one epoch), which is the guard ``main.py`` needs for its
    ``train_scheduler = "inflora_cosine"`` option.
    """
    if total_epochs <= 1:
        return 1.0
    return math.cos(99.0 * math.pi * epoch / (200.0 * (total_epochs - 1)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_threshold(threshold):
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"DualGPM threshold must lie in (0, 1), got {threshold}")


def _basis_array(basis):
    """None / empty -> None; otherwise a float64 (d, cols) array."""
    if basis is None:
        return None
    M = np.asarray(basis, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] == 0:
        return None
    return M


def _energy_count(S, threshold):
    """Number of leading singular values whose cumulative squared energy stays
    strictly below ``threshold`` (upstream ``np.sum(np.cumsum(ratio) < thr)``)."""
    s2 = np.asarray(S, dtype=np.float64) ** 2
    ratio = s2 / s2.sum()
    return int(np.sum(np.cumsum(ratio) < threshold))


# ---------------------------------------------------------------------------
# Down-projection design (InfLoRA._train)
# ---------------------------------------------------------------------------

def design_down_projection(gram, basis, retain, rank):
    """Fixed down-projection A_t (``rank`` x d, float32) for the current task.

    Upstream ``InfLoRA._train``: task 0 takes the left singular vectors of the
    raw Gram; later tasks first project the Gram one-sidedly with
    ``feature_mat = F F^T``: ``cur - feature_mat @ cur`` for a 'remove' layer
    (``retain`` False, F = M) or ``feature_mat @ cur`` for a 'retain' layer
    (``retain`` True, F = M_perp).  The top-``rank`` left singular vectors are
    transposed and divided by sqrt(3); the same matrix serves K and V.
    With an empty ``basis`` the raw Gram is used (task 0).
    """
    G = np.asarray(gram, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"gram must be square, got shape {G.shape}")
    M = _basis_array(basis)
    if M is not None:
        feature_mat = np.dot(M, M.T)
        G = np.dot(feature_mat, G) if retain else G - np.dot(feature_mat, G)
    U = np.linalg.svd(G, full_matrices=False)[0]
    return np.ascontiguousarray(U[:, :rank].T / SQRT3, dtype=np.float32)


def basis_leakage(A, basis, retain):
    """Diagnostic for the log line: how much of A_t lies where it should not.

    'remove' (``retain`` False): max |A M|, the overlap with the protected
    subspace (should be ~0).  'retain' (True): max |A - (A M_perp) M_perp^T|,
    the part of the rows outside the retained subspace (should be ~0).  0.0
    when the basis is empty.
    """
    M = _basis_array(basis)
    if M is None:
        return 0.0
    A = np.asarray(A, dtype=np.float64)
    AM = np.dot(A, M)
    if retain:
        return float(np.abs(A - np.dot(AM, M.T)).max())
    return float(np.abs(AM).max())


# ---------------------------------------------------------------------------
# DualGPM memory (InfLoRA.update_DualGPM / update_GPM)
# ---------------------------------------------------------------------------

def dualgpm_init(gram, threshold, dual=True):
    """First-task basis: ``(M, retain)``.

    ``r = sum(cumsum(S^2 / sum S^2) < threshold)`` on the SVD of the Gram.
    ``r < d/2`` (always the case at the published thresholds): store
    ``U[:, :max(r, 1)]`` as a 'remove' basis.  Plain GPM (``dual`` False,
    upstream ``update_GPM``) always does that.  With ``dual`` True and
    ``r >= d/2`` upstream would store the same columns labelled 'retain' and
    then fail its own assertion; the port stores the orthogonal complement
    ``U[:, r:]`` as a 'retain' basis and warns.
    """
    _check_threshold(threshold)
    G = np.asarray(gram, dtype=np.float64)
    d = G.shape[0]
    U, S, _ = np.linalg.svd(G, full_matrices=False)
    r = _energy_count(S, threshold)
    if (not dual) or r < d / 2:
        return U[:, :max(r, 1)], False
    warnings.warn(f"DualGPM init: {r} of {d} directions needed at threshold "
                  f"{threshold:.4f}; storing the {d - r}-column complement as a "
                  "'retain' basis (upstream would fail its cols <= d/2 assert)")
    return U[:, r:], True


def dualgpm_update(gram, basis, retain, threshold):
    """One DualGPM update of an existing basis: ``(M, retain, changed)``.

    'remove' layer (``retain`` False, M = protected subspace), upstream GPM
    expansion: ``act_hat = G - (M M^T) G``; with ``sval_total = sum S_G^2`` and
    ``sval_hat = sum S_hat^2``, start from
    ``acc = (sval_total - sval_hat) / sval_total`` and append the left singular
    vectors of ``act_hat`` while ``acc < threshold`` (``acc += S_i^2 /
    sval_total``).  No direction -> the basis is returned unchanged.  The
    stacked basis is capped at d columns.  ``changed`` = columns appended.

    'retain' layer (``retain`` True, M = M_perp): ``act_hat = (M M^T) G``,
    ``acc = sval_hat / sval_total``; count directions while
    ``acc >= 1 - threshold`` (``acc -= S_i^2 / sval_total``); then M_perp
    becomes the first ``cols - r`` left singular vectors of
    ``M - (U_r U_r^T) M``.  ``changed`` = columns removed.
    """
    _check_threshold(threshold)
    G = np.asarray(gram, dtype=np.float64)
    M = np.asarray(basis, dtype=np.float64)
    d = G.shape[0]
    if M.ndim != 2 or M.shape[0] != d:
        raise ValueError(f"basis shape {M.shape} does not match gram {G.shape}")
    cols = M.shape[1]
    U1, S1, _ = np.linalg.svd(G, full_matrices=False)
    sval_total = float((S1 ** 2).sum())

    if not retain:
        # Projected representation: the part of G outside the stored subspace.
        act_hat = G - np.dot(np.dot(M, M.T), G)
        U, S, _ = np.linalg.svd(act_hat, full_matrices=False)
        sval_hat = float((S ** 2).sum())
        sval_ratio = (S ** 2) / sval_total
        accumulated_sval = (sval_total - sval_hat) / sval_total
        r = 0
        for ii in range(sval_ratio.shape[0]):
            if accumulated_sval < threshold:
                accumulated_sval += sval_ratio[ii]
                r += 1
            else:
                break
        if r == 0:
            return M, False, 0                      # upstream: 'Skip Updating DualGPM'
        Ui = np.hstack((M, U[:, :r]))
        if Ui.shape[1] > Ui.shape[0]:               # cap at d columns
            Ui = Ui[:, :Ui.shape[0]]
        return Ui, False, int(Ui.shape[1] - cols)

    # 'retain': the part of G inside the retained complement.
    act_hat = np.dot(np.dot(M, M.T), G)
    U, S, _ = np.linalg.svd(act_hat, full_matrices=False)
    sval_hat = float((S ** 2).sum())
    sval_ratio = (S ** 2) / sval_total
    accumulated_sval = sval_hat / sval_total
    r = 0
    for ii in range(sval_ratio.shape[0]):
        if accumulated_sval >= (1 - threshold):
            accumulated_sval -= sval_ratio[ii]
            r += 1
        else:
            break
    if r == 0:
        return M, True, 0
    r = min(r, cols)   # unreachable for threshold < 1 (act_hat has rank <= cols)
    act_feature = M - np.dot(np.dot(U[:, :r], U[:, :r].T), M)
    Ui = np.linalg.svd(act_feature)[0]              # full U, as upstream
    return Ui[:, :cols - r], True, int(r)


def dualgpm_finalize(basis, retain, dual=True):
    """Post-update bookkeeping of upstream's 'Gradient Constraints Summary'.

    ``dual`` True: a 'remove' basis with more than d/2 columns is replaced by
    its orthogonal complement (``svd(M)[0][:, cols:]``) and becomes 'retain';
    a 'retain' basis must have at most d/2 columns (AssertionError otherwise,
    like upstream's assert).  ``dual`` False (plain GPM) never switches.
    """
    M = np.asarray(basis, dtype=np.float64)
    d, cols = M.shape
    if dual and (not retain) and cols > d / 2:
        U = np.linalg.svd(M)[0]                     # full U: columns cols.. = complement
        return U[:, cols:], True
    if retain and cols > d / 2:
        raise AssertionError(f"'retain' basis has {cols} > d/2 = {d / 2} columns")
    return M, retain


def dualgpm_step(gram, basis, retain, threshold, dual=True):
    """One ``update_DualGPM`` call for one layer: ``(M, retain, changed)``.

    Empty ``basis`` (None or 0 columns) -> ``dualgpm_init``; otherwise
    ``dualgpm_update``; then ``dualgpm_finalize``.  ``changed`` is the number
    of columns stored (init), appended ('remove') or removed ('retain') before
    the finalize step.  ``dual`` False gives plain ``update_GPM``.
    """
    M = _basis_array(basis)
    if M is None:
        M, retain = dualgpm_init(gram, threshold, dual)
        changed = int(M.shape[1])
    else:
        M, retain, changed = dualgpm_update(gram, M, retain, threshold)
    M, retain = dualgpm_finalize(M, retain, dual)
    return M, bool(retain), int(changed)


# ---------------------------------------------------------------------------
# Fixed-size storage
# ---------------------------------------------------------------------------

def pack_basis(basis, d, dtype=np.float64):
    """Zero-padded (d, d) array holding ``basis`` in its first columns.

    The InfLoRAQKV wrapper keeps the basis in fixed-shape buffers (``basis``
    d x d, ``basis_cols``, ``basis_retain``) so ``trainable_state`` round-trips
    it without variable shapes.  ``basis`` None/empty -> all zeros.
    """
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
