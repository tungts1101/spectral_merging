"""NumPy-only per-class training-set splits for the rebuttal experiments.

Used by ``utils.data_manager.DataManager`` (hook ``_apply_splits``) for
E13 (few-shot per task) and E14 (validation-split selection). It depends on
NumPy alone, so it imports on the torch-free PC and inside the DataManager on
the server.

Protocol (rebuttal plan, items E13/E14):
  1. ``carve_val`` moves a stratified fraction of every class of the remapped
     training set into a validation split, at least one image per class in
     each split.
  2. ``subsample`` then keeps ``fewshot_counts`` = min(k, n_c) images of every
     remaining training class.
  The test set, the class order and the task increments are never touched.

Randomness: every function draws from a private ``np.random.RandomState(seed)``
created inside the call and visits the classes in ascending label order. The
global NumPy stream (which ``DataManager._setup_data`` re-seeds for the class
order) is never read or advanced, and ``apply_splits`` with the default
arguments performs no draw and no copy at all.

Index arrays are always sorted int64, so they fancy-index both the uint8 image
arrays (CIFAR) and the ``np.array`` of path strings (ImageFolder datasets)
without changing their dtype.
"""
import numpy as np

__all__ = [
    "class_indices", "fewshot_counts", "subsample", "carve_val", "describe",
    "apply_splits",
]


def _as_targets(targets):
    """Validate a 1-D integer label array; returns it as an ndarray (no copy)."""
    t = np.asarray(targets)
    if t.ndim != 1:
        raise ValueError("targets must be 1-D, got shape %s" % (t.shape,))
    if t.size and not np.issubdtype(t.dtype, np.integer):
        raise ValueError("targets must be integers, got dtype %s" % t.dtype)
    return t


def _check_count(name, value):
    """A positive Python/NumPy integer (bool excluded), returned as int."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError("%s must be a positive integer, got %r" % (name, value))
    return int(value)


def _sorted_index(parts):
    """Concatenate index chunks into one sorted int64 array (empty-safe)."""
    if not parts:
        return np.zeros(0, dtype=np.int64)
    out = np.concatenate(parts).astype(np.int64, copy=False)
    out.sort()
    return out


def class_indices(targets):
    """Map every class label to the sorted int64 indices of its samples.

    Keys are Python ints in ascending label order, which is the order in which
    every seeded draw in this module visits the classes.
    """
    t = _as_targets(targets)
    if t.size == 0:
        return {}
    order = np.argsort(t, kind="stable").astype(np.int64)  # stable: ascending within a class
    labels, starts = np.unique(t[order], return_index=True)
    ends = np.append(starts[1:], t.size)
    return {int(c): order[s:e] for c, s, e in zip(labels, starts, ends)}


def fewshot_counts(targets, k):
    """Few-shot budget per class, ``{c: min(k, n_c)}`` (plan E13).

    A class with fewer than ``k`` training images keeps all of them (ImageNet-R
    has such classes), so the realised counts are quoted from the ``[Data]``
    log line instead of being assumed to equal ``k``.
    """
    k = _check_count("fewshot k", k)
    return {c: min(k, int(idx.size)) for c, idx in class_indices(targets).items()}


def subsample(targets, counts, seed):
    """Per-class subsample: keep the first ``counts[c]`` of a seeded permutation.

    One ``np.random.RandomState(seed)`` serves the whole call; it draws one
    permutation of every class's sample positions, in ascending label order,
    whether or not that class is reduced. The result therefore depends only on
    ``(targets, counts, seed)``, never on the dict order of ``counts``, and the
    subset drawn for a class is the same whichever other classes are listed.
    Classes absent from ``counts`` keep all their samples; a count outside
    ``[0, n_c]`` raises ValueError. Returns sorted int64 indices into
    ``targets``.
    """
    if seed is None:
        raise ValueError("subsample needs an integer seed (got None)")
    rs = np.random.RandomState(seed)
    keep = []
    for c, idx in class_indices(targets).items():
        perm = rs.permutation(idx.size)
        if c in counts:
            n = int(counts[c])
            if n < 0 or n > idx.size:
                raise ValueError("count for class %d must lie in [0, %d], got %d"
                                 % (c, idx.size, n))
            keep.append(idx[perm[:n]])
        else:
            keep.append(idx)
    return _sorted_index(keep)


def _n_val(n_c, val_frac, min_val, min_train):
    """Validation count of one class: max(min_val, round-half-up(val_frac * n_c)),
    capped so that at least ``min_train`` images stay in train."""
    n_val = max(int(min_val), int(np.floor(val_frac * n_c + 0.5 + 1e-9)))
    return min(n_val, max(n_c - int(min_train), 0))


def carve_val(targets, val_frac, seed, min_val=1, min_train=1):
    """Stratified validation carve (plan E14).

    Per class with ``n_c`` images, ``n_val = max(min_val, round(val_frac * n_c))``
    (round half up), capped at ``n_c - min_train`` so that at least ``min_train``
    images stay in train. With the defaults every class with ``n_c >= 2`` has at
    least one image in each split; a class with a single training image keeps
    it in train (``n_val = 0``).

    A private ``np.random.RandomState(seed)`` draws one permutation per class in
    ascending label order; the first ``n_val`` positions go to val, the rest
    stay in train. Returns ``(train_idx, val_idx)``: sorted int64 arrays that
    are disjoint and together cover ``arange(len(targets))``.
    """
    val_frac = float(val_frac)
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must lie in (0, 1), got %r" % val_frac)
    if seed is None:
        raise ValueError("carve_val needs an integer seed (got None)")
    rs = np.random.RandomState(seed)
    train, val = [], []
    for c, idx in class_indices(targets).items():
        n_val = _n_val(idx.size, val_frac, min_val, min_train)
        perm = rs.permutation(idx.size)
        val.append(idx[perm[:n_val]])
        train.append(idx[perm[n_val:]])
    return _sorted_index(train), _sorted_index(val)


def describe(targets, name, n_classes=None):
    """One token of the ``[Data]`` log line, for example
    ``"train: n=50000 C=100 per-class min/median/max=500/500/500"``;
    an empty split gives ``"<name>: none"``.

    With ``n_classes`` the per-class counts are
    ``np.bincount(targets, minlength=n_classes)``, so a class missing from the
    split shows up as ``min=0``; without it only the labels present are counted.
    The median is printed with ``%.10g`` (``500`` or ``10.5``).
    """
    t = _as_targets(targets)
    if t.size == 0:
        return "%s: none" % name
    if n_classes is None:
        counts = np.bincount(t)
        counts = counts[counts > 0]
    else:
        counts = np.bincount(t, minlength=int(n_classes))
    return "%s: n=%d C=%d per-class min/median/max=%d/%s/%d" % (
        name, t.size, counts.size, counts.min(), "%.10g" % np.median(counts), counts.max())


def apply_splits(data, targets, *, val_frac=0.0, val_seed=None,
                 fewshot_k=None, subsample_seed=None):
    """The DataManager hook: validation carve first, then few-shot subsampling.

    Returns ``(train_data, train_targets, val_data, val_targets)``.

    With ``val_frac`` = 0 and ``fewshot_k`` = None (the defaults) the inputs
    come back as the same objects (no copy, no draw) and the val arrays are
    empty slices of them, so the default DataManager behaviour is unchanged.
    Otherwise ``data`` must be a NumPy array (uint8 images or ``str`` paths);
    it is fancy-indexed with int64 index arrays, which keeps its dtype. The
    carve uses ``val_seed`` and the few-shot draw ``subsample_seed``; a seed
    must be given whenever its step is active (the DataManager fills in the
    run seed).
    """
    val_frac = 0.0 if val_frac is None else float(val_frac)
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must lie in [0, 1), got %r" % val_frac)
    if fewshot_k is not None:
        fewshot_k = _check_count("fewshot_k", fewshot_k)
    t = _as_targets(targets)
    if val_frac == 0.0 and fewshot_k is None:
        return data, targets, data[:0], targets[:0]
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array (images or path strings), got %s"
                        % type(data).__name__)
    if len(data) != t.size:
        raise ValueError("data and targets differ in length: %d vs %d" % (len(data), t.size))
    train_idx = np.arange(t.size, dtype=np.int64)
    val_idx = np.zeros(0, dtype=np.int64)
    if val_frac > 0.0:
        train_idx, val_idx = carve_val(t, val_frac, val_seed)
    if fewshot_k is not None:
        remaining = t[train_idx]
        sub = subsample(remaining, fewshot_counts(remaining, fewshot_k), subsample_seed)
        train_idx = train_idx[sub]
    return data[train_idx], t[train_idx], data[val_idx], t[val_idx]
