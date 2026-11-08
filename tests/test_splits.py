"""NumPy-only tests for utils/splits.py (rebuttal plan E13/E14).

Run on the torch-free PC with ``python tests/test_splits.py`` (or pytest).
Covers class_indices, fewshot_counts, subsample, carve_val, describe and
apply_splits (the DataManager hook): invariants, determinism, seed
sensitivity, the rounding rule, argument validation, and that the global NumPy
RNG is never touched. The last test exercises utils.data_manager.DataManager
end-to-end on a synthetic dataset; it needs torch/torchvision (imported by
utils.data_manager), so it is skipped here and runs on the server CPU.
"""
import os
import sys
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.splits import (apply_splits, carve_val, class_indices, describe,  # noqa: E402
                          fewshot_counts, subsample)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def make_targets(seed=0, n_classes=23, dtype=np.int64):
    """Uneven class sizes in 1..60 with the samples interleaved (not contiguous).

    Class 0 has a single image (must stay in train), class 1 has two (the
    smallest class that can feed both splits), class 2 has sixty.
    """
    rs = np.random.RandomState(seed)
    sizes = rs.randint(1, 61, size=n_classes)
    sizes[0], sizes[1], sizes[2] = 1, 2, 60
    targets = np.repeat(np.arange(n_classes), sizes).astype(dtype)
    rs.shuffle(targets)
    return targets, sizes


def make_paths(n):
    return np.array(["/data/img_%05d.jpg" % i for i in range(n)])


def expected_n_val(n_c, frac):
    """The carve rule: round half up, at least 1, at most n_c - 1 (0 if n_c == 1)."""
    return min(max(1, int(np.floor(frac * n_c + 0.5 + 1e-9))), max(n_c - 1, 0))


def expect_raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError("%s not raised by %s(%r, %r)" % (exc.__name__, fn.__name__, args, kwargs))


def same_state(a, b):
    """Two np.random.get_state() tuples are identical."""
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:], "global RNG state changed"


def is_sorted_unique(idx):
    return idx.dtype == np.int64 and idx.ndim == 1 and bool(np.all(np.diff(idx) > 0))


# ----------------------------------------------------------------------------
# class_indices / fewshot_counts
# ----------------------------------------------------------------------------
def test_class_indices():
    for dtype in (np.int64, np.int32):
        t, sizes = make_targets(dtype=dtype)
        ci = class_indices(t)
        assert list(ci.keys()) == list(range(sizes.size))
        assert all(type(c) is int for c in ci)
        total = 0
        for c, idx in ci.items():
            assert is_sorted_unique(idx)
            assert np.all(t[idx] == c)
            assert idx.size == sizes[c]
            total += idx.size
        assert total == t.size
    assert class_indices(np.zeros(0, dtype=np.int64)) == {}
    assert list(class_indices([3, 3, 1]).keys()) == [1, 3]        # list input accepted
    assert np.array_equal(class_indices([3, 3, 1])[3], [0, 1])
    expect_raises(ValueError, class_indices, np.zeros((2, 2), dtype=int))
    expect_raises(ValueError, class_indices, np.array([0.5, 1.0]))


def test_fewshot_counts():
    t, sizes = make_targets()
    assert fewshot_counts(t, 5) == {c: min(5, int(n)) for c, n in enumerate(sizes)}
    assert fewshot_counts(t, np.int64(5)) == fewshot_counts(t, 5)
    assert fewshot_counts(t, 10 ** 6) == {c: int(n) for c, n in enumerate(sizes)}
    assert fewshot_counts(t, 1) == {c: 1 for c in range(sizes.size)}
    for bad in (0, -1, 2.5, True, None, "5"):
        expect_raises(ValueError, fewshot_counts, t, bad)


# ----------------------------------------------------------------------------
# subsample
# ----------------------------------------------------------------------------
def test_subsample():
    t, sizes = make_targets()
    counts = fewshot_counts(t, 5)
    keep = subsample(t, counts, seed=1993)
    assert is_sorted_unique(keep)
    assert np.array_equal(np.bincount(t[keep], minlength=sizes.size), np.minimum(sizes, 5))
    # determinism and seed sensitivity
    assert np.array_equal(keep, subsample(t, counts, 1993))
    assert not np.array_equal(keep, subsample(t, counts, 1994))
    # the dict order of counts does not matter
    assert np.array_equal(keep, subsample(t, dict(reversed(list(counts.items()))), 1993))
    # classes absent from counts keep everything, and one permutation is drawn
    # per class regardless, so the subset of class 2 is the same in both calls
    keep2 = subsample(t, {0: 1, 2: 5}, 1993)
    expected = sizes.copy()
    expected[2] = 5
    assert np.array_equal(np.bincount(t[keep2], minlength=sizes.size), expected)
    assert np.array_equal(keep[t[keep] == 2], keep2[t[keep2] == 2])
    # a count of 0 drops the class
    keep3 = subsample(t, {2: 0}, 1)
    assert not np.any(t[keep3] == 2) and keep3.size == t.size - 60
    # validation
    expect_raises(ValueError, subsample, t, {0: 2}, 1)   # class 0 has one image
    expect_raises(ValueError, subsample, t, {2: -1}, 1)
    expect_raises(ValueError, subsample, t, counts, None)
    assert subsample(np.zeros(0, dtype=np.int64), {}, 1).size == 0


# ----------------------------------------------------------------------------
# carve_val
# ----------------------------------------------------------------------------
def test_carve_val_invariants():
    t, sizes = make_targets()
    for frac, seed in ((0.1, 1993), (0.2, 7), (0.5, 0)):
        tr, va = carve_val(t, frac, seed)
        assert is_sorted_unique(tr) and is_sorted_unique(va)
        assert np.intersect1d(tr, va).size == 0
        assert np.array_equal(np.union1d(tr, va), np.arange(t.size))
        n_tr = np.bincount(t[tr], minlength=sizes.size)
        n_va = np.bincount(t[va], minlength=sizes.size)
        for c, n in enumerate(sizes):
            exp_val = expected_n_val(int(n), frac)
            assert n_va[c] == exp_val, (frac, c, n, n_va[c], exp_val)
            assert n_tr[c] == n - exp_val
            assert n_tr[c] >= 1                      # every class keeps a train image
            assert (n_va[c] >= 1) == (n >= 2)        # and a val image whenever it can
        assert n_va[0] == 0 and n_tr[0] == 1         # the one-image class
        assert n_va[1] == 1 and n_tr[1] == 1         # the two-image class


def test_carve_val_rounding_table():
    # (val_frac, n_c) -> n_val: round half up, then max(1, .), then cap at n_c - 1
    table = {(0.1, 1): 0, (0.1, 2): 1, (0.1, 5): 1, (0.1, 14): 1, (0.1, 15): 2,
             (0.1, 24): 2, (0.1, 25): 3, (0.1, 500): 50, (0.2, 30): 6, (0.2, 41): 8,
             (0.5, 2): 1, (0.5, 3): 2, (0.9, 10): 9, (0.99, 10): 9}
    for (frac, n), expected in table.items():
        tr, va = carve_val(np.zeros(n, dtype=np.int64), frac, 0)
        assert va.size == expected and tr.size == n - expected, (frac, n, va.size, expected)
    # min_val / min_train
    tr, va = carve_val(np.zeros(5, dtype=np.int64), 0.1, 0, min_val=2, min_train=3)
    assert va.size == 2 and tr.size == 3
    tr, va = carve_val(np.zeros(3, dtype=np.int64), 0.1, 0, min_val=2, min_train=3)
    assert va.size == 0 and tr.size == 3


def test_carve_val_determinism():
    t, _ = make_targets()
    a = carve_val(t, 0.1, 1993)
    b = carve_val(t, 0.1, 1993)
    c = carve_val(t, 0.1, 1994)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])
    for bad in (0.0, 1.0, -0.1, 1.5):
        expect_raises(ValueError, carve_val, t, bad, 0)
    expect_raises(ValueError, carve_val, t, 0.1, None)


# ----------------------------------------------------------------------------
# describe
# ----------------------------------------------------------------------------
def test_describe():
    t = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    assert describe(t, "train") == "train: n=10 C=4 per-class min/median/max=1/2.5/4"
    assert describe(t, "train", 5) == "train: n=10 C=5 per-class min/median/max=0/2/4"
    assert describe(np.zeros(0, dtype=np.int64), "val") == "val: none"
    assert describe(np.full(500, 7), "test", 8) == "test: n=500 C=8 per-class min/median/max=0/0/500"
    t2 = np.repeat(np.arange(100), 500)
    assert describe(t2, "train", 100) == "train: n=50000 C=100 per-class min/median/max=500/500/500"


# ----------------------------------------------------------------------------
# apply_splits (the DataManager hook)
# ----------------------------------------------------------------------------
def test_apply_splits_default_noop():
    t, _ = make_targets()
    rs = np.random.RandomState(1)
    imgs = rs.randint(0, 256, size=(t.size, 4, 4, 3)).astype(np.uint8)
    paths = make_paths(t.size)
    for data in (imgs, paths):
        np.random.seed(99)
        state = np.random.get_state()
        tr_d, tr_t, va_d, va_t = apply_splits(data, t)
        assert tr_d is data and tr_t is t                      # same objects, no copy
        assert va_d.shape == (0,) + data.shape[1:] and va_d.dtype == data.dtype
        assert va_t.shape == (0,) and va_t.dtype == t.dtype
        same_state(state, np.random.get_state())               # and no draw
    tr_d, tr_t, va_d, va_t = apply_splits(paths, t, val_frac=0.0, val_seed=None,
                                          fewshot_k=None, subsample_seed=None)
    assert tr_d is paths and tr_t is t and va_t.size == 0
    tr_d, tr_t, va_d, va_t = apply_splits(paths, t, val_frac=None)
    assert tr_d is paths and tr_t is t and va_t.size == 0


def test_apply_splits_carve_then_fewshot():
    t, sizes = make_targets()
    paths = make_paths(t.size)
    k = 5
    tr_d, tr_t, va_d, va_t = apply_splits(paths, t, val_frac=0.1, val_seed=1993,
                                          fewshot_k=k, subsample_seed=1993)
    # equals the explicit composition: carve the full set, subsample the remainder
    tr_idx, va_idx = carve_val(t, 0.1, 1993)
    sub = subsample(t[tr_idx], fewshot_counts(t[tr_idx], k), 1993)
    assert np.array_equal(tr_d, paths[tr_idx][sub]) and np.array_equal(tr_t, t[tr_idx][sub])
    assert np.array_equal(va_d, paths[va_idx]) and np.array_equal(va_t, t[va_idx])
    # dtypes preserved: paths stay str arrays, labels keep their dtype
    assert tr_d.dtype == paths.dtype and va_d.dtype == paths.dtype
    assert tr_t.dtype == t.dtype and va_t.dtype == t.dtype
    # realised counts: val from the carve rule, train = min(k, n_c - n_val) >= 1
    n_va = np.bincount(va_t, minlength=sizes.size)
    n_tr = np.bincount(tr_t, minlength=sizes.size)
    for c, n in enumerate(sizes):
        exp_val = expected_n_val(int(n), 0.1)
        assert n_va[c] == exp_val and n_tr[c] == min(k, n - exp_val) and n_tr[c] >= 1
    assert np.intersect1d(tr_d, va_d).size == 0            # disjoint images
    assert np.union1d(np.union1d(tr_d, va_d), paths).size == paths.size
    # the val split does not depend on the few-shot step
    _, _, va_d2, va_t2 = apply_splits(paths, t, val_frac=0.1, val_seed=1993)
    assert np.array_equal(va_d, va_d2) and np.array_equal(va_t, va_t2)
    # few-shot only
    tr_d3, tr_t3, _, va_t3 = apply_splits(paths, t, fewshot_k=k, subsample_seed=1993)
    assert va_t3.size == 0
    assert np.array_equal(np.bincount(tr_t3, minlength=sizes.size), np.minimum(sizes, k))
    assert np.array_equal(tr_d3, paths[subsample(t, fewshot_counts(t, k), 1993)])
    # seed sensitivity through the hook
    tr_d4, _, _, _ = apply_splits(paths, t, fewshot_k=k, subsample_seed=1994)
    assert not np.array_equal(tr_d3, tr_d4)
    # image arrays (CIFAR layout) are indexed the same way
    rs = np.random.RandomState(1)
    imgs = rs.randint(0, 256, size=(t.size, 4, 4, 3)).astype(np.uint8)
    tr_i, tr_ti, va_i, va_ti = apply_splits(imgs, t, val_frac=0.1, val_seed=1993,
                                            fewshot_k=k, subsample_seed=1993)
    assert tr_i.dtype == np.uint8 and tr_i.shape[1:] == (4, 4, 3) and va_i.shape[1:] == (4, 4, 3)
    assert np.array_equal(tr_i, imgs[tr_idx][sub]) and np.array_equal(va_i, imgs[va_idx])
    assert np.array_equal(tr_ti, tr_t) and np.array_equal(va_ti, va_t)


def test_apply_splits_validation():
    t, _ = make_targets()
    paths = make_paths(t.size)
    expect_raises(ValueError, apply_splits, paths, t, val_frac=0.1)              # seed required
    expect_raises(ValueError, apply_splits, paths, t, fewshot_k=5)               # seed required
    expect_raises(ValueError, apply_splits, paths, t, val_frac=1.0, val_seed=1)
    expect_raises(ValueError, apply_splits, paths, t, val_frac=-0.1, val_seed=1)
    expect_raises(ValueError, apply_splits, paths, t, fewshot_k=0, subsample_seed=1)
    expect_raises(ValueError, apply_splits, paths, t, fewshot_k=True, subsample_seed=1)
    expect_raises(TypeError, apply_splits, list(paths), t, fewshot_k=5, subsample_seed=1)
    expect_raises(ValueError, apply_splits, paths[:-1], t, fewshot_k=5, subsample_seed=1)


def test_global_rng_untouched():
    t, _ = make_targets()
    paths = make_paths(t.size)
    np.random.seed(12345)
    state = np.random.get_state()
    class_indices(t)
    fewshot_counts(t, 5)
    subsample(t, fewshot_counts(t, 5), 7)
    carve_val(t, 0.1, 3)
    describe(t, "train", 23)
    apply_splits(paths, t, val_frac=0.1, val_seed=1, fewshot_k=3, subsample_seed=2)
    same_state(state, np.random.get_state())
    assert np.random.randint(0, 10 ** 6) == np.random.RandomState(12345).randint(0, 10 ** 6)


# ----------------------------------------------------------------------------
# DataManager hook end-to-end (server CPU: needs torch/torchvision/PIL)
# ----------------------------------------------------------------------------
def test_data_manager_hook_optional():
    """Builds DataManager on a synthetic dataset with and without the hook.

    Checks the default path is unchanged (same arrays, same global RNG state,
    same class order and task sizes), that the carve happens before the
    few-shot step, that source="val" works and that describe() has the
    documented format. Skipped where utils.data_manager cannot be imported.
    """
    try:
        from torchvision import transforms
        from utils import data_manager as dm
    except Exception as exc:  # torch-free PC
        print("  skipped (needs torch/torchvision): %s" % exc)
        return

    class SyntheticData:
        use_path = False
        class_order = list(range(20))

        def __init__(self):
            rs = np.random.RandomState(0)
            sizes = rs.randint(3, 31, size=20)
            sizes[0] = 1
            self.train_targets = np.repeat(np.arange(20), sizes)
            rs.shuffle(self.train_targets)
            self.train_data = rs.randint(0, 256, size=(self.train_targets.size, 8, 8, 3)).astype(np.uint8)
            self.test_targets = np.repeat(np.arange(20), 4)
            self.test_data = rs.randint(0, 256, size=(80, 8, 8, 3)).astype(np.uint8)
            self.train_trsf = [transforms.ToTensor()]
            self.test_trsf = [transforms.ToTensor()]
            self.common_trsf = []

        def download_data(self):
            pass

    made = []

    def fake_get_idata(name, use_input_norm):
        made.append(SyntheticData())
        return made[-1]

    original = dm._get_idata
    dm._get_idata = fake_get_idata
    try:
        ref = dm.DataManager("synthetic", True, 1993, 2, 2)
        state_ref = np.random.get_state()
        explicit = dm.DataManager("synthetic", True, 1993, 2, 2, False, val_frac=0.0,
                                  val_seed=None, fewshot_k=None, subsample_seed=None)
        hooked = dm.DataManager("synthetic", True, 1993, 2, 2, val_frac=0.1, fewshot_k=5)
        state_hooked = np.random.get_state()
        same_state(state_ref, state_hooked)                      # the hook never touches np.random
        # default path: the arrays are the loader's own objects, no copy
        assert ref._train_data is made[0].train_data
        assert np.array_equal(explicit._train_data, ref._train_data)
        assert np.array_equal(explicit._train_targets, ref._train_targets)
        assert not ref.has_val and not explicit.has_val and hooked.has_val
        expect_raises(ValueError, ref.get_dataset, np.arange(2), "val", "test")
        expect_raises(ValueError, ref.get_dataset, np.arange(2), "bogus", "test")
        # untouched: test set, class order, task sizes
        assert np.array_equal(hooked._test_targets, ref._test_targets)
        assert np.array_equal(hooked._test_data, ref._test_data)
        assert hooked._class_order == ref._class_order
        assert hooked.get_task_sizes() == ref.get_task_sizes() == [2] * 10
        assert hooked.nb_tasks == ref.nb_tasks == 10
        # realised counts: carve first (round half up, >= 1, <= n_c - 1), then min(5, rest)
        full = ref.train_class_counts()
        tr = hooked.train_class_counts()
        va = hooked.val_class_counts()
        assert full.sum() == ref.train_set_size and tr.sum() == hooked.train_set_size
        for c in range(20):
            exp_val = expected_n_val(int(full[c]), 0.1)
            assert va[c] == exp_val and tr[c] == min(5, full[c] - exp_val) and tr[c] >= 1
        assert np.array_equal(ref.val_class_counts(), np.zeros(20, dtype=np.int64))
        # the [Data] line
        line = hooked.describe()
        assert line.startswith("[Data] dataset=synthetic | train: n=%d C=20 per-class min/median/max="
                               % hooked.train_set_size), line
        assert " | val: n=%d C=20 per-class min/median/max=" % va.sum() in line
        assert "(val_frac=0.1, val_seed=1993) | test: n=80 C=20 per-class min/median/max=4/4/4 | " in line
        assert line.endswith("| fewshot: k=5 (subsample_seed=1993)"), line
        assert ref.describe().endswith("| val: none | test: n=80 C=20 per-class min/median/max=4/4/4"
                                       " | fewshot: none"), ref.describe()
        # source="val" serves the carved images through the usual dataset
        vds = hooked.get_dataset(np.arange(0, 2), source="val", mode="test")
        assert len(vds) == va[:2].sum()
        _, _, img, label = vds[0]
        assert tuple(img.shape) == (3, 8, 8) and 0 <= int(label) < 2
        tds = hooked.get_dataset(np.arange(0, 20), source="train", mode="test")
        assert len(tds) == hooked.train_set_size
        assert np.array_equal(np.bincount(tds.labels, minlength=20), tr)
        # the image arrays stay uint8 and val images are train images of the full set
        assert hooked._train_data.dtype == np.uint8 and hooked._val_data.dtype == np.uint8
        full_imgs = ref._train_data.reshape(ref.train_set_size, -1)
        for arr in (hooked._train_data, hooked._val_data):
            flat = arr.reshape(len(arr), -1)
            assert all(np.any(np.all(full_imgs == row, axis=1)) for row in flat[:5])
        # argument validation reaches the constructor
        expect_raises(ValueError, dm.DataManager, "synthetic", True, 1993, 2, 2, val_frac=1.0)
        expect_raises(ValueError, dm.DataManager, "synthetic", True, 1993, 2, 2, fewshot_k=0)
    finally:
        dm._get_idata = original


# ----------------------------------------------------------------------------
def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok    %s" % name)
        except Exception:
            failed += 1
            print("FAIL  %s" % name)
            traceback.print_exc()
    print("%d/%d tests passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
