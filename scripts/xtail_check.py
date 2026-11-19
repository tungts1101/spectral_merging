"""Per-domain readiness check of the X-TAIL stream (CPU only, no torch model).

For every domain of utils.data.XTAIL_DOMAINS: image directory found, split
loaded, first and last image present, train / test counts, classes. Missing
domains are reported, not fatal. Then the full `xtail` loader is run when
every domain is ready.

    XTAIL_ROOT=/mnt/data/greenhorn/xtail python scripts/xtail_check.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import data as D   # noqa: E402

ready = True
for name, n_cls, kind, cands, split_file, template in D.XTAIL_DOMAINS:
    t0 = time.time()
    try:
        d = D._xtail_dir(cands)
        if kind == "coop":
            train, test, names = D._xtail_coop(d, split_file, n_cls)
        elif kind == "aircraft":
            train, test, names = D._xtail_aircraft(d)
        else:
            train, test, names = D._xtail_mnist(d)
        missing = [p for p in (train[0][0], train[0][-1], test[0][0], test[0][-1]) if not os.path.isfile(p)]
        per_class = [train[1].count(c) for c in range(n_cls)]
        print(f"{'OK  ' if not missing else 'MISS'} {name:15s} dir={d} train={len(train[0])} test={len(test[0])} "
              f"classes={len(names)}/{n_cls} train/class min={min(per_class)} "
              f"{'missing ' + missing[0] if missing else ''} ({time.time() - t0:.1f}s)", flush=True)
        ready = ready and not missing
    except Exception as e:                      # noqa: BLE001
        print(f"FAIL {name:15s} {type(e).__name__}: {str(e)[:160]}", flush=True)
        ready = False
if ready:
    t0 = time.time()
    x = D.xtail()
    x.download_data()
    print(f"FULL xtail: train={len(x.train_data)} test={len(x.test_data)} classes={len(x.classnames)} "
          f"max label={int(x.train_targets.max())} ({time.time() - t0:.1f}s)")
else:
    print("NOT READY: some domains are missing")
