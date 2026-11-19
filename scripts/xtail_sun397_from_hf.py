"""Rebuild the SUN397 directory tree from the Hugging Face copy 1aurent/SUN397.

Princeton no longer serves SUN397.tar.gz, so the images come from the parquet
copy, whose build script globbed `SUN397/**/*.jpg` of the original archive in
sorted order and stored each file through datasets.Image (bytes plus the file's
basename in the `path` field) with the class directory as the label. This
script writes every image back to <root>/sun397/SUN397/<class dir>/<basename>,
the layout the CoOp split (split_zhou_SUN397.json) expects, then checks that
every path of the split exists. It refuses to guess: if the `path` field is
missing, it stops.

    python scripts/xtail_sun397_from_hf.py /mnt/data/greenhorn/xtail
"""
import glob
import json
import os
import sys
import time

import pyarrow.parquet as pq

root = sys.argv[1].rstrip("/")
src = os.path.join(root, "sun397_hf", "data")
dst = os.path.join(root, "sun397", "SUN397")
split_file = os.path.join(root, "splits", "split_zhou_SUN397.json")

# class names from the README's YAML (names '0'..'396', e.g. '/a/abbey')
readme = open(os.path.join(root, "sun397_hf", "README.md"), encoding="utf-8").read()
names = {}
in_names = False
for line in readme.split("\n"):
    s = line.strip()
    if s == "names:":
        in_names = True
        continue
    if in_names:
        if s.startswith("'") and "': " in s:
            k, v = s.split("': ", 1)
            names[int(k.strip("'"))] = v.strip().strip("'\"")
        elif s and not s.startswith("'"):
            break
assert len(names) == 397, f"parsed {len(names)} class names from README"

files = sorted(glob.glob(os.path.join(src, "train-*.parquet")))
assert len(files) == 93, f"{len(files)} parquet files (expected 93)"
t0 = time.time()
n, n_nopath, written = 0, 0, 0
for f in files:
    tbl = pq.read_table(f, columns=["image", "label"])
    images = tbl.column("image").to_pylist()
    labels = tbl.column("label").to_pylist()
    for im, lab in zip(images, labels):
        n += 1
        path = im.get("path") if isinstance(im, dict) else None
        if not path:
            n_nopath += 1
            if n_nopath == 1:
                print("WARN: a row has no `path` field; cannot name files", flush=True)
            continue
        cls = names[int(lab)].lstrip("/")
        out = os.path.join(dst, cls, os.path.basename(path))
        if not os.path.isfile(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as fh:
                fh.write(im["bytes"])
            written += 1
    print(f"{os.path.basename(f)}: rows so far {n}, written {written}, no-path {n_nopath}, "
          f"{time.time() - t0:.0f}s", flush=True)
print(f"rows {n} written {written} no-path {n_nopath}")
if n_nopath:
    sys.exit("STOP: rows without file names; the positional fallback is not implemented on purpose")

split = json.load(open(split_file))
missing = {k: [p for p, _, _ in split[k] if not os.path.isfile(os.path.join(dst, p))] for k in ("train", "val", "test")}
for k, m in missing.items():
    print(f"split {k}: {len(split[k])} entries, {len(m)} missing" + (f", e.g. {m[:3]}" if m else ""))
print("SUN397_DONE" if not any(missing.values()) else "SUN397_INCOMPLETE")
