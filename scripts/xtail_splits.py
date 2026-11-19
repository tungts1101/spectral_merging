"""Fetch the CoOp split files (split_zhou_*.json) that RAIL's loaders read.

Links from CoOp's DATASETS.md (github.com/KaiyangZhou/CoOp). Files land in
<root>/splits/. gdown is used for the Google Drive ids; a failure is reported
and the next file is tried.

    python scripts/xtail_splits.py /mnt/data/greenhorn/xtail
"""
import os
import sys

import gdown

root = sys.argv[1].rstrip("/")
out = os.path.join(root, "splits")
os.makedirs(out, exist_ok=True)
FILES = {
    "split_zhou_Caltech101.json": "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN",
    "split_zhou_OxfordPets.json": "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs",
    "split_zhou_StanfordCars.json": "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT",
    "split_zhou_OxfordFlowers.json": "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT",
    "cat_to_name.json": "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0",
    "split_zhou_Food101.json": "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl",
    "split_zhou_SUN397.json": "1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq",
    "split_zhou_DescribableTextures.json": "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x",
    "split_zhou_EuroSAT.json": "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o",
}
for name, fid in FILES.items():
    dst = os.path.join(out, name)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"HAVE {name} {os.path.getsize(dst)} bytes", flush=True)
        continue
    try:
        gdown.download(id=fid, output=dst, quiet=True)
        print(f"OK   {name} {os.path.getsize(dst)} bytes", flush=True)
    except Exception as e:                      # noqa: BLE001
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
print("DONE", flush=True)
