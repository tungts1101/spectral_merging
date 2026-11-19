"""Download the X-TAIL datasets of RAIL (Xu et al., NeurIPS 2024) with torchvision.

Nine of the ten domains (Stanford Cars is already under DATA_ROOT/cars). Each
dataset goes to <root>/<name>/ in torchvision's own layout; a failed download
is reported and the next one starts. CPU and network only.

    python scripts/xtail_download.py /mnt/data/greenhorn/xtail
"""
import sys
import time

import torchvision.datasets as D

root = sys.argv[1].rstrip("/")
JOBS = [
    ("mnist", lambda: [D.MNIST(f"{root}/mnist", train=t, download=True) for t in (True, False)]),
    ("eurosat", lambda: D.EuroSAT(f"{root}/eurosat", download=True)),
    ("dtd", lambda: [D.DTD(f"{root}/dtd", split=s, download=True) for s in ("train", "val", "test")]),
    ("flowers102", lambda: [D.Flowers102(f"{root}/flowers102", split=s, download=True) for s in ("train", "val", "test")]),
    ("oxford_pets", lambda: [D.OxfordIIITPet(f"{root}/oxford_pets", split=s, download=True) for s in ("trainval", "test")]),
    ("fgvc_aircraft", lambda: [D.FGVCAircraft(f"{root}/fgvc_aircraft", split=s, download=True) for s in ("train", "val", "test")]),
    ("caltech101", lambda: D.Caltech101(f"{root}/caltech101", download=True)),
    ("food101", lambda: [D.Food101(f"{root}/food101", split=s, download=True) for s in ("train", "test")]),
    ("sun397", lambda: D.SUN397(f"{root}/sun397", download=True)),
]
for name, fn in JOBS:
    t0 = time.time()
    try:
        out = fn()
        sizes = [len(d) for d in (out if isinstance(out, list) else [out])]
        print(f"OK   {name}: sizes {sizes} in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:                        # noqa: BLE001
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:200]} ({time.time() - t0:.0f}s)", flush=True)
print("DONE", flush=True)
