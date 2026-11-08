"""SVD cost of the spectral merge per LoRA rank (rebuttal E11), independent of
training and of helper.py (imports merge_ops only, so a CPU torch wheel suffices).

For r in RANKS and the three adapted modules of ViT-B/16
    qkv: in 768,  out 2304     fc1: in 768,  out 3072     fc2: in 3072, out 768
it times, on CPU and (when available) CUDA with synchronisation,
  variant 'only_A'          merge_ops.truncate_energy(dA, p) on a random fp32 [r, in]
                            matrix -- the SVD the headline merge performs per layer
                            (merge_ops.truncate_energy is the verbatim copy of
                            helper.truncate_energy, so the LAPACK path is the same)
  variant 'combine'         torch.linalg.svd of a dense random [out, in] update
                            (the 'combine' variant's full-size SVD, for reference)
  variant 'combine_lowrank' the same on a rank-2r structured update
                            B1 A1 - B0 A0 (the actual effective update's shape)
and reports the median over --repeats runs after --warmup warm-ups. Per-merge
cost = 12 blocks x (qkv + fc1 + fc2) per-layer medians.

Output: figures_data/e11_svd_time.csv with columns
    r, device, variant, module, in_dim, out_dim, ms_per_layer, ms_per_merge, n_repeats
('ms_per_merge' repeats the per-merge total on each module row of the same
(r, device, variant) so the CSV joins on any row).

Run:  python bench_svd.py                 # ranks 8 16 32 64 128, CPU + CUDA
      python bench_svd.py --quick         # fewer repeats (under a minute on a laptop CPU)
      python bench_svd.py --ranks 64 --devices cpu --threads 8
"""
import argparse
import csv
import os
import statistics
import sys
import time

RANKS = (8, 16, 32, 64, 128)
MODULES = (("qkv", 768, 2304), ("fc1", 768, 3072), ("fc2", 3072, 768))
BLOCKS = 12
VARIANTS = ("only_A", "combine", "combine_lowrank")
COLUMNS = ["r", "device", "variant", "module", "in_dim", "out_dim",
           "ms_per_layer", "ms_per_merge", "n_repeats"]


def _sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def time_fn(fn, device, repeats, warmup):
    """Median wall-clock (ms) of fn() over `repeats` timed calls."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def bench_variant(variant, r, in_dim, out_dim, device, p, repeats, warmup, gen):
    """Per-layer median ms of one (variant, r, module) on `device`."""
    import torch
    import merge_ops as mo
    if variant == "only_A":
        dA = torch.randn(r, in_dim, generator=gen).to(device)
        return time_fn(lambda: mo.truncate_energy(dA, p), device, repeats, warmup)
    if variant == "combine":
        D = torch.randn(out_dim, in_dim, generator=gen).to(device)
        return time_fn(lambda: torch.linalg.svd(D, full_matrices=False), device, repeats, warmup)
    if variant == "combine_lowrank":
        A0 = torch.randn(r, in_dim, generator=gen)
        B0 = torch.randn(out_dim, r, generator=gen)
        A1 = A0 + 0.1 * torch.randn(r, in_dim, generator=gen)
        B1 = B0 + 0.1 * torch.randn(out_dim, r, generator=gen)
        D = (B1 @ A1 - B0 @ A0).to(device)
        return time_fn(lambda: torch.linalg.svd(D, full_matrices=False), device, repeats, warmup)
    raise ValueError(variant)


def run_bench(ranks, devices, variants, p=0.9, repeats=20, warmup=3, seed=0, blocks=BLOCKS):
    """List of CSV rows (dicts) over ranks x devices x variants x modules."""
    import torch
    rows = []
    for device in devices:
        for r in ranks:
            for variant in variants:
                gen = torch.Generator().manual_seed(seed)
                per_layer = {}
                for module, in_dim, out_dim in MODULES:
                    per_layer[module] = bench_variant(variant, int(r), in_dim, out_dim, device, p,
                                                      repeats, warmup, gen)
                per_merge = blocks * sum(per_layer.values())
                for module, in_dim, out_dim in MODULES:
                    rows.append({"r": int(r), "device": device, "variant": variant,
                                 "module": module, "in_dim": in_dim, "out_dim": out_dim,
                                 "ms_per_layer": f"{per_layer[module]:.4f}",
                                 "ms_per_merge": f"{per_merge:.3f}", "n_repeats": repeats})
                print(f"[bench_svd] {device:<5} r={int(r):<4} {variant:<16} "
                      + " ".join(f"{m}={per_layer[m]:.3f}ms" for m, _, _ in MODULES)
                      + f"  per merge {per_merge:.2f} ms")
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", nargs="+", type=int, default=list(RANKS))
    ap.add_argument("--devices", nargs="+", default=None,
                    help="default: cpu plus cuda when torch.cuda.is_available()")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--p", type=float, default=0.9)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="repeats 5, warm-up 1")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads for the CPU rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("figures_data", "e11_svd_time.csv"))
    args = ap.parse_args(argv)
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    repeats, warmup = (5, 1) if args.quick else (args.repeats, args.warmup)
    print(f"[bench_svd] torch {torch.__version__}, devices {devices}, threads {torch.get_num_threads()}, "
          f"repeats {repeats}, warm-up {warmup}, p {args.p}")
    rows = run_bench(args.ranks, devices, args.variants, args.p, repeats, warmup, args.seed)
    write_csv(args.out, rows)
    print(f"[bench_svd] wrote {args.out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
