"""Wall-clock benchmark of the Stage-2 drift solve and of the covariance transport
(rebuttal E3, concern A6: "the ridge solve is a d x d system; what does it cost at
larger d?").

Production (main.fit_drift_map) solves the identity-anchored ridge on CPU in
float64: A = F_old^T F_old + lambda I_d, Delta = solve(A, Cd^T)^T, W = I + Delta,
a (d, d) system with d right-hand sides; driftsolve.ridge_identity_dense is that
code verbatim and driftsolve.ridge_identity_woodbury the push-through form
Delta = (F_new - F_old)^T (F_old F_old^T + lambda I_n)^-1 F_old, an (n, n) system
that is cheaper when n < d (train_drift_solver='woodbury').  The transport of the
class statistics (main.transport_prototypes) is K times W S W^T on (d, d)
covariances -- O(K d^3), the real large-d cost -- or, in the E5 diagonal mode,
one (K, d) @ (d, d) product (covstats.transport_diag).

Grid: d in {768, 1024, 4096} x n in {480, 600, 830, 1000, 2400, 5000} (the per-task
feature counts: ImageNet-R 50x4 ~480, CUB ~600, Cars ~830, CIFAR 50x2 1000,
ImageNet-R 2400, CIFAR 5000) for the solve, and d x K in {100, 200} for the
transport.  Synthetic float64 features F_old = randn(n, d) (LayerNorm scale),
F_new = F_old (I + 0.05 G / sqrt(d))^T + 0.01 noise, lambda = 100 (the headline
value); every solver runs on the same data.  Timing: median of --repeats
synchronous perf_counter measurements after one warm-up call; the solve rows
also time the Gram product and the linear solve alone (breakdown, same
arithmetic).  Woodbury rows record max|W_woodbury - W_dense| (expected < 1e-8).
Both solvers work in float64 whatever the input dtype (driftsolve casts), so
the dtype column is always float64; --device cuda times the GPU path.  A full
transport whose projected time (K x one call) exceeds --max-transport-seconds is
not run K times; its total is the projection and the note says so.

Output: --out (default figures_data/e3_drift_solve_bench.csv; the directory is
created; never /tmp) with the columns
    kind (solve|transport), d, n, K, solver (dense|woodbury|full|diag), device,
    dtype, gram_s, solve_s, total_s, flops_est, gflops, maxdiff_vs_dense,
    repeats, note, threads, torch_version, cpu
and a Markdown table for RESULTS.md.  Run it on the idle server CPU (thread
count and load change the numbers; torch.get_num_threads() is recorded).

Usage
    python bench_drift_solve.py --dry-run                       # prints the grid, no torch
    python bench_drift_solve.py                                 # ~15 min on the server CPU
    python bench_drift_solve.py --quick --out <scratch>/bench.csv   # seconds, smoke test
    python bench_drift_solve.py --d 768 --n 2400 5000 --K 200 --repeats 5
    python bench_drift_solve.py --device cuda --out figures_data/e3_drift_solve_bench_cuda.csv
"""
import argparse
import csv
import os
import platform
import statistics
import sys
import time

DEFAULT_D = (768, 1024, 4096)
DEFAULT_N = (480, 600, 830, 1000, 2400, 5000)
DEFAULT_K = (100, 200)
LAMBDA = 100.0                      # headline train_drift_lambda (CIFAR, ImageNet-R, Cars)
COLUMNS = ("kind", "d", "n", "K", "solver", "device", "dtype", "gram_s", "solve_s", "total_s",
           "flops_est", "gflops", "maxdiff_vs_dense", "repeats", "note", "threads", "torch_version", "cpu")


def cpu_name():
    """Human-readable CPU model (platform.processor is empty on many Linux boxes)."""
    name = platform.processor() or ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return name or platform.machine()


def _timeit(fn, repeats, sync=None):
    """Median wall-clock seconds of `fn()` over `repeats` runs after one warm-up."""
    fn()
    if sync:
        sync()
    times = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def make_features(torch, n, d, device, seed=0):
    """Synthetic paired features (float64): F_old ~ N(0, I), F_new = F_old W_true^T + noise."""
    gen = torch.Generator().manual_seed(seed)
    F_old = torch.randn(n, d, generator=gen, dtype=torch.float64)
    W_true = torch.eye(d, dtype=torch.float64) + 0.05 * torch.randn(d, d, generator=gen, dtype=torch.float64) / d ** 0.5
    F_new = F_old @ W_true.T + 0.01 * torch.randn(n, d, generator=gen, dtype=torch.float64)
    return F_old.to(device), F_new.to(device)


def bench_solve(torch, driftsolve, d, n, repeats, ridge, device, sync, meta):
    """Two rows (dense, woodbury) for one (d, n): total time of driftsolve.solve_drift,
    the Gram and the linear-solve breakdown, FLOP estimate and the solver difference."""
    F_old, F_new = make_features(torch, n, d, device)
    lam = LAMBDA
    Y = (F_new - F_old) if ridge == "identity" else F_new
    I_d = torch.eye(d, dtype=torch.float64, device=device)
    I_n = torch.eye(n, dtype=torch.float64, device=device)
    rows, W_dense = [], None
    for solver in ("dense", "woodbury"):
        total = _timeit(lambda: driftsolve.solve_drift(F_old, F_new, lam, ridge=ridge, solver=solver),
                        repeats, sync)
        if solver == "dense":
            gram = _timeit(lambda: F_old.T @ F_old, repeats, sync)
            A = F_old.T @ F_old + lam * I_d
            rhs = (Y.T @ F_old).T
            solve = _timeit(lambda: torch.linalg.solve(A, rhs), repeats, sync)
            flops = driftsolve.flops_dense(n, d)
        else:
            gram = _timeit(lambda: F_old @ F_old.T, repeats, sync)
            G = F_old @ F_old.T + lam * I_n
            solve = _timeit(lambda: torch.linalg.solve(G, F_old), repeats, sync)
            flops = driftsolve.flops_woodbury(n, d)
        W = driftsolve.solve_drift(F_old, F_new, lam, ridge=ridge, solver=solver)
        if solver == "dense":
            W_dense, maxdiff = W, 0.0
        else:
            maxdiff = float((W - W_dense).abs().max())
        rows.append(dict(kind="solve", d=d, n=n, K="", solver=solver, device=device, dtype="float64",
                         gram_s=f"{gram:.6f}", solve_s=f"{solve:.6f}", total_s=f"{total:.6f}",
                         flops_est=f"{flops:.4g}", gflops=f"{flops / total / 1e9:.2f}",
                         maxdiff_vs_dense=f"{maxdiff:.3e}", repeats=repeats, note="", **meta))
        print(f"  solve d={d:<5d} n={n:<5d} {solver:<9s} total {total * 1e3:9.2f} ms "
              f"(gram {gram * 1e3:8.2f} ms, solve {solve * 1e3:8.2f} ms) maxdiff {maxdiff:.2e}")
    return rows


def bench_transport(torch, covstats, driftsolve, d, K, repeats, device, sync, max_seconds, meta):
    """Two rows (full, diag) for one (d, K): K sequential W S W^T transports of a
    (d, d) float32 covariance (the per-class loop of transport_prototypes) vs one
    batched diagonal transport of (K, d) variances (the E5 diagonal mode)."""
    gen = torch.Generator().manual_seed(1)
    W = (torch.eye(d) + 0.05 * torch.randn(d, d, generator=gen) / d ** 0.5).to(device)
    A = torch.randn(d, d, generator=gen)
    S = ((A @ A.T) / d + torch.eye(d)).to(device)              # SPD float32 like _class_covs
    v = S.diagonal().clone().unsqueeze(0).repeat(K, 1)         # (K, d) variances
    rows = []
    # full: time one call, project K calls, run them only within the cap
    t_one = _timeit(lambda: covstats.transport_full(S, W), repeats, sync)
    projected = K * t_one
    if projected > max_seconds:
        total_full, note = projected, f"projected from one call ({t_one:.4f} s x K={K}); cap {max_seconds:g} s"
    else:
        # one timed pass of the K-loop (after _timeit's warm-up pass): the single
        # call above already shows the spread, and a pass may take up to the cap
        total_full = _timeit(lambda: [covstats.transport_full(S, W) for _ in range(K)], 1, sync)
        note = ""
    flops_full = driftsolve.flops_transport_full(K, d)
    rows.append(dict(kind="transport", d=d, n="", K=K, solver="full", device=device, dtype="float64",
                     gram_s="", solve_s="", total_s=f"{total_full:.6f}", flops_est=f"{flops_full:.4g}",
                     gflops=f"{flops_full / total_full / 1e9:.2f}", maxdiff_vs_dense="", repeats=repeats,
                     note=note, **meta))
    total_diag = _timeit(lambda: covstats.transport_diag(v, W, jitter=1e-4), repeats, sync)
    flops_diag = driftsolve.flops_transport_diag(K, d)
    rows.append(dict(kind="transport", d=d, n="", K=K, solver="diag", device=device, dtype="float64",
                     gram_s="", solve_s="", total_s=f"{total_diag:.6f}", flops_est=f"{flops_diag:.4g}",
                     gflops=f"{flops_diag / total_diag / 1e9:.2f}", maxdiff_vs_dense="", repeats=repeats,
                     note="", **meta))
    print(f"  transport d={d:<5d} K={K:<4d} full {total_full:9.3f} s{' (projected)' if note else ''}"
          f"  diag {total_diag * 1e3:9.2f} ms  ratio {total_full / total_diag:.0f}x")
    return rows


def write_csv(rows, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out}: {len(rows)} rows")


def print_markdown(rows):
    """Markdown tables for RESULTS.md: solve (per d, rows n) and transport (per d, rows K)."""
    solve = [r for r in rows if r["kind"] == "solve"]
    trans = [r for r in rows if r["kind"] == "transport"]
    for d in sorted({r["d"] for r in solve}):
        print(f"\nDrift solve, d = {d} (float64, lambda = {LAMBDA:g}; seconds, median of repeats)")
        print("| n | dense | woodbury | woodbury / dense | max abs diff | dense GFLOP (est.) |")
        print("|---|---|---|---|---|---|")
        for n in sorted({r["n"] for r in solve if r["d"] == d}):
            dense = next(r for r in solve if r["d"] == d and r["n"] == n and r["solver"] == "dense")
            wood = next(r for r in solve if r["d"] == d and r["n"] == n and r["solver"] == "woodbury")
            td, tw = float(dense["total_s"]), float(wood["total_s"])
            print(f"| {n} | {td:.4f} | {tw:.4f} | {tw / td:.2f} | {wood['maxdiff_vs_dense']} | "
                  f"{float(dense['flops_est']) / 1e9:.2f} |")
    if trans:
        print("\nCovariance transport per task (seconds; full = K x W S W^T on (d, d), diag = (K, d) @ (W*W)^T)")
        print("| d | K | full | diag | full / diag | note |")
        print("|---|---|---|---|---|---|")
        for d in sorted({r["d"] for r in trans}):
            for K in sorted({r["K"] for r in trans if r["d"] == d}):
                full = next(r for r in trans if r["d"] == d and r["K"] == K and r["solver"] == "full")
                diag = next(r for r in trans if r["d"] == d and r["K"] == K and r["solver"] == "diag")
                tf, tdg = float(full["total_s"]), float(diag["total_s"])
                print(f"| {d} | {K} | {tf:.3f} | {tdg:.4f} | {tf / tdg:.0f} | {full['note']} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", nargs="+", type=int, default=list(DEFAULT_D), help="feature dimensions")
    ap.add_argument("--n", nargs="+", type=int, default=list(DEFAULT_N), help="paired-feature counts")
    ap.add_argument("--K", nargs="+", type=int, default=list(DEFAULT_K), help="class counts of the transport")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ridge", choices=("identity", "zero"), default="identity")
    ap.add_argument("--device", default="cpu", help="cpu (production) or cuda")
    ap.add_argument("--max-transport-seconds", type=float, default=120.0,
                    help="a full transport projected beyond this is recorded as the projection")
    ap.add_argument("--no-transport", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="tiny grid (d 64 128, n 32 256, K 4, 1 repeat); writes *_quick.csv unless --out is given")
    ap.add_argument("--out", default=None,
                    help="output CSV (default figures_data/e3_drift_solve_bench.csv, or ..._quick.csv with --quick)")
    ap.add_argument("--dry-run", action="store_true", help="print the grid; imports no torch")
    args = ap.parse_args(argv)
    if args.quick:
        args.d, args.n, args.K, args.repeats = [64, 128], [32, 256], [4], 1
    if args.out is None:
        args.out = os.path.join("figures_data", "e3_drift_solve_bench" + ("_quick" if args.quick else "") + ".csv")

    n_solve = 2 * len(args.d) * len(args.n)
    n_trans = 0 if args.no_transport else 2 * len(args.d) * len(args.K)
    print(f"grid: d {args.d} x n {args.n} ({n_solve} solve rows), K {args.K} ({n_trans} transport rows), "
          f"repeats {args.repeats}, ridge {args.ridge}, device {args.device} -> {args.out}")
    if args.dry_run:
        return 0

    import torch                                  # lazy: --dry-run needs no torch
    import covstats
    import driftsolve
    sync = (lambda: torch.cuda.synchronize()) if args.device.startswith("cuda") else None
    meta = dict(threads=torch.get_num_threads(), torch_version=torch.__version__, cpu=cpu_name())
    print(f"torch {meta['torch_version']}, {meta['threads']} threads, {meta['cpu']}")

    rows = []
    for d in args.d:
        for n in args.n:
            rows += bench_solve(torch, driftsolve, d, n, args.repeats, args.ridge, args.device, sync, meta)
        if not args.no_transport:
            for K in args.K:
                rows += bench_transport(torch, covstats, driftsolve, d, K, args.repeats, args.device, sync,
                                        args.max_transport_seconds, meta)
    write_csv(rows, args.out)
    print_markdown(rows)
    worst = max((float(r["maxdiff_vs_dense"]) for r in rows if r["kind"] == "solve" and r["solver"] == "woodbury"),
                default=0.0)
    print(f"\nmax |W_woodbury - W_dense| over the grid: {worst:.3e} (expected < 1e-8 in float64)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
