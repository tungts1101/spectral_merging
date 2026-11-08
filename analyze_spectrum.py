"""Spectrum of the merge update, replayed from the cached LoRA chain (rebuttal E4;
reused by E11 with --prefix/--tag and by the E7 offline calibration).

The merged checkpoints M_t are deleted by run_config_sweep's cleanup, so this
script rebuilds them on the CPU from the cached per-task states
    {ckpt_dir}/{seed}_{ds}_vit_base_patch16_224_lora_{train_prefix}_backbone_{t}.pt
(Learner.backbone_checkpoint) with M_0 = W_0 and
    M_t = helper.pspectral_merging(M_{t-1}, W_t, p, alpha, variant)
i.e. the call Learner.merge makes, so the replay equals the in-run merged
state up to CPU/GPU round-off (--verify-merged checks that against a surviving
_merged_*.pt once). For every task t >= 1 and every LoRA factor pair (36 for
qkv/fc1/fc2 x 12 blocks) it records two spectra:

  space 'A'   : dA = A_t - A_{M,t-1}  ([r, in]) -- the matrix helper.truncate_energy
                truncates in the only_A variant; the kept rank k(p) uses the
                exact rule k = #(cumsum(S^2)/sum(S^2) < p) + 1 clamped to [1, r]
                (merge_ops.energy_rank, same LAPACK path as truncate_energy).
  space 'eff' : scale * (B_t A_t - B_M A_M)  ([out, in], rank <= 2r) with
                scale = model_lora_alpha / model_lora_r (2 for the headline recipe);
                the effective-weight update the paper's Eq. 2 is stated on.

Outputs (figures_data/, utf-8):
  e4_spectrum_{ds}{tag}.csv : seed, task, block, module, space, n_sv, sigma1, fro,
                              k_p0.5, k_p0.7, k_p0.9, k_p0.95, k_p0.99,
                              energy_top1, energy_top8      (one row per pair/space)
  e4_singvals_{ds}{tag}.csv : seed, task, block, module, space, idx, sigma, sigma_norm
A degenerate pair (sigma_1 == 0, truncate_energy returns A.clone()) keeps n_sv and
leaves the k columns empty. The script prints the per-task mean sigma_1 of both
spaces (the E7 'specnorm' statistic) and the kept-k table per module.

Run (server, CPU, about 1 min per seed):
    python analyze_spectrum.py --ds imagenetr
    python analyze_spectrum.py --ds cifar224 --prefix-family best
    python analyze_spectrum.py --ds imagenetr --prefix main_inr_r8_qkvfc --tag _r8 \
        --ckpt-dir checkpoints_main/rebuttal/e11_rank8          # E11 chains
    python analyze_spectrum.py --ds imagenetr --seeds 1993 --verify-merged
Needs torch + helper.py (timm, peft); the CSV helpers import without torch.
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import rebuttal_common as rc

DEFAULT_PS = (0.5, 0.7, 0.9, 0.95, 0.99)
SPACES = ("A", "eff")
MODULE_ORDER = ("qkv", "fc1", "fc2")


def k_col(p):
    """CSV column of the kept rank at energy fraction p (0.9 -> 'k_p0.9')."""
    return f"k_p{float(p):g}"


def spectrum_columns(ps=DEFAULT_PS):
    return (["seed", "task", "block", "module", "space", "n_sv", "sigma1", "fro"]
            + [k_col(p) for p in ps] + ["energy_top1", "energy_top8"])


SINGVAL_COLUMNS = ["seed", "task", "block", "module", "space", "idx", "sigma", "sigma_norm"]

_BACKBONE_RE = re.compile(r"_backbone_(\d+)\.pt$")


# ---------------------------------------------------------------------------
# Cached chain
# ---------------------------------------------------------------------------

def chain_files(ckpt_dir, stem):
    """Sorted [(t, path)] of the contiguous task states t = 0, 1, ... of the chain
    `stem` (= rebuttal_common.chain_stem) under ckpt_dir; symlinks resolve."""
    found = {}
    for path in glob.glob(os.path.join(ckpt_dir, stem + "_backbone_*.pt")):
        m = _BACKBONE_RE.search(os.path.basename(path))
        if m and os.path.exists(path):
            found[int(m.group(1))] = path
    out = []
    t = 0
    while t in found:
        out.append((t, found[t]))
        t += 1
    return out


def load_state(path):
    """CPU state dict of plain tensors (the cache holds nn.Parameter objects)."""
    import torch
    try:
        state = torch.load(path, map_location="cpu")
    except Exception:  # torch >= 2.6 weights_only default on an older pickle
        state = torch.load(path, map_location="cpu", weights_only=False)
    return {k: (v.detach() if hasattr(v, "detach") else v) for k, v in state.items()}


def replay_chain(ckpt_dir, stem, p, alpha, variant="only_A", update_op="trunc",
                 max_tasks=None):
    """Yield (t, W_t, M_{t-1}, M_t) for t >= 1 along the cached chain.

    M_0 = W_0; M_t = helper.pspectral_merging(M_{t-1}, W_t, p=p, alpha=alpha,
    variant=variant) -- the default-keyword call of Learner.merge (update_op is
    passed only when it differs from 'trunc', so the default replay goes through
    exactly the regression-tested path).
    """
    import helper  # torch, timm, peft (server)
    files = chain_files(ckpt_dir, stem)
    if not files:
        raise FileNotFoundError(f"no {stem}_backbone_<t>.pt under {ckpt_dir}")
    if max_tasks is not None:
        files = files[: int(max_tasks)]
    M = load_state(files[0][1])
    for t, path in files[1:]:
        W = load_state(path)
        kw = {} if update_op == "trunc" else {"update_op": update_op}
        M_new = helper.pspectral_merging(M, W, p=p, alpha=alpha, variant=variant, **kw)
        yield t, W, M, M_new
        M = M_new


def infer_rank(state):
    """LoRA rank r from the first lora_A factor ([r, in])."""
    import merge_ops as mo
    pairs = mo.lora_pairs(state)
    if not pairs:
        raise ValueError("state holds no LoRA factor pair")
    return int(state[pairs[0][0]].shape[0])


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def _fmt(x):
    return f"{x:.6g}"


def pair_rows(seed, task, w_prev, w_new, scale, ps=DEFAULT_PS, with_singvals=True):
    """Spectrum rows (and singular-value rows) of every factor pair for one task.

    w_prev = M_{t-1}, w_new = W_t. Space 'A' uses merge_ops.energy_rank (the
    truncate_energy SVD path, bit-identical k); space 'eff' uses svdvals of the
    effective update and the same rank rule.
    """
    import torch
    import merge_ops as mo
    rows, svals = [], []
    for an, bn in mo.lora_pairs(w_new):
        block, module = mo.pair_label(an)
        dA = w_new[an].float() - w_prev[an].float()
        eff = mo.effective_update(w_prev, w_new, an, bn, scale)
        k0, S_A, _ = mo.energy_rank(dA, ps[0])
        S_e = torch.linalg.svdvals(eff)
        for space, S, d in (("A", S_A, dA), ("eff", S_e, eff)):
            n_sv = int(S.numel())
            degenerate = n_sv == 0 or float(S[0]) == 0.0
            row = {"seed": seed, "task": task, "block": block, "module": module,
                   "space": space, "n_sv": n_sv,
                   "sigma1": _fmt(float(S[0])) if n_sv else "0",
                   "fro": _fmt(float(d.norm()))}
            for p in ps:
                k = 0 if degenerate else mo.rank_from_svals(S, p)
                row[k_col(p)] = "" if k == 0 else k
            row["energy_top1"] = _fmt(mo.energy_share(S, 1))
            row["energy_top8"] = _fmt(mo.energy_share(S, 8))
            rows.append(row)
            if with_singvals and n_sv and not degenerate:
                s1 = float(S[0])
                for i, s in enumerate(S.tolist()):
                    svals.append({"seed": seed, "task": task, "block": block, "module": module,
                                  "space": space, "idx": i + 1, "sigma": _fmt(s),
                                  "sigma_norm": _fmt(s / s1)})
    return rows, svals


def verify_merged(ckpt_dir, stem, tag, task, M_t):
    """Max |replay - file| over the keys of a surviving merged checkpoint
    {stem}_merged_{tag}_{task}.pt, or None when the file does not exist."""
    path = os.path.join(ckpt_dir, f"{stem}_merged_{tag}_{task}.pt")
    if not os.path.exists(path):
        return None
    ref = load_state(path)
    worst = 0.0
    for k, v in M_t.items():
        if k not in ref:
            continue
        worst = max(worst, float((v.float() - ref[k].float()).abs().max()))
    return worst


# ---------------------------------------------------------------------------
# CSV + summary (torch-free)
# ---------------------------------------------------------------------------

def write_csv(path, rows, cols):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_spectrum_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows, p=0.9):
    """Summary dict of e4_spectrum rows (strings or numbers):
      sigma1_by_task[space][task] = mean over seeds and pairs of sigma_1
      (the 'specnorm' statistic s_t for stat 'eff' / 'A', averaged over seeds),
      k_by_module[space][module] = (mean, min, max) of k(p) over seeds/tasks/pairs,
      energy[space] = (mean energy_top1, mean energy_top8), n_rows, seeds, tasks.
    """
    kc = k_col(p)
    out = {"sigma1_by_task": {}, "k_by_module": {}, "energy": {}, "n_rows": len(rows),
           "seeds": sorted({int(r["seed"]) for r in rows}),
           "tasks": sorted({int(r["task"]) for r in rows})}
    for space in SPACES:
        sub = [r for r in rows if r["space"] == space]
        by_task = {}
        for r in sub:
            by_task.setdefault(int(r["task"]), []).append(float(r["sigma1"]))
        out["sigma1_by_task"][space] = {t: _mean(v) for t, v in sorted(by_task.items())}
        by_mod = {}
        for r in sub:
            if r.get(kc, "") not in ("", None):
                by_mod.setdefault(r["module"], []).append(int(float(r[kc])))
        out["k_by_module"][space] = {m: (_mean(v), min(v), max(v)) for m, v in by_mod.items()}
        out["energy"][space] = (_mean(float(r["energy_top1"]) for r in sub),
                                _mean(float(r["energy_top8"]) for r in sub))
    return out


def print_summary(summary, p=0.9, label=""):
    tasks = summary["tasks"]
    span = f"{tasks[0]}..{tasks[-1]}" if tasks else "-"
    print(f"\n== spectrum summary {label} (p = {p:g}; seeds {summary['seeds']}, "
          f"tasks {span}, {summary['n_rows']} rows) ==")
    print("| task | mean sigma1 (A) | mean sigma1 (eff) |")
    print("|---:|---:|---:|")
    tasks = sorted(set(summary["sigma1_by_task"]["A"]) | set(summary["sigma1_by_task"]["eff"]))
    for t in tasks:
        a = summary["sigma1_by_task"]["A"].get(t, float("nan"))
        e = summary["sigma1_by_task"]["eff"].get(t, float("nan"))
        print(f"| {t} | {a:.5g} | {e:.5g} |")
    print(f"\n| space | module | mean k(p={p:g}) | min | max | energy top-1 | energy top-8 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for space in SPACES:
        e1, e8 = summary["energy"][space]
        for m in sorted(summary["k_by_module"][space], key=lambda x: (MODULE_ORDER.index(x)
                                                                      if x in MODULE_ORDER else 99, x)):
            mean, lo, hi = summary["k_by_module"][space][m]
            print(f"| {space} | {m} | {mean:.2f} | {lo} | {hi} | {e1:.3f} | {e8:.3f} |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", required=True, help="dataset (imagenetr, cifar224, cub, cars)")
    ap.add_argument("--prefix", default=None,
                    help="train_prefix of the chain (default: rebuttal_common.prefix(ds))")
    ap.add_argument("--prefix-family", choices=rc.PREFIX_FAMILIES, default="main")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(rc.SEEDS))
    ap.add_argument("--ckpt-dir", default=rc.SEQFT_CKPT_DIR,
                    help="directory of the cached chain (default checkpoints_main/seqft)")
    ap.add_argument("--model-backbone", default=rc.MODEL_BACKBONE)
    ap.add_argument("--alpha", type=float, default=None,
                    help="damping alpha of the replayed merge (default: headline alpha of ds)")
    ap.add_argument("--p", type=float, default=None,
                    help="energy fraction of the replayed merge (default: headline p = 0.9)")
    ap.add_argument("--variant", default="only_A")
    ap.add_argument("--update-op", default="trunc", choices=("trunc", "clip", "scale_fro"))
    ap.add_argument("--ps", default=",".join(f"{p:g}" for p in DEFAULT_PS),
                    help="comma-separated energy fractions of the kept-rank columns")
    ap.add_argument("--lora-alpha", type=float, default=None,
                    help="model_lora_alpha for the effective scale (default 2 * r, i.e. scale 2)")
    ap.add_argument("--scale", type=float, default=None,
                    help="effective-update scale alpha/r (overrides --lora-alpha)")
    ap.add_argument("--max-tasks", type=int, default=None, help="replay only the first N states")
    ap.add_argument("--tag", default="", help="suffix of the CSV names, e.g. _r8 (E11)")
    ap.add_argument("--out-dir", default="figures_data")
    ap.add_argument("--no-singvals", action="store_true", help="skip the e4_singvals CSV")
    ap.add_argument("--verify-merged", action="store_true",
                    help="compare the replay with surviving _merged_{tag}_{t}.pt files")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    rc.set_prefix_family(args.prefix_family)
    if args.ds not in rc.POSTURE:
        raise SystemExit(f"unknown dataset {args.ds!r}; known {sorted(rc.POSTURE)}")
    prefix = args.prefix or rc.prefix(args.ds)
    alpha = rc.HEADLINE_ALPHA[args.ds] if args.alpha is None else args.alpha
    p = rc.POSTURE[args.ds]["pspectral_p"] if args.p is None else args.p
    ps = tuple(float(x) for x in args.ps.split(",") if x.strip())
    cols = spectrum_columns(ps)
    merged_tag = rc.merge_tag(rc.cfg(args.ds, train_merge_alpha=alpha, pspectral_p=p,
                                     spectral_variant=args.variant, merge_update_op=args.update_op))
    print(f"[E4] ds={args.ds} prefix={prefix} ckpt_dir={args.ckpt_dir} seeds={args.seeds} "
          f"p={p:g} alpha={alpha:g} variant={args.variant} op={args.update_op} tag={merged_tag}")

    all_rows, all_svals = [], []
    for seed in args.seeds:
        stem = rc.chain_stem(seed, args.ds, prefix, args.model_backbone)
        files = chain_files(args.ckpt_dir, stem)
        if len(files) < 2:
            print(f"[E4] seed {seed}: {len(files)} cached state(s) for {stem} -- skipped")
            continue
        t0 = time.time()
        r = infer_rank(load_state(files[0][1]))
        scale = args.scale if args.scale is not None else \
            (args.lora_alpha if args.lora_alpha is not None else 2.0 * r) / r
        print(f"[E4] seed {seed}: {len(files)} states, r = {r}, scale = {scale:g}")
        for t, W, M_prev, M_t in replay_chain(args.ckpt_dir, stem, p, alpha, args.variant,
                                              args.update_op, args.max_tasks):
            rows, svals = pair_rows(seed, t, M_prev, W, scale, ps, not args.no_singvals)
            all_rows.extend(rows)
            all_svals.extend(svals)
            if args.verify_merged:
                diff = verify_merged(args.ckpt_dir, stem, merged_tag, t, M_t)
                print(f"[E4]   task {t}: merged file " + ("absent" if diff is None else
                                                          f"max |replay - file| = {diff:.3e}"))
        print(f"[E4] seed {seed}: done in {time.time() - t0:.1f}s ({len(all_rows)} rows so far)")

    if not all_rows:
        raise SystemExit("[E4] nothing replayed: check --ckpt-dir / --prefix / --seeds "
                         "(python check_cache.py lists the cached chains)")
    out_spec = os.path.join(args.out_dir, f"e4_spectrum_{args.ds}{args.tag}.csv")
    write_csv(out_spec, all_rows, cols)
    print(f"[E4] wrote {out_spec}: {len(all_rows)} rows")
    if not args.no_singvals:
        out_sv = os.path.join(args.out_dir, f"e4_singvals_{args.ds}{args.tag}.csv")
        write_csv(out_sv, all_svals, SINGVAL_COLUMNS)
        print(f"[E4] wrote {out_sv}: {len(all_svals)} rows")
    print_summary(summarize(all_rows, p), p, label=f"{args.ds}{args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
