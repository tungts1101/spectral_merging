"""What the discarded directions carry: kept vs discarded right-singular
directions of the merge update against the old- and new-task layer-INPUT
subspaces (rebuttal E6). GPU + data, no training.

For every task t >= 1 of a cached chain the script replays M_{t-1}
(analyze_spectrum.replay_chain), loads it into a Model21 built directly from
helper.get_backbone (never a Learner, whose __init__ rewrites the base file),
and collects the uncentred input covariances Sigma = X^T X / n of the 36
adapted layers (backbones.compute_covariances with
backbones.peft_adapted_modules) over
  (i)  a seeded subsample of the OLD classes' training images (classes
       0..known-1, deterministic test transform, as compute_multivariate_normal
       uses), and
  (ii) a seeded subsample of the NEW task's training images,
both under M_{t-1}. With U S Vh = svd(dA), dA = A_t - A_{M,t-1}, and k the
truncate_energy rank at p (merge_ops.energy_rank), V_keep = Vh[:k]^T and
V_drop = Vh[k:]^T live in the layer-input space, so per layer and task:

  energy_share_{keep,drop}  weight energy sum_{i<=k} s_i^2 / sum s_i^2 and its complement
  tr_{old,new}_{full,keep,drop,cross}
                            tr(dA Sigma dA^T), tr(dA_keep Sigma dA_keep^T),
                            tr(dA_drop Sigma dA_drop^T) and the cross term
                            2 tr(dA_keep Sigma dA_drop^T) (full = keep + drop + cross);
                            tr(dA_keep Sigma_old dA_keep^T) / tr(dA Sigma_old dA^T) is the
                            Sigma-weighted deviation ratio of truncation vs the full step
                            at the same alpha (the 'typical inputs' sentence of Sec. 4.5)
  resp_{old,new}_{keep,drop} the response shares tr_*_keep / tr_*_full, tr_*_drop / tr_*_full
  ratio_{old,new}           resp_*_drop / energy_share_drop (> 1: the discarded
                            directions are over-represented on those inputs)
  overlap_{keep,drop}_{old,new}
                            ||Q^T V||_F^2 / dim(V) = mean cos^2 of the principal
                            angles between V and Q, with Q the top-m eigenvectors
                            of Sigma holding `energy` (90 %) of its trace
  m_{old,new}, chance_{old,new} = m / in_dim   the subspace size and the chance level

Output: figures_data/e6_subspace_{ds}{tag}.csv (one row per seed/task/layer) and
a summary table; scripts/fig8_subspace.py draws the per-task curves.

Run (server GPU, about 5 min per seed):
    python analyze_subspace.py --ds imagenetr
    python analyze_subspace.py --ds cifar224 --n-old 64 --n-new 64 --seeds 1993   # data-path smoke
The pure metric function direction_metrics() needs torch only (CPU is fine).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

import rebuttal_common as rc
import analyze_spectrum as spec

COLUMNS = ["seed", "task", "block", "module", "in_dim", "n_sv", "k",
           "energy_share_keep", "energy_share_drop",
           "tr_old_full", "tr_old_keep", "tr_old_drop", "tr_old_cross",
           "tr_new_full", "tr_new_keep", "tr_new_drop", "tr_new_cross",
           "resp_old_keep", "resp_old_drop", "resp_new_keep", "resp_new_drop",
           "ratio_old", "ratio_new", "m_old", "m_new", "chance_old", "chance_new",
           "overlap_keep_old", "overlap_drop_old", "overlap_keep_new", "overlap_drop_new",
           "n_old", "n_new"]

# module name of a factor pair -> key of backbones.peft_adapted_modules
MODULE_KEY = {"qkv": "attn_qkv", "fc1": "mlp_fc1", "fc2": "mlp_fc2"}


# ---------------------------------------------------------------------------
# Pure metric function (torch only)
# ---------------------------------------------------------------------------

def top_energy_subspace(Sigma, energy=0.9):
    """(Q, m) with Q the top-m eigenvectors of the symmetric PSD Sigma holding at
    least `energy` of its trace (smallest such m, clamped to [1, d]); eigh in
    Sigma's dtype, so pass float32 on the GPU for speed."""
    import torch
    evals, evecs = torch.linalg.eigh(Sigma)
    evals = evals.flip(0).clamp(min=0.0)
    evecs = evecs.flip(1)
    tot = float(evals.sum())
    if tot <= 0.0:
        return evecs[:, :1], 1
    cum = torch.cumsum(evals, 0) / tot
    m = int((cum < energy).sum().item()) + 1
    m = max(1, min(m, int(evals.numel())))
    return evecs[:, :m], m


def _trace_quad(X, Sigma):
    """tr(X Sigma X^T) for X [r, d], Sigma [d, d] (float64)."""
    return float(((X @ Sigma) * X).sum())


def _overlap(V, Q):
    """||Q^T V||_F^2 / dim(V): mean squared cosine of the principal angles between
    span(V) and span(Q); NaN for an empty V."""
    if V.shape[1] == 0:
        return float("nan")
    return float((Q.T.double() @ V).pow(2).sum()) / V.shape[1]


def direction_metrics(dA, Sigma_old, Sigma_new, p=0.9, energy=0.9):
    """Metrics of the kept / discarded directions of one factor update dA ([r, in])
    against two layer-input covariances ([in, in]); see the module docstring.

    k follows the exact truncate_energy rule on the float32 SVD
    (merge_ops.energy_rank); the subspaces and traces use a float64 SVD of dA.
    Returns a dict with the COLUMNS entries from 'in_dim' to 'overlap_drop_new'.
    """
    import torch
    import merge_ops as mo
    dA32 = dA.float()
    k, S32, _ = mo.energy_rank(dA32, p)
    d64 = dA.double()
    U, S, Vh = torch.linalg.svd(d64, full_matrices=False)
    n_sv = int(S.numel())
    e = S ** 2
    tot = float(e.sum())
    out = {"in_dim": int(dA.shape[1]), "n_sv": n_sv, "k": k}
    if k == 0 or tot <= 0.0:                      # degenerate update
        for c in COLUMNS[7:-2]:
            out[c] = float("nan")
        return out
    out["energy_share_keep"] = float(e[:k].sum()) / tot
    out["energy_share_drop"] = 1.0 - out["energy_share_keep"]
    dA_keep = (U[:, :k] * S[:k]) @ Vh[:k]
    dA_drop = d64 - dA_keep
    V_keep, V_drop = Vh[:k].T, Vh[k:n_sv].T
    for name, Sigma in (("old", Sigma_old), ("new", Sigma_new)):
        Sd = Sigma.to(d64.device).double()
        full, keep, drop = _trace_quad(d64, Sd), _trace_quad(dA_keep, Sd), _trace_quad(dA_drop, Sd)
        out[f"tr_{name}_full"], out[f"tr_{name}_keep"], out[f"tr_{name}_drop"] = full, keep, drop
        out[f"tr_{name}_cross"] = full - keep - drop
        out[f"resp_{name}_keep"] = keep / full if full > 0 else float("nan")
        out[f"resp_{name}_drop"] = drop / full if full > 0 else float("nan")
        esd = out["energy_share_drop"]
        out[f"ratio_{name}"] = (out[f"resp_{name}_drop"] / esd) if esd > 0 and full > 0 else float("nan")
        Q, m = top_energy_subspace(Sigma.to(d64.device).float(), energy)
        out[f"m_{name}"] = m
        out[f"chance_{name}"] = m / dA.shape[1]
        out[f"overlap_keep_{name}"] = _overlap(V_keep, Q)
        out[f"overlap_drop_{name}"] = _overlap(V_drop, Q)
    return out


# ---------------------------------------------------------------------------
# Data path (server)
# ---------------------------------------------------------------------------

def load_into_model(model, state):
    """Learner.load_backbone (main.py) without a Learner: 'norm.*' -> model.norm,
    the PEFT keys -> model.backbone (strict=False)."""
    peft_params, norm_params = {}, {}
    for name, param in state.items():
        if name.startswith("norm."):
            norm_params[name[5:]] = param
        else:
            peft_params[name] = param
    model.backbone.load_state_dict(peft_params, strict=False)
    if norm_params and getattr(model, "norm", None) is not None:
        model.norm.load_state_dict(norm_params, strict=True)


def build_model(ds, r, lora_alpha, targets=("qkv", "fc1", "fc2"), device="cuda"):
    """Model21(helper.get_backbone(cfg)) on `device`, eval mode, LoRA r / alpha /
    targets of the chain; cfg = BASE_CONFIG + the headline posture (the keys
    get_backbone reads: model_backbone, model_lora_*, model_outdim, model_use_norm)."""
    import main
    import helper
    cfg = dict(main.BASE_CONFIG)
    cfg.update(rc.cfg(ds))
    cfg.update({"model_lora_r": int(r), "model_lora_alpha": float(lora_alpha),
                "model_lora_target_modules": list(targets), "dataset_name": ds})
    model = main.Model21(cfg, helper.get_backbone(cfg))
    return model.to(device).eval()


def subsample_loader(dm, class_range, n_max, rs, batch, workers):
    """DataLoader (shuffle False, private generator) over at most n_max images of
    `class_range` drawn without replacement by the RandomState rs; returns
    (loader, n_images)."""
    import torch
    from torch.utils.data import DataLoader
    from utils.data_manager import DummyDataset
    import helper
    data, targets, dset = dm.get_dataset(np.asarray(class_range), source="train", mode="test",
                                         ret_data=True)
    n = len(targets)
    if n_max and n > n_max:
        keep = np.sort(rs.choice(n, int(n_max), replace=False))
        data, targets = data[keep], targets[keep]
    sub = DummyDataset(data, targets, dset.trsf, dset.use_path)
    gen = torch.Generator().manual_seed(0)
    loader = DataLoader(sub, batch_size=batch, shuffle=False, num_workers=workers,
                        worker_init_fn=helper.seed_worker, generator=gen)
    return loader, len(targets)


def layer_covariances(model, loader, device="cuda"):
    """{block_<i>_attn_qkv|mlp_fc1|mlp_fc2: Sigma [in, in]} under the model's
    current state (backbones.compute_covariances with peft_adapted_modules)."""
    import torch
    from backbones import compute_covariances, peft_adapted_modules
    with torch.no_grad():
        covs = compute_covariances(model.backbone, loader, device=device,
                                   modules=peft_adapted_modules(model.backbone))
    return {k: v for k, v in covs.items() if v is not None}


def analyze_seed(args, seed, prefix):
    """Rows of one seed's chain."""
    import torch
    import merge_ops as mo
    from utils.data_manager import DataManager
    stem = rc.chain_stem(seed, args.ds, prefix, args.model_backbone)
    files = spec.chain_files(args.ckpt_dir, stem)
    if len(files) < 2:
        print(f"[E6] seed {seed}: {len(files)} cached state(s) for {stem} -- skipped")
        return []
    r = spec.infer_rank(spec.load_state(files[0][1]))
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2.0 * r
    model = build_model(args.ds, r, lora_alpha, device=args.device)
    _, init_cls, increment = rc.DATA_TABLE[args.ds]
    dm = DataManager(args.ds, not args.ds.endswith("dil"), seed, init_cls, increment, False)
    incs = dm.get_task_sizes()
    rows = []
    p = args.p
    for t, W, M_prev, _M_t in spec.replay_chain(args.ckpt_dir, stem, p, args.alpha, args.variant,
                                                max_tasks=args.max_tasks):
        t0 = time.time()
        known = int(sum(incs[:t]))
        new_range = np.arange(known, known + incs[t])
        load_into_model(model, M_prev)
        model.eval()
        rs = np.random.RandomState(args.subsample_seed if args.subsample_seed is not None else seed)
        loader_old, n_old = subsample_loader(dm, np.arange(0, known), args.n_old, rs, args.batch,
                                             args.workers)
        loader_new, n_new = subsample_loader(dm, new_range, args.n_new, rs, args.batch, args.workers)
        covs_old = layer_covariances(model, loader_old, args.device)
        covs_new = layer_covariances(model, loader_new, args.device)
        for an, bn in mo.lora_pairs(W):
            block, module = mo.pair_label(an)
            key = f"block_{block}_{MODULE_KEY.get(module, module)}"
            if key not in covs_old or key not in covs_new:
                print(f"[E6]   no covariance for {key}; skipped")
                continue
            dA = (W[an].float() - M_prev[an].float()).to(args.device)
            met = direction_metrics(dA, covs_old[key], covs_new[key], p, args.energy)
            row = {"seed": seed, "task": t, "block": block, "module": module,
                   "n_old": n_old, "n_new": n_new}
            row.update(met)
            rows.append(row)
        del covs_old, covs_new
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[E6] seed {seed} task {t}: {n_old} old / {n_new} new images, "
              f"{len(mo.lora_pairs(W))} layers in {time.time() - t0:.1f}s")
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# CSV + summary (torch-free)
# ---------------------------------------------------------------------------

def _fmt(v):
    if isinstance(v, float):
        return "" if v != v else f"{v:.6g}"
    return v


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})


def summarize(rows):
    """Means over seeds, tasks and layers of the headline columns (NaN-aware)."""
    keys = ["energy_share_keep", "resp_old_keep", "resp_new_keep", "energy_share_drop",
            "resp_old_drop", "resp_new_drop", "ratio_old", "ratio_new",
            "overlap_keep_old", "overlap_drop_old", "chance_old",
            "overlap_keep_new", "overlap_drop_new", "chance_new"]
    out = {}
    for k in keys:
        vals = np.array([float(r[k]) for r in rows if r.get(k) not in ("", None)], dtype=float)
        out[k] = float(np.nanmean(vals)) if vals.size else float("nan")
    out["n_rows"] = len(rows)
    return out


def print_summary(summary, label=""):
    print(f"\n== subspace summary {label} ({summary['n_rows']} rows; means over seeds, tasks, layers) ==")
    print("| quantity | kept | discarded | chance |")
    print("|---|---:|---:|---:|")
    print(f"| weight energy share | {summary['energy_share_keep']:.3f} | {summary['energy_share_drop']:.3f} | - |")
    print(f"| response share, old inputs | {summary['resp_old_keep']:.3f} | {summary['resp_old_drop']:.3f} | - |")
    print(f"| response share, new inputs | {summary['resp_new_keep']:.3f} | {summary['resp_new_drop']:.3f} | - |")
    print(f"| overlap with old-input subspace | {summary['overlap_keep_old']:.3f} | "
          f"{summary['overlap_drop_old']:.3f} | {summary['chance_old']:.3f} |")
    print(f"| overlap with new-input subspace | {summary['overlap_keep_new']:.3f} | "
          f"{summary['overlap_drop_new']:.3f} | {summary['chance_new']:.3f} |")
    print(f"| ratio resp(drop)/energy(drop) | - | old {summary['ratio_old']:.3f} / new "
          f"{summary['ratio_new']:.3f} | 1 |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(rc.SEEDS))
    ap.add_argument("--prefix", default=None, help="train_prefix (default rebuttal_common.prefix(ds))")
    ap.add_argument("--prefix-family", choices=rc.PREFIX_FAMILIES, default="main")
    ap.add_argument("--ckpt-dir", default=rc.SEQFT_CKPT_DIR)
    ap.add_argument("--model-backbone", default=rc.MODEL_BACKBONE)
    ap.add_argument("--alpha", type=float, default=None, help="replay alpha (default headline)")
    ap.add_argument("--p", type=float, default=None, help="energy fraction (default headline 0.9)")
    ap.add_argument("--variant", default="only_A")
    ap.add_argument("--lora-alpha", type=float, default=None,
                    help="model_lora_alpha of the chain (default 2 * r)")
    ap.add_argument("--n-old", type=int, default=2048, help="old-class images per task (0 = all)")
    ap.add_argument("--n-new", type=int, default=2048, help="new-task images (0 = all)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--energy", type=float, default=0.9,
                    help="trace fraction defining the input subspaces Q_old / Q_new")
    ap.add_argument("--subsample-seed", type=int, default=None,
                    help="RandomState seed of the image subsample (default: the run seed)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="figures_data")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    rc.set_prefix_family(args.prefix_family)
    if args.ds not in rc.POSTURE:
        raise SystemExit(f"unknown dataset {args.ds!r}; known {sorted(rc.POSTURE)}")
    prefix = args.prefix or rc.prefix(args.ds)
    if args.alpha is None:
        args.alpha = rc.HEADLINE_ALPHA[args.ds]
    if args.p is None:
        args.p = rc.POSTURE[args.ds]["pspectral_p"]
    print(f"[E6] ds={args.ds} prefix={prefix} ckpt_dir={args.ckpt_dir} seeds={args.seeds} "
          f"p={args.p:g} alpha={args.alpha:g} n_old={args.n_old} n_new={args.n_new} "
          f"energy={args.energy:g} device={args.device}")
    rows = []
    for seed in args.seeds:
        rows.extend(analyze_seed(args, seed, prefix))
    if not rows:
        raise SystemExit("[E6] nothing analysed: check --ckpt-dir / --prefix / --seeds")
    out = os.path.join(args.out_dir, f"e6_subspace_{args.ds}{args.tag}.csv")
    write_rows(out, rows)
    print(f"[E6] wrote {out}: {len(rows)} rows")
    print_summary(summarize(rows), label=f"{args.ds}{args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
