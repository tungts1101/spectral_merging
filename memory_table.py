"""Storage accounting for ISM and for the baselines that keep class statistics (E5).

Pure Python: every number is derived from constants mirrored from the code
(sources in the comments next to each constant), so the table answers the
"constant memory" concern (rebuttal A4: b28C-W3, xkNt-W3) from the code and
not from guesses.  The adapter is constant; the per-class Gaussians grow with
the number of classes and exceed the adapter after a few dozen classes.

Prints (and with --out writes) one row per dataset x statistics mode:
    full       mean + full covariance per class  (train_ca_cov_mode "class", the paper)
    full_sym   the same with only the upper triangle stored (storage bound, not implemented)
    diag       mean + diagonal variance           (train_ca_cov_mode "diag", E5)
    lowrank k  mean + rank-k factor + residual diagonal ("lowrank", train_ca_cov_rank k)
    shared     mean per class + one pooled covariance   ("shared")
plus baseline rows (SLCA, LCA, E2-LoRA, RanPAC, SLDC) with their formula and
source, labelled `paper` / `code (port)` / `to verify`.

Usage:
    python memory_table.py                                   # tables to stdout
    python memory_table.py --out figures_data/e5_storage_table.csv
    python memory_table.py --checkpoint checkpoints_main/seqft/<seed>_<ds>_vit_base_patch16_224_lora_<prefix>_backbone_0.pt
                                                             # server: counts the floats of a real state (torch)
    python memory_table.py --results figures_data/e5_storage_accuracy.csv --out figures_data/e5_storage_joined.csv
                                                             # joins FA/AA/FF of the cov_* cells (CSV with dataset +
                                                             # config|mode columns) onto the storage rows, e.g. to
                                                             # cross-check the parser's storage columns
figures_data/e5_storage_accuracy.csv itself (FA vs storage, read by scripts/fig_e5_storage.py)
is written by `python parse_logs.py e5`, the single producer; this module provides the
storage formulas and the baseline rows.  MB are decimal (1 MB = 1e6 bytes) at fp32 unless stated.
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, OrderedDict

# ----------------------------------------------------------------------------
# Constants mirrored from the code
# ----------------------------------------------------------------------------

D = 768                       # BASE_CONFIG["model_outdim"] (main.py:1768); Model21.feature_dim = ViT-B/16 num_features
LORA_R = 64                   # BASE_CONFIG["model_lora_r"] (main.py:1771); run_best.BASE
LORA_ALPHA = 128              # BASE_CONFIG["model_lora_alpha"] (main.py:1772); scale alpha/r = 2
BLOCKS = 12                   # timm vit_base_patch16_224 depth
# PEFT LoRA on the timm ViT-B/16 Linear layers (helper.get_backbone, helper.py:342-355):
# lora_A: [r, in_features], lora_B: [out_features, r]; per module floats = r * (in + out).
TARGET_SHAPES = OrderedDict([          # module -> (in_features, out_features)
    ("qkv", (768, 2304)),              # blocks.i.attn.qkv  (3 * 768 outputs)
    ("fc1", (768, 3072)),              # blocks.i.mlp.fc1   (mlp_ratio 4)
    ("fc2", (3072, 768)),              # blocks.i.mlp.fc2
])                                     # run_best/run_sota_merge: model_lora_target_modules ["qkv","fc1","fc2"]
NORM_FLOATS = 2 * D           # Model21.norm = nn.LayerNorm(768) weight + bias, saved with every
                              # state (main.py:180-184, 204-213; model_use_norm True, main.py:1770)
HEAD_PER_CLASS = D            # ContinualLinear heads: nn.Linear(768, n, bias=False), no LayerNorm because
                              # model_classifier_use_norm defaults to `not model_use_norm` = False
                              # (helper.py:468-481, main.py:666-667)
STATS_JITTER_NOTE = "cov + 1e-4 I"   # compute_multivariate_normal (main.py:1156-1161)
BYTES_FP32, BYTES_FP64 = 4, 8

# main.TOTAL_CLASSES (main.py:1741-1743); DIL labels are offset per domain
# (main.py:1736-1740), so every (domain, class) pair keeps its own Gaussian.
DATASETS = OrderedDict([
    ("cifar224", ("CIFAR-100", 100)),
    ("imagenetr", ("ImageNet-R", 200)),
    ("cub", ("CUB-200", 200)),
    ("cars", ("Stanford Cars", 196)),
    ("officehome_dil", ("Office-Home (DIL, 4 x 65)", 260)),
    ("dndil", ("DomainNet (DIL, 6 x 345)", 2070)),
])

RANPAC_M = 10_000             # RanPAC random-projection width M (McDonnell et al., NeurIPS'23)


# ----------------------------------------------------------------------------
# Formulas
# ----------------------------------------------------------------------------

def lora_floats(r=LORA_R, shapes=TARGET_SHAPES, blocks=BLOCKS, norm_floats=NORM_FLOATS):
    """(total floats, {module: floats per block}) of the deployed LoRA state
    saved by Model21.get_backbone_trainable_params (12 blocks x (A + B) + norm)."""
    per_block = OrderedDict((name, r * (i + o)) for name, (i, o) in shapes.items())
    total = blocks * sum(per_block.values()) + norm_floats
    return total, per_block


def head_floats(n_classes):
    return n_classes * HEAD_PER_CLASS


def stats_per_class(mode, d=D, rank=None):
    """Floats stored per class (the per-class part only)."""
    if mode == "full":
        return d + d * d                  # mean + covariance (main.py:1103-1104)
    if mode == "full_sym":
        return d + d * (d + 1) // 2       # upper triangle only (storage bound)
    if mode == "diag":
        return 2 * d                      # mean + variance
    if mode == "lowrank":
        if not rank:
            raise ValueError("lowrank needs a rank")
        return d * (rank + 2)             # mean + U (d x k) + residual variance
    if mode == "shared":
        return d                          # mean only; one pooled covariance for all classes
    raise ValueError(f"unknown mode {mode!r}")


def stats_shared(mode, d=D):
    """Floats stored once, independent of the class count."""
    return d * d if mode == "shared" else 0


def stats_floats(mode, n_classes, d=D, rank=None):
    return n_classes * stats_per_class(mode, d, rank) + stats_shared(mode, d)


def align_gpu_floats(mode, n_classes, d=D, rank=None):
    """Approximate GPU floats held by the LCA sampling distributions (align,
    main.py:1271-1275): MultivariateNormal keeps covariance + scale_tril per
    class; the cheap modes keep their own parameters (and shared one Cholesky)."""
    if mode in ("full", "full_sym"):
        return 2 * n_classes * d * d
    if mode == "diag":
        return 2 * n_classes * d
    if mode == "lowrank":
        return n_classes * (d * rank + 2 * d) + n_classes * rank * rank
    if mode == "shared":
        return n_classes * d + 2 * d * d
    raise ValueError(mode)


def mb(floats, bytes_per=BYTES_FP32):
    return floats * bytes_per / 1e6


MODES = [("full", None), ("full_sym", None), ("diag", None),
         ("lowrank", 16), ("lowrank", 64), ("shared", None)]
MODE_LABEL = {"full": "full covariance (paper)", "full_sym": "full, upper triangle (bound)",
              "diag": "diagonal", "lowrank": "low-rank k + diagonal", "shared": "shared covariance"}
MODE_CONFIG = {"full": "train_ca_cov_mode=class", "full_sym": "(storage only)",
               "diag": "train_ca_cov_mode=diag", "lowrank": "train_ca_cov_mode=lowrank, train_ca_cov_rank=k",
               "shared": "train_ca_cov_mode=shared"}


def build_rows(datasets=None, modes=MODES, ranks=None):
    """Storage rows for ISM: one per dataset x mode (fp32)."""
    datasets = datasets or list(DATASETS)
    if ranks:
        modes = [m for m in modes if m[0] != "lowrank"] + [("lowrank", k) for k in ranks]
    lora, _ = lora_floats()
    rows = []
    for ds in datasets:
        label, C = DATASETS[ds]
        for mode, rank in modes:
            st = stats_floats(mode, C, rank=rank)
            hd = head_floats(C)
            rows.append(OrderedDict([
                ("kind", "ours"), ("method", "ISM (ours)"), ("dataset", ds), ("dataset_label", label),
                ("C", C), ("mode", mode), ("rank", rank or ""),
                ("config", MODE_CONFIG[mode]),
                ("lora_floats", lora), ("lora_MB", round(mb(lora), 2)),
                ("head_floats", hd), ("head_MB", round(mb(hd), 2)),
                ("stats_per_class", stats_per_class(mode, rank=rank)),
                ("stats_floats", st), ("stats_MB", round(mb(st), 2)),
                ("total_MB", round(mb(lora + hd + st), 2)),
                ("stats_over_lora", round(st / lora, 2)),
                ("align_gpu_MB", round(mb(align_gpu_floats(mode, C, rank=rank)), 1)),
                ("source", "code: helper.get_backbone, main.Model21, main.compute_multivariate_normal"),
            ]))
    return rows


def baseline_rows(datasets=None):
    """Statistics storage of the baselines that also keep class statistics.
    `status` says whether the formula comes from a paper, from a port in this
    repo, or still needs verification; nothing here is measured."""
    datasets = datasets or list(DATASETS)
    specs = [
        # method, per-class floats, per-class bytes, once floats, formula, source, status
        ("SLCA", D + D * D, BYTES_FP32 * (D + D * D), 0,
         "mean + full covariance per class (classifier alignment)",
         "Zhang et al., SLCA, ICCV'23, Sec. 3 'Classifier Alignment' (sampling from per-class Gaussians)",
         "paper"),
        ("LCA", D + D * D, BYTES_FP32 * (D + D * D), 0,
         "mean + full covariance per class (local classifier alignment, the Stage-2 rule ISM adopts)",
         "LCA, ICLR'26 (the alignment this repo's Stage 2 follows)", "paper"),
        ("E2-LoRA", D + D * D, BYTES_FP64 * D + BYTES_FP32 * D * D, 0,
         "mean (float64 in the port) + full covariance per class",
         "port in main.py Learner._e2_compute_class_stats (main.py:1408-1416) of the E2-LoRA ICML'26 recipe",
         "code (port)"),
        ("RanPAC", RANPAC_M, BYTES_FP32 * RANPAC_M, RANPAC_M * RANPAC_M,
         f"prototype row of width M={RANPAC_M:,} per class + Gram matrix M x M once "
         f"(random projection d x M regenerable from its seed, +{D * RANPAC_M:,} floats if stored)",
         "McDonnell et al., RanPAC, NeurIPS'23, Sec. 3 / Alg. 1 (M = 10,000)", "paper"),
        ("SLDC", D + D * D, BYTES_FP32 * (D + D * D), 0,
         "mean + full covariance per class (assumed SLCA-style alignment)",
         "SLDC, AAAI'26 -- TO VERIFY against the paper before quoting", "to verify"),
    ]
    rows = []
    for ds in datasets:
        label, C = DATASETS[ds]
        for method, per_cls, per_cls_bytes, once, formula, source, status in specs:
            floats = C * per_cls + once
            nbytes = C * per_cls_bytes + once * BYTES_FP32
            rows.append(OrderedDict([
                ("kind", "baseline"), ("method", method), ("dataset", ds), ("dataset_label", label),
                ("C", C), ("mode", ""), ("rank", ""), ("config", ""),
                ("lora_floats", ""), ("lora_MB", ""), ("head_floats", ""), ("head_MB", ""),
                ("stats_per_class", per_cls),
                ("stats_floats", floats), ("stats_MB", round(nbytes / 1e6, 2)),
                ("total_MB", ""), ("stats_over_lora", round(floats / lora_floats()[0], 2)),
                ("align_gpu_MB", ""),
                ("source", f"{status}: {source}; {formula}"),
            ]))
    return rows


# ----------------------------------------------------------------------------
# Results join (accuracy vs storage)
# ----------------------------------------------------------------------------

_COV_CONFIG_RE = re.compile(r"^cov_(?P<m>class|diag|shared|lowrank)(?:_k(?P<k>\d+))?(?:_|$)")
METRIC_COLS = ("n_seeds", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd")


def _mode_of_result(row):
    """(mode, rank) of a result row: from `mode`/`rank` columns, or from a config
    name cov_class | cov_diag | cov_lowrank_k<k> | cov_shared."""
    mode = (row.get("mode") or "").strip()
    rank = (row.get("rank") or "").strip()
    if not mode:
        name = (row.get("config") or row.get("config_name") or row.get("name") or "").strip()
        m = _COV_CONFIG_RE.match(name)
        if not m:
            return None, None
        mode, rank = m.group("m"), m.group("k") or ""
    mode = {"class": "full"}.get(mode, mode)
    rank = int(rank) if rank else None
    return mode, rank


def join_results(rows, results_csv):
    """Attach FA/AA/FF of the cov_* cells to the matching ISM storage rows and
    return only the rows that have results."""
    with open(results_csv, encoding="utf-8", newline="") as fh:
        results = list(csv.DictReader(fh))
    lookup = {}
    for r in results:
        mode, rank = _mode_of_result(r)
        ds = (r.get("dataset") or "").strip()
        if mode is None or not ds:
            continue
        lookup[(ds, mode, rank)] = r
    joined = []
    for row in rows:
        if row["kind"] != "ours":
            continue
        key = (row["dataset"], row["mode"], row["rank"] or None)
        r = lookup.get(key)
        if r is None:
            continue
        out = OrderedDict(row)
        for c in METRIC_COLS:
            out[c] = r.get(c, "")
        joined.append(out)
    missing = set(lookup) - {(r["dataset"], r["mode"], r["rank"] or None) for r in rows if r["kind"] == "ours"}
    if missing:
        print(f"note: {len(missing)} result row(s) without a storage row: {sorted(missing)}")
    return joined


# ----------------------------------------------------------------------------
# Checkpoint verification (server, torch)
# ----------------------------------------------------------------------------

def verify_checkpoint(path):
    """Count the floats of a cached `_backbone_*.pt` state and compare with lora_floats()."""
    try:
        import torch
    except ImportError:
        print("torch is not installed here; run --checkpoint on the server")
        return 2
    try:
        state = torch.load(path, map_location="cpu")
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=False)
    groups = Counter()
    total = 0
    for key, val in state.items():
        n = int(val.numel())
        total += n
        if key.startswith("norm."):
            groups["norm"] += n
        elif ".attn.qkv." in key:
            groups["qkv"] += n
        elif ".mlp.fc1." in key:
            groups["fc1"] += n
        elif ".mlp.fc2." in key:
            groups["fc2"] += n
        else:
            groups["other"] += n
    expected, per_block = lora_floats()
    print(f"{path}: {total:,} floats in {len(state)} tensors")
    for g, n in sorted(groups.items()):
        ref = per_block.get(g, NORM_FLOATS if g == "norm" else 0) * (BLOCKS if g != "norm" else 1)
        print(f"  {g:<5} {n:>12,}  (expected {ref:,})")
    ok = total == expected
    print("OK: matches lora_floats()" if ok else f"MISMATCH: expected {expected:,}")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}: {len(rows)} rows")


def _md(headers, lines):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for ln in lines:
        print("| " + " | ".join(str(x) for x in ln) + " |")


def print_header():
    total, per_block = lora_floats()
    parts = " + ".join(f"{k} {v:,}" for k, v in per_block.items())
    print(f"LoRA module (PEFT r={LORA_R}, alpha={LORA_ALPHA}, {'+'.join(TARGET_SHAPES)}, {BLOCKS} blocks): "
          f"lora floats {total:,} = {BLOCKS} x ({parts}) + {NORM_FLOATS:,} norm -> {mb(total):.2f} MB fp32")
    full = stats_per_class("full")
    print(f"Class statistics, full covariance ({STATS_JITTER_NOTE}): {full:,} floats per class "
          f"= {D} + {D}^2 -> {mb(full):.2f} MB fp32 per class; "
          f"upper triangle {stats_per_class('full_sym'):,}; diag {stats_per_class('diag'):,}; "
          f"lowrank k16 {stats_per_class('lowrank', rank=16):,}; k64 {stats_per_class('lowrank', rank=64):,}; "
          f"shared {stats_per_class('shared'):,} per class + {stats_shared('shared'):,} once")
    print(f"Head: {HEAD_PER_CLASS} floats per class (Linear 768 -> C, bias False, no norm)")
    print()


def print_ours(rows):
    print("ISM storage per dataset and statistics mode (fp32, decimal MB)")
    _md(["dataset", "C", "statistics", "floats/class", "stats MB", "LoRA MB", "head MB", "total MB",
         "stats/LoRA", "LCA GPU MB"],
        [(r["dataset_label"], r["C"], MODE_LABEL[r["mode"]] + (f" k={r['rank']}" if r["rank"] else ""),
          f"{r['stats_per_class']:,}", f"{r['stats_MB']:.2f}", f"{r['lora_MB']:.2f}", f"{r['head_MB']:.2f}",
          f"{r['total_MB']:.2f}", f"{r['stats_over_lora']:.2f}", f"{r['align_gpu_MB']:.1f}")
         for r in rows if r["kind"] == "ours"])
    print()


def print_baselines(rows):
    print("Baselines that also keep class statistics (formula from the stated source; not measured)")
    _md(["method", "dataset", "C", "floats", "MB", "source"],
        [(r["method"], r["dataset_label"], r["C"], f"{r['stats_floats']:,}", f"{r['stats_MB']:.2f}", r["source"])
         for r in rows if r["kind"] == "baseline"])
    print()


def print_joined(rows):
    print("Accuracy vs storage (cov_* cells)")
    _md(["dataset", "statistics", "stats MB", "total MB", "n", "FA", "AA", "FF"],
        [(r["dataset_label"], MODE_LABEL[r["mode"]] + (f" k={r['rank']}" if r["rank"] else ""),
          f"{r['stats_MB']:.2f}", f"{r['total_MB']:.2f}", r.get("n_seeds", ""),
          f"{r.get('fa', '')} +- {r.get('fa_sd', '')}", f"{r.get('aa', '')} +- {r.get('aa_sd', '')}",
          f"{r.get('ffm', '')} +- {r.get('ffm_sd', '')}") for r in rows])
    print()


def self_check():
    total, per_block = lora_floats()
    assert per_block["qkv"] == 196_608 and per_block["fc1"] == 245_760 and per_block["fc2"] == 245_760, per_block
    assert total == 8_259_072, total
    assert stats_per_class("full") == 590_592
    assert stats_per_class("full_sym") == 296_064
    assert stats_per_class("diag") == 1_536
    assert stats_per_class("lowrank", rank=16) == 13_824
    assert stats_per_class("lowrank", rank=64) == 50_688
    assert stats_floats("shared", 200) == 200 * 768 + 768 * 768


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--ranks", nargs="+", type=int, default=None,
                    help="low-rank ranks to tabulate (default 16 64)")
    ap.add_argument("--out", default=None, help="CSV path (storage table, or the joined table with --results)")
    ap.add_argument("--results", default=None,
                    help="aggregated results CSV (columns dataset + config|mode[,rank] + fa,fa_sd,...) to join")
    ap.add_argument("--checkpoint", default=None, help="a cached _backbone_*.pt to count (needs torch)")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args(argv)

    self_check()
    if args.checkpoint:
        return verify_checkpoint(args.checkpoint)

    print_header()
    rows = build_rows(args.datasets, ranks=args.ranks)
    if not args.no_baselines:
        rows += baseline_rows(args.datasets)
    if args.results:
        joined = join_results(rows, args.results)
        print_joined(joined)
        if args.out:
            write_csv(joined, args.out)
        return 0
    print_ours(rows)
    if not args.no_baselines:
        print_baselines(rows)
    if args.out:
        write_csv(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
