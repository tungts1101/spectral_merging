"""E15a: OPCM and NUFILT as merge rules on the cached chains (merging_sota.py
opcm_pair / nufilt_pair through the SOTA branch of Learner.merge; Stage-2-only
re-runs, about 10 min per seed).

Sequential regime (default, the paper's chain): base = _backbone_base.pt = W_0,
w_prev = M_{t-1} (task-0 state at t = 1), w_new = W_t, task vector tau_t = W_t -
W_0, old knowledge dM = M_{t-1} - W_0 (the convention of the DARE/DELLA/KnOTS
rows).  Known failure mode to state with the rows: projecting the cumulative
update away from span(M_{t-1} - W_0) also removes shared knowledge.
  --stage 1   grid on ImageNet-R, 8 cells x 3 seeds = 24 runs (about 4 GPU-h)
                opcm    alpha in {0.5, 0.7} x lam rule in {adaptive, sqrt}
                nufilt  r_p in {32, 128} x adapt in {off, on}
              (r_p 128 removes the whole rank-64 old row space of the lora_A
              tensors; 32 is the partial filter)
  --stage 2   the best stage-1 cell per rule (FA of the SUMMARY line; `--best
              RULE=TAG` overrides, the paper defaults a0.5_adaptive / rp128_ad0
              are used when no complete stage-1 log exists) on cifar224, cub,
              cars: 6 cells x 3 seeds = 18 runs (about 3 GPU-h)
Config names e15_{rule}_{tag} on every dataset (e15_opcm_a0.5_adaptive,
e15_nufilt_rp128_ad0); merged tags opcm_a{alpha}_{lam} / nufilt_rp{rp}_rv{rv}_
rl{rl}_ad{0|1} (main.merge_tag).

Independent regime (`--regime independent`, the setting both papers assume):
every task was trained from the same fresh initial module (seqft_independent
True, Appendix E chain on ImageNet-R; RESULTS.md "Sequential versus independent
task initialisation").  The base checkpoint stays that initial module, so tau_t =
W_t - W_0 and dM = M_{t-1} - W_0 are exactly the papers' task vectors.  Cells
(ImageNet-R, 3 seeds, cached): the best stage-1 cell per rule plus the ISM and
averaging reference rows of that table on the same chain, named
e15_indep_{rule}_{tag}, e15_indep_ism_a{alpha}, e15_indep_avg_c1.0
(4 cells, 12 runs, about 2 GPU-h).  The chain's train_prefix comes from
`--prefix` (or is detected from the cache inventory when exactly one complete
ImageNet-R chain with 'indep' in its prefix exists: `python check_cache.py`).
--stage is ignored in this regime.

Outputs: logs_main/<ds>/e15_*.log with the '[Merging] opcm alpha=.. lam_t=..' /
'[Merging] nufilt rp=.. adapt=.. filtered_frac=..' lines and the SUMMARY block;
then `python parse_logs.py e15` -> two rows in the Tab. 6 layout plus the grid.

Run:
    python run_e15_opcm_nufilt.py --stage 1 --dry-run
    python run_e15_opcm_nufilt.py --stage 1 --smoke        # opcm + nufilt, ImageNet-R, 2 tasks
    python run_e15_opcm_nufilt.py --stage 1
    python run_e15_opcm_nufilt.py --stage 2                # reads the stage-1 SUMMARY lines
    python run_e15_opcm_nufilt.py --regime independent --prefix <indep_prefix>
Cells run in checkpoints_main/rebuttal/e15_<ds> (e15_indep for the independent
chain); one process per directory.  reset_merge stays True (rc.COMMON): OPCM's
lam state lives in Learner._merge_rule_state for one run only.
"""
import argparse
import re
import sys
import time
from collections import OrderedDict

import rebuttal_common as rc

GRIDS = OrderedDict([
    ("opcm", [(f"a{a}_{lam}", {"merge_opcm_alpha": a, "merge_opcm_lam": lam})
              for a in (0.5, 0.7) for lam in ("adaptive", "sqrt")]),
    ("nufilt", [(f"rp{rp}_ad{int(ad)}", {"merge_nufilt_rp": rp, "merge_nufilt_adapt": ad})
                for rp in (32, 128) for ad in (False, True)]),
])
PAPER_DEFAULT = {"opcm": "a0.5_adaptive", "nufilt": "rp128_ad0"}   # fallback for stage 2
STAGE1_DATASET = "imagenetr"
STAGE2_DATASETS = ("cifar224", "cub", "cars")
INDEP_DATASET = "imagenetr"
INDEP_PLACEHOLDER = "UNKNOWN_INDEP_PREFIX"      # dry-run only, when no cache is present
JOB = "e15"

_SUMMARY_HEAD = re.compile(r"SUMMARY  (?P<ds>\S+) [—-] (?P<cfg>.+?)\s*$", re.M)
_SUMMARY_FAA = re.compile(r"FAA: ([\d.]+) ± ([\d.]+)")


def cell_name(rule, tag):
    return f"{JOB}_{rule}_{tag}"


def rule_cfg(ds, rule, overrides, ckpt, **extra):
    return rc.cfg(ds, train_merge=rule, **overrides, merge_log_stats=True, checkpoint_dir=ckpt, **extra)


def summary_faa(log_path, config_name):
    """(mean, sd) of the FAA field of the LAST 'SUMMARY  DS — CONFIG' block for
    config_name in the log, or None (utf-8; the dash is U+2014, the sign U+00B1)."""
    text = rc.read_log(log_path)
    heads = list(_SUMMARY_HEAD.finditer(text))
    out = None
    for i, m in enumerate(heads):
        if m.group("cfg").strip() != config_name.upper():
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        mm = _SUMMARY_FAA.search(text, m.end(), end)
        if mm:
            out = (float(mm.group(1)), float(mm.group(2)))
    return out


def best_stage1(rule, n_seeds, log_root=None, override=None, verbose=True):
    """Tag of the best complete stage-1 cell of `rule` by SUMMARY FA; `override`
    (from --best) wins; the paper default is used when nothing is complete."""
    tags = [t for t, _ in GRIDS[rule]]
    if override:
        if override not in tags:
            raise SystemExit(f"--best {rule}={override}: unknown tag; grid: {tags}")
        return override
    scores = {}
    for tag in tags:
        name = cell_name(rule, tag)
        lp = rc.log_path(STAGE1_DATASET, name, log_root)
        if rc.has_summary(lp, name, n_seeds):
            fa = summary_faa(lp, name)
            if fa:
                scores[tag] = fa
    if not scores:
        if verbose:
            print(f"[{JOB}] no complete stage-1 log for {rule} under {rc.log_path(STAGE1_DATASET, '*', log_root)}; "
                  f"using the paper default {PAPER_DEFAULT[rule]} (override with --best {rule}=TAG)")
        return PAPER_DEFAULT[rule]
    best = max(scores, key=lambda t: scores[t][0])
    if verbose:
        table = ", ".join(f"{t}: {m:.2f}±{s:.2f}" for t, (m, s) in scores.items())
        print(f"[{JOB}] stage-1 {rule}: {table} -> best {best}"
              + (f" ({len(scores)}/{len(tags)} cells complete)" if len(scores) < len(tags) else ""))
    return best


def stage1_cells(rules, job_suffix=""):
    ds = STAGE1_DATASET
    ckpt = rc.isolated_ckpt_path(f"{JOB}_{ds}{job_suffix}")
    cells = OrderedDict()
    for rule in rules:
        for tag, ov in GRIDS[rule]:
            cells[cell_name(rule, tag)] = rule_cfg(ds, rule, ov, ckpt)
    return OrderedDict([(ds, cells)])


def stage2_cells(rules, datasets, best, job_suffix=""):
    per_ds = OrderedDict()
    for ds in datasets:
        ckpt = rc.isolated_ckpt_path(f"{JOB}_{ds}{job_suffix}")
        cells = OrderedDict()
        for rule in rules:
            tag = best[rule]
            cells[cell_name(rule, tag)] = rule_cfg(ds, rule, dict(GRIDS[rule])[tag], ckpt)
        per_ds[ds] = cells
    return per_ds


def detect_indep_prefix(ckpt_src, seeds):
    """The train_prefix of the independent-init ImageNet-R chain: exactly one
    complete chain (all seeds) whose prefix contains 'indep'; None otherwise."""
    try:
        import check_cache
    except ImportError:
        return None, []
    chains, _ = check_cache.inventory(ckpt_src)
    candidates = {}
    for ch in chains.values():
        if ch.ds == INDEP_DATASET and "indep" in ch.prefix.lower():
            candidates.setdefault(ch.prefix, set()).add(ch.seed if ch.complete else None)
    complete = sorted(p for p, s in candidates.items() if set(seeds) <= s)
    if len(complete) == 1:
        return complete[0], sorted(candidates)
    return None, sorted(candidates)


def independent_cells(rules, best, prefix, with_reference=True, job_suffix=""):
    """The two rules at their best stage-1 setting plus the ISM and averaging
    reference rows (RESULTS.md 'Sequential versus independent task
    initialisation') on the independent-init chain."""
    ds = INDEP_DATASET
    ckpt = rc.isolated_ckpt_path(f"{JOB}_indep{job_suffix}")
    common = dict(seqft_independent=True, train_prefix=prefix)
    cells = OrderedDict()
    for rule in rules:
        tag = best[rule]
        cells[f"{JOB}_indep_{rule}_{tag}"] = rule_cfg(ds, rule, dict(GRIDS[rule])[tag], ckpt, **common)
    if with_reference:
        ism = rc.cfg(ds, merge_log_stats=True, checkpoint_dir=ckpt, **common)
        cells[f"{JOB}_indep_ism_a{ism['train_merge_alpha']}"] = ism
        # Task Arithmetic over ALL independent experts: W_0 + mean_i (W_i - W_0)
        # (train_merge_incremental False, the default); c = 1.0 as in that table.
        cells[f"{JOB}_indep_avg_c1.0"] = rc.cfg(ds, train_merge="avg", train_merge_coef=1.0,
                                                 train_merge_topk=100, merge_log_stats=True,
                                                 checkpoint_dir=ckpt, **common)
    return OrderedDict([(ds, cells)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1,
                    help="1 = ImageNet-R grid (default); 2 = best cell per rule on the other datasets")
    ap.add_argument("--regime", choices=("sequential", "independent"), default="sequential",
                    help="sequential (default, the paper's chain) or independent (Appendix E chain)")
    ap.add_argument("--rules", nargs="+", choices=list(GRIDS), default=list(GRIDS),
                    help="rules (default opcm nufilt)")
    ap.add_argument("--best", nargs="*", default=[], metavar="RULE=TAG",
                    help="stage-2 / independent cell per rule, e.g. opcm=a0.7_sqrt nufilt=rp32_ad1")
    ap.add_argument("--prefix", default=None,
                    help="train_prefix of the independent-init ImageNet-R chain (--regime independent)")
    ap.add_argument("--no-reference", action="store_true",
                    help="independent regime: skip the ISM and averaging reference rows")
    ap.add_argument("--smoke-all", action="store_true",
                    help="with --smoke: smoke every cell instead of one per rule")
    ap.add_argument("--job-suffix", default="",
                    help="suffix of the checkpoint directory name (run two processes on one dataset)")
    args = rc.apply_common_args(ap.parse_args(argv))

    overrides = {}
    for item in args.best:
        if "=" not in item:
            raise SystemExit(f"--best expects RULE=TAG, got {item!r}")
        rule, tag = item.split("=", 1)
        if rule not in GRIDS:
            raise SystemExit(f"--best: unknown rule {rule!r}; known: {list(GRIDS)}")
        overrides[rule] = tag

    explicit_ds = args.datasets != list(rc.DATASETS)
    if args.regime == "independent":
        prefix = args.prefix
        if prefix is None:
            prefix, candidates = detect_indep_prefix(args.ckpt_src, args.seeds)
            if prefix is None:
                msg = (f"[{JOB}] independent regime: pass --prefix <train_prefix of the Appendix E "
                       f"independent-init ImageNet-R chain>; `python check_cache.py` lists the chains"
                       + (f"; candidates with 'indep' in the prefix: {candidates}" if candidates else ""))
                if not args.dry_run:
                    raise SystemExit(msg)
                print(msg + f" -- dry-run continues with the placeholder {INDEP_PLACEHOLDER}")
                prefix = INDEP_PLACEHOLDER
            else:
                print(f"[{JOB}] independent-init chain detected: {prefix}")
        best = {r: best_stage1(r, len(args.seeds), args.log_root, overrides.get(r)) for r in args.rules}
        per_ds = independent_cells(args.rules, best, prefix, not args.no_reference, args.job_suffix)
        job_of = {INDEP_DATASET: f"{JOB}_indep{args.job_suffix}"}
    elif args.stage == 1:
        per_ds = stage1_cells(args.rules, args.job_suffix)
        job_of = {STAGE1_DATASET: f"{JOB}_{STAGE1_DATASET}{args.job_suffix}"}
    else:
        datasets = [d for d in args.datasets if d != STAGE1_DATASET] if explicit_ds else list(STAGE2_DATASETS)
        best = {r: best_stage1(r, len(args.seeds), args.log_root, overrides.get(r)) for r in args.rules}
        per_ds = stage2_cells(args.rules, datasets, best, args.job_suffix)
        job_of = {ds: f"{JOB}_{ds}{args.job_suffix}" for ds in datasets}

    if args.smoke and not args.smoke_all:
        # one cell per rule on the first dataset (opcm and nufilt both exercise merging_sota)
        ds0 = next(iter(per_ds))
        picked = OrderedDict()
        for rule in args.rules:
            for name, cfg in per_ds[ds0].items():
                if cfg["train_merge"] == rule:
                    picked[name] = cfg
                    break
        per_ds = OrderedDict([(ds0, picked)])

    total = sum(len(c) for c in per_ds.values())
    for ds, cells in per_ds.items():
        print(f"[{JOB}] {args.regime} stage {args.stage if args.regime == 'sequential' else '-'} {ds}: "
              f"{len(cells)} cells x {1 if args.smoke else len(args.seeds)} seed(s)")

    t0 = time.time()
    for ds, cells in per_ds.items():
        if not cells:
            continue
        if not args.smoke:
            prefix = next(iter(cells.values()))["train_prefix"]
            rc.isolated_ckpt_dir(ds, prefix, args.seeds, job_of[ds], src_dir=args.ckpt_src,
                                 dry_run=args.dry_run)
        rc.launch(ds, cells, args.seeds, **rc.launch_kwargs(args))

    if args.dry_run:
        rc.print_cells(total)
        return 0
    print(f"[{JOB}] done: {total} cells. Next: python parse_logs.py e15"
          + (" ; then --stage 2" if args.regime == "sequential" and args.stage == 1 else ""))
    print(f"Total: {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
