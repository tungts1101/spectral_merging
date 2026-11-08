"""Read-only inventory of the cached sequential LoRA chains.

Lists the files of checkpoints_main/seqft (or --ckpt-dir) as (seed, dataset,
train_prefix) chains using the naming of Learner.prefix / backbone_checkpoint /
classifier_checkpoint / merged_checkpoint (main.py:1661-1710):

    {seed}_{dataset}_{model_backbone}_{train_prefix}_backbone_base.pt   (W_0 after task 0)
    {seed}_{dataset}_{model_backbone}_{train_prefix}_backbone_{t}.pt    (W_t)
    {seed}_{dataset}_{model_backbone}_{train_prefix}_classifier_{t}.pt  (unaligned heads)
    {seed}_{dataset}_{model_backbone}_{train_prefix}_merged_{tag}_{t}.pt (M_t, deleted by cleanup)

For every chain it reports the task count (contiguous _backbone_t files from
t = 0), whether the classifier files and the base file are present, which
merged tags survive, and whether the chain is complete for the dataset's
standard split (DATA_TABLE) or for --T.  `chain_status()` is the helper every
driver's --dry-run calls through rebuttal_common.launch().

Usage (server day 0):
    python check_cache.py                       # table of every chain under checkpoints_main/seqft
    python check_cache.py --ckpt-dir checkpoints_main/rebuttal/e8
    python check_cache.py --ds imagenetr --prefix main_inr_r64_qkvfc --seeds 1993 1994 1995 --T 10
    python check_cache.py --csv figures_data/cache_inventory.csv
No torch is needed; nothing is written unless --csv is given.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

from rebuttal_common import (DATA_TABLE, TOTAL_CLASSES, MODEL_BACKBONE, SEQFT_CKPT_DIR,
                             SEEDS, POSTURE, BEST_PREFIX, chain_stem, task_increments)

# Known dataset names (main.TOTAL_CLASSES), longest first so 'officehome_dil'
# is not cut at its underscore; a generic token is the fallback.
_DS_ALT = "|".join(sorted(TOTAL_CLASSES, key=len, reverse=True)) + r"|[a-z0-9]+"
FILE_RE = re.compile(
    r"^(?P<seed>\d+)_(?P<ds>" + _DS_ALT + r")_(?P<rest>.+)_"
    r"(?P<kind>backbone|classifier|covs|state|e2state|merged_(?P<tag>.+))_(?P<t>base|\d+)\.pt$")
# rest = {model_backbone}[_{train_prefix}]; the seqft backbone name ends in '_lora'
# (helper.get_backbone), the merge-family dirs use the recipe name instead.
_BB_RE = re.compile(r"^(?P<bb>.+?_lora)(?:_(?P<prefix>.+))?$")
_RECIPES = ("e2lora", "basic_lora", "sgp_lora", "nsp_lora", "full_nsp", "full",
            "sgp_seqft", "nsp_seqft", "first_task_lora", "joint_lora", "joint_full", "inflora")


def parse_filename(name):
    """Dict(seed, ds, backbone, prefix, kind, tag, t) for a checkpoint file name,
    or None when the name does not follow the Learner naming."""
    m = FILE_RE.match(name)
    if not m:
        return None
    rest = m.group("rest")
    bm = _BB_RE.match(rest)
    if bm:
        bb, pre = bm.group("bb"), bm.group("prefix") or ""
    else:
        bb, pre = rest, ""
        for r in sorted(_RECIPES, key=len, reverse=True):
            if rest == r or rest.startswith(r + "_"):
                bb, pre = r, rest[len(r) + 1:]
                break
    t = m.group("t")
    return {"seed": int(m.group("seed")), "ds": m.group("ds"), "backbone": bb, "prefix": pre,
            "kind": "merged" if m.group("tag") else m.group("kind"), "tag": m.group("tag"),
            "t": -1 if t == "base" else int(t), "file": name}


class Chain:
    """All files of one (seed, dataset, backbone, train_prefix)."""

    def __init__(self, seed, ds, backbone, prefix):
        self.seed, self.ds, self.backbone, self.prefix = seed, ds, backbone, prefix
        self.backbone_tasks = set()
        self.classifier_tasks = set()
        self.base = False
        self.merged = defaultdict(set)      # tag -> tasks
        self.other = defaultdict(set)       # kind -> tasks (covs, state, e2state)

    @property
    def key(self):
        return (self.seed, self.ds, self.backbone, self.prefix)

    @property
    def n_tasks(self):
        """Contiguous task states from t = 0 (the number of cached tasks)."""
        t = 0
        while t in self.backbone_tasks:
            t += 1
        return t

    @property
    def expected_T(self):
        """Task count of the dataset's standard split (None for unknown datasets)."""
        if self.ds not in DATA_TABLE:
            return None
        _, init, inc = DATA_TABLE[self.ds]
        return len(task_increments(TOTAL_CLASSES[self.ds], init, inc))

    def complete_for(self, T):
        tasks = set(range(T))
        return (tasks <= self.backbone_tasks) and (tasks <= self.classifier_tasks) and self.base

    @property
    def complete(self):
        T = self.expected_T
        return self.complete_for(T) if T else False

    @property
    def stray_backbone(self):
        """Backbone tasks beyond the contiguous run (a long split or leftovers)."""
        return sorted(t for t in self.backbone_tasks if t >= self.n_tasks)

    def row(self):
        T = self.expected_T
        return {"seed": self.seed, "dataset": self.ds, "backbone": self.backbone,
                "prefix": self.prefix, "n_tasks": self.n_tasks,
                "n_classifier": len(self.classifier_tasks), "base": int(self.base),
                "expected_T": T if T else "", "complete": int(self.complete),
                "max_task": max(self.backbone_tasks) if self.backbone_tasks else "",
                "stray": " ".join(map(str, self.stray_backbone)),
                "merged_tags": "; ".join(f"{tag}:{len(ts)}" for tag, ts in sorted(self.merged.items())),
                "other": "; ".join(f"{k}:{len(ts)}" for k, ts in sorted(self.other.items()))}


def inventory(ckpt_dir=SEQFT_CKPT_DIR):
    """{(seed, ds, backbone, prefix): Chain} for every .pt file in ckpt_dir
    (symlinks count as present only when their target exists)."""
    chains = {}
    unparsed = []
    if not os.path.isdir(ckpt_dir):
        return chains, unparsed
    for name in sorted(os.listdir(ckpt_dir)):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(ckpt_dir, name)
        if not os.path.exists(path):          # dangling symlink
            unparsed.append(name + " (dangling link)")
            continue
        info = parse_filename(name)
        if info is None:
            unparsed.append(name)
            continue
        key = (info["seed"], info["ds"], info["backbone"], info["prefix"])
        ch = chains.get(key)
        if ch is None:
            ch = chains[key] = Chain(*key)
        kind, t = info["kind"], info["t"]
        if kind == "backbone":
            if t < 0:
                ch.base = True
            else:
                ch.backbone_tasks.add(t)
        elif kind == "classifier":
            ch.classifier_tasks.add(t)
        elif kind == "merged":
            ch.merged[info["tag"]].add(t)
        else:
            ch.other[kind].add(t)
    return chains, unparsed


def chain_status(ds, prefix, seeds, T, ckpt_dir=SEQFT_CKPT_DIR, model_backbone=MODEL_BACKBONE):
    """Per-seed presence of the files a T-task cache-served run needs.

    Returns {"dir": ckpt_dir, "exists": bool, "T": T, "complete": all seeds complete,
             "seeds": {seed: {"base": bool, "backbone_missing": [t...],
                              "classifier_missing": [t...], "n_backbone": int,
                              "complete": bool}}}.
    Existence is checked file by file (symlinks resolve; dangling ones count as
    missing), so it also works on an isolated directory.
    """
    out = {"dir": ckpt_dir, "exists": os.path.isdir(ckpt_dir), "T": T, "seeds": {}}
    for seed in seeds:
        stem = chain_stem(seed, ds, prefix, model_backbone)
        base = os.path.exists(os.path.join(ckpt_dir, stem + "_backbone_base.pt"))
        bmiss = [t for t in range(T) if not os.path.exists(os.path.join(ckpt_dir, f"{stem}_backbone_{t}.pt"))]
        cmiss = [t for t in range(T) if not os.path.exists(os.path.join(ckpt_dir, f"{stem}_classifier_{t}.pt"))]
        out["seeds"][seed] = {"base": base, "backbone_missing": bmiss, "classifier_missing": cmiss,
                              "n_backbone": T - len(bmiss),
                              "complete": base and not bmiss and not cmiss}
    out["complete"] = all(s["complete"] for s in out["seeds"].values()) if seeds else False
    return out


def format_status(status):
    """One line per seed for a dry-run."""
    lines = [f"cache {status['dir']} (T={status['T']}): " + ("complete" if status["complete"] else "INCOMPLETE")]
    for seed, s in status["seeds"].items():
        if s["complete"]:
            lines.append(f"  seed {seed}: complete")
        else:
            lines.append(f"  seed {seed}: base={'yes' if s['base'] else 'NO'} "
                         f"backbone missing {s['backbone_missing']} classifier missing {s['classifier_missing']}")
    return "\n".join(lines)


def families_present(chains):
    """{dataset: {prefix: [seeds]}} over the chains, to pick --prefix-family."""
    fam = defaultdict(lambda: defaultdict(list))
    for ch in chains.values():
        fam[ch.ds][ch.prefix].append(ch.seed)
    return {ds: {p: sorted(s) for p, s in d.items()} for ds, d in fam.items()}


def headline_families(chains):
    """Which headline family (main_*_r64_qkvfc vs best_<ds>_r64) is complete
    for the three seeds, per dataset."""
    out = {}
    for ds in POSTURE:
        main_pre, best_pre = POSTURE[ds]["train_prefix"], BEST_PREFIX[ds]
        got = {}
        for fam, pre in (("main", main_pre), ("best", best_pre)):
            seeds_ok = sorted(ch.seed for ch in chains.values()
                              if ch.ds == ds and ch.prefix == pre and ch.complete)
            got[fam] = seeds_ok
        out[ds] = got
    return out


def print_table(chains, unparsed):
    cols = ["seed", "dataset", "prefix", "n_tasks", "n_classifier", "base", "expected_T",
            "complete", "stray", "merged_tags", "other"]
    rows = [ch.row() for ch in sorted(chains.values(), key=lambda c: (c.ds, c.prefix, c.seed))]
    if not rows:
        print("(no chains found)")
    else:
        widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
        print("  ".join(c.ljust(widths[c]) for c in cols))
        for r in rows:
            print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    if unparsed:
        print(f"\n{len(unparsed)} file(s) not following the Learner naming:")
        for u in unparsed[:20]:
            print("  " + u)
        if len(unparsed) > 20:
            print(f"  ... {len(unparsed) - 20} more")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", nargs="+", default=[SEQFT_CKPT_DIR],
                    help="checkpoint directories to inventory (default checkpoints_main/seqft)")
    ap.add_argument("--ds", default=None, help="with --prefix: report chain_status for this dataset")
    ap.add_argument("--prefix", default=None, help="train_prefix for chain_status")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--T", type=int, default=None,
                    help="task count for chain_status (default: the dataset's standard split)")
    ap.add_argument("--backbone", default=MODEL_BACKBONE)
    ap.add_argument("--csv", default=None, help="write the chain table to this CSV")
    args = ap.parse_args(argv)

    all_rows = []
    for d in args.ckpt_dir:
        print(f"=== {d} ===")
        if not os.path.isdir(d):
            print("(directory does not exist)")
            continue
        chains, unparsed = inventory(d)
        print_table(chains, unparsed)
        fams = headline_families(chains)
        print("\nheadline families complete for seeds (main = main_*_r64_qkvfc, best = best_<ds>_r64):")
        for ds, got in fams.items():
            print(f"  {ds:<10} main {got['main'] or '-'}   best {got['best'] or '-'}")
        present = families_present(chains)
        long_chains = [ch for ch in chains.values() if ch.expected_T and ch.n_tasks > ch.expected_T]
        if long_chains:
            print("\nchains longer than the standard split (20/50-task candidates):")
            for ch in sorted(long_chains, key=lambda c: (c.ds, c.prefix, c.seed)):
                print(f"  {ch.ds} {ch.prefix} seed {ch.seed}: {ch.n_tasks} tasks")
        print(f"\nprefixes per dataset: " + "; ".join(
            f"{ds}: {', '.join(sorted(p))}" for ds, p in sorted(present.items())))
        for ch in chains.values():
            r = ch.row()
            r["dir"] = d
            all_rows.append(r)
        print()

    if args.ds and args.prefix:
        T = args.T
        if T is None:
            _, init, inc = DATA_TABLE[args.ds]
            T = len(task_increments(TOTAL_CLASSES[args.ds], init, inc))
        for d in args.ckpt_dir:
            print(format_status(chain_status(args.ds, args.prefix, args.seeds, T, d, args.backbone)))

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        cols = ["dir", "seed", "dataset", "backbone", "prefix", "n_tasks", "n_classifier", "base",
                "expected_T", "complete", "max_task", "stray", "merged_tags", "other"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"wrote {args.csv}: {len(all_rows)} chain(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
