"""parse_logs.py -- the one torch-free parser for this repo's experiment logs.

Reads every ``logs_main/<dataset>/<config>.log`` written by ``main.run_config_sweep``
(and the ``logs_main/sota_merge.log`` style files written by ``run_sota_merge.py``),
recomputes the paper metrics from the per-task ``Grouped:`` vectors with a numpy port of
``helper.compute_metrics`` (helper.py:22-43), and turns the result into per-seed and
seed-averaged CSVs plus markdown tables.  Only numpy is needed, so it runs on the
Windows PC as well as on the server.

Log layout parsed here (every line carries the prefix
``'YYYY-MM-DD HH:MM:SS,mmm [main.py] => '`` from the format string in
main.py:1988-1996; lines without that prefix, e.g. traceback continuations and the
bare ``====`` rulers, are tolerated):

    Experiment: {ds} -- {cfg} -- seed {seed}           (main.py:2000; '--' is an em dash)
    Configuration:                                     (main.py:1913)
      key: value                                       (main.py:1915; strings unquoted)
    [Freeze] Non-PEFT frozen (n params), PEFT 'lora_' trainable (n params)   (main.py:329)
    [Training] Task t (...) | Reusing cached trained checkpoint for task t | Epoch e/E, ...
    [Merging] pspectral {variant} (p=.., alpha=..) of {file} + task t         (main.py:1542)
    [Merging] SOTA {method} (lamb=..) of {file} + task t                      (main.py:1588)
    [Merging] Merging N task checkpoints                                      (main.py:1633)
    [Drift] W shape (d, d), ||W||=.., ||W-I||=.., mean cos(W.f_(t-1), f_t)=..
            over N samples (lambda=.., ridge=..[, solver=..])                (main.py:1216)
    [Drift] lambda_rule=gcv chosen lambda=.. | gcv 1:.. 3:.. ...  -> run['drift_gcv'] (E7b)
    [Merging] opcm alpha=.. lam_t=.. ... / [Merging] nufilt rp=.. adapt=.. ...
                                                   -> run['merge_state'] (E15a state lines)
    [Evaluation] Task t / Total Acc: x, Grouped: [..] / FAA: .., FFM: .., FFD: .., ASA: ..
                                                                              (main.py:1035-1081)
    [Evaluation-val] ...  (same two lines on the validation split, E14)
    Time: Xs                                                                  (main.py:2003)
    SUMMARY  {DS} -- {CFG}                                                    (main.py:2011)
      ASA: m +- s | FAA: m +- s | FFM: m +- s        (np.std, ddof=0; main.py:2014)
      [val] ASA: ... (E14, optional)
    SWEEP SUMMARY [sota-{ds}] {name}: FA m +- s | AA m +- s    (run_sota_merge.py:55)
    [Experiment {ds}_{cfg}] {ExcType}: {msg}         (failed seed, main.py:1966; its FA
                                                      enters the SUMMARY as 0.0)

Rules that matter for correctness:
  * Files are opened with encoding='utf-8' (em dashes, the plus-minus sign, the Greek
    lambda and the middle dot are all in the logs; the locale default is cp1252 here).
  * Logs are opened in append mode by the runner, so a config name run twice has two
    blocks per seed.  ``select_runs`` keeps the LAST block with status ``ok`` per
    (dataset, config, seed), else the last block (so a crashed re-run is reported).
  * ``n_tasks_expected`` replicates DataManager.__init__ (utils/data_manager.py:15-20)
    from dataset_init_cls/dataset_increment and TOTAL_CLASSES (main.py:1741); never a
    hard-coded 10.
  * FA/AA/FF are recomputed from the matrix and cross-checked against the last logged
    ``[Evaluation] FAA`` line (``metrics_match``).  Std is ddof=0 like the SUMMARY lines.
  * Failed seeds (error line, or FA == 0.0) get status ``crashed`` and are never
    averaged; ``aggregate`` reports them in ``n_failed`` / ``failed_seeds``.
  * The per-task ``Total Acc`` (sample-weighted) curve is kept as ``totals`` so that
    ``last_total`` / ``inc_total`` can be reported where the baseline tables are
    sample-weighted (E1 long splits, DIL).
  * Lines whose tag is not parsed here (``[MergeStat]``, ``[StepNorm]``, ``[FeatShift]``,
    ``[Timing]``, ``[DriftDiag]``, ``[Data]``, ``[L2M]``, ``[SpecNorm]``, ``[InfLoRA]``,
    ``[Evaluation-til]`` ...) are kept verbatim in ``run['tags'][tag]`` so later
    subcommands can parse them without touching the core.

CLI
    python parse_logs.py [--roots logs_main ...] [--glob PATTERN ...]
                         [--per-seed out.csv] [--agg out.csv] [--keys dataset config_name]
    python parse_logs.py e1 --roots logs_main --out-dir figures_data      (long splits)
    python parse_logs.py e2 --roots logs_main logs_exp21 --out-dir figures_data [--validate]
    python parse_logs.py e3|e5|e7|e8|e9|e10|e11|e12|e13|e14|e15 --help   (rebuttal tables;
                         registry at the bottom; every one writes figures_data/e<N>_*.csv)
"""
import argparse
import ast
import csv
import glob
import os
import re
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Constants of the code base, read from rebuttal_common (torch-free): the one
# copy of the posture tables that the run scripts build their configs from and
# that rebuttal_common.check_posture() verifies against run_best / run_sota_merge
# on the server (plan ground rule 1: postures are copied once).
# ---------------------------------------------------------------------------
import rebuttal_common as _rc   # noqa: E402  (stdlib only; see its module docstring)

SEEDS = tuple(_rc.SEEDS)

# main.TOTAL_CLASSES (needed to replicate DataManager's task count from the split keys).
TOTAL_CLASSES = dict(_rc.TOTAL_CLASSES)

# Headline posture per dataset: run_best.CONFIGS (train_merge_alpha, train_drift_lambda)
# and run_best.BASE (pspectral, only_A, p 0.9) via rebuttal_common.POSTURE / COMMON.
HEADLINE = {ds: {"alpha": _rc.HEADLINE_ALPHA[ds], "lambda": _rc.HEADLINE_LAMBDA[ds]}
            for ds in _rc.DATASETS}
_HEADLINE_PS = {_rc.POSTURE[ds]["pspectral_p"] for ds in _rc.DATASETS}
assert len(_HEADLINE_PS) == 1, f"one headline p expected, got {_HEADLINE_PS}"
HEADLINE_P = _HEADLINE_PS.pop()
HEADLINE_VARIANT = _rc.COMMON["spectral_variant"]
HEADLINE_TARGETS = tuple(_rc.COMMON["model_lora_target_modules"])
HEADLINE_RANK = _rc.COMMON["model_lora_r"]

DEFAULT_ROOTS = ("logs_main",)

# Metrics that aggregate() averages when present in the rows.
DEFAULT_METRICS = ("faa", "asa", "ffm", "ffd", "last_total", "inc_total",
                   "old_mean", "new_acc", "first5_mean",
                   "faa_val", "asa_val", "ffm_val", "last_total_val", "inc_total_val",
                   "dil_last", "dil_inc", "time_s")

# Columns written with two decimals (everything else keeps its Python repr).
METRIC_COLS = set(DEFAULT_METRICS) | {
    "fa", "aa", "ffm", "fa_sd", "aa_sd", "ffm_sd", "faa_logged", "ffm_logged",
    "ffd_logged", "asa_logged", "faa_val_logged", "acc_mean", "acc_sd", "total_mean",
    "total_sd", "last", "last_sd", "inc", "inc_sd", "summary_fa", "summary_aa",
    "summary_ffm", "best_fa", "headline_fa", "time_mean", "time_min", "time_max",
    "hours_mean", "cube_fa", "fa_cube"}

# High-volume tags that are parsed (or deliberately dropped) instead of captured raw.
TAG_SKIP = frozenset({"Evaluation", "Evaluation-val", "Training", "Alignment", "Merging",
                      "Drift", "Experiment", "E2", "E2 CA", "Projection", "Optimizer",
                      "Cleanup", "Export", "StateMerge", "Stats", "EarlyStop"})

# ---------------------------------------------------------------------------
# Regex set (one per log line format listed in the module docstring)
# ---------------------------------------------------------------------------
PREFIX_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{1,6})?) \[([^\]]*)\] => (.*)$")
_DASH = r"\s[—–-]+\s"          # ' -- ' written as an em dash (U+2014) by main.py
EXPERIMENT_RE = re.compile(r"^Experiment: (\S+)" + _DASH + r"(.+?)" + _DASH + r"seed (\d+)\s*$")
CONFIG_LINE_RE = re.compile(r"^  ([^\s:]+): (.*)$")
_NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf)"
GROUPED_RE = re.compile(r"^\[Evaluation(-val)?\] Total Acc: (" + _NUM + r"), Grouped: \[([^\]]*)\]")
METRICS_RE = re.compile(r"^\[Evaluation(-val)?\] FAA: (" + _NUM + r"), FFM: (" + _NUM
                        + r"), FFD: (" + _NUM + r"), ASA: (" + _NUM + r")")
EVAL_TASK_RE = re.compile(r"^\[Evaluation\] Task (\d+)")
DIL_RE = re.compile(r"^\[Evaluation\] DIL Last-Acc: (" + _NUM + r"), Inc-Acc: (" + _NUM + r")")
DRIFT_RE = re.compile(
    r"^\[Drift\] W shape \((\d+), (\d+)\), \|\|W\|\|=(" + _NUM + r"), \|\|W-I\|\|=(" + _NUM
    + r"), mean cos\(.*?\)=(" + _NUM + r") over (\d+) samples "
    r"\((?:λ|lambda|lam)=([^,)]+), ridge=([^,)]+)(?:, ([^)]*))?\)")
MERGE_PSPEC_RE = re.compile(r"^\[Merging\] pspectral (\S+) \(p=(" + _NUM + r"), alpha=(" + _NUM + r")\)")
MERGE_KSPEC_RE = re.compile(r"^\[Merging\] kspectral (\S+) \(k=(\d+), alpha=(" + _NUM + r")\)")
MERGE_SOTA_RE = re.compile(r"^\[Merging\] SOTA (\w+) \(lamb=(" + _NUM + r")\)")
MERGE_N_RE = re.compile(r"^\[Merging\] Merging (\d+) task checkpoints")
MERGE_ADAPT_RE = re.compile(r"^\[Merging\] adaptive alpha (.*?)->\s*alpha=(" + _NUM + r")")
# '[Merging] opcm alpha=.. lam_t=.. ...' / '[Merging] nufilt rp=.. adapt=.. ...' (main._log_merge_rule_state, E15a)
MERGE_STATE_RE = re.compile(r"^\[Merging\] (opcm|nufilt) (.+)$")
# '[Drift] lambda_rule=gcv chosen lambda={lam} | gcv {lam:g}:{score:.6g} ...' (main.fit_drift_map, E7b)
DRIFT_GCV_RE = re.compile(r"^\[Drift\] lambda_rule=gcv chosen lambda=(\S+) \| gcv (.*)$")
TIME_RE = re.compile(r"^Time: (" + _NUM + r")s")
ERROR_RE = re.compile(r"^\[Experiment ([^\]]*)\] (\w+): (.*)$")
SUMMARY_HDR_RE = re.compile(r"^SUMMARY\s+(\S+)" + _DASH + r"(.+?)\s*$")
_PM = r"(?:±|\+-|\+/-)"
SUMMARY_LINE_RE = re.compile(
    r"^\s*(\[val\] )?ASA: (" + _NUM + r") " + _PM + r" (" + _NUM + r") \| FAA: (" + _NUM
    + r") " + _PM + r" (" + _NUM + r") \| FFM: (" + _NUM + r") " + _PM + r" (" + _NUM + r")")
SWEEP_RE = re.compile(r"^SWEEP SUMMARY \[([^\]]+)\] (\S+?): (.+)$")
SWEEP_BEST_RE = re.compile(r"^SWEEP SUMMARY \[([^\]]+)\] (\S+) best=(\S+) FA (" + _NUM + r")")
SWEEP_PAIR_RE = re.compile(r"([A-Za-z][A-Za-z ]*?) (" + _NUM + r") " + _PM + r" (" + _NUM + r")")
FREEZE_RE = re.compile(r"^\[Freeze\] Non-PEFT frozen \(([\d,]+) params\), PEFT '([^']*)' trainable \(([\d,]+) params\)")
CACHED_RE = re.compile(r"^\[Training\] Reusing cached trained checkpoint for task (\d+)")
EPOCH_RE = re.compile(r"^\[Training\] Epoch (\d+)/(\d+)")
EARLYSTOP_RE = re.compile(r"^\[EarlyStop\] Pruned at task (\d+)")
TAG_RE = re.compile(r"^\[([^\]]+)\]")
STAGE_RE = re.compile(
    r"^(\[Training\] Task \d+|\[Training\] Reusing cached trained checkpoint for task \d+|"
    r"\[Training\] Epoch \d+/\d+|\[Merging\] Task \d+|\[Merging\] Saving merged checkpoint for task \d+|"
    r"\[Drift\] Fitting ridge drift map for task \d+|\[Drift\] W shape|\[Drift\] Transporting|"
    r"\[Alignment\] Task \d+|\[Alignment\] Epoch \d+/\d+|\[Evaluation\] Task \d+|Time: )")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def literal(value):
    """ast.literal_eval of a '  key: value' dump value with a raw-string fallback.

    main.py:1915 writes ``f"  {key}: {value}"``, so strings come out unquoted
    ('seqft', 'only_A', 'main_inr_r64_qkvfc') and literal_eval raises on them.
    """
    s = value.strip()
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return s


def parse_ts(ts):
    """'YYYY-MM-DD HH:MM:SS,mmm' -> datetime (None if unparsable)."""
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def task_increments(total, init_cls, increment):
    """Verbatim port of DataManager.__init__ (utils/data_manager.py:15-20)."""
    incs = [init_cls]
    while sum(incs) + increment < total:
        incs.append(increment)
    offset = total - sum(incs)
    if offset > 0:
        incs.append(offset)
    return incs


def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def expected_tasks(cfg, dataset=None):
    """Number of tasks the run should have evaluated, from the Configuration dump."""
    ds = cfg.get("dataset_name", dataset)
    init, inc = cfg.get("dataset_init_cls"), cfg.get("dataset_increment")
    total = TOTAL_CLASSES.get(ds)
    n = None
    if cfg.get("joint"):
        n = 1
    elif total and _is_int(init) and _is_int(inc) and init > 0 and inc > 0:
        n = len(task_increments(total, init, inc))
    elif _is_int(cfg.get("dataset_num_task")):
        n = int(cfg["dataset_num_task"])
    stop = cfg.get("train_stop_at_task")
    if n is not None and _is_int(stop) and stop >= 0:      # smoke runs: BASE default is -1
        n = min(n, stop + 1)
    return n


def split_desc(cfg, dataset=None):
    """'10x10', '16+9x20', '50x2' ... from the split keys (None if unknown)."""
    ds = cfg.get("dataset_name", dataset)
    init, inc = cfg.get("dataset_init_cls"), cfg.get("dataset_increment")
    total = TOTAL_CLASSES.get(ds)
    if not (total and _is_int(init) and _is_int(inc) and init > 0 and inc > 0):
        return None
    incs = task_increments(total, init, inc)
    if len(set(incs)) == 1:
        return f"{len(incs)}x{incs[0]}"
    return f"{incs[0]}+{len(incs) - 1}x{incs[1]}"


def metrics_from_matrix(mat):
    """numpy port of helper.compute_metrics (helper.py:22-43): FAA, FFM, FFD, ASA.

    ``mat`` is the (T, T) accuracy matrix with zeros above the diagonal, exactly the
    array Learner.eval builds from the Grouped rows (main.py:1073-1077).
    """
    mat = np.asarray(mat, dtype=float)
    faa = float(np.mean(mat[-1]))
    session_averages = [float(np.mean(mat[i, : i + 1])) for i in range(mat.shape[0])]
    asa = float(np.mean(session_averages))
    if mat.shape[0] == 1:
        return faa, 0.0, 0.0, asa
    final_acc_per_task = mat[-1]
    max_acc_per_task = np.max(mat, axis=0)
    forget = max_acc_per_task[:-1] - final_acc_per_task[:-1]
    ffm = float(np.mean(forget))
    ffd = float(np.max(forget) - np.min(forget))
    return faa, ffm, ffd, asa


def matrix_from_rows(rows):
    """Grouped rows (row i holds i+1 entries) -> (T, T) matrix, zeros above the diagonal."""
    n = len(rows)
    mat = np.zeros((n, n), dtype=float)
    for i, row in enumerate(rows):
        k = min(len(row), n)
        mat[i, :k] = row[:k]
    return mat


def final_row_split(row):
    """old_mean / new_acc / first5_mean of the final Grouped row (export_mc3d_full.parse)."""
    v = list(row)
    if not v:
        return None, None, None
    old = float(np.mean(v[:-1])) if len(v) > 1 else float(v[0])
    first5 = float(np.mean(v[:5]))
    return old, float(v[-1]), first5


def _hashable(v):
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, np.generic):
        return v.item()
    return v


def get_field(run, key):
    """A run field if present, else the config value of that key, else None."""
    if key in run:
        return run[key]
    return (run.get("config") or {}).get(key)


# ---------------------------------------------------------------------------
# Reading and segmenting a log file
# ---------------------------------------------------------------------------
def split_prefix(raw):
    """One raw line -> (timestamp or None, source file or None, message)."""
    m = PREFIX_RE.match(raw)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, raw


def read_lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [split_prefix(line.rstrip("\r\n")) for line in fh]


def segment_blocks(lines):
    """Cut the (ts, src, msg) line list into run blocks.

    A block starts at an ``Experiment: ds -- cfg -- seed N`` header (run_config_sweep)
    or, when no header precedes it, at a ``Configuration:`` line (run_sota_merge-style
    logs).  Each block is a dict with mode, dataset, config_name, seed, start, end and
    the line slice.
    """
    blocks, cur = [], None

    def close(end):
        if cur is not None:
            cur["end"] = end
            cur["lines"] = lines[cur["start"]:end]
            blocks.append(cur)

    for i, (_ts, _src, msg) in enumerate(lines):
        m = EXPERIMENT_RE.match(msg)
        if m:
            close(i)
            cur = {"mode": "experiment", "dataset": m.group(1),
                   "config_name": m.group(2).strip(), "seed": int(m.group(3)),
                   "start": i, "saw_config": False}
            continue
        if msg.strip() == "Configuration:":
            if cur is None or cur["mode"] == "configuration" or cur["saw_config"]:
                close(i)
                cur = {"mode": "configuration", "dataset": None, "config_name": None,
                       "seed": None, "start": i, "saw_config": True}
            else:
                cur["saw_config"] = True
    close(len(lines))
    return blocks


def split_seed_blocks(lines_or_text, dedup=True, path=None):
    """Segment a log and (by default) keep the LAST complete block per (config, seed).

    Accepts the raw text or the (ts, src, msg) line list.  With ``dedup`` the blocks are
    parsed and reduced through ``select_runs``; without it the raw segments are returned.
    """
    if isinstance(lines_or_text, str):
        lines = [split_prefix(line) for line in lines_or_text.splitlines()]
    else:
        lines = lines_or_text
    segs = segment_blocks(lines)
    if not dedup:
        return segs
    return select_runs(_parse_segments(segs, lines, path))


def _parse_segments(segs, lines, path=None, capture_tags=None):
    """parse_block over the segments, then name Configuration-only blocks from the
    SWEEP SUMMARY lines that follow them."""
    runs = []
    for i, seg in enumerate(segs):
        run = parse_block(seg["lines"], dataset=seg["dataset"], config_name=seg["config_name"],
                          seed=seg["seed"], mode=seg["mode"], path=path, block_index=i,
                          capture_tags=capture_tags)
        run["_start"], run["_end"] = seg["start"], seg["end"]
        runs.append(run)
    _name_configuration_runs(runs, lines)
    for run in runs:
        run.pop("_start", None)
        run.pop("_end", None)
    return runs


# ---------------------------------------------------------------------------
# One block -> one run dict
# ---------------------------------------------------------------------------
def _floats(csv_text):
    return [float(x) for x in csv_text.split(",") if x.strip()]


def parse_block(lines, dataset=None, config_name=None, seed=None, mode="experiment",
                path=None, block_index=0, capture_tags=None):
    """Parse one seed block into a flat run dict (see the module docstring).

    ``capture_tags``: iterable of tag names to keep verbatim; None keeps every bracketed
    tag that is not in TAG_SKIP (MergeStat, StepNorm, FeatShift, Timing, DriftDiag, Data,
    L2M, Freeze, SpecNorm, InfLoRA, Evaluation-til, ...).
    """
    capture = set(capture_tags) if capture_tags is not None else None
    cfg, in_config = {}, False
    rows, rows_val, totals, totals_val = [], [], [], []
    logged, logged_val = None, None
    drift, n_ckpts, alphas, adaptive = [], [], [], []
    merge_state, drift_gcv = [], []
    merge = {"rule": None, "variant": None, "p": None, "alpha": None, "k": None,
             "sota_method": None, "coef": None}
    errors, events = [], []
    tags = defaultdict(list)
    time_s, dil_last, dil_inc, pruned_at = None, None, None, None
    trainable, frozen, peft_key = None, None, None
    cached_tasks, trained_tasks, n_epoch_lines = set(), 0, 0
    first_ts = last_ts = None

    for ts, _src, msg in lines:
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
        if in_config:
            m = CONFIG_LINE_RE.match(msg)
            if m:
                cfg[m.group(1)] = literal(m.group(2))
                continue
            in_config = False
        s = msg.strip()
        if not s:
            continue
        if s == "Configuration:":
            in_config, cfg = True, {}
            continue
        m = GROUPED_RE.match(s)
        if m:
            vals = _floats(m.group(3))
            if m.group(1):
                rows_val.append(vals)
                totals_val.append(float(m.group(2)))
            else:
                rows.append(vals)
                totals.append(float(m.group(2)))
            continue
        m = METRICS_RE.match(s)
        if m:
            rec = {"faa": float(m.group(2)), "ffm": float(m.group(3)),
                   "ffd": float(m.group(4)), "asa": float(m.group(5))}
            if m.group(1):
                logged_val = rec
            else:
                logged = rec
            continue
        if s.startswith("Time: "):
            m = TIME_RE.match(s)
            if m:
                time_s = float(m.group(1))
                events.append((ts, "Time:"))
            continue
        m = ERROR_RE.match(s)
        if m:
            errors.append(f"{m.group(2)}: {m.group(3)}")
            continue
        m = STAGE_RE.match(s)
        if m:
            events.append((ts, m.group(1)))
        if s.startswith("[Drift] lambda_rule="):
            m = DRIFT_GCV_RE.match(s)
            if m:
                scores = OrderedDict()
                for tok in m.group(2).split():
                    if ":" in tok:
                        k, v = tok.split(":", 1)
                        try:
                            scores[float(k)] = float(v)
                        except ValueError:
                            continue
                drift_gcv.append({"lambda": literal(m.group(1)), "scores": scores})
            continue
        if s.startswith("[Drift] W shape"):
            m = DRIFT_RE.match(s)
            if m:
                drift.append({"d": int(m.group(1)), "w_norm": float(m.group(3)),
                              "w_minus_i": float(m.group(4)), "cos": float(m.group(5)),
                              "n": int(m.group(6)), "lambda": literal(m.group(7)),
                              "ridge": m.group(8).strip(), "extra": m.group(9)})
            continue
        if s.startswith("[Merging]"):
            m = MERGE_PSPEC_RE.match(s)
            if m:
                merge.update(rule="pspectral", variant=m.group(1), p=float(m.group(2)))
                if merge["alpha"] is None:
                    merge["alpha"] = float(m.group(3))
                alphas.append(float(m.group(3)))
                continue
            m = MERGE_KSPEC_RE.match(s)
            if m:
                merge.update(rule="kspectral", variant=m.group(1), k=int(m.group(2)))
                if merge["alpha"] is None:
                    merge["alpha"] = float(m.group(3))
                alphas.append(float(m.group(3)))
                continue
            m = MERGE_SOTA_RE.match(s)
            if m:
                merge.update(rule="sota", sota_method=m.group(1), coef=float(m.group(2)))
                continue
            m = MERGE_N_RE.match(s)
            if m:
                merge["rule"] = merge["rule"] or "classical"
                n_ckpts.append(int(m.group(1)))
                continue
            m = MERGE_ADAPT_RE.match(s)
            if m:
                adaptive.append((m.group(1).strip(), float(m.group(2))))
                continue
            m = MERGE_STATE_RE.match(s)
            if m:                      # kv_pairs is defined with the subcommand helpers below
                merge_state.append((m.group(1), kv_pairs(m.group(2))))
            continue
        if s.startswith("[Training]"):
            m = CACHED_RE.match(s)
            if m:
                cached_tasks.add(int(m.group(1)))
                continue
            m = EPOCH_RE.match(s)
            if m:
                n_epoch_lines += 1
                if m.group(1) == "1":
                    trained_tasks += 1
            continue
        if s.startswith("[Evaluation]"):
            m = DIL_RE.match(s)
            if m:
                dil_last, dil_inc = float(m.group(1)), float(m.group(2))
            continue
        if s.startswith("[EarlyStop]"):
            m = EARLYSTOP_RE.match(s)
            if m:
                pruned_at = int(m.group(1))
            continue
        if s.startswith("[Freeze]"):
            m = FREEZE_RE.match(s)
            if m:
                frozen = int(m.group(1).replace(",", ""))
                peft_key = m.group(2)
                trainable = int(m.group(3).replace(",", ""))
        m = TAG_RE.match(s)
        if m:
            tag = m.group(1)
            if (capture is None and tag not in TAG_SKIP
                    and not re.match(r"^(Before Task|Task) \d+$", tag)) or \
                    (capture is not None and tag in capture):
                tags[tag].append(s)

    # --- identity -------------------------------------------------------------
    if dataset is None:
        dataset = cfg.get("dataset_name")
    if seed is None:
        seed = cfg.get("seed")
    name_inferred = False
    if config_name is None:
        config_name, name_inferred = _inferred_name(cfg), True

    # --- metrics ---------------------------------------------------------------
    run = OrderedDict()
    run.update(path=path, file=os.path.basename(path) if path else None,
               block_index=block_index, mode=mode, dataset=dataset,
               config_name=config_name, name_inferred=name_inferred, seed=seed,
               config=cfg, n_tasks=len(rows), n_tasks_expected=expected_tasks(cfg, dataset),
               split=None, split_desc=split_desc(cfg, dataset))
    run["split"] = f"t{run['n_tasks_expected']}" if run["n_tasks_expected"] else None
    run["rows"], run["totals"] = rows, totals
    run["rows_val"], run["totals_val"] = rows_val, totals_val
    _fill_metrics(run, rows, totals, logged, suffix="")
    _fill_metrics(run, rows_val, totals_val, logged_val, suffix="_val")
    run["dil_last"], run["dil_inc"] = dil_last, dil_inc

    # --- drift / merge / training facts -----------------------------------------
    run["drift"] = drift
    run["drift_lambda"] = drift[-1]["lambda"] if drift else None
    run["drift_ridge"] = drift[-1]["ridge"] if drift else None
    run["drift_n"] = [d["n"] for d in drift]
    run["drift_w_minus_i"] = [d["w_minus_i"] for d in drift]
    run["drift_cos"] = [d["cos"] for d in drift]
    run["merge_rule"] = merge["rule"]
    run["merge_variant"] = merge["variant"]
    run["merge_p"] = merge["p"]
    run["merge_alpha"] = merge["alpha"]
    run["merge_alphas"] = alphas
    run["merge_k"] = merge["k"]
    run["merge_sota_method"] = merge["sota_method"]
    run["merge_coef"] = merge["coef"]
    run["merge_n_ckpts"] = n_ckpts
    # [Merging] Merging N task checkpoints: incremental pairwise merges always report 2;
    # the non-incremental branch reports t+1, so the test is only decisive from task 2.
    run["merge_incremental_observed"] = (max(n_ckpts) == 2) if len(n_ckpts) >= 2 else None
    run["adaptive_alphas"] = adaptive
    run["merge_state"] = merge_state           # [(method, {lam_t, kept_frac, ...})] per opcm/nufilt merge
    run["drift_gcv"] = drift_gcv               # [{'lambda', 'scores'}] per gcv-chosen drift fit
    run["time_s"] = time_s
    run["first_ts"], run["last_ts"] = first_ts, last_ts
    run["trainable_params"], run["frozen_params"], run["peft_key"] = trainable, frozen, peft_key
    run["n_cached_tasks"], run["n_trained_tasks"] = len(cached_tasks), trained_tasks
    run["n_epoch_lines"] = n_epoch_lines
    run["pruned_at"] = pruned_at
    run["errors"] = errors
    run["tags"] = dict(tags)
    run["events"] = events

    # --- status ------------------------------------------------------------------
    status, reason = "ok", None
    if errors:
        status, reason = "crashed", errors[0]
    elif pruned_at is not None:
        status, reason = "pruned", f"pruned at task {pruned_at}"
    elif not rows:
        status, reason = "incomplete", "no Grouped lines"
    elif run["n_tasks_expected"] is not None and len(rows) < run["n_tasks_expected"]:
        status, reason = "incomplete", f"{len(rows)}/{run['n_tasks_expected']} tasks"
    elif run["faa"] is not None and run["faa"] == 0.0:
        status, reason = "crashed", "FA 0.0"
    run["status"], run["fail_reason"] = status, reason
    run["n_blocks"] = 1
    return run


def _fill_metrics(run, rows, totals, logged, suffix):
    keys = ("faa", "ffm", "ffd", "asa")
    if rows:
        faa, ffm, ffd, asa = metrics_from_matrix(matrix_from_rows(rows))
        vals = dict(zip(keys, (faa, ffm, ffd, asa)))
        old, new, first5 = final_row_split(rows[-1])
        run["last_total" + suffix] = totals[-1] if totals else None
        run["inc_total" + suffix] = float(np.mean(totals)) if totals else None
    else:
        vals = dict.fromkeys(keys)
        old = new = first5 = None
        run["last_total" + suffix] = run["inc_total" + suffix] = None
    for k in keys:
        run[k + suffix] = vals[k]
        run[k + suffix + "_logged"] = logged[k] if logged else None
    run["old_mean" + suffix], run["new_acc" + suffix], run["first5_mean" + suffix] = old, new, first5
    if rows and logged:
        run["metrics_match" + suffix] = all(
            abs(vals[k] - logged[k]) <= 0.0101 for k in keys if np.isfinite(vals[k]))
    else:
        run["metrics_match" + suffix] = None


def _inferred_name(cfg):
    """Config name for a block without an Experiment header and no SWEEP SUMMARY after it."""
    rule = cfg.get("train_merge", "?")
    method = cfg.get("train_method", "seqft")
    bits = [f"cfg:{method}" if method != "seqft" else "cfg", str(rule)]
    if rule == "pspectral":
        bits.append(f"{cfg.get('spectral_variant')}_p{cfg.get('pspectral_p')}_a{cfg.get('train_merge_alpha')}")
    elif rule not in ("none", None):
        bits.append(f"c{cfg.get('train_merge_coef')}")
    return "_".join(bits)


# ---------------------------------------------------------------------------
# File level: SUMMARY / SWEEP SUMMARY blocks, naming, selection, cross-checks
# ---------------------------------------------------------------------------
def parse_summaries(lines):
    """'SUMMARY  DS -- CFG' blocks -> {(ds_lower, cfg_lower): {asa, asa_sd, ..., val}}.

    The last block for a config wins (append-mode re-runs).
    """
    out = {}
    for i, (_ts, _src, msg) in enumerate(lines):
        m = SUMMARY_HDR_RE.match(msg.strip())
        if not m:
            continue
        rec = {"dataset": m.group(1), "config_name": m.group(2).strip(), "line": i}
        for _ts2, _src2, msg2 in lines[i + 1: i + 8]:
            m2 = SUMMARY_LINE_RE.match(msg2)
            if not m2:
                continue
            vals = {"asa": float(m2.group(2)), "asa_sd": float(m2.group(3)),
                    "faa": float(m2.group(4)), "faa_sd": float(m2.group(5)),
                    "ffm": float(m2.group(6)), "ffm_sd": float(m2.group(7))}
            if m2.group(1):
                rec["val"] = vals
            else:
                rec.update(vals)
        out[(rec["dataset"].lower(), rec["config_name"].lower())] = rec
    return out


def parse_sweep_summaries(lines):
    """run_sota_merge / run_inflora 'SWEEP SUMMARY [...]' lines -> list of dicts."""
    out = []
    for i, (_ts, _src, msg) in enumerate(lines):
        s = msg.strip()
        m = SWEEP_BEST_RE.match(s)
        if m:
            out.append({"line": i, "group": m.group(1), "kind": "best", "method": m.group(2),
                        "best": m.group(3), "fa": float(m.group(4)), "name": None,
                        "dataset": None, "metrics": {}})
            continue
        m = SWEEP_RE.match(s)
        if not m:
            continue
        group = m.group(1)
        ds = group.split("-", 1)[1] if "-" in group else None
        metrics = OrderedDict()
        for mm in SWEEP_PAIR_RE.finditer(m.group(3)):
            metrics[mm.group(1).strip()] = (float(mm.group(2)), float(mm.group(3)))
        out.append({"line": i, "group": group, "kind": "cell", "dataset": ds,
                    "name": m.group(2), "metrics": metrics})
    return out


def _name_matches(name, cfg):
    if name.startswith("sota_"):
        return str(cfg.get("train_merge")) in name
    return True


def _name_configuration_runs(runs, lines):
    """Give Configuration-only blocks the name of the SWEEP SUMMARY line that follows them.

    Blocks that START between two consecutive SWEEP SUMMARY cell lines belong to the
    second line's cell when the dataset matches (and, for 'sota_*' names, the merge rule
    is in the name).  The block's end index is not used: a block only closes at the next
    'Configuration:' line, which lies beyond its own SWEEP SUMMARY line.
    """
    sweeps = [s for s in parse_sweep_summaries(lines) if s["kind"] == "cell"]
    prev = -1
    for sw in sweeps:
        for run in runs:
            if run["mode"] != "configuration" or not run["name_inferred"]:
                continue
            if prev < run["_start"] < sw["line"] and \
                    (sw["dataset"] is None or run["dataset"] == sw["dataset"]) and \
                    _name_matches(sw["name"], run["config"]):
                run["config_name"], run["name_inferred"] = sw["name"], False
                run["sweep"] = sw["metrics"]
        prev = sw["line"]


def select_runs(runs):
    """Keep one run per (dataset, config_name, seed): the last ``ok`` block, else the last."""
    groups = OrderedDict()
    for r in runs:
        groups.setdefault((r["dataset"], r["config_name"], r["seed"]), []).append(r)
    out = []
    for rs in groups.values():
        ok = [r for r in rs if r["status"] == "ok"]
        chosen = ok[-1] if ok else rs[-1]
        chosen["n_blocks"] = len(rs)
        chosen["block_statuses"] = [r["status"] for r in rs]
        stamps = [r["first_ts"] for r in rs if r["first_ts"]]
        chosen["first_seen_ts"] = min(stamps) if stamps else chosen["first_ts"]
        stamps = [r["last_ts"] for r in rs if r["last_ts"]]
        chosen["last_seen_ts"] = max(stamps) if stamps else chosen["last_ts"]
        out.append(chosen)
    return out


def check_summary(runs_of_config, summary, tol=0.0101):
    """Compare the SUMMARY line with mean/std (ddof=0) over the selected seeds.

    Failed seeds enter the real SUMMARY as 0.0 (main.py:1968), so they count as 0.0 here.
    """
    if not summary or "faa" not in summary:
        return "n/a"
    vals = {}
    for k in ("faa", "ffm", "asa"):
        vals[k] = [r[k] if (r["status"] == "ok" and r[k] is not None) else 0.0
                   for r in runs_of_config]
    for k in ("faa", "ffm", "asa"):
        if abs(float(np.mean(vals[k])) - summary[k]) > tol or \
                abs(float(np.std(vals[k])) - summary[k + "_sd"]) > tol:
            return "mismatch"
    return "match"


def parse_file_full(path, capture_tags=None):
    """(selected runs, all blocks, summaries, sweep lines) of one log file."""
    lines = read_lines(path)
    runs = _parse_segments(segment_blocks(lines), lines, path, capture_tags)
    summaries = parse_summaries(lines)
    sweeps = parse_sweep_summaries(lines)
    selected = select_runs(runs)
    by_cfg = defaultdict(list)
    for r in selected:
        by_cfg[(str(r["dataset"]).lower(), str(r["config_name"]).lower())].append(r)
    # Experiment-mode runs also get the metrics of a 'SWEEP SUMMARY [...] {name}: ...' line
    # that names their config (run_inflora.py writes one per cell; the last line wins).
    cells = {}
    for sw in sweeps:
        if sw["kind"] == "cell":
            cells[(str(sw["dataset"]).lower() if sw["dataset"] else None, str(sw["name"]).lower())] = sw["metrics"]
    for key, rs in by_cfg.items():
        summ = summaries.get(key)
        verdict = check_summary(rs, summ)
        for r in rs:
            r["summary"] = summ
            r["summary_check"] = verdict
            if "sweep" not in r:
                r["sweep"] = cells.get(key, cells.get((None, key[1])))
    return selected, runs, summaries, sweeps


def parse_file(path, capture_tags=None):
    """list of per-seed run dicts of one log (last complete block per config and seed)."""
    return parse_file_full(path, capture_tags=capture_tags)[0]


def parse_glob(pattern, capture_tags=None):
    """parse_file over every file matching a glob (recursive ``**`` allowed)."""
    runs = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        if os.path.isfile(path):
            runs.extend(parse_file(path, capture_tags=capture_tags))
    return runs


def load_runs(roots=None, globs=None, capture_tags=None, skip_smoke=False, verbose=True):
    """Every selected run under the given roots (recursive *.log) and glob patterns."""
    files = []
    for root in roots or ():
        if not os.path.isdir(root):
            if verbose:
                print(f"[parse_logs] note: root {root!r} does not exist, skipped", file=sys.stderr)
            continue
        for dirpath, _dirs, names in os.walk(root):
            files.extend(os.path.join(dirpath, n) for n in names if n.endswith(".log"))
    for pattern in globs or ():
        files.extend(p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p))
    seen, runs = set(), []
    for path in sorted(files):
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        if skip_smoke and "smoke" in os.path.normpath(path).split(os.sep):
            continue
        runs.extend(parse_file(path, capture_tags=capture_tags))
    return runs


# ---------------------------------------------------------------------------
# Aggregation, CSV and markdown output
# ---------------------------------------------------------------------------
def aggregate(rows, keys, metrics=None, ddof=0, require_ok=True):
    """Group runs by ``keys`` (run fields or config keys) and average the metrics.

    Only runs with status 'ok' enter the means (``require_ok``); the others are
    counted in ``n_failed`` / ``failed_seeds``.  Std uses ``ddof`` (0 = SUMMARY lines).
    """
    metrics = list(metrics or DEFAULT_METRICS)
    present = [m for m in metrics if any(get_field(r, m) is not None for r in rows)]
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(tuple(_hashable(get_field(r, k)) for k in keys), []).append(r)
    out = []
    for key, rs in groups.items():
        ok = [r for r in rs if r.get("status", "ok") == "ok"] if require_ok else list(rs)
        ok_ids = {id(r) for r in ok}
        row = OrderedDict(zip(keys, key))
        row["n_seeds"] = len(ok)
        row["seeds"] = sorted(r.get("seed") for r in ok if r.get("seed") is not None)
        row["n_failed"] = len(rs) - len(ok)
        row["failed_seeds"] = sorted(r.get("seed") for r in rs
                                     if id(r) not in ok_ids and r.get("seed") is not None)
        row["n_runs"] = sum(r.get("n_blocks", 1) for r in rs)
        for m in present:
            vals = [float(get_field(r, m)) for r in ok if get_field(r, m) is not None]
            row[m] = float(np.mean(vals)) if vals else None
            row[m + "_sd"] = float(np.std(vals, ddof=ddof)) if len(vals) > ddof else None
        row["status"] = "complete" if ok and row["n_failed"] == 0 else ("partial" if ok else "failed")
        out.append(row)
    return out


def fmt_value(col, v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{v:.2f}" if (col in METRIC_COLS or col.endswith("_sd")) else f"{v:g}"
    if isinstance(v, (list, tuple)):
        return ";".join(fmt_value(col, x) for x in v)
    if isinstance(v, dict):
        return ";".join(f"{k}={fmt_value(col, x)}" for k, x in v.items())
    return str(v)


def write_csv(path, rows, cols=None):
    """Write dict rows (floats :.2f for metric columns, lists ';'-joined)."""
    if cols is None:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt_value(c, r.get(c)) for c in cols])
    return path


def pm(mean, sd):
    """'m +- s' cell text (ASCII +- so it survives any terminal encoding)."""
    if mean is None:
        return "-"
    return f"{mean:.2f}" if sd is None else f"{mean:.2f} +- {sd:.2f}"


def markdown_table(rows, cols, headers=None):
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt_value(c, r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


PER_SEED_COLUMNS = [
    "dataset", "config_name", "seed", "status", "fail_reason", "n_tasks", "n_tasks_expected",
    "split", "split_desc", "faa", "asa", "ffm", "ffd", "last_total", "inc_total", "old_mean",
    "new_acc", "first5_mean", "faa_val", "asa_val", "ffm_val", "last_total_val", "inc_total_val",
    "dil_last", "dil_inc", "metrics_match", "summary_check", "time_s", "merge_rule",
    "merge_variant", "merge_p", "merge_alpha", "merge_sota_method", "merge_coef",
    "merge_incremental_observed", "drift_lambda", "drift_ridge", "train_method",
    "train_merge", "train_prefix", "train_epochs", "trainable_params", "n_cached_tasks",
    "n_trained_tasks", "n_blocks", "first_ts", "last_ts", "errors", "path"]


def run_to_row(run, config_keys=()):
    """Flat per-seed CSV row: PER_SEED_COLUMNS (run fields or config keys) + extra keys."""
    row = OrderedDict()
    for c in PER_SEED_COLUMNS:
        row[c] = get_field(run, c)
    for k in config_keys:
        row[k] = run["config"].get(k)
    row["path"] = (run.get("path") or "").replace("\\", "/")
    return row


# ---------------------------------------------------------------------------
# Subcommand registry
# ---------------------------------------------------------------------------
SUBCOMMANDS = OrderedDict()


def subcommand(name, help_text=""):
    """Register ``fn(argv) -> int`` as ``python parse_logs.py <name> ...``."""
    def deco(fn):
        SUBCOMMANDS[name] = (fn, help_text)
        return fn
    return deco


def _common_args(p, roots=DEFAULT_ROOTS):
    p.add_argument("--roots", nargs="*", default=list(roots),
                   help="log roots walked recursively for *.log (default: %(default)s)")
    p.add_argument("--glob", dest="globs", nargs="*", default=[],
                   help="extra glob patterns (recursive ** allowed)")
    p.add_argument("--include-smoke", action="store_true",
                   help="also read logs under a 'smoke' directory (skipped by default)")


def _load(args):
    runs = load_runs(args.roots, args.globs, skip_smoke=not args.include_smoke)
    if not runs:
        print("[parse_logs] no runs found", file=sys.stderr)
    return runs


def cmd_generic(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py",
                                description="Per-seed and aggregated metrics from the logs. "
                                            "Subcommands: " + ", ".join(SUBCOMMANDS))
    _common_args(p)
    p.add_argument("--per-seed", help="write the per-seed CSV here")
    p.add_argument("--agg", help="write the aggregated CSV here")
    p.add_argument("--keys", nargs="+", default=["dataset", "config_name"],
                   help="grouping keys for --agg and the table (run fields or config keys)")
    p.add_argument("--no-table", action="store_true")
    args = p.parse_args(argv)
    runs = _load(args)
    if args.per_seed:
        write_csv(args.per_seed, [run_to_row(r) for r in runs])
        print(f"wrote {args.per_seed}: {len(runs)} runs")
    agg = aggregate(runs, args.keys)
    if args.agg:
        write_csv(args.agg, agg)
        print(f"wrote {args.agg}: {len(agg)} cells")
    if not args.no_table and agg:
        for row in agg:
            row["FA"] = pm(row.get("faa"), row.get("faa_sd"))
            row["AA"] = pm(row.get("asa"), row.get("asa_sd"))
            row["FF"] = pm(row.get("ffm"), row.get("ffm_sd"))
            row["Last/Inc (sample-wtd)"] = (
                f"{row['last_total']:.2f} / {row['inc_total']:.2f}"
                if row.get("last_total") is not None else "-")
        print(markdown_table(agg, list(args.keys) + ["n_seeds", "n_failed", "FA", "AA", "FF",
                                                      "Last/Inc (sample-wtd)", "status"]))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        fn, _help = SUBCOMMANDS[argv[0]]
        return fn(argv[1:])
    if argv and re.match(r"^e\d+[a-z]?$", argv[0]):
        print(f"unknown subcommand {argv[0]!r}; known: {', '.join(SUBCOMMANDS)}", file=sys.stderr)
        return 2
    return cmd_generic(argv)


# ===========================================================================
# Experiment subcommands.  Add new ones below as separate functions registered
# with @subcommand('eN'); they only use the API above (load_runs, aggregate,
# write_csv, markdown_table, run['tags'][...] for the experiment-specific lines).
# ===========================================================================

# --- E1: long splits (20/50 tasks) --------------------------------------------------
LONG_INCREMENTS = (2, 4, 5, 10)
LONG_DATASETS = ("cifar224", "imagenetr", "cub")
E1_AGG_KEYS = ("dataset", "split", "config_name", "train_merge_alpha", "train_drift_lambda",
               "train_epochs", "train_prefix")


def _baselines(path):
    """figures_data/e1_long_baselines.csv rows (method, dataset, split, last, last_sd, inc,
    inc_sd, source) grouped by (dataset, split); {} when the file is absent."""
    out = defaultdict(list)
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["dataset"].strip(), r["split"].strip())].append(r)
    return out


def _ref_config(cells, dataset):
    """Reference cell of a (dataset, split): headline alpha/lambda if present, else the
    cell with the most seeds (ties: earliest)."""
    head = HEADLINE.get(dataset, {})
    for c in cells:
        if c["n_seeds"] and c.get("train_merge_alpha") == head.get("alpha") and \
                c.get("train_drift_lambda") == head.get("lambda"):
            return c
    return max(cells, key=lambda c: (c["n_seeds"], -cells.index(c))) if cells else None


@subcommand("e1", "20/50-task results: per-seed/agg CSVs, trajectories, table vs baselines")
def cmd_e1(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e1", description=cmd_e1.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--baselines", default=None,
                   help="baseline CSV (default <out-dir>/e1_long_baselines.csv)")
    p.add_argument("--t10-configs", nargs="*", default=None,
                   help="10-task reference config names (default best_<ds>)")
    p.add_argument("--datasets", nargs="*", default=list(LONG_DATASETS))
    args = p.parse_args(argv)
    runs = _load(args)
    t10 = set(args.t10_configs if args.t10_configs is not None
              else [f"best_{ds}" for ds in args.datasets])
    sel = [r for r in runs if r["dataset"] in args.datasets and
           r["config"].get("dataset_increment") in LONG_INCREMENTS and
           r["n_tasks_expected"] in (20, 50)]
    ref = [r for r in runs if r["dataset"] in args.datasets and r["n_tasks_expected"] == 10
           and r["config_name"] in t10]
    rows = sel + ref
    os.makedirs(args.out_dir, exist_ok=True)
    per_seed = os.path.join(args.out_dir, "e1_long_per_seed.csv")
    write_csv(per_seed, [run_to_row(r) for r in rows])
    cells = aggregate(rows, E1_AGG_KEYS)
    for c in cells:
        c["alpha"], c["lambda"] = c.pop("train_merge_alpha"), c.pop("train_drift_lambda")
        c["epochs"], c["prefix"] = c.pop("train_epochs"), c.pop("train_prefix")
        c["complete"] = c["n_seeds"] >= 3 and c["n_failed"] == 0
        members = [r for r in rows if r["dataset"] == c["dataset"] and r["split"] == c["split"]
                   and r["config_name"] == c["config_name"]]
        summ = next((r["summary"] for r in members if r.get("summary")), None)
        c["summary_fa"] = summ["faa"] if summ else None
        c["summary_check"] = members[0]["summary_check"] if members else "n/a"
        c["files"] = sorted({r["path"] for r in members if r["path"]})
    cols = ["dataset", "split", "config_name", "alpha", "lambda", "epochs", "prefix", "n_seeds",
            "n_failed", "complete", "faa", "faa_sd", "asa", "asa_sd", "ffm", "ffm_sd",
            "last_total", "last_total_sd", "inc_total", "inc_total_sd", "old_mean", "new_acc",
            "first5_mean", "summary_fa", "summary_check", "files"]
    agg_path = os.path.join(args.out_dir, "e1_long_agg.csv")
    write_csv(agg_path, cells, cols)
    print(f"wrote {per_seed}: {len(rows)} runs; {agg_path}: {len(cells)} cells")

    # Trajectories (per-task task-equal mean of each Grouped row and the sample-weighted
    # total) of the reference cell of every (dataset, split).
    by_ds_split = OrderedDict()
    for c in cells:
        by_ds_split.setdefault((c["dataset"], c["split"]), []).append(c)
    for (ds, split), cs in by_ds_split.items():
        refc = _ref_config(cs, ds)
        if refc is None or not refc["n_seeds"]:
            continue
        ok = [r for r in rows if r["status"] == "ok" and r["dataset"] == ds and r["split"] == split
              and r["config_name"] == refc["config_name"]]
        T = min(len(r["rows"]) for r in ok)
        acc = np.array([[float(np.mean(r["rows"][t])) for t in range(T)] for r in ok])
        tot = np.array([[r["totals"][t] for t in range(T)] for r in ok])
        traj = [OrderedDict(task=t, acc_mean=float(acc[:, t].mean()), acc_sd=float(acc[:, t].std()),
                            total_mean=float(tot[:, t].mean()), total_sd=float(tot[:, t].std()),
                            n_seeds=len(ok), config_name=refc["config_name"]) for t in range(T)]
        path = os.path.join(args.out_dir, f"e1_long_traj_{ds}_{split}.csv")
        write_csv(path, traj)
        print(f"wrote {path}: {T} tasks from {refc['config_name']}")

    # Markdown: baselines (sample-weighted Last/Inc, typed from RESULTS.md section 4) next
    # to ISM with both the task-equal FA/AA and the sample-weighted last/inc totals.
    base = _baselines(args.baselines or os.path.join(args.out_dir, "e1_long_baselines.csv"))
    if not base:
        print("(no baseline file found; pass --baselines figures_data/e1_long_baselines.csv)")
    order = sorted(by_ds_split, key=lambda k: (args.datasets.index(k[0]) if k[0] in args.datasets
                                               else 99, int(k[1][1:]) if k[1] else 0))
    for ds, split in order:
        cs = by_ds_split[(ds, split)]
        desc = next((r["split_desc"] for r in rows if r["dataset"] == ds and r["split"] == split), "")
        print(f"\n### {ds} -- {desc} ({split})")
        table = []
        for b in base.get((ds, split), []):
            table.append(OrderedDict(method=b["method"],
                                     last=pm(_f(b.get("last")), _f(b.get("last_sd"))),
                                     inc=pm(_f(b.get("inc")), _f(b.get("inc_sd"))),
                                     fa="-", aa="-", ff="-", n="-", source=b.get("source", "")))
        for c in cs:
            table.append(OrderedDict(
                method=f"ISM (ours) [{c['config_name']}, alpha {c['alpha']}, lambda {c['lambda']}, "
                       f"{c['epochs']} ep]",
                last=pm(c["last_total"], c["last_total_sd"]), inc=pm(c["inc_total"], c["inc_total_sd"]),
                fa=pm(c["faa"], c["faa_sd"]), aa=pm(c["asa"], c["asa_sd"]), ff=pm(c["ffm"], c["ffm_sd"]),
                n=f"{c['n_seeds']}" + (f" (+{c['n_failed']} failed)" if c["n_failed"] else ""),
                source=";".join(os.path.basename(f) for f in c["files"])))
        print(markdown_table(table, ["method", "last", "inc", "fa", "aa", "ff", "n", "source"],
                             ["Method", "Last-Acc (sample-wtd)", "Inc-Acc (sample-wtd)",
                              "FA (task-equal)", "AA (task-equal)", "FF", "seeds", "source"]))
    bad = [c for c in cells if c["summary_check"] == "mismatch"]
    for c in bad:
        print(f"WARNING: SUMMARY line of {c['dataset']}/{c['config_name']} disagrees with the "
              f"recomputed seeds (failed seeds count as 0.0 in the SUMMARY): "
              f"summary FA {c['summary_fa']} vs recomputed {c['faa']}")
    return 0


def _f(x):
    try:
        return float(x) if x not in (None, "", "-") else None
    except (TypeError, ValueError):
        return None


# --- E2: tuning record and cost table ------------------------------------------------
CELL_KEYS = ("dataset", "train_method", "train_merge", "spectral_variant", "pspectral_p",
             "train_merge_alpha", "train_merge_k", "train_merge_coef", "train_drift_lambda",
             "train_drift_ridge", "train_ca_robust_weight", "train_ca_cov_shrinkage",
             "train_weight_decay", "model_lora_target_modules", "model_lora_r", "train_prefix",
             "dataset_init_cls", "dataset_increment", "train_epochs", "merge_dare_q",
             "merge_della_q", "merge_bc_beta", "merge_bc_gamma", "merge_knots_topk",
             "train_merge_topk", "train_merge_incremental", "specreg_weight", "speccap",
             "speccap_mode", "speccap_anchor", "merge_update_op", "merge_alpha_rule",
             "merge_alpha_param", "train_ca_cov_mode", "train_ca_cov_rank",
             "data_fewshot_k", "data_val_frac", "train_drift_lambda_rule", "l2m_weight")
# Rule-specific keys shown in the compact 'extras' column (the Configuration dump prints
# every BASE_CONFIG key, so only the keys the rule reads are informative; main.py:1587-1640).
_E2_RULE_KEYS = {"dare": ("merge_dare_q",), "della": ("merge_della_q",),
                 "breadcrumbs": ("merge_bc_beta", "merge_bc_gamma"), "knots": ("merge_knots_topk",),
                 "kspectral": ("train_merge_k",),
                 "avg": ("train_merge_topk", "train_merge_incremental"),
                 "ties": ("train_merge_topk", "train_merge_incremental"),
                 "max_abs": ("train_merge_topk", "train_merge_incremental"),
                 "max": ("train_merge_topk", "train_merge_incremental"),
                 "min": ("train_merge_topk", "train_merge_incremental")}
# Optional keys shown whenever they leave their default (default-off instrumentation and
# the E10/E7/E5/E13/E14 switches; a missing key means the default).
_E2_OPTIONAL_KEYS = (("specreg_weight", (0.0, 0, None)), ("speccap", (None, 0, 0.0, False)),
                     ("speccap_mode", ("rescale",)), ("speccap_anchor", ("base",)),
                     ("merge_update_op", ("trunc",)), ("merge_alpha_rule", ("fixed",)),
                     ("merge_alpha_param", ()), ("train_ca_cov_mode", ("class",)),
                     ("train_ca_cov_rank", ()), ("data_fewshot_k", (None,)), ("data_val_frac", (0.0, 0, None)),
                     ("train_drift_ridge", ("identity",)), ("l2m_weight", (0.0, 0, None)),
                     ("train_epochs_small_task", (None,)), ("train_drift_lambda_rule", (None, "fixed")),
                     ("train_drift_solver", ("dense",)))


def _extras(cfg, rule):
    """Compact 'key=value' list: the rule's own keys plus optional keys off their default."""
    bits = [f"{k}={cfg[k]}" for k in _E2_RULE_KEYS.get(rule, ()) if k in cfg]
    shown = set(_E2_RULE_KEYS.get(rule, ()))
    for k, defaults in _E2_OPTIONAL_KEYS:
        if k in cfg and k not in shown and cfg[k] not in defaults:
            if k in ("speccap_mode", "speccap_anchor") and cfg.get("speccap") in (None, 0, 0.0, False):
                continue                      # only meaningful when the cap is on
            bits.append(f"{k}={cfg[k]}")
    return ";".join(bits)


def _rule(cfg):
    method = cfg.get("train_method", "seqft")
    return str(cfg.get("train_merge")) if method == "seqft" else str(method)


def _is_truthy(v):
    return v not in (None, 0, 0.0, False, "None", "none", "")


def _is_headline(cell):
    ds = cell["dataset"]
    head = HEADLINE.get(ds)
    if not head or _rule(cell) != "pspectral":
        return False
    return (cell.get("spectral_variant") == HEADLINE_VARIANT
            and _close(cell.get("pspectral_p"), HEADLINE_P)
            and _close(cell.get("train_merge_alpha"), head["alpha"])
            and _close(cell.get("train_drift_lambda"), head["lambda"])
            and tuple(cell.get("model_lora_target_modules") or ()) == HEADLINE_TARGETS
            and cell.get("model_lora_r") == HEADLINE_RANK
            and cell.get("split") == "t10")


def _close(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _same_except(a, b, key):
    return all(a.get(k) == b.get(k) for k in CELL_KEYS if k != key)


@subcommand("e2", "tuning record (chronological cells), budget per dataset x rule, "
                  "specreg/speccap runs and the per-seed Time table")
def cmd_e2(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e2", description=cmd_e2.__doc__)
    _common_args(p, roots=("logs_main", "logs_exp21"))
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--dataset", default=None, help="restrict the narrative to one dataset")
    p.add_argument("--validate", action="store_true",
                   help="compare the pspectral cells with figures_data/mc3d_<ds>_full.csv")
    p.add_argument("--cubes-dir", default="figures_data")
    p.add_argument("--cube-datasets", nargs="*", default=["imagenetr", "cars"])
    args = p.parse_args(argv)
    runs = _load(args)
    for r in runs:
        r["rule"] = _rule(r["config"])

    # ---- record: one row per hyper-parameter cell, chronological -------------------
    cells = aggregate(runs, CELL_KEYS, metrics=("faa", "asa", "ffm"))
    for c in cells:
        members = [r for r in runs if all(_hashable(get_field(r, k)) == c[k] for k in CELL_KEYS)]
        c["rule"] = _rule(members[0]["config"])
        c["split"] = members[0]["split"]
        c["split_desc"] = members[0]["split_desc"]
        stamps = [r["first_seen_ts"] for r in members if r.get("first_seen_ts")]
        c["first_ts"] = min(stamps) if stamps else ""
        stamps = [r["last_seen_ts"] for r in members if r.get("last_seen_ts")]
        c["last_ts"] = max(stamps) if stamps else ""
        c["files"] = sorted({_rel(r["path"]) for r in members if r["path"]})
        c["config_names"] = sorted({str(r["config_name"]) for r in members})
        c["extras"] = _extras(members[0]["config"], c["rule"])
        c["headline"] = _is_headline(c)
        flags = []
        if c["n_failed"]:
            flags.append(f"failed seeds {c['failed_seeds']}")
        if any(r.get("summary_check") == "mismatch" for r in members):
            flags.append("SUMMARY mismatch (0.0 seeds averaged?)")
        if any(r.get("metrics_match") is False for r in members):
            flags.append("recomputed metrics differ from the logged FAA line")
        if tuple(c.get("model_lora_target_modules") or ()) != HEADLINE_TARGETS:
            flags.append("non-headline targets")
        if c["n_runs"] > len(members):
            flags.append(f"{c['n_runs'] - len(members)} duplicate block(s) dropped")
        c["flags"] = "; ".join(flags)
    cells.sort(key=lambda c: (str(c["dataset"]), c["first_ts"]))
    os.makedirs(args.out_dir, exist_ok=True)
    rec_cols = ["dataset", "train_method", "rule", "spectral_variant", "pspectral_p", "train_merge_alpha",
                "train_merge_coef", "train_drift_lambda", "train_ca_robust_weight",
                "train_ca_cov_shrinkage", "train_weight_decay", "model_lora_target_modules",
                "model_lora_r", "train_prefix", "split", "split_desc", "train_epochs", "extras",
                "n_seeds", "seeds", "n_failed", "n_runs", "faa", "faa_sd", "asa", "asa_sd", "ffm",
                "ffm_sd", "first_ts", "last_ts", "files", "config_names", "headline", "flags"]
    rec_path = os.path.join(args.out_dir, "e2_tuning_record.csv")
    write_csv(rec_path, cells, rec_cols)
    print(f"wrote {rec_path}: {len(cells)} cells from {len(runs)} runs")

    # ---- budget: dataset x rule --------------------------------------------------------
    budget = []
    by_dr = OrderedDict()
    for c in cells:
        by_dr.setdefault((c["dataset"], c["rule"]), []).append(c)
    for (ds, rule), cs in by_dr.items():
        varied = OrderedDict()
        for k in CELL_KEYS:
            if k == "dataset":
                continue
            vals = sorted({fmt_value(k, c.get(k)) for c in cs})
            if len(vals) > 1:
                varied[k] = vals
        full = [c for c in cs if c["n_seeds"] >= 3 and c["n_failed"] == 0]
        best = max(full, key=lambda c: c["faa"]) if full else None
        head = next((c for c in cs if c["headline"]), None)
        row = OrderedDict(dataset=ds, rule=rule, n_cells=len(cs),
                          n_seed_runs=sum(c["n_runs"] for c in cs),
                          n_complete_cells=len(full),
                          varied_keys=";".join(varied),
                          grid="; ".join(f"{k}={{{','.join(v)}}}" for k, v in varied.items()),
                          best_cell=(f"{best['config_names'][0]} ({best['split']})" if best else ""),
                          best_fa=best["faa"] if best else None,
                          headline_present=head is not None,
                          headline_fa=head["faa"] if head else None,
                          headline_is_argmax=(best is not None and head is not None
                                              and _close(best["faa"], head["faa"])),
                          first_ts=min(c["first_ts"] for c in cs), last_ts=max(c["last_ts"] for c in cs))
        budget.append(row)
    bud_path = os.path.join(args.out_dir, "e2_tuning_budget.csv")
    write_csv(bud_path, budget)
    print(f"wrote {bud_path}: {len(budget)} dataset x rule rows\n")
    print("### Tuning budget (cells = distinct hyper-parameter settings, runs = seed blocks)")
    print(markdown_table(budget, ["dataset", "rule", "n_cells", "n_seed_runs", "n_complete_cells",
                                  "grid", "best_cell", "best_fa", "headline_present",
                                  "headline_fa", "headline_is_argmax"]))

    # ---- what the logs can and cannot reconstruct ---------------------------------------
    print("\n### Headline cells (alpha / lambda choices) and whether a sweep contains them")
    for ds, head in HEADLINE.items():
        cs = [c for c in cells if c["dataset"] == ds and c["rule"] == "pspectral"]
        hcs = [c for c in cs if c["headline"]]
        if not hcs:
            print(f"- {ds}: headline cell (only_A, p {HEADLINE_P}, alpha {head['alpha']}, lambda "
                  f"{head['lambda']}, 10 tasks) is in none of the parsed logs: the choice cannot "
                  f"be reconstructed from these logs.")
            continue

        def _sweep(hc, key):
            return sorted({c[key] for c in cs if _same_except(c, hc, key)})

        # The headline posture may exist under several prefixes (best_* and main_*); report
        # the copy that sits inside an alpha sweep, since that is the one that documents the choice.
        hc = max(hcs, key=lambda c: (len(_sweep(c, "train_merge_alpha")), -hcs.index(c)))
        alpha_sweep = _sweep(hc, "train_merge_alpha")
        lam_sweep = _sweep(hc, "train_drift_lambda")
        fa_by_alpha = {c["train_merge_alpha"]: c["faa"] for c in cs
                       if _same_except(c, hc, "train_merge_alpha") and c["faa"] is not None}
        argmax = max(fa_by_alpha, key=fa_by_alpha.get) if fa_by_alpha else None
        prefixes = sorted({str(c["train_prefix"]) for c in hcs})
        print(f"- {ds}: headline cell present under prefix(es) {prefixes} (FA {pm(hc['faa'], hc['faa_sd'])}, "
              f"{hc['n_seeds']} seeds, first run {hc['first_ts']}, {';'.join(hc['config_names'])}); "
              f"same-posture alpha sweep "
              f"{alpha_sweep if len(alpha_sweep) > 1 else 'ABSENT (alpha choice not reconstructible)'}; "
              f"lambda sweep {lam_sweep if len(lam_sweep) > 1 else 'ABSENT'}"
              + (f"; argmax alpha of that sweep = {argmax}"
                 + (" (= headline)" if _close(argmax, head["alpha"]) else " (NOT the headline)")
                 if len(alpha_sweep) > 1 else ""))

    # ---- specreg / speccap runs (E10 training-time arm) ---------------------------------
    spec = [c for c in cells if _is_truthy(c.get("specreg_weight")) or _is_truthy(c.get("speccap"))]
    spec_path = os.path.join(args.out_dir, "e2_specreg_runs.csv")
    write_csv(spec_path, spec, rec_cols)
    print(f"\n### Existing specreg / speccap cells (E10): {len(spec)} (written to {spec_path})")
    if spec:
        print(markdown_table(spec, ["dataset", "config_names", "specreg_weight", "speccap", "speccap_mode",
                                    "speccap_anchor", "train_merge_alpha", "n_seeds", "faa", "faa_sd",
                                    "asa", "ffm", "first_ts"]))

    # ---- per-seed Time table (the cost table for the runbook) ---------------------------
    trows = []
    for r in runs:
        if r["time_s"] is None:
            continue
        n = r["n_tasks"] or 0
        kind = "uncached" if (r["n_cached_tasks"] == 0 and r["n_trained_tasks"] > 0) else \
               ("cached" if (n and r["n_cached_tasks"] >= n) else "partial")
        trows.append(OrderedDict(dataset=r["dataset"], config_name=r["config_name"], seed=r["seed"],
                                 status=r["status"], kind=kind, time_s=r["time_s"], n_tasks=n,
                                 n_cached_tasks=r["n_cached_tasks"], n_trained_tasks=r["n_trained_tasks"],
                                 train_epochs=r["config"].get("train_epochs"), split=r["split"],
                                 train_prefix=r["config"].get("train_prefix"), first_ts=r["first_ts"],
                                 path=_rel(r["path"])))
    time_path = os.path.join(args.out_dir, "e2_time_per_seed.csv")
    write_csv(time_path, trows)
    print(f"\n### Per-seed wall-clock ('Time:' lines, status ok), written to {time_path}")
    cost = []
    for kind in ("uncached", "cached", "partial"):
        for ds in sorted({t["dataset"] for t in trows}, key=str):
            ts_ = [t["time_s"] for t in trows if t["dataset"] == ds and t["kind"] == kind
                   and t["status"] == "ok" and t["split"] == "t10"]
            if ts_:
                cost.append(OrderedDict(dataset=ds, kind=kind, n=len(ts_), time_mean=float(np.mean(ts_)),
                                        time_min=float(np.min(ts_)), time_max=float(np.max(ts_)),
                                        hours_mean=float(np.mean(ts_)) / 3600.0))
    print(markdown_table(cost, ["dataset", "kind", "n", "time_mean", "time_min", "time_max", "hours_mean"],
                         ["dataset", "kind (10-task runs)", "n", "mean s", "min s", "max s", "mean h"]))

    # ---- narrative: chronological sweeps per dataset --------------------------------------
    print("\n### Chronological record per dataset (cell: rule / varied values / FA / seeds / log)")
    for ds in sorted({c["dataset"] for c in cells}, key=str):
        if args.dataset and ds != args.dataset:
            continue
        print(f"\n**{ds}**")
        cs = [c for c in cells if c["dataset"] == ds]
        for c in cs:
            setting = (f"{c['spectral_variant']} p{c['pspectral_p']} a{c['train_merge_alpha']}"
                       if c["rule"] == "pspectral" else f"c{c['train_merge_coef']}")
            print(f"- {c['first_ts'][:16] if c['first_ts'] else '????':16s} {c['rule']:12s} {setting:24s} "
                  f"lambda {c['train_drift_lambda']} robust {c['train_ca_robust_weight']} wd "
                  f"{c['train_weight_decay']} {c['split'] or '?'} ep {c['train_epochs']} | FA "
                  f"{pm(c['faa'], c['faa_sd'])} ({c['n_seeds']} seeds"
                  f"{', ' + str(c['n_failed']) + ' failed' if c['n_failed'] else ''}) | "
                  f"{';'.join(c['config_names'])}" + (f" | {c['extras']}" if c["extras"] else "")
                  + (f" | {c['flags']}" if c["flags"] else ""))

    # ---- validation against the two committed cubes --------------------------------------
    if args.validate:
        print("\n### Validation against figures_data/mc3d_<ds>_full.csv")
        for ds in args.cube_datasets:
            path = os.path.join(args.cubes_dir, f"mc3d_{ds}_full.csv")
            if not os.path.isfile(path):
                print(f"- {ds}: {path} not found")
                continue
            with open(path, encoding="utf-8", newline="") as fh:
                cube = list(csv.DictReader(fh))
            matched = mismatched = missing = 0
            worst = 0.0
            for row in cube:
                cand = [c for c in cells if c["dataset"] == ds and c["rule"] == "pspectral"
                        and c["spectral_variant"] == row["variant"] and _close(c["pspectral_p"], row["p"])
                        and _close(c["train_merge_alpha"], row["alpha"]) and c["split"] == "t10"
                        and c["n_seeds"] >= 3]
                mc3d = [c for c in cand if any(n.startswith("mc3d_") for n in c["config_names"])]
                cand = mc3d or cand
                if not cand:
                    missing += 1
                    continue
                diff = abs(cand[0]["faa"] - float(row["fa"]))
                worst = max(worst, diff)
                if diff <= 0.0101:
                    matched += 1
                else:
                    mismatched += 1
                    print(f"  mismatch {ds} {row['variant']} p{row['p']} a{row['alpha']}: cube {row['fa']} "
                          f"vs logs {cand[0]['faa']:.2f} ({';'.join(cand[0]['config_names'])})")
            print(f"- {ds}: {len(cube)} cube cells: {matched} matched, {mismatched} mismatched, "
                  f"{missing} missing in the logs (max |diff| {worst:.3f})")
    return 0


def _rel(path):
    try:
        return os.path.relpath(path).replace(os.sep, "/")
    except ValueError:            # different drive on Windows
        return path


# ===========================================================================
# Rebuttal subcommands e3 e5 e7 e8 e9 e10 e11 e12 e13 e14 e15.
#
# Every function below uses only the API above (load_runs, aggregate, write_csv,
# markdown_table, run['tags'][...], run['events'], run['adaptive_alphas'],
# run['merge_state'], run['drift_gcv']).  The gated line formats come from main.py
# (B1-B4 rebuttal edits); each parser quotes the f-string it reads.
# ===========================================================================

# ---------------------------------------------------------------------------
# key=value line parsers for the gated lines
# ---------------------------------------------------------------------------
_PLAIN_NUM_RE = re.compile(r"^[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|nan|inf)$")
# 'key=value' pairs; the key must start a word (so '||W_t-anchor||_F=' is not a key).
_KV_RE = re.compile(r"(?<![\w|.\-])([A-Za-z_][A-Za-z0-9_]*)=([^\s|]+)")
_TASK_RE = re.compile(r"^\[[^\]]+\] (?:task|Task) (\d+)")
TIMING_RE = re.compile(r"^\[Timing\] task (\d+) (\w+)=(" + _NUM + r")s(.*)$")
L2M_ANCHOR_RE = re.compile(r"^\[L2M\] anchor=(\S+) weight=(" + _NUM + r") tensors=(\d+)")
L2M_DIST_RE = re.compile(r"^\[L2M\] end-of-task \|\|W_t-anchor\|\|_F=(" + _NUM + r")")
TIL_RE = re.compile(r"^\[Evaluation-til\] (?:Total|Task-IL) Acc: (" + _NUM + r"), Grouped: \[([^\]]*)\]")
INFLORA_DESIGN_RE = re.compile(
    r"^\[InfLoRA\] Task (\d+): A_t designed for (\d+) blocks \(remove (\d+) / retain (\d+)\) \| max\|A M\| (" + _NUM + r")")
INFLORA_GPM_RE = re.compile(
    r"^\[InfLoRA\] DualGPM task (\d+) threshold (" + _NUM + r") \| cols (\[[^\]]*\]) \| types (\[[^\]]*\]) \| (?:added|changed) (\[[^\]]*\])")
INFLORA_SEED_RE = re.compile(
    r"^\[InfLoRA\] seed (\d+): FA (" + _NUM + r") \| AA (" + _NUM + r") \| FF (" + _NUM + r") \| micro ACC ("
    + _NUM + r") \| AVG (" + _NUM + r") \| curve (\[[^\]]*\])")
INFLORA_BACKBONE_RE = re.compile(r"^\[InfLoRA\] backbone (\S+) -> (\S+)")
# '[KeepLoRA] ...' lines (main._keeplora_before_train / _after_train, run_keeplora.run_cell, backbones factory)
KEEPLORA_DESIGN_RE = re.compile(
    r"^\[KeepLoRA\] Task (\d+): A/B designed for (\d+) slices from (\d+) batches \| basis cols min (\d+) max (\d+) "
    r"\| max\|U\^T A\| (" + _NUM + r") \| B init (\w+)")
KEEPLORA_SUBSPACE_RE = re.compile(
    r"^\[KeepLoRA\] subspace task (\d+) eps_f=(" + _NUM + r") mode=(\w+) \| cols min (\d+) max (\d+) mean (" + _NUM
    + r") \| added min (\d+) max (\d+) \| full (\d+)/(\d+)(?: \| skipped batches (\d+))? \| cols (\[[^\]]*\]) \| added (\[[^\]]*\])")
KEEPLORA_SEED_RE = re.compile(
    r"^\[KeepLoRA\] seed (\d+): FA (" + _NUM + r") \| AA (" + _NUM + r") \| FF (" + _NUM + r") \| micro ACC ("
    + _NUM + r") \| AVG (" + _NUM + r") \| curve (\[[^\]]*\])")
KEEPLORA_BACKBONE_RE = re.compile(r"^\[KeepLoRA\] backbone (\S+) -> (\S+)")
SMALL_TASK_RE = re.compile(r"^\[Training\] small task \((\d+) classes\): epochs=(\d+)")


def _num(token):
    """'0.812s' -> 0.812, '12' -> 12, '1e-3' -> 0.001, 'trunc' -> 'trunc'."""
    t = token.strip().rstrip(",;")
    if t.endswith("s") and _PLAIN_NUM_RE.match(t[:-1]):      # durations: merge_time=0.812s
        t = t[:-1]
    if _PLAIN_NUM_RE.match(t):
        try:
            return int(t)
        except ValueError:
            return float(t)
    return t


def kv_pairs(text):
    """'k=v k2=v2 | k3=v3s' -> OrderedDict with numbers parsed (see _num)."""
    out = OrderedDict()
    for m in _KV_RE.finditer(text):
        out[m.group(1)] = _num(m.group(2))
    return out


def task_of(line):
    """Task index of a '[Tag] task t ...' / '[Tag] Task t ...' line (None if absent)."""
    m = _TASK_RE.match(line)
    return int(m.group(1)) if m else None


def parse_norm_line(line):
    """[MergeStat] / [StepNorm] line -> dict(task, degenerate, <fields>).

    main.py (_log_merge_stat / _log_step_norm):
      [MergeStat] task t alpha=.. op=.. k_mean=.. k_min=.. k_max=.. | A_sigma_max_mean=..
        A_sigma_max_max=.. | eff_sigma_max_mean=.. eff_sigma_max_max=.. eff_fro_mean=.. |
        raw_sigma_max_mean=.. raw_eff_sigma_max_mean=.. | merge_time=..s
      [MergeStat] task t | A_sigma_max_mean=.. ... | merge_time=..s      (SOTA / classical)
      [MergeStat] task t no LoRA factor pairs | merge_time=..s           (degenerate)
      [StepNorm] task t k_mean=.. k_min=.. k_max=.. | A_sigma_max_mean=.. ... eff_fro_mean=..
    Fields absent from a line are simply missing from the dict.
    """
    rec = OrderedDict(task=task_of(line), degenerate=("no LoRA factor pairs" in line))
    rec.update(kv_pairs(line.split("]", 1)[1]))
    return rec


def parse_timing(line):
    """'[Timing] task t stage=1.234s k=v ...' -> dict(task, stage, seconds, extra) or None."""
    m = TIMING_RE.match(line)
    if not m:
        return None
    return OrderedDict(task=int(m.group(1)), stage=m.group(2), seconds=float(m.group(3)),
                       extra=kv_pairs(m.group(4)))


def parse_featshift(line):
    """'[FeatShift] task t shift=.. n=..' -> dict(task, shift, n)."""
    rec = OrderedDict(task=task_of(line))
    rec.update(kv_pairs(line.split("]", 1)[1]))
    return rec


def parse_driftdiag(line):
    """'[DriftDiag] task t K=.. n_img=.. src=.. | mean_rel raw=.. before=.. after=.. |
    cov_relF raw=.. before=.. after=.. [| onestep mean_rel I=.. W=.. oracle=.. cov_relF I=..
    W=.. oracle=..]' -> dict with the DRIFTDIAG_CSV_HEADER names (mean_rel_raw, ...,
    os_mean_rel_I, ...).  Returns None for the degenerate warning lines."""
    if "no old" in line or "skipping" in line:
        return None
    sections = [s.strip() for s in line.split("]", 1)[1].split("|")]
    rec = OrderedDict(task=task_of(line))
    rec.update(kv_pairs(sections[0]))                       # K, n_img, src
    for sec in sections[1:]:
        words = sec.split()
        if not words:
            continue
        if words[0] == "onestep":
            sub = None
            for w in words[1:]:
                if "=" in w:
                    k, v = w.split("=", 1)
                    rec[f"os_{sub}_{k}"] = _num(v)
                else:
                    sub = w
        else:
            for w in words[1:]:
                if "=" in w:
                    k, v = w.split("=", 1)
                    rec[f"{words[0]}_{k}"] = _num(v)
    return rec


def parse_data_line(line):
    """DataManager.describe() line -> dict(dataset, train, val, test, fewshot_k, val_frac,
    val_seed, subsample_seed); train/val/test are dicts n, C, min, median, max (None when
    the token reads 'none').  Format (utils/data_manager.py):
      [Data] dataset=cifar224 | train: n=50000 C=100 per-class min/median/max=500/500/500
      | val: none | test: n=10000 C=100 per-class min/median/max=100/100/100 | fewshot: none
    with the hook active: 'val: n=.. (val_frac=0.1, val_seed=1993)', 'fewshot: k=5
    (subsample_seed=1993)'."""
    out = OrderedDict(dataset=None, train=None, val=None, test=None, fewshot_k=None,
                      val_frac=None, val_seed=None, subsample_seed=None)
    for part in line.split(" | "):
        part = part.strip()
        if part.startswith("[Data] dataset="):
            out["dataset"] = part.split("=", 1)[1].strip()
            continue
        m = re.match(r"^(train|val|test): (.*)$", part)
        if m:
            name, rest = m.group(1), m.group(2)
            if rest.startswith("none"):
                out[name] = None
            else:
                mm = re.match(r"n=(\d+) C=(\d+) per-class min/median/max=(\d+)/([\d.]+)/(\d+)", rest)
                if mm:
                    out[name] = OrderedDict(n=int(mm.group(1)), C=int(mm.group(2)), min=int(mm.group(3)),
                                            median=float(mm.group(4)), max=int(mm.group(5)))
                mm = re.search(r"val_frac=([^,)]+), val_seed=([^)]*)\)", rest)
                if mm:
                    out["val_frac"], out["val_seed"] = _num(mm.group(1)), _num(mm.group(2))
            continue
        m = re.match(r"^fewshot: k=(\d+) \(subsample_seed=([^)]*)\)", part)
        if m:
            out["fewshot_k"], out["subsample_seed"] = int(m.group(1)), _num(m.group(2))
    return out


def parse_til_rows(run):
    """Task-incremental Grouped rows of the '[Evaluation-til]' lines (eval_til=True)."""
    rows = []
    for line in run["tags"].get("Evaluation-til", []):
        m = TIL_RE.match(line)
        if m:
            rows.append(_floats(m.group(2)))
    return rows


def parse_inflora_lines(run):
    """[InfLoRA] lines -> dict(backbone, design [per task], gpm [per task], seed_line)."""
    out = {"backbone": None, "design": [], "gpm": [], "seed_line": None}
    for line in run["tags"].get("InfLoRA", []):
        m = INFLORA_BACKBONE_RE.match(line)
        if m:
            out["backbone"] = (m.group(1), m.group(2))
            continue
        m = INFLORA_DESIGN_RE.match(line)
        if m:
            out["design"].append(OrderedDict(task=int(m.group(1)), blocks=int(m.group(2)),
                                             remove=int(m.group(3)), retain=int(m.group(4)),
                                             leak=float(m.group(5))))
            continue
        m = INFLORA_GPM_RE.match(line)
        if m:
            out["gpm"].append(OrderedDict(task=int(m.group(1)), threshold=float(m.group(2)),
                                          cols=literal(m.group(3)), types=literal(m.group(4)),
                                          added=literal(m.group(5))))
            continue
        m = INFLORA_SEED_RE.match(line)
        if m:
            out["seed_line"] = OrderedDict(seed=int(m.group(1)), fa=float(m.group(2)), aa=float(m.group(3)),
                                           ff=float(m.group(4)), micro_acc=float(m.group(5)),
                                           avg=float(m.group(6)), curve=literal(m.group(7)))
    return out


def parse_keeplora_lines(run):
    """[KeepLoRA] lines -> dict(backbone, design [per task], subspace [per task], seed_line)."""
    out = {"backbone": None, "design": [], "subspace": [], "seed_line": None}
    for line in run["tags"].get("KeepLoRA", []):
        m = KEEPLORA_BACKBONE_RE.match(line)
        if m:
            out["backbone"] = (m.group(1), m.group(2))
            continue
        m = KEEPLORA_DESIGN_RE.match(line)
        if m:
            out["design"].append(OrderedDict(task=int(m.group(1)), slices=int(m.group(2)), batches=int(m.group(3)),
                                             cols_min=int(m.group(4)), cols_max=int(m.group(5)),
                                             leak=float(m.group(6)), b_init=m.group(7)))
            continue
        m = KEEPLORA_SUBSPACE_RE.match(line)
        if m:
            out["subspace"].append(OrderedDict(task=int(m.group(1)), eps_f=float(m.group(2)), mode=m.group(3),
                                               cols_min=int(m.group(4)), cols_max=int(m.group(5)),
                                               cols_mean=float(m.group(6)), added_min=int(m.group(7)),
                                               added_max=int(m.group(8)), full=int(m.group(9)),
                                               n_slices=int(m.group(10)),
                                               skipped=int(m.group(11)) if m.group(11) else 0,
                                               cols=literal(m.group(12)), added=literal(m.group(13))))
            continue
        m = KEEPLORA_SEED_RE.match(line)
        if m:
            out["seed_line"] = OrderedDict(seed=int(m.group(1)), fa=float(m.group(2)), aa=float(m.group(3)),
                                           ff=float(m.group(4)), micro_acc=float(m.group(5)),
                                           avg=float(m.group(6)), curve=literal(m.group(7)))
    return out


def tag_records(run, tag, parser):
    """parser() over the raw lines of one tag (None results dropped)."""
    out = []
    for line in run["tags"].get(tag, []):
        rec = parser(line)
        if rec is not None:
            out.append(rec)
    return out


def _mean(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _sd(vals, ddof=0):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.std(vals, ddof=ddof)) if len(vals) > ddof else None


def _ok(runs):
    return [r for r in runs if r["status"] == "ok"]


def _cfgf(run, key, default=None):
    """float(config[key]) or default."""
    v = run["config"].get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _load_tags(args, capture_tags=None):
    runs = load_runs(args.roots, args.globs, capture_tags=capture_tags,
                     skip_smoke=not args.include_smoke)
    if not runs:
        print("[parse_logs] no runs found", file=sys.stderr)
    return runs


def _read_refs(path):
    """Typed reference numbers (method, dataset, fa, aa, ffm, source ...) from a CSV;
    [] when the path is None or missing."""
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return [{k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _std_metrics(cell):
    """fa/aa/ffm naming (export_mc3d_full.py schema) of an aggregate() cell."""
    return OrderedDict(n_seeds=cell["n_seeds"], fa=cell.get("faa"), fa_sd=cell.get("faa_sd"),
                       aa=cell.get("asa"), aa_sd=cell.get("asa_sd"), ffm=cell.get("ffm"), ffm_sd=cell.get("ffm_sd"),
                       old_mean=cell.get("old_mean"), new_acc=cell.get("new_acc"), first5_mean=cell.get("first5_mean"))


_CELL_METRICS = ("faa", "asa", "ffm", "old_mean", "new_acc", "first5_mean")


def _members(runs, cell, keys):
    return [r for r in runs if all(_hashable(get_field(r, k)) == cell[k] for k in keys)]


def _names(members):
    return sorted({str(r["config_name"]) for r in members})


def _is_headline_posture(cfg, ds):
    """The paper's ISM cell (run_best posture) with every rebuttal switch at its default."""
    head = HEADLINE.get(ds)
    if not head or cfg.get("train_merge") != "pspectral" or cfg.get("train_method", "seqft") != "seqft":
        return False
    variant = cfg.get("spectral_variant", cfg.get("pspectral_variant", "only_A"))
    return (variant == HEADLINE_VARIANT and _close(cfg.get("pspectral_p"), HEADLINE_P)
            and _close(cfg.get("train_merge_alpha"), head["alpha"])
            and _close(cfg.get("train_drift_lambda"), head["lambda"])
            and cfg.get("merge_alpha_rule", "fixed") == "fixed"
            and cfg.get("merge_update_op", "trunc") == "trunc"
            and cfg.get("train_ca_cov_mode", "class") == "class"
            and cfg.get("train_drift_lambda_rule") in (None, "fixed")
            and not _is_truthy(cfg.get("data_fewshot_k")) and not _is_truthy(cfg.get("data_val_frac"))
            and not _is_truthy(cfg.get("l2m_weight")))


def _reference_cell(runs, ds, prefix=None, prefer=(), exclude=()):
    """Paired ISM reference on the same chain: the headline-posture 10-task cell of ``ds``
    (same train_prefix when given) with the most ok seeds; names in ``prefer`` win ties.
    Returns the aggregate() cell (with 'config_name') or None."""
    cands = [r for r in runs if r["dataset"] == ds and r["status"] == "ok" and r["split"] == "t10"
             and _is_headline_posture(r["config"], ds)
             and (prefix is None or r["config"].get("train_prefix") == prefix)
             and r["config_name"] not in exclude]
    if not cands:
        return None
    cells = aggregate(cands, ("config_name",), metrics=_CELL_METRICS)

    def rank(c):
        name = str(c["config_name"])
        pref = next((i for i, p in enumerate(prefer) if name.startswith(p)), len(prefer))
        return (-c["n_seeds"], pref, name)
    best = sorted(cells, key=rank)[0]
    best["train_prefix"] = next(r["config"].get("train_prefix") for r in cands
                                if r["config_name"] == best["config_name"])
    return best


def _fmt_cell(mean, sd=None, nd=2):
    if mean is None:
        return "-"
    return f"{mean:.{nd}f}" if sd is None else f"{mean:.{nd}f} +- {sd:.{nd}f}"


def _spearman(x, y):
    """Spearman rank correlation (average ranks for ties); None with < 3 points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or len(x) != len(y):
        return None

    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _wrote(path, n, what="rows"):
    print(f"wrote {path}: {n} {what}")


# --- E3: per-stage durations and the share of total time -----------------------------
E3_STAGES = ("train", "merge", "drift", "drift_features", "drift_solve", "transport", "stats",
             "transport_stats", "align", "eval", "driftdiag")
E3_COLS = ["dataset", "config_name", "seed", "task", "source", "cached"] + [s + "_s" for s in E3_STAGES] + ["task_s"]


def _secs(a, b):
    return (b - a).total_seconds() if (a is not None and b is not None) else None


def stage_times_from_events(run):
    """Per-task stage durations (s) from the asctime stamps of the standard lines.

    Segments (main.py line order): training = '[Training] Task t' -> last '[Training]'
    line of the task (last epoch line, or the cache-hit line); merge = '[Merging] Task t'
    -> first non-[Merging] line (includes the checkpoint save); drift = '[Drift] Fitting'
    -> '[Drift] W shape' (previous-extractor reload + two feature passes + solve);
    transport_stats = '[Drift] Transporting' -> '[Alignment] Task t' (transport AND the
    class-statistics pass: no line separates them, only the [Timing] rows do); align =
    '[Alignment] Task t' -> last '[Alignment] Epoch'; eval = last alignment line ->
    '[Evaluation] Task t' (the test forward pass; at task 0, which has no alignment, it
    starts at the last training line and so includes the task-0 statistics pass);
    task = '[Training] Task t' -> '[Evaluation] Task t'.
    """
    ev = [(parse_ts(ts), lbl) for ts, lbl in run["events"] if ts]
    starts = [i for i, (_t, lbl) in enumerate(ev) if lbl.startswith("[Training] Task ")]
    out = []
    for n, i in enumerate(starts):
        t = int(ev[i][1].rsplit(" ", 1)[1])
        seg = ev[i:(starts[n + 1] if n + 1 < len(starts) else len(ev))]

        def first(pred, start=0):
            for j in range(start, len(seg)):
                if pred(seg[j][1]):
                    return j
            return None

        def last(pred):
            idx = [j for j in range(len(seg)) if pred(seg[j][1])]
            return idx[-1] if idx else None

        row = OrderedDict((c, None) for c in E3_COLS)
        row.update(dataset=run["dataset"], config_name=run["config_name"], seed=run["seed"], task=t,
                   source="timestamps",
                   cached=int(any(lbl.startswith("[Training] Reusing cached") for _t, lbl in seg)))
        j_tr_last = last(lambda lbl: lbl.startswith("[Training]"))
        if j_tr_last is not None and j_tr_last > 0:
            row["train_s"] = _secs(seg[0][0], seg[j_tr_last][0])
        j_m = first(lambda lbl: lbl == f"[Merging] Task {t}")
        if j_m is not None:
            j_me = first(lambda lbl: not lbl.startswith("[Merging]"), j_m + 1)
            row["merge_s"] = _secs(seg[j_m][0], seg[j_me][0]) if j_me is not None else None
        j_d = first(lambda lbl: lbl.startswith("[Drift] Fitting"))
        if j_d is not None:
            j_w = first(lambda lbl: lbl == "[Drift] W shape", j_d + 1)
            row["drift_s"] = _secs(seg[j_d][0], seg[j_w][0]) if j_w is not None else None
        j_tp = first(lambda lbl: lbl == "[Drift] Transporting")
        j_al = first(lambda lbl: lbl == f"[Alignment] Task {t}")
        if j_tp is not None and j_al is not None:
            row["transport_stats_s"] = _secs(seg[j_tp][0], seg[j_al][0])
        j_ep = last(lambda lbl: lbl.startswith("[Alignment] Epoch"))
        if j_al is not None and j_ep is not None:
            row["align_s"] = _secs(seg[j_al][0], seg[j_ep][0])
        j_ev = first(lambda lbl: lbl == f"[Evaluation] Task {t}")
        j_es = j_ep if j_ep is not None else (j_al if j_al is not None else j_tr_last)
        if j_ev is not None and j_es is not None:
            row["eval_s"] = _secs(seg[j_es][0], seg[j_ev][0])
        if j_ev is not None:
            row["task_s"] = _secs(seg[0][0], seg[j_ev][0])
        out.append(row)
    return out


def stage_times_from_timing(run):
    """Per-task stage seconds from the '[Timing]' lines (stage_timing=True)."""
    per_task = OrderedDict()
    for rec in tag_records(run, "Timing", parse_timing):
        row = per_task.get(rec["task"])
        if row is None:
            row = OrderedDict((c, None) for c in E3_COLS)
            row.update(dataset=run["dataset"], config_name=run["config_name"], seed=run["seed"],
                       task=rec["task"], source="timing")
            per_task[rec["task"]] = row
        row[rec["stage"] + "_s"] = rec["seconds"]
        if rec["stage"] == "train" and "cached" in rec["extra"]:
            row["cached"] = rec["extra"]["cached"]
    for row in per_task.values():
        parts = [row.get(k + "_s") for k in ("train", "merge", "drift", "transport", "stats", "align", "eval", "driftdiag")]
        row["task_s"] = sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None
    return list(per_task.values())


def stage_times_from_csv(path):
    """Rows of one stage_timing_csv file (dataset,seed,task,stage,seconds,extra) -> per-task
    rows (source 'csv'); the config name is taken from the file name timing_<config>.csv."""
    name = os.path.basename(path)
    cfg_name = re.sub(r"^timing_", "", name[:-4]) if name.endswith(".csv") else name
    per = OrderedDict()
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                key = (r["dataset"], int(r["seed"]), int(r["task"]))
                sec = float(r["seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            row = per.get(key)
            if row is None:
                row = OrderedDict((c, None) for c in E3_COLS)
                row.update(dataset=key[0], config_name=cfg_name, seed=key[1], task=key[2], source="csv")
                per[key] = row
            row[r["stage"] + "_s"] = sec
            if r["stage"] == "train":
                kv = kv_pairs(r.get("extra") or "")
                row["cached"] = kv.get("cached")
    for row in per.values():
        parts = [row.get(k + "_s") for k in ("train", "merge", "drift", "transport", "stats", "align", "eval", "driftdiag")]
        row["task_s"] = sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None
    return list(per.values())


def _e3_share(rows, time_lookup, epochs_lookup):
    """(dataset, config, source) -> mean per-task seconds per stage, the share of the
    per-seed 'Time:' total, and the drift block as a share of training time."""
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r["dataset"], r["config_name"], r["source"]), []).append(r)
    out = []
    for (ds, name, src), rs in groups.items():
        seeds = sorted({r["seed"] for r in rs})
        row = OrderedDict(dataset=ds, config_name=name, source=src, n_seeds=len(seeds),
                          n_tasks=max(len([r for r in rs if r["seed"] == s]) for s in seeds))
        for st in E3_STAGES:
            row[st + "_s"] = _mean([r[st + "_s"] for r in rs])
        row["task_s"] = _mean([r["task_s"] for r in rs])
        totals, shares = [], defaultdict(list)
        drift_over_train = []
        for s in seeds:
            mine = [r for r in rs if r["seed"] == s]
            total = time_lookup.get((ds, name, s))
            if total:
                totals.append(total)
                for st in E3_STAGES + ("task",):
                    vals = [r[st + "_s"] for r in mine if r[st + "_s"] is not None]
                    if vals:
                        shares[st].append(sum(vals) / total)
            if epochs_lookup.get((ds, name, s), 0) > 0:
                tr = [r["train_s"] for r in mine if r["train_s"] is not None]
                dr = [r["drift_s"] for r in mine if r["drift_s"] is not None]
                if tr and dr and sum(tr) > 0:
                    drift_over_train.append(sum(dr) / sum(tr))
        row["time_total_s"] = _mean(totals)
        for st in E3_STAGES + ("task",):
            row["share_" + st] = _mean(shares[st])
        row["drift_over_train"] = _mean(drift_over_train)
        out.append(row)
    return out


@subcommand("e3", "per-stage durations from log timestamps / [Timing] lines / timing CSVs and "
                  "their share of the per-seed total (e3_stage_times_per_task.csv, e3_stage_share.csv)")
def cmd_e3(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e3", description=cmd_e3.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--configs", nargs="*", default=None,
                   help="config-name globs to include (default: every run)")
    p.add_argument("--timing-csv", nargs="*", default=[],
                   help="stage_timing_csv files or globs (logs_main/*/timing_*.csv)")
    args = p.parse_args(argv)
    runs = _ok(_load_tags(args, capture_tags=("Timing",)))
    if args.configs:
        import fnmatch
        runs = [r for r in runs if any(fnmatch.fnmatch(str(r["config_name"]), g) for g in args.configs)]
    rows = []
    for r in runs:
        rows.extend(stage_times_from_events(r))
        rows.extend(stage_times_from_timing(r))
    for pattern in args.timing_csv:
        for path in sorted(glob.glob(pattern, recursive=True)):
            rows.extend(stage_times_from_csv(path))
    if not rows:
        print("(no runs with stage timestamps / [Timing] lines / timing CSVs found; nothing written)")
        return 0
    os.makedirs(args.out_dir, exist_ok=True)
    per_path = os.path.join(args.out_dir, "e3_stage_times_per_task.csv")
    write_csv(per_path, rows, E3_COLS)
    _wrote(per_path, len(rows), "task rows")
    time_lookup = {(r["dataset"], r["config_name"], r["seed"]): r["time_s"] for r in runs if r["time_s"]}
    epochs_lookup = {(r["dataset"], r["config_name"], r["seed"]): r["n_epoch_lines"] for r in runs}
    share = _e3_share(rows, time_lookup, epochs_lookup)
    share_path = os.path.join(args.out_dir, "e3_stage_share.csv")
    write_csv(share_path, share)
    _wrote(share_path, len(share), "dataset x config x source rows")
    for row in share:
        for st in ("train", "merge", "drift", "transport_stats", "transport", "stats", "align", "eval", "task"):
            v = row.get("share_" + st)
            row["pct_" + st] = f"{100 * v:.1f}%" if v is not None else "-"
        row["drift/train"] = f"{row['drift_over_train']:.3f}" if row.get("drift_over_train") is not None else "-"
    print("\n### Mean seconds per task (timestamps lump transport with the class statistics; "
          "[Timing] rows separate every stage)")
    print(markdown_table(share, ["dataset", "config_name", "source", "n_seeds", "n_tasks", "train_s", "merge_s",
                                 "drift_s", "drift_features_s", "drift_solve_s", "transport_stats_s", "transport_s",
                                 "stats_s", "align_s", "eval_s", "task_s", "time_total_s"]))
    print("\n### Share of the per-seed 'Time:' total (sum over tasks / total) and drift block over training time")
    print(markdown_table(share, ["dataset", "config_name", "source", "pct_train", "pct_merge", "pct_drift",
                                 "pct_transport_stats", "pct_transport", "pct_stats", "pct_align", "pct_eval",
                                 "pct_task", "drift/train"]))
    return 0


# --- E5: accuracy vs storage of the cheap class statistics ----------------------------
COV_NAME_RE = re.compile(r"^cov_(class|diag|shared|lowrank)(?:_k(\d+))?(?:_|$)")   # memory_table._COV_CONFIG_RE
CHEAP_MODES = ("diag", "lowrank", "shared")
D_FEAT = 768                     # BASE_CONFIG model_outdim: ViT-B/16 feature dim (main.py)
LORA_FLOATS_PER_RANK = 129_024   # 12 blocks x [(768+2304) + (768+3072) + (3072+768)] floats per unit rank
NORM_FLOATS = 2 * D_FEAT         # Model21.norm LayerNorm weight + bias saved with every LoRA state
E5_COLS = ["dataset", "config", "mode", "rank", "C", "d", "lora_r", "stats_per_class", "stats_floats", "stats_MB",
           "lora_floats", "lora_MB", "head_floats", "head_MB", "total_MB", "stats_over_lora", "n_seeds",
           "n_failed", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd", "old_mean", "new_acc", "first5_mean",
           "dfa_vs_class", "config_names"]


def stats_floats(mode, d, n_classes, rank=None):
    """(total floats, floats per class) of the stored class statistics; the formulas of
    covstats.storage_floats / memory_table.stats_per_class: class|full d + d^2 per class,
    diag 2d, lowrank d(k+2), shared d per class + d^2 once."""
    if mode in ("class", "full", "shared_corr"):
        per, once = d + d * d, 0
    elif mode == "diag":
        per, once = 2 * d, 0
    elif mode == "lowrank":
        if not rank:
            raise ValueError("lowrank needs a rank")
        per, once = d * (int(rank) + 2), 0
    elif mode == "shared":
        per, once = d, d * d
    else:
        raise ValueError(f"unknown cov mode {mode!r}")
    return n_classes * per + once, per


def _mb(floats):
    return floats * 4 / 1e6          # fp32, decimal MB (memory_table.mb)


@subcommand("e5", "FA vs storage of the train_ca_cov_mode variants (class/diag/lowrank/shared) "
                  "-> e5_storage_accuracy.csv (memory_table --results compatible)")
def cmd_e5(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e5", description=cmd_e5.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--all-pspectral", action="store_true",
                   help="also include runs not named cov_* whose train_ca_cov_mode is set (default: cov_* names "
                        "plus any run in a cheap mode)")
    args = p.parse_args(argv)
    runs = _load_tags(args, capture_tags=())
    sel = [r for r in runs if COV_NAME_RE.match(str(r["config_name"]))
           or r["config"].get("train_ca_cov_mode", "class") in CHEAP_MODES
           or (args.all_pspectral and "train_ca_cov_mode" in r["config"])]
    for r in sel:
        mode = str(r["config"].get("train_ca_cov_mode", "class"))
        m = COV_NAME_RE.match(str(r["config_name"]))
        if m and m.group(1) != mode:
            print(f"WARNING: {r['dataset']}/{r['config_name']} is named {m.group(1)} but runs cov mode {mode}",
                  file=sys.stderr)
        r["_mode"] = mode
        r["_rank"] = int(r["config"].get("train_ca_cov_rank", 64)) if mode == "lowrank" else None
    if not sel:
        print("(no cov_* / cheap-statistics runs found; nothing written)")
        return 0
    keys = ("dataset", "_mode", "_rank")
    cells = aggregate(sel, keys, metrics=_CELL_METRICS)
    rows = []
    for c in cells:
        members = _members(sel, c, keys)
        cfg = members[0]["config"]
        ds, mode, rank = c["dataset"], c["_mode"], c["_rank"]
        C = TOTAL_CLASSES.get(ds)
        d = int(cfg.get("model_outdim", D_FEAT) or D_FEAT)
        r_lora = int(cfg.get("model_lora_r", 64) or 64)
        row = OrderedDict(dataset=ds, config=_names(members)[0], mode=mode, rank=rank, C=C, d=d, lora_r=r_lora)
        if C is not None:
            total, per = stats_floats(mode, d, C, rank)
            lora = LORA_FLOATS_PER_RANK * r_lora + NORM_FLOATS
            head = C * d
            row.update(stats_per_class=per, stats_floats=total, stats_MB=_mb(total), lora_floats=lora,
                       lora_MB=_mb(lora), head_floats=head, head_MB=_mb(head), total_MB=_mb(total + lora + head),
                       stats_over_lora=total / lora)
        row.update(_std_metrics(c))
        row["n_failed"] = c["n_failed"]
        row["config_names"] = _names(members)
        rows.append(row)
    for row in rows:
        ref = next((x for x in rows if x["dataset"] == row["dataset"] and x["mode"] == "class"), None)
        row["dfa_vs_class"] = (row["fa"] - ref["fa"]) if (ref and ref["fa"] is not None and row["fa"] is not None) else None
    rows.sort(key=lambda r: (str(r["dataset"]), -(r.get("total_MB") or 0)))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "e5_storage_accuracy.csv")
    write_csv(path, rows, E5_COLS)
    _wrote(path, len(rows), "dataset x mode cells")
    for row in rows:
        row["FA"] = pm(row["fa"], row["fa_sd"])
        row["AA"] = pm(row["aa"], row["aa_sd"])
        row["FF"] = pm(row["ffm"], row["ffm_sd"])
        row["stats MB"] = f"{row['stats_MB']:.1f}" if row.get("stats_MB") is not None else "-"
        row["total MB"] = f"{row['total_MB']:.1f}" if row.get("total_MB") is not None else "-"
        row["dFA"] = f"{row['dfa_vs_class']:+.2f}" if row.get("dfa_vs_class") is not None else "-"
    print("\n### Accuracy vs storage (fp32 MB; LoRA module 33.0 MB at r=64, head C x 768 floats)")
    print(markdown_table(rows, ["dataset", "config", "mode", "rank", "C", "stats_per_class", "stats MB", "total MB",
                                "n_seeds", "FA", "AA", "FF", "dFA"]))
    missing = [ds for ds in HEADLINE if not any(r["dataset"] == ds for r in rows)]
    if missing:
        print(f"(no cov_* runs for {missing})")
    return 0


# --- E7: adaptive alpha and data-free lambda --------------------------------------------
E7_TAGS = ("MergeStat", "FeatShift", "Drift", "StepNorm")
E7_ALPHA_COLS = ["dataset", "config_name", "rule", "stat", "param", "seed", "task", "s_t", "alpha_t"]
E7_SUMMARY_COLS = ["dataset", "config_name", "rule", "stat", "param", "n_seeds", "n_failed", "alpha_mean", "alpha_sd",
                   "alpha_min", "alpha_max", "alpha_first_task", "alpha_last_task", "headline_alpha", "fa", "fa_sd",
                   "aa", "aa_sd", "ffm", "ffm_sd", "ref_config", "ref_fa", "ref_fa_sd", "fa_gap", "s_t_mean"]
E7_GCV_COLS = ["dataset", "config_name", "seed", "task", "lambda", "scores"]
E7_GCV_SUMMARY_COLS = ["dataset", "config_name", "n_seeds", "n_failed", "lambda_mode", "lambda_geomean",
                       "lambda_per_task", "headline_lambda", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd",
                       "ref_config", "ref_fa", "fa_gap"]


def _alpha_series(run):
    """[(task, s_t, alpha_t, rule, stat)] of one run: the '[Merging] adaptive alpha' lines
    (one per merge, tasks 1..T-1) or, under the fixed rule, the realised alphas of the
    '[Merging] pspectral' lines paired with the '[FeatShift]' statistic when logged."""
    rule = str(run["config"].get("merge_alpha_rule", "fixed"))
    out = []
    if run["adaptive_alphas"]:
        for i, (desc, alpha) in enumerate(run["adaptive_alphas"]):
            kv = kv_pairs(desc)
            out.append((i + 1, kv.get("s_t"), alpha, kv.get("rule", rule), kv.get("stat")))
        return out
    shifts = {rec["task"]: rec.get("shift") for rec in tag_records(run, "FeatShift", parse_featshift)}
    for i, alpha in enumerate(run["merge_alphas"]):
        t = i + 1
        out.append((t, shifts.get(t), alpha, "fixed", "shift" if t in shifts else None))
    return out


def _gcv_series(run):
    """[(task, lambda, scores)] from the '[Drift] lambda_rule=gcv' lines (run['drift_gcv']),
    aligned with the drift fits of tasks 1..T-1."""
    return [(i + 1, rec["lambda"], rec["scores"]) for i, rec in enumerate(run.get("drift_gcv") or [])]


@subcommand("e7", "realised adaptive alpha (specnorm / featshift) per dataset vs the tuned alphas, with the "
                  "FA gap to the paired headline cell; GCV-chosen lambda per task (e7_*.csv)")
def cmd_e7(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e7", description=cmd_e7.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    args = p.parse_args(argv)
    runs = _load_tags(args, capture_tags=E7_TAGS)
    adaptive = [r for r in runs if str(r["config"].get("merge_alpha_rule", "fixed")) != "fixed"
                or r["adaptive_alphas"] or _is_truthy(r["config"].get("merge_log_featshift"))]
    gcv = [r for r in runs if r["config"].get("train_drift_lambda_rule") == "gcv" or r.get("drift_gcv")]
    if not adaptive and not gcv:
        print("(no adaptive-alpha or gcv runs found; nothing written)")
        return 0
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- per-task realised alphas ----------------------------------------------------
    alpha_rows = []
    for r in adaptive:
        cfg = r["config"]
        for t, s_t, a, rule, stat in _alpha_series(r):
            alpha_rows.append(OrderedDict(dataset=r["dataset"], config_name=r["config_name"], rule=rule,
                                          stat=stat or cfg.get("merge_alpha_stat"), param=cfg.get("merge_alpha_param"),
                                          seed=r["seed"], task=t, s_t=s_t, alpha_t=a))
    path = os.path.join(args.out_dir, "e7_adaptive_alpha.csv")
    write_csv(path, alpha_rows, E7_ALPHA_COLS)
    _wrote(path, len(alpha_rows), "task rows")

    # ---- summary per (dataset, config) with the paired headline reference ------------
    keys = ("dataset", "config_name")
    summary = []
    for c in aggregate(adaptive, keys, metrics=_CELL_METRICS):
        members = _members(adaptive, c, keys)
        ok = _ok(members)
        cfg = members[0]["config"]
        ds = c["dataset"]
        per_seed = [[a for (_t, _s, a, _r, _st) in _alpha_series(r)] for r in ok]
        per_seed = [v for v in per_seed if v]
        flat = [a for v in per_seed for a in v]
        s_vals = [s for r in ok for (_t, s, _a, _r, _st) in _alpha_series(r) if s is not None]
        ref = _reference_cell(runs, ds, prefix=cfg.get("train_prefix"), prefer=("e7_", "e10_trunc", "mc3d_", "cov_class"),
                              exclude=(c["config_name"],))
        rule = str(cfg.get("merge_alpha_rule", "fixed"))
        stat = ("shift" if rule == "featshift" else cfg.get("merge_alpha_stat", "eff")) if rule != "fixed" \
            else ("shift" if s_vals else None)
        row = OrderedDict(dataset=ds, config_name=c["config_name"], rule=rule, stat=stat,
                          param=cfg.get("merge_alpha_param"), n_seeds=c["n_seeds"], n_failed=c["n_failed"],
                          alpha_mean=_mean([np.mean(v) for v in per_seed]) if per_seed else None,
                          alpha_sd=_sd([np.mean(v) for v in per_seed]) if per_seed else None,
                          alpha_min=min(flat) if flat else None, alpha_max=max(flat) if flat else None,
                          alpha_first_task=_mean([v[0] for v in per_seed]) if per_seed else None,
                          alpha_last_task=_mean([v[-1] for v in per_seed]) if per_seed else None,
                          headline_alpha=HEADLINE.get(ds, {}).get("alpha"))
        row.update(_std_metrics(c))
        row["ref_config"] = ref["config_name"] if ref else None
        row["ref_fa"] = ref["faa"] if ref else None
        row["ref_fa_sd"] = ref.get("faa_sd") if ref else None
        row["fa_gap"] = (row["fa"] - ref["faa"]) if (ref and row["fa"] is not None and ref["faa"] is not None) else None
        row["s_t_mean"] = _mean(s_vals)
        summary.append(row)
    summary.sort(key=lambda r: (str(r["rule"]), str(r["dataset"]), str(r["config_name"])))
    path = os.path.join(args.out_dir, "e7_summary.csv")
    write_csv(path, summary, E7_SUMMARY_COLS)
    _wrote(path, len(summary), "configs")
    for row in summary:
        row["alpha (realised)"] = _fmt_cell(row["alpha_mean"], row["alpha_sd"], 3)
        row["range"] = (f"{row['alpha_min']:.3f}-{row['alpha_max']:.3f}" if row["alpha_min"] is not None else "-")
        row["FA"] = pm(row["fa"], row["fa_sd"])
        row["ref FA (paired)"] = pm(row["ref_fa"], row["ref_fa_sd"])
        row["gap"] = f"{row['fa_gap']:+.2f}" if row["fa_gap"] is not None else "-"
        row["AA"], row["FF"] = pm(row["aa"], row["aa_sd"]), pm(row["ffm"], row["ffm_sd"])
    print("\n### Adaptive alpha: realised alpha per dataset next to the tuned headline alpha "
          "(reference = headline-posture cell on the same chain)")
    print(markdown_table(summary, ["dataset", "config_name", "rule", "stat", "param", "n_seeds", "alpha (realised)",
                                   "range", "headline_alpha", "FA", "AA", "FF", "ref_config", "ref FA (paired)", "gap"]))

    # ---- GCV lambda -------------------------------------------------------------------
    gcv_rows, gcv_summary = [], []
    for r in gcv:
        for t, lam, scores in _gcv_series(r):
            gcv_rows.append(OrderedDict(dataset=r["dataset"], config_name=r["config_name"], seed=r["seed"], task=t,
                                        **{"lambda": lam}, scores=" ".join(f"{k:g}:{v:.6g}" for k, v in scores.items())))
    path = os.path.join(args.out_dir, "e7_gcv_lambda.csv")
    write_csv(path, gcv_rows, E7_GCV_COLS)
    _wrote(path, len(gcv_rows), "task rows")
    for c in aggregate(gcv, keys, metrics=_CELL_METRICS):
        members = _members(gcv, c, keys)
        ok = _ok(members)
        ds = c["dataset"]
        lams = [lam for r in ok for (_t, lam, _s) in _gcv_series(r)]
        per_task = defaultdict(list)
        for r in ok:
            for t, lam, _s in _gcv_series(r):
                per_task[t].append(lam)
        mode = max(sorted(set(lams)), key=lams.count) if lams else None
        ref = _reference_cell(runs, ds, prefix=members[0]["config"].get("train_prefix"),
                              prefer=("e7_", "e10_trunc", "mc3d_", "cov_class"), exclude=(c["config_name"],))
        row = OrderedDict(dataset=ds, config_name=c["config_name"], n_seeds=c["n_seeds"], n_failed=c["n_failed"],
                          lambda_mode=mode,
                          lambda_geomean=float(np.exp(np.mean(np.log(lams)))) if lams and min(lams) > 0 else None,
                          lambda_per_task=[round(float(np.mean(per_task[t])), 3) for t in sorted(per_task)],
                          headline_lambda=HEADLINE.get(ds, {}).get("lambda"))
        row.update(_std_metrics(c))
        row["ref_config"] = ref["config_name"] if ref else None
        row["ref_fa"] = ref["faa"] if ref else None
        row["fa_gap"] = (row["fa"] - ref["faa"]) if (ref and row["fa"] is not None and ref["faa"] is not None) else None
        gcv_summary.append(row)
    path = os.path.join(args.out_dir, "e7_gcv_summary.csv")
    write_csv(path, gcv_summary, E7_GCV_SUMMARY_COLS)
    _wrote(path, len(gcv_summary), "configs")
    if gcv_summary:
        for row in gcv_summary:
            row["FA"], row["gap"] = pm(row["fa"], row["fa_sd"]), (f"{row['fa_gap']:+.2f}" if row["fa_gap"] is not None else "-")
            row["lambda per task"] = " ".join(f"{v:g}" for v in row["lambda_per_task"])
        print("\n### GCV-chosen ridge lambda (train_drift_lambda_rule=gcv) vs the tuned headline lambda")
        print(markdown_table(gcv_summary, ["dataset", "config_name", "n_seeds", "lambda_mode", "lambda_geomean",
                                           "lambda per task", "headline_lambda", "FA", "ref_config", "ref_fa", "gap"]))
    return 0


# --- E8: truncation (p = 0.9) vs damped interpolation (p = 1) ---------------------------
MC3D_RE = re.compile(r"^mc3d_(\w+?)_p([\d.]+)_a([\d.]+)$")      # export_mc3d_full.main
MC3D_COLS = ["variant", "p", "alpha", "n_seeds", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd",
             "old_mean", "new_acc", "first5_mean"]
MC3D_PROTECTED = ("mc3d_imagenetr_full.csv", "mc3d_cars_full.csv")   # committed cubes: never rewritten
E8_DATASETS = ("cifar224", "cub", "cars")


def _mc3d_key(variant, p, alpha):
    return (str(variant), round(float(p), 2), round(float(alpha), 2))


def read_mc3d(path):
    """Rows of an export_mc3d_full-style CSV (floats parsed) keyed by (variant, p, alpha)."""
    out = OrderedDict()
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            row = OrderedDict()
            for k, v in r.items():
                if k in ("variant",):
                    row[k] = v
                elif k == "n_seeds":
                    row[k] = int(float(v)) if v not in (None, "") else None
                else:
                    row[k] = float(v) if v not in (None, "") else None
            out[_mc3d_key(row["variant"], row["p"], row["alpha"])] = row
    return out


def write_mc3d(path, rows):
    """export_mc3d_full.py schema: p/alpha and every metric with two decimals."""
    base = os.path.basename(path)
    if base in MC3D_PROTECTED:
        raise RuntimeError(f"refusing to rewrite the committed cube {base}; the new cells go to the _extra file")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MC3D_COLS)
        for r in rows:
            out = []
            for c in MC3D_COLS:
                v = r.get(c)
                if c == "variant":
                    out.append(v)
                elif c == "n_seeds":
                    out.append("" if v is None else int(v))
                else:
                    out.append("" if v is None else f"{float(v):.2f}")
            w.writerow(out)
    return path


def mc3d_cells(runs, ds):
    """aggregate() of the mc3d_* runs of one dataset by the (variant, p, alpha) of the
    config NAME (export_mc3d_full semantics); the config values are cross-checked."""
    sel = []
    for r in runs:
        m = MC3D_RE.match(str(r["config_name"]))
        if r["dataset"] != ds or not m:
            continue
        r["_variant"], r["_p"], r["_alpha"] = m.group(1), round(float(m.group(2)), 4), round(float(m.group(3)), 4)
        sel.append(r)
    keys = ("_variant", "_p", "_alpha")
    rows = []
    for c in aggregate(sel, keys, metrics=_CELL_METRICS):
        members = _members(sel, c, keys)
        row = OrderedDict(variant=c["_variant"], p=c["_p"], alpha=c["_alpha"])
        row.update(_std_metrics(c))
        row["n_failed"] = c["n_failed"]
        row["config_names"] = _names(members)
        row["flags"] = []
        cfg = members[0]["config"]
        if cfg and not (_close(cfg.get("pspectral_p"), c["_p"], 1e-6) and _close(cfg.get("train_merge_alpha"), c["_alpha"], 1e-6)
                        and cfg.get("spectral_variant", cfg.get("pspectral_variant")) == c["_variant"]):
            row["flags"].append("config dump disagrees with the config name")
        if c["n_failed"]:
            row["flags"].append(f"failed seeds {c['failed_seeds']}")
        if any(r.get("summary_check") == "mismatch" for r in members):
            row["flags"].append("SUMMARY mismatch")
        rows.append(row)
    rows.sort(key=lambda r: (r["variant"], r["p"], r["alpha"]))
    return rows


def _e8_tables(ds, rows, cube):
    """Print the tab:alpha layout per dataset plus the paired p=0.9 vs p=1.0 differences."""
    table = []
    seen = set()
    for r in rows:
        if r["variant"] != HEADLINE_VARIANT:
            continue
        seen.add(_mc3d_key(r["variant"], r["p"], r["alpha"]))
        table.append(OrderedDict(p=r["p"], alpha=r["alpha"], FA=pm(r["fa"], r["fa_sd"]), AA=pm(r["aa"], r["aa_sd"]),
                                 FF=pm(r["ffm"], r["ffm_sd"]), old=_fmt_cell(r["old_mean"]), new=_fmt_cell(r["new_acc"]),
                                 oldest=_fmt_cell(r["first5_mean"]), n=r["n_seeds"], source="logs"))
    for key, r in cube.items():
        if r.get("variant") != HEADLINE_VARIANT or key in seen:
            continue
        table.append(OrderedDict(p=r["p"], alpha=r["alpha"], FA=pm(r["fa"], r["fa_sd"]), AA=pm(r["aa"], r["aa_sd"]),
                                 FF=pm(r["ffm"], r["ffm_sd"]), old=_fmt_cell(r["old_mean"]), new=_fmt_cell(r["new_acc"]),
                                 oldest=_fmt_cell(r["first5_mean"]), n=r["n_seeds"], source="cube"))
    table.sort(key=lambda r: (round(r["alpha"], 4), round(r["p"], 4)))
    head = HEADLINE.get(ds, {}).get("alpha")
    print(f"\n### {ds}: only_A cells (headline alpha {head}); p = 1.0 is damped interpolation, p = 0.9 truncation")
    print(markdown_table(table, ["p", "alpha", "FA", "AA", "FF", "old", "new", "oldest", "n", "source"]))
    by = {(round(r["p"], 4), round(r["alpha"], 4)): r for r in table}
    paired = []
    for (pp, a), r in sorted(by.items()):
        if abs(pp - HEADLINE_P) > 1e-6 or (1.0, a) not in by:
            continue
        r1 = by[(1.0, a)]
        f9, f1 = _f(r["FA"].split(" ")[0]), _f(r1["FA"].split(" ")[0])
        g9, g1 = _f(r["FF"].split(" ")[0]), _f(r1["FF"].split(" ")[0])
        paired.append(OrderedDict(alpha=a, **{"FA p0.9": r["FA"], "FA p1.0": r1["FA"],
                                               "dFA (trunc - damp)": f"{f9 - f1:+.2f}" if None not in (f9, f1) else "-",
                                               "FF p0.9": r["FF"], "FF p1.0": r1["FF"],
                                               "dFF": f"{g9 - g1:+.2f}" if None not in (g9, g1) else "-",
                                               "headline": "<-" if _close(a, head) else ""}))
    if paired:
        print(f"\n#### {ds}: paired p=0.9 vs p=1.0 at the same alpha")
        print(markdown_table(paired, list(paired[0].keys())))
    large = [r for r in table if _close(r["alpha"], 1.0)]
    if len(large) > 1:
        print(f"\n#### {ds}: large-step claim, alpha = 1.0 across p (keeping less energy should forget less)")
        print(markdown_table(sorted(large, key=lambda r: r["p"]), ["p", "FA", "AA", "FF", "old", "new", "n", "source"]))


@subcommand("e8", "truncation vs damped interpolation: mc3d_* cells -> mc3d_cifar224_full.csv, mc3d_cub_full.csv, "
                  "mc3d_cars_extra.csv (committed cubes untouched) + the tab:alpha tables")
def cmd_e8(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e8", description=cmd_e8.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data",
                   help="where the mc3d_*_full.csv / _extra.csv files live (default figures_data)")
    p.add_argument("--datasets", nargs="*", default=list(E8_DATASETS))
    args = p.parse_args(argv)
    runs = _load_tags(args, capture_tags=())
    for ds in args.datasets:                      # write_mc3d creates the out-dir when it writes
        rows = mc3d_cells(runs, ds)
        full_name = f"mc3d_{ds}_full.csv"
        full_path = os.path.join(args.out_dir, full_name)
        cube = read_mc3d(full_path) if full_name in MC3D_PROTECTED else OrderedDict()
        if not rows:
            print(f"\n### {ds}: no mc3d_* runs found; nothing written")
            if cube:
                _e8_tables(ds, rows, cube)
            continue
        if full_name in MC3D_PROTECTED:
            new = [r for r in rows if _mc3d_key(r["variant"], r["p"], r["alpha"]) not in cube]
            path = os.path.join(args.out_dir, f"mc3d_{ds}_extra.csv")
            write_mc3d(path, new)
            print(f"wrote {path}: {len(new)} cells not in the committed {full_name} "
                  f"({len(cube)} cube cells read; {len(rows) - len(new)} already there)")
            dup = [r for r in rows if _mc3d_key(r["variant"], r["p"], r["alpha"]) in cube]
            for r in dup:
                c = cube[_mc3d_key(r["variant"], r["p"], r["alpha"])]
                if c.get("fa") is not None and r["fa"] is not None and abs(c["fa"] - r["fa"]) > 0.0101:
                    print(f"  note: {ds} {r['variant']} p{r['p']} a{r['alpha']}: logs FA {r['fa']:.2f} vs cube {c['fa']:.2f}")
        else:
            write_mc3d(full_path, rows)
            _wrote(full_path, len(rows), "cells")
        for r in rows:
            if r["flags"]:
                print(f"  WARNING {ds} {r['variant']} p{r['p']} a{r['alpha']}: {'; '.join(r['flags'])}")
        _e8_tables(ds, rows, cube)
    return 0


# --- E9: drift-map diagnostics ----------------------------------------------------------
DRIFTDIAG_METRICS = ("mean_rel_raw", "mean_rel_before", "mean_rel_after", "mean_cos_raw", "mean_cos_before",
                     "mean_cos_after", "cov_relF_raw", "cov_relF_before", "cov_relF_after", "symkl_raw",
                     "symkl_before", "symkl_after", "os_mean_rel_I", "os_mean_rel_W", "os_mean_rel_oracle",
                     "os_cov_relF_I", "os_cov_relF_W", "os_cov_relF_oracle", "W_minus_I_F", "Wor_minus_I_F",
                     "W_minus_Wor_F", "n_fit_new", "n_fit_old", "n_classes", "n_img")
# Per-seed CSVs of run_driftdiag.py: driftdiag_t<N>.csv only; the --validate files
# (driftdiag_t10_validate.csv) and anything else are skipped with a note.
DRIFTDIAG_FILE_RE = re.compile(r"^driftdiag_(t\d+)\.csv$")


def read_driftdiag_csv(path):
    """Rows of one logs_main/<ds>/driftdiag_<split>.csv (main.DRIFTDIAG_CSV_HEADER) with the
    split taken from the file name and numeric fields parsed ('' -> None); [] (with a
    printed note) for a file whose name is not driftdiag_t<N>.csv."""
    m = DRIFTDIAG_FILE_RE.match(os.path.basename(path))
    if not m:
        print(f"(skip {path}: not a driftdiag_t<N>.csv file)")
        return []
    split = m.group(1)
    out = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            row = OrderedDict(dataset=r.get("dataset"), split=split, seed=_num(r.get("seed") or ""),
                              task=_num(r.get("task") or ""), group=str(r.get("group")), source="csv", path=path)
            for k in DRIFTDIAG_METRICS:
                v = r.get(k)
                row[k] = _num(v) if v not in (None, "") else None
                if isinstance(row[k], str):
                    row[k] = None
            out.append(row)
    return out


def driftdiag_rows_from_runs(runs):
    """'[DriftDiag]' log lines -> rows in the CSV layout (group 'all' only)."""
    out = []
    for r in runs:
        for rec in tag_records(r, "DriftDiag", parse_driftdiag):
            row = OrderedDict(dataset=r["dataset"], split=r["split"], seed=r["seed"], task=rec["task"], group="all",
                              source="log", path=r["path"])
            for k in DRIFTDIAG_METRICS:
                row[k] = rec.get(k)
            row["n_img"] = rec.get("n_img")
            out.append(row)
    return out


def _e9_aggregate(rows):
    """(dataset, split, task, group) -> n_seeds, seeds, source and, per metric, the seed
    mean under the bare metric name plus ``<metric>_sd`` (std, ddof 0, like the SUMMARY
    lines; the schema scripts/fig_e9_driftdiag.py reads).  A seed that appears twice for
    one key (an appended re-run of the CSV) counts once with its LAST row."""
    groups = OrderedDict()
    for r in rows:
        key = (r["dataset"], r["split"], r["task"], r["group"])
        groups.setdefault(key, OrderedDict())[r["seed"]] = r        # last row per seed wins
    out = []
    for (ds, split, task, group), by_seed in groups.items():
        rs = list(by_seed.values())
        seeds = sorted(s for s in by_seed if s is not None)
        row = OrderedDict(dataset=ds, split=split, task=task, group=group, n_seeds=len(by_seed),
                          seeds=";".join(str(int(s)) if float(s).is_integer() else str(s) for s in seeds),
                          source=";".join(sorted({r["source"] for r in rs})))
        for k in DRIFTDIAG_METRICS:
            row[k] = _mean([r[k] for r in rs])
            row[k + "_sd"] = _sd([r[k] for r in rs])
        out.append(row)
    out.sort(key=lambda r: (str(r["dataset"]), str(r["split"]), r["task"] if r["task"] is not None else -1,
                            (0, 0) if r["group"] == "all" else (1, _num(r["group"]) if isinstance(_num(r["group"]), int) else 0)))
    return out


@subcommand("e9", "drift-map diagnostics: seed-averaged e9_driftdiag_<ds>_<split>.csv from the per-seed "
                  "driftdiag_<split>.csv files (or the [DriftDiag] lines) + the error table")
def cmd_e9(argv):
    """The one collector of the E9 figure data: figures_data/e9_driftdiag_<ds>_<split>.csv
    with columns dataset, split, task, group, n_seeds, seeds, source and, for every
    DRIFTDIAG_METRICS entry, <metric> (seed mean) and <metric>_sd (std, ddof 0); one row
    per (task, group), group 'all' = every old class.  Input: the per-seed CSVs
    logs_main/<ds>/driftdiag_t<N>.csv of run_driftdiag.py (appended across seeds; the last
    row of a seed wins), else the '[DriftDiag]' log lines (group 'all' only).  The
    printed table gives, per dataset and split, the last-task raw vs after errors, the
    one-step identity / W / oracle means and the slope of the error against the task
    index.  scripts/fig_e9_driftdiag.py reads the CSVs."""
    p = argparse.ArgumentParser(prog="parse_logs.py e9", description=cmd_e9.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--csv-glob", nargs="*", default=None,
                   help="per-seed CSVs (default <root>/*/driftdiag_*.csv for every root)")
    args = p.parse_args(argv)
    patterns = args.csv_glob
    if patterns is None:
        patterns = [os.path.join(root, "*", "driftdiag_*.csv") for root in args.roots]
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            rows.extend(read_driftdiag_csv(path))
    have_csv = {(r["dataset"], r["split"]) for r in rows}
    runs = _ok(_load_tags(args, capture_tags=("DriftDiag",)))
    log_rows = [r for r in driftdiag_rows_from_runs(runs) if (r["dataset"], r["split"]) not in have_csv]
    rows.extend(log_rows)
    if not rows:
        print("(no driftdiag CSVs or [DriftDiag] lines found)")
        return 0
    agg = _e9_aggregate(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["dataset", "split", "task", "group", "n_seeds", "seeds", "source"] + \
        [k + s for k in DRIFTDIAG_METRICS for s in ("", "_sd")]
    for (ds, split) in sorted({(r["dataset"], r["split"]) for r in agg}, key=str):
        part = [r for r in agg if r["dataset"] == ds and r["split"] == split]
        path = os.path.join(args.out_dir, f"e9_driftdiag_{ds}_{split}.csv")
        write_csv(path, part, cols)
        _wrote(path, len(part), "task x group rows")
    table = []
    for (ds, split) in sorted({(r["dataset"], r["split"]) for r in agg}, key=str):
        allg = [r for r in agg if r["dataset"] == ds and r["split"] == split and r["group"] == "all"
                and r["task"] is not None]
        if not allg:
            continue
        allg.sort(key=lambda r: r["task"])
        lastr = allg[-1]
        tasks = np.array([r["task"] for r in allg], float)

        def slope(key):
            ys = [r[key] for r in allg]
            ok = [(t, y) for t, y in zip(tasks, ys) if y is not None]
            if len(ok) < 2:
                return None
            tt, yy = np.array([o[0] for o in ok]), np.array([o[1] for o in ok])
            return float(np.polyfit(tt, yy, 1)[0])
        table.append(OrderedDict(
            dataset=ds, split=split, n_seeds=lastr["n_seeds"], last_task=int(lastr["task"]),
            **{"mean_rel raw (last)": _fmt_cell(lastr["mean_rel_raw"], lastr["mean_rel_raw_sd"], 4),
               "mean_rel after (last)": _fmt_cell(lastr["mean_rel_after"], lastr["mean_rel_after_sd"], 4),
               "cov_relF raw (last)": _fmt_cell(lastr["cov_relF_raw"], None, 4),
               "cov_relF after (last)": _fmt_cell(lastr["cov_relF_after"], None, 4),
               "onestep I / W / oracle (mean_rel, task mean)": " / ".join(
                   _fmt_cell(_mean([r[k] for r in allg]), None, 4)
                   for k in ("os_mean_rel_I", "os_mean_rel_W", "os_mean_rel_oracle")),
               "slope after/task": _fmt_cell(slope("mean_rel_after"), None, 5),
               "slope raw/task": _fmt_cell(slope("mean_rel_raw"), None, 5),
               "source": lastr["source"]}))
    print("\n### Drift-map diagnostics (group 'all'; relative L2 error of the old-class means, raw = never "
          "corrected, after = transported; one-step: identity vs fitted W vs oracle W)")
    print(markdown_table(table, list(table[0].keys()) if table else ["dataset"]))
    return 0


# --- E10: truncation vs other step limits -----------------------------------------------
E10_TAGS = ("MergeStat", "StepNorm", "L2M")
E10_NORM_FIELDS = ("k_mean", "k_min", "k_max", "A_sigma_max_mean", "A_sigma_max_max", "eff_sigma_max_mean",
                   "eff_sigma_max_max", "eff_fro_mean", "raw_sigma_max_mean", "raw_eff_sigma_max_mean", "merge_time")
E10_TASK_COLS = ["dataset", "config_name", "rule", "seed", "task", "source", "alpha_t"] + list(E10_NORM_FIELDS) + ["l2m_dist"]
E10_COLS = ["dataset", "rule", "p", "alpha_or_weight", "config_names", "n_seeds", "n_failed", "fa", "fa_sd", "aa", "aa_sd",
            "ffm", "ffm_sd", "old_mean", "new_acc", "first5_mean", "norm_source", "k_mean", "k_max",
            "eff_sigma_max_mean", "eff_sigma_max_max", "eff_fro_mean", "A_sigma_max_mean", "raw_sigma_max_mean",
            "raw_eff_sigma_max_mean", "merge_time_s", "l2m_dist_mean", "source"]


def e10_rule(cfg):
    """Rule label of a step-control cell: trunc / clip / scale_fro (merge-time ops at p < 1),
    scale_p1 (full-update scaling = pspectral at p = 1), l2m / specreg / speccap (training-time
    arms, train_merge none); None for anything else."""
    rule = str(cfg.get("train_merge", "none"))
    if rule == "pspectral":
        if _close(cfg.get("pspectral_p"), 1.0):
            return "scale_p1"
        return str(cfg.get("merge_update_op", "trunc"))
    if rule == "none":
        if _is_truthy(cfg.get("l2m_weight")):
            return "l2m"
        if _is_truthy(cfg.get("specreg_weight")):
            return "specreg"
        if _is_truthy(cfg.get("speccap")):
            return "speccap"
    return None


def e10_value(cfg, rule):
    if rule in ("trunc", "clip", "scale_fro", "scale_p1"):
        return _cfgf({"config": cfg}, "train_merge_alpha")
    if rule == "l2m":
        return _cfgf({"config": cfg}, "l2m_weight")
    if rule == "specreg":
        return _cfgf({"config": cfg}, "specreg_weight")
    if rule == "speccap":
        return _cfgf({"config": cfg}, "speccap")
    return None


def step_norm_rows(run):
    """Per-task norm records of one run from [MergeStat] (merging runs) or [StepNorm]
    (merge-less runs) plus the [L2M] end-of-task distance."""
    out = []
    rule = e10_rule(run["config"])
    dists = {}
    for line in run["tags"].get("L2M", []):
        m = L2M_DIST_RE.match(line)
        if m:
            dists[len(dists) + 1] = float(m.group(1))      # one per trained task t >= 1, in order
    for tag in ("MergeStat", "StepNorm"):
        for rec in tag_records(run, tag, parse_norm_line):
            if rec["degenerate"]:
                continue
            row = OrderedDict(dataset=run["dataset"], config_name=run["config_name"], rule=rule, seed=run["seed"],
                              task=rec["task"], source=tag,
                              alpha_t=rec.get("alpha", run["merge_alphas"][rec["task"] - 1]
                                              if rec["task"] and len(run["merge_alphas"]) >= rec["task"] else None))
            for k in E10_NORM_FIELDS:
                row[k] = rec.get(k)
            row["l2m_dist"] = dists.get(rec["task"])
            out.append(row)
    if not out and dists:
        for t, d in dists.items():
            row = OrderedDict(dataset=run["dataset"], config_name=run["config_name"], rule=rule, seed=run["seed"],
                              task=t, source="L2M", alpha_t=None)
            for k in E10_NORM_FIELDS:
                row[k] = None
            row["l2m_dist"] = d
            out.append(row)
    return out


@subcommand("e10", "step-control table: truncation / clipping / Frobenius-matched scaling / p=1 scaling / "
                   "L2-to-anchor training with their realised step norms (e10_step_control.csv)")
def cmd_e10(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e10", description=cmd_e10.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--cubes-dir", default="figures_data", help="mc3d_<ds>_full.csv / _extra.csv for the p=1 rows")
    p.add_argument("--datasets", nargs="*", default=None)
    args = p.parse_args(argv)
    runs = _load_tags(args, capture_tags=E10_TAGS)
    sel = [r for r in runs if (str(r["config_name"]).startswith("e10_")
                               or str(r["config"].get("merge_update_op", "trunc")) != "trunc"
                               or _is_truthy(r["config"].get("l2m_weight"))
                               or _is_truthy(r["config"].get("specreg_weight")) or _is_truthy(r["config"].get("speccap")))
           and e10_rule(r["config"]) is not None]
    if args.datasets:
        sel = [r for r in sel if r["dataset"] in args.datasets]
    if not sel:
        print("(no e10 step-control runs found; nothing written)")
        return 0
    datasets = sorted({r["dataset"] for r in sel}, key=str)
    # p = 1 scaling rows: the pspectral p=1 cells of the same datasets at the alphas of the op cells.
    alphas = {ds: sorted({round(e10_value(r["config"], e10_rule(r["config"])), 4) for r in sel
                          if r["dataset"] == ds and e10_rule(r["config"]) in ("trunc", "clip", "scale_fro")
                          and e10_value(r["config"], e10_rule(r["config"])) is not None}) for ds in datasets}
    for r in runs:
        cfg = r["config"]
        if (r["dataset"] in alphas and cfg.get("train_merge") == "pspectral" and _close(cfg.get("pspectral_p"), 1.0)
                and cfg.get("spectral_variant", cfg.get("pspectral_variant", "only_A")) == HEADLINE_VARIANT
                and any(_close(cfg.get("train_merge_alpha"), a) for a in alphas[r["dataset"]]) and r not in sel):
            sel.append(r)
    for r in sel:
        r["_rule"] = e10_rule(r["config"])
        r["_val"] = round(e10_value(r["config"], r["_rule"]), 6) if e10_value(r["config"], r["_rule"]) is not None else None
        r["_p"] = _cfgf(r, "pspectral_p") if r["config"].get("train_merge") == "pspectral" else None
    keys = ("dataset", "_rule", "_p", "_val")
    task_rows, cells = [], []
    for r in sel:
        task_rows.extend(step_norm_rows(r))
    for c in aggregate(sel, keys, metrics=_CELL_METRICS):
        members = _members(sel, c, keys)
        trs = [t for t in task_rows if t["dataset"] == c["dataset"] and any(t["config_name"] == m["config_name"]
                                                                            and t["seed"] == m["seed"] for m in _ok(members))]
        row = OrderedDict(dataset=c["dataset"], rule=c["_rule"], p=c["_p"], alpha_or_weight=c["_val"],
                          config_names=_names(members))
        row.update(_std_metrics(c))
        row["n_failed"] = c["n_failed"]
        srcs = sorted({t["source"] for t in trs})
        row["norm_source"] = ";".join(srcs) if srcs else None
        for k in ("k_mean", "k_max", "eff_sigma_max_mean", "eff_sigma_max_max", "eff_fro_mean", "A_sigma_max_mean",
                  "raw_sigma_max_mean", "raw_eff_sigma_max_mean"):
            row[k] = _mean([t[k] for t in trs if t["source"] in ("MergeStat", "StepNorm")])
        row["merge_time_s"] = _mean([t["merge_time"] for t in trs])
        row["l2m_dist_mean"] = _mean([t["l2m_dist"] for t in trs])
        row["source"] = "logs"
        cells.append(row)
    # p = 1 rows missing from the logs: take FA from the cubes (no norms there).
    for ds in datasets:
        cube = OrderedDict()
        for name in (f"mc3d_{ds}_full.csv", f"mc3d_{ds}_extra.csv"):
            cube.update(read_mc3d(os.path.join(args.cubes_dir, name)))
        for a in alphas[ds]:
            if any(c["dataset"] == ds and c["rule"] == "scale_p1" and _close(c["alpha_or_weight"], a) for c in cells):
                continue
            cr = cube.get(_mc3d_key(HEADLINE_VARIANT, 1.0, a))
            if cr:
                row = OrderedDict(dataset=ds, rule="scale_p1", p=1.0, alpha_or_weight=a, config_names=["cube"],
                                  n_seeds=cr.get("n_seeds"), fa=cr.get("fa"), fa_sd=cr.get("fa_sd"), aa=cr.get("aa"),
                                  aa_sd=cr.get("aa_sd"), ffm=cr.get("ffm"), ffm_sd=cr.get("ffm_sd"),
                                  old_mean=cr.get("old_mean"), new_acc=cr.get("new_acc"), first5_mean=cr.get("first5_mean"),
                                  n_failed=0, norm_source=None, source="cube")
                for k in ("k_mean", "k_max", "eff_sigma_max_mean", "eff_sigma_max_max", "eff_fro_mean", "A_sigma_max_mean",
                          "raw_sigma_max_mean", "raw_eff_sigma_max_mean", "merge_time_s", "l2m_dist_mean"):
                    row[k] = None
                cells.append(row)
    order = {"trunc": 0, "scale_p1": 1, "clip": 2, "scale_fro": 3, "l2m": 4, "specreg": 5, "speccap": 6}
    cells.sort(key=lambda c: (str(c["dataset"]), order.get(c["rule"], 9), c["alpha_or_weight"] or 0))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "e10_step_control.csv")
    write_csv(path, cells, E10_COLS)
    _wrote(path, len(cells), "cells")
    path = os.path.join(args.out_dir, "e10_step_norms_per_task.csv")
    write_csv(path, task_rows, E10_TASK_COLS)
    _wrote(path, len(task_rows), "task rows")
    for c in cells:
        c["FA"], c["AA"], c["FF"] = pm(c["fa"], c["fa_sd"]), pm(c["aa"], c["aa_sd"]), pm(c["ffm"], c["ffm_sd"])
        c["eff sigma_max"] = _fmt_cell(c["eff_sigma_max_mean"], None, 4)
        c["eff fro"] = _fmt_cell(c["eff_fro_mean"], None, 4)
        c["raw sigma_max"] = _fmt_cell(c["raw_eff_sigma_max_mean"], None, 4)
        c["k"] = _fmt_cell(c["k_mean"], None, 1)
        c["merge s"] = _fmt_cell(c["merge_time_s"], None, 3)
        c["||W-anchor||"] = _fmt_cell(c["l2m_dist_mean"], None, 3)
        c["value"] = f"{c['alpha_or_weight']:g}" if c["alpha_or_weight"] is not None else "-"
    print("\n### Step control at matched spectral norm (eff = realised effective-weight step scale*(BA - B'A'); "
          "raw = the untruncated update; p=1 rows from the cubes carry no norms)")
    print(markdown_table(cells, ["dataset", "rule", "p", "value", "n_seeds", "FA", "AA", "FF", "k", "eff sigma_max",
                                 "eff fro", "raw sigma_max", "merge s", "||W-anchor||", "norm_source", "source"]))
    return 0


# --- E11: LoRA rank sweep on ImageNet-R -------------------------------------------------
E11_COLS = ["r", "lora_alpha", "train_prefix", "config_names", "n_seeds", "n_failed", "trainable_params",
            "expected_params", "params_match", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd", "old_mean", "new_acc",
            "first5_mean", "k_mean", "k_max_mean", "frac_tasks_kmax_eq_r", "merge_time_s", "eff_sigma_max_mean",
            "e4_k_mean", "e4_frac_layers_k_eq_r", "e4_rows", "svd_ms_per_merge_cpu", "svd_ms_per_merge_cuda"]
TRAINABLE_PER_RANK = 129_024    # [Freeze] counts the PEFT 'lora_' tensors only (norm.* excluded): 12 x 10,752 x r


def read_e4_spectrum(path, p=0.9):
    """(mean k, fraction of rows with k == r, n rows) of an analyze_spectrum.py CSV
    (columns seed,task,block,module,space,n_sv,sigma1,fro,k_p0.5..k_p0.99,...), A space."""
    if not path or not os.path.isfile(path):
        return None
    ks = []
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        # analyze_spectrum writes k_p0.9; accept k_p0.90 / k_p0.900 as well
        col = next((c for c in (rd.fieldnames or []) if c.startswith("k_p") and _close(_f(c[3:]), p, 1e-6)), None)
        if col is None:
            return None
        for r in rd:
            if (r.get("space") or "A") != "A":
                continue
            try:
                ks.append(int(float(r[col])))
            except (TypeError, ValueError):
                continue
    return ks


def read_svd_time(path):
    """figures_data/e11_svd_time.csv (r, device, module, ms_per_layer, ms_per_merge) ->
    {(r, device): sum over modules of ms_per_merge}."""
    out = defaultdict(float)
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                out[(int(float(r["r"])), r["device"].strip())] += float(r["ms_per_merge"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


@subcommand("e11", "LoRA rank sweep: FA, kept k, trainable params ([Freeze]), merge time ([MergeStat]), "
                   "E4 spectra and SVD bench joined per rank (e11_rank_imagenetr.csv)")
def cmd_e11(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e11", description=cmd_e11.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--dataset", default="imagenetr")
    p.add_argument("--pattern", default=r"^e11_", help="config-name regex of the rank cells")
    p.add_argument("--spectrum-dir", default="figures_data",
                   help="e4_spectrum_<ds>_r<r>.csv files from analyze_spectrum.py --tag _r<r>")
    p.add_argument("--svd-csv", default=None, help="bench_svd.py output (default <spectrum-dir>/e11_svd_time.csv)")
    args = p.parse_args(argv)
    runs = _load_tags(args, capture_tags=("MergeStat", "Freeze"))
    sel = [r for r in runs if r["dataset"] == args.dataset and re.search(args.pattern, str(r["config_name"]))]
    if not sel:
        print(f"(no runs matching {args.pattern!r} on {args.dataset}; nothing written)")
        return 0
    for r in sel:
        r["_r"] = int(r["config"].get("model_lora_r", 64) or 64)
    svd = read_svd_time(args.svd_csv or os.path.join(args.spectrum_dir, "e11_svd_time.csv"))
    rows = []
    for c in aggregate(sel, ("_r",), metrics=_CELL_METRICS):
        members = _members(sel, c, ("_r",))
        ok = _ok(members)
        r_ = c["_r"]
        recs = [rec for m in ok for rec in tag_records(m, "MergeStat", parse_norm_line) if not rec["degenerate"]]
        params = sorted({m["trainable_params"] for m in members if m["trainable_params"] is not None})
        ks = read_e4_spectrum(os.path.join(args.spectrum_dir, f"e4_spectrum_{args.dataset}_r{r_}.csv"))
        row = OrderedDict(r=r_, lora_alpha=members[0]["config"].get("model_lora_alpha"),
                          train_prefix=members[0]["config"].get("train_prefix"), config_names=_names(members))
        row.update(_std_metrics(c))
        row["n_failed"] = c["n_failed"]
        row["trainable_params"] = params[0] if len(params) == 1 else (params if params else None)
        row["expected_params"] = TRAINABLE_PER_RANK * r_
        row["params_match"] = (params == [TRAINABLE_PER_RANK * r_]) if params else None
        row["k_mean"] = _mean([rec.get("k_mean") for rec in recs])
        row["k_max_mean"] = _mean([rec.get("k_max") for rec in recs])
        kmax = [rec.get("k_max") for rec in recs if rec.get("k_max") is not None]
        row["frac_tasks_kmax_eq_r"] = (sum(1 for k in kmax if k == r_) / len(kmax)) if kmax else None
        row["merge_time_s"] = _mean([rec.get("merge_time") for rec in recs])
        row["eff_sigma_max_mean"] = _mean([rec.get("eff_sigma_max_mean") for rec in recs])
        row["e4_k_mean"] = _mean(ks) if ks else None
        row["e4_frac_layers_k_eq_r"] = (sum(1 for k in ks if k == r_) / len(ks)) if ks else None
        row["e4_rows"] = len(ks) if ks else None
        row["svd_ms_per_merge_cpu"] = svd.get((r_, "cpu"))
        row["svd_ms_per_merge_cuda"] = svd.get((r_, "cuda"), svd.get((r_, "gpu")))
        rows.append(row)
    rows.sort(key=lambda r: r["r"])
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"e11_rank_{args.dataset}.csv")
    write_csv(path, rows, E11_COLS)
    _wrote(path, len(rows), "ranks")
    for row in rows:
        row["FA"], row["AA"], row["FF"] = pm(row["fa"], row["fa_sd"]), pm(row["aa"], row["aa_sd"]), pm(row["ffm"], row["ffm_sd"])
        row["params"] = (f"{row['trainable_params']:,}" if isinstance(row["trainable_params"], int) else str(row["trainable_params"]))
        row["k (MergeStat)"] = _fmt_cell(row["k_mean"], None, 1)
        row["k_max = r"] = _fmt_cell(row["frac_tasks_kmax_eq_r"], None, 2)
        row["k (E4)"] = _fmt_cell(row["e4_k_mean"], None, 1)
        row["layers k = r"] = _fmt_cell(row["e4_frac_layers_k_eq_r"], None, 2)
        row["merge s"] = _fmt_cell(row["merge_time_s"], None, 3)
        row["svd ms cpu/cuda"] = f"{_fmt_cell(row['svd_ms_per_merge_cpu'], None, 1)} / {_fmt_cell(row['svd_ms_per_merge_cuda'], None, 1)}"
        row["params ok"] = {True: "yes", False: "NO", None: "-"}[row["params_match"]]
    print(f"\n### LoRA rank sweep on {args.dataset} (k = kept directions at p=0.9; params = [Freeze] trainable, "
          f"expected {TRAINABLE_PER_RANK:,} x r)")
    print(markdown_table(rows, ["r", "lora_alpha", "n_seeds", "params", "params ok", "FA", "AA", "FF", "k (MergeStat)",
                                "k_max = r", "k (E4)", "layers k = r", "merge s", "svd ms cpu/cuda"]))
    return 0


# --- E12: equal tuning budget for the other merge rules ----------------------------------
E12_CLASSICAL = ("avg", "ties", "max_abs", "max", "min")          # helper.merge rules (incremental chain expected)
E12_SOTA = ("dare", "della", "knots", "breadcrumbs", "modelstock")
E12_RULES = E12_CLASSICAL + E12_SOTA
E12_COLS = ["dataset", "rule", "coef", "config_names", "n_seeds", "n_failed", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd",
            "old_mean", "new_acc", "first5_mean", "incremental_observed", "extras", "flags"]
E12_ORDER = ("cifar224", "cub", "imagenetr", "cars")


def _rule_label(rule):
    return {"avg": "Task Arithmetic (avg)", "ties": "TIES", "max_abs": "max-abs", "dare": "DARE", "della": "DELLA",
            "knots": "KnOTS", "breadcrumbs": "Breadcrumbs", "modelstock": "Model Stock"}.get(rule, rule)


# Cells with these keys off their default are not ISM tuning cells (other experiments' arms).
_E12_NOT_TUNING = ("data_val_frac", "data_fewshot_k", "merge_alpha_rule", "merge_update_op", "train_ca_cov_mode",
                   "train_drift_lambda_rule", "l2m_weight", "train_epochs_small_task")


def _ism_budget_from_record(path, ds):
    """ISM cells per dataset from figures_data/e2_tuning_record.csv: full-data 10-task
    pspectral cells with every rebuttal switch at its default ('only_A, p=0.9' alpha cells
    and all pspectral cells)."""
    if not path or not os.path.isfile(path):
        return None
    n_all, n_head = 0, 0
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("dataset") != ds or r.get("rule") != "pspectral" or r.get("split") not in ("t10", ""):
                continue
            if any(k + "=" in (r.get("extras") or "") for k in _E12_NOT_TUNING):
                continue
            n_all += 1
            if r.get("spectral_variant") == HEADLINE_VARIANT and _close(_f(r.get("pspectral_p")), HEADLINE_P):
                n_head += 1
    return {"all_pspectral_cells": n_all, "p0.9_only_A_alpha_cells": n_head}


def _ism_budget_from_cubes(cubes_dir, ds):
    cube = OrderedDict()
    for name in (f"mc3d_{ds}_full.csv", f"mc3d_{ds}_extra.csv"):
        cube.update(read_mc3d(os.path.join(cubes_dir, name)))
    head = [k for k in cube if k[0] == HEADLINE_VARIANT and _close(k[1], HEADLINE_P)]
    return {"all_pspectral_cells": len(cube), "p0.9_only_A_alpha_cells": len(head)} if cube else None


def _ta_interpolation_row(cubes_dir, runs, ds):
    """'Task Arithmetic, interpolation form, tuned per dataset': best FA over alpha of the
    pspectral only_A p=1 cells (logs first, then the cubes)."""
    cands = OrderedDict()
    for k, r in read_mc3d(os.path.join(cubes_dir, f"mc3d_{ds}_full.csv")).items():
        if k[0] == HEADLINE_VARIANT and _close(k[1], 1.0):
            cands[k[2]] = (r.get("fa"), r.get("fa_sd"), r.get("aa"), r.get("aa_sd"), r.get("ffm"), r.get("ffm_sd"), "cube")
    for k, r in read_mc3d(os.path.join(cubes_dir, f"mc3d_{ds}_extra.csv")).items():
        if k[0] == HEADLINE_VARIANT and _close(k[1], 1.0):
            cands[k[2]] = (r.get("fa"), r.get("fa_sd"), r.get("aa"), r.get("aa_sd"), r.get("ffm"), r.get("ffm_sd"), "extra")
    sel = [r for r in runs if r["dataset"] == ds and r["status"] == "ok" and r["config"].get("train_merge") == "pspectral"
           and _close(r["config"].get("pspectral_p"), 1.0)
           and r["config"].get("spectral_variant", r["config"].get("pspectral_variant", "only_A")) == HEADLINE_VARIANT
           and r["split"] == "t10"]
    for r in sel:
        r["_alpha"] = round(_cfgf(r, "train_merge_alpha"), 4)
    for c in aggregate(sel, ("_alpha",), metrics=_CELL_METRICS):
        if c["n_seeds"]:
            cands[c["_alpha"]] = (c["faa"], c["faa_sd"], c["asa"], c["asa_sd"], c["ffm"], c["ffm_sd"], "logs")
    cands = {a: v for a, v in cands.items() if v[0] is not None}
    if not cands:
        return None
    a = max(cands, key=lambda k: cands[k][0])
    fa, fa_sd, aa, aa_sd, ffm, ffm_sd, src = cands[a]
    return OrderedDict(alpha=a, fa=fa, fa_sd=fa_sd, aa=aa, aa_sd=aa_sd, ffm=ffm, ffm_sd=ffm_sd, n_cells=len(cands), source=src)


@subcommand("e12", "merge-rule coefficient grid: e12_merge_grid.csv and the two tables (per-dataset best "
                   "coefficient; ImageNet-R-selected coefficient reused) against the ISM cell on the same chain")
def cmd_e12(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e12", description=cmd_e12.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--e2-record", default=None,
                   help="figures_data/e2_tuning_record.csv for ISM's real budget (default <out-dir>/e2_tuning_record.csv)")
    p.add_argument("--cubes-dir", default="figures_data")
    p.add_argument("--select-on", default="imagenetr", help="dataset whose best coefficient is reused (paper protocol)")
    p.add_argument("--pattern", default=r"^e12_", help="config-name regex of the grid cells (plus any run with a "
                                                        "grid rule and train_merge_coef when --all-rules)")
    p.add_argument("--all-rules", action="store_true")
    args = p.parse_args(argv)
    runs = _load_tags(args, capture_tags=())
    grid = [r for r in runs if str(r["config"].get("train_merge")) in E12_RULES
            and (re.search(args.pattern, str(r["config_name"])) or args.all_rules)]
    if not grid:
        print("(no e12 grid runs found; nothing written)")
        return 0
    for r in grid:
        r["_rule"] = str(r["config"].get("train_merge"))
        r["_coef"] = round(_cfgf(r, "train_merge_coef", 1.0), 6)
    keys = ("dataset", "_rule", "_coef")
    cells = []
    for c in aggregate(grid, keys, metrics=_CELL_METRICS):
        members = _members(grid, c, keys)
        row = OrderedDict(dataset=c["dataset"], rule=c["_rule"], coef=c["_coef"], config_names=_names(members))
        row.update(_std_metrics(c))
        row["n_failed"] = c["n_failed"]
        obs = {m["merge_incremental_observed"] for m in members}
        row["incremental_observed"] = (True if obs == {True} else (False if False in obs else None))
        row["extras"] = _extras(members[0]["config"], c["_rule"])
        flags = []
        if c["_rule"] in E12_CLASSICAL and row["incremental_observed"] is False:
            flags.append("NOT the incremental pairwise chain ('[Merging] Merging N task checkpoints' with N > 2)")
        if c["_rule"] in E12_CLASSICAL and not members[0]["config"].get("train_merge_incremental", False):
            flags.append("train_merge_incremental is False")
        if c["n_failed"]:
            flags.append(f"failed seeds {c['failed_seeds']}")
        if any(m.get("summary_check") == "mismatch" for m in members):
            flags.append("SUMMARY mismatch")
        row["flags"] = "; ".join(flags)
        cells.append(row)
    cells.sort(key=lambda r: (E12_ORDER.index(r["dataset"]) if r["dataset"] in E12_ORDER else 9, r["rule"], r["coef"]))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "e12_merge_grid.csv")
    write_csv(path, cells, E12_COLS)
    _wrote(path, len(cells), "cells")
    for c in cells:
        if c["flags"]:
            print(f"  WARNING {c['dataset']} {c['rule']} c{c['coef']}: {c['flags']}")
    datasets = [d for d in E12_ORDER if any(c["dataset"] == d for c in cells)] + \
               sorted({c["dataset"] for c in cells if c["dataset"] not in E12_ORDER}, key=str)
    rules = sorted({c["rule"] for c in cells}, key=lambda r: E12_RULES.index(r) if r in E12_RULES else 99)
    rec_path = args.e2_record or os.path.join(args.out_dir, "e2_tuning_record.csv")

    def fa_aa(c):
        return f"{_fmt_cell(c['fa'])} / {_fmt_cell(c['aa'])}" if c and c.get("fa") is not None else "-"

    def complete(cs):
        return [c for c in cs if c["n_seeds"] >= 3 and c["n_failed"] == 0] or [c for c in cs if c["n_seeds"]]

    # ISM rows: the paired headline cell on the same chain (prefer e12_ism*), budget per dataset.
    ism = {ds: _reference_cell(runs, ds, prefer=("e12_ism", "e8_", "mc3d_", "e10_trunc", "cov_class", "best_"))
           for ds in datasets}
    budget_ism = {}
    for ds in datasets:
        b = _ism_budget_from_record(rec_path, ds) or _ism_budget_from_cubes(args.cubes_dir, ds)
        budget_ism[ds] = b
    ta = {ds: _ta_interpolation_row(args.cubes_dir, runs, ds) for ds in datasets}

    # (a) per-dataset best coefficient (equal budget, test-selected)
    rows_a, rows_b = [], []
    for rule in rules:
        ra = OrderedDict(rule=_rule_label(rule))
        rb = OrderedDict(rule=_rule_label(rule))
        for ds in datasets:
            cs = complete([c for c in cells if c["dataset"] == ds and c["rule"] == rule])
            best = max(cs, key=lambda c: c["fa"]) if cs else None
            ra[ds] = f"c={best['coef']:g}: {fa_aa(best)}" if best else "-"
            ra[f"{ds} cells"] = len([c for c in cells if c["dataset"] == ds and c["rule"] == rule])
        sel_cs = complete([c for c in cells if c["dataset"] == args.select_on and c["rule"] == rule])
        c_sel = max(sel_cs, key=lambda c: c["fa"])["coef"] if sel_cs else None
        rb["coef (from " + args.select_on + ")"] = f"{c_sel:g}" if c_sel is not None else "-"
        for ds in datasets:
            cell = next((c for c in cells if c["dataset"] == ds and c["rule"] == rule and _close(c["coef"], c_sel)), None) \
                if c_sel is not None else None
            rb[ds] = fa_aa(cell) if cell else "-"
        rows_a.append(ra)
        rows_b.append(rb)
    ism_a = OrderedDict(rule="ISM (ours), headline cell on the same chain")
    ism_b = OrderedDict(rule="ISM (ours), headline cell on the same chain", **{"coef (from " + args.select_on + ")": "alpha " + "/".join(
        f"{HEADLINE[d]['alpha']:g}" for d in datasets if d in HEADLINE)})
    for ds in datasets:
        c = ism[ds]
        txt = (f"{_fmt_cell(c['faa'])} / {_fmt_cell(c['asa'])} [{c['config_name']}]" if c else "- (no headline cell in the logs)")
        ism_a[ds] = txt
        b = budget_ism[ds]
        ism_a[f"{ds} cells"] = (f"{b['p0.9_only_A_alpha_cells']} alpha cells at p=0.9 ({b['all_pspectral_cells']} pspectral cells)"
                                if b else "n/a")
        ism_b[ds] = txt
    rows_a.append(ism_a)
    rows_b.append(ism_b)
    ta_a = OrderedDict(rule="Task Arithmetic, interpolation form (pspectral p=1), tuned per dataset")
    for ds in datasets:
        t = ta[ds]
        ta_a[ds] = f"alpha={t['alpha']:g}: {_fmt_cell(t['fa'])} / {_fmt_cell(t['aa'])} ({t['source']})" if t else "-"
        ta_a[f"{ds} cells"] = t["n_cells"] if t else "-"
    rows_a.append(ta_a)
    print("\n### (a) Per-dataset best coefficient (equal budget, TEST-selected; ISM budget from "
          f"{'the E2 record' if os.path.isfile(rec_path) else 'the cubes'}; full-data 10-task ISM cells only)")
    cols_a = ["rule"] + [x for ds in datasets for x in (ds, f"{ds} cells")]
    print(markdown_table(rows_a, cols_a, ["rule"] + [x for ds in datasets for x in (f"{ds} FA / AA", "cells")]))
    print(f"\n### (b) Coefficient selected on {args.select_on} and reused (paper protocol, TEST-selected)")
    cols_b = ["rule", "coef (from " + args.select_on + ")"] + datasets
    print(markdown_table(rows_b, cols_b, ["rule", "coef"] + [f"{ds} FA / AA" for ds in datasets]))
    return 0


# --- E13: few-shot per task -------------------------------------------------------------
E13_METHODS = ("ism", "nomerge", "avg", "e2lora", "sgplora")
E13_NAME_RE = re.compile(r"^e13_(fs|lt)(\d+)_(\w+)$")
E13_COLS = ["dataset", "method", "shots", "scenario", "config_names", "train_prefix", "n_seeds", "n_failed", "fa", "fa_sd",
            "aa", "aa_sd", "ffm", "ffm_sd", "old_mean", "new_acc", "first5_mean", "til_fa", "til_fa_sd",
            "train_n", "train_median", "train_min", "train_max", "ref_fa", "ref_aa", "ref_ffm", "ref_source",
            "degradation_fa"]
E13_PER_SEED_COLS = ["dataset", "method", "shots", "config_name", "seed", "status", "faa", "asa", "ffm", "old_mean",
                     "new_acc", "first5_mean", "til_fa", "train_median", "time_s"]


def e13_method(run):
    """Method label of a run: from the e13_fs{K}_{method} name, else from the recipe keys."""
    m = E13_NAME_RE.match(str(run["config_name"]))
    if m:
        return m.group(3).replace("_", "")
    cfg = run["config"]
    method = str(cfg.get("train_method", "seqft"))
    if method != "seqft":
        return method.replace("_", "")
    rule = str(cfg.get("train_merge", "none"))
    if rule == "pspectral":
        return "ism"
    if rule == "none":
        return "nomerge"
    return rule


def _til_fa(run):
    rows = parse_til_rows(run)
    return float(np.mean(rows[-1])) if rows else None


def _data_counts(run):
    for line in run["tags"].get("Data", []):
        d = parse_data_line(line)
        if d.get("train"):
            return d
    return None


def _ref_from_cell(cell, train_prefix=None):
    """Reference dict (fa/aa/ffm with sd, n_seeds, config_name, train_prefix, source) from an
    aggregate() cell keyed by config_name."""
    return OrderedDict(fa=cell["faa"], fa_sd=cell.get("faa_sd"), aa=cell["asa"], aa_sd=cell.get("asa_sd"),
                       ffm=cell["ffm"], ffm_sd=cell.get("ffm_sd"), n_seeds=cell["n_seeds"],
                       config_name=str(cell["config_name"]), train_prefix=train_prefix,
                       source=f"logs:{cell['config_name']}")


def _full_data_reference(runs, ds, method, refs):
    """Full-data reference of a method on ``ds``: typed --refs row first (columns method,
    dataset, fa, aa, ffm, source and optionally fa_sd, aa_sd, ffm_sd, n_seeds), else the
    matching full-data runs in the logs (ISM: headline posture cell; nomerge/avg: same recipe
    without data_fewshot_k; e2lora/sgplora: that train_method without data_fewshot_k, e.g.
    the e13_full_<row> cells of run_e13_fewshot.py --full-ref).  The dict has the keys of
    _ref_from_cell; it is both the ref_* columns of the K rows and the shots='full' row."""
    for r in refs:
        if r.get("dataset") == ds and r.get("method", "").replace("_", "") == method:
            n = _f(r.get("n_seeds"))
            return OrderedDict(fa=_f(r.get("fa")), fa_sd=_f(r.get("fa_sd")), aa=_f(r.get("aa")),
                               aa_sd=_f(r.get("aa_sd")), ffm=_f(r.get("ffm")), ffm_sd=_f(r.get("ffm_sd")),
                               n_seeds=int(n) if n is not None else None, config_name=None,
                               train_prefix=r.get("train_prefix") or None, source=r.get("source") or "refs")
    if method == "ism":
        c = _reference_cell(runs, ds, prefer=("best_", "mc3d_", "e8_", "cov_class", "e10_trunc"))
        return _ref_from_cell(c, c.get("train_prefix")) if c else None
    cands = [r for r in runs if r["dataset"] == ds and r["status"] == "ok" and r["split"] == "t10"
             and not _is_truthy(r["config"].get("data_fewshot_k")) and not _is_truthy(r["config"].get("data_val_frac"))
             and not E13_NAME_RE.match(str(r["config_name"])) and e13_method(r) == method
             and not _is_truthy(r["config"].get("specreg_weight")) and not _is_truthy(r["config"].get("l2m_weight"))
             and not _is_truthy(r["config"].get("speccap"))]
    if method == "avg":
        cands = [r for r in cands if r["config"].get("train_merge_incremental") and _close(r["config"].get("train_merge_coef"), 1.0)]
    if not cands:
        return None
    cells = aggregate(cands, ("config_name",), metrics=_CELL_METRICS)
    c = sorted(cells, key=lambda c: (-c["n_seeds"], str(c["config_name"])))[0]
    prefix = next((r["config"].get("train_prefix") for r in cands if r["config_name"] == c["config_name"]), None)
    return _ref_from_cell(c, prefix)


def _full_row(ds, method, ref):
    """The shots='full' row of e13_fewshot_<ds>.csv (E13_COLS) for one method: its full-data
    reference as a cell of its own, so scripts/fig_e13_fewshot.py draws the 'full data' group
    from the same file; the K rows keep the reference in their ref_* columns."""
    row = OrderedDict(dataset=ds, method=method, shots="full", scenario="full",
                      config_names=[ref["config_name"] or ref["source"]], train_prefix=ref["train_prefix"],
                      n_seeds=ref["n_seeds"], n_failed=None, fa=ref["fa"], fa_sd=ref["fa_sd"], aa=ref["aa"],
                      aa_sd=ref["aa_sd"], ffm=ref["ffm"], ffm_sd=ref["ffm_sd"])
    for k in ("old_mean", "new_acc", "first5_mean", "til_fa", "til_fa_sd", "train_n", "train_median",
              "train_min", "train_max"):
        row[k] = None
    row.update(ref_fa=ref["fa"], ref_aa=ref["aa"], ref_ffm=ref["ffm"], ref_source=ref["source"], degradation_fa=None)
    return row


@subcommand("e13", "few-shot-per-task tables with each method's full-data reference (e13_fewshot_<ds>.csv, "
                   "e13_fewshot_<ds>_per_seed.csv)")
def cmd_e13(argv):
    """figures_data/e13_fewshot_<ds>.csv: one row per (method, shots) with shots = K for the
    few-shot cells and shots = 'full' for the method's full-data reference (the contract of
    scripts/fig_e13_fewshot.py: the 'full data' group is drawn from the shots='full' rows);
    the K rows also carry the reference in ref_fa / ref_aa / ref_ffm / ref_source and the
    drop in degradation_fa.  References: --refs CSV rows first (method, dataset, fa, aa, ffm,
    source, optional fa_sd/aa_sd/ffm_sd/n_seeds), else the full-data runs in the logs."""
    p = argparse.ArgumentParser(prog="parse_logs.py e13", description=cmd_e13.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--refs", default=None,
                   help="typed full-data references CSV (method,dataset,fa,aa,ffm,source[,fa_sd,aa_sd,ffm_sd,n_seeds])")
    args = p.parse_args(argv)
    runs = _load_tags(args, capture_tags=("Data", "Evaluation-til"))
    refs = _read_refs(args.refs)
    sel = [r for r in runs if E13_NAME_RE.match(str(r["config_name"])) or _is_truthy(r["config"].get("data_fewshot_k"))
           or _is_truthy(r["config"].get("data_lt_ratio"))]
    if args.datasets:
        sel = [r for r in sel if r["dataset"] in args.datasets]
    if not sel:
        print("(no few-shot runs found; nothing written)")
        return 0
    for r in sel:
        r["_method"] = e13_method(r)
        k = r["config"].get("data_fewshot_k")
        lt = r["config"].get("data_lt_ratio")
        r["_scenario"] = "fewshot" if _is_truthy(k) else ("longtail" if _is_truthy(lt) else "?")
        r["_shots"] = int(k) if _is_truthy(k) else (f"lt{lt}" if _is_truthy(lt) else None)
        r["til_fa"] = _til_fa(r)
        d = _data_counts(r)
        r["train_median"] = d["train"]["median"] if d else None
    os.makedirs(args.out_dir, exist_ok=True)
    keys = ("dataset", "_method", "_shots")
    all_cells = []
    for ds in sorted({r["dataset"] for r in sel}, key=str):
        mine = [r for r in sel if r["dataset"] == ds]
        per_seed = []
        for r in mine:
            row = OrderedDict(dataset=ds, method=r["_method"], shots=r["_shots"], config_name=r["config_name"], seed=r["seed"],
                              status=r["status"], faa=r["faa"], asa=r["asa"], ffm=r["ffm"], old_mean=r["old_mean"],
                              new_acc=r["new_acc"], first5_mean=r["first5_mean"], til_fa=r["til_fa"],
                              train_median=r["train_median"], time_s=r["time_s"])
            per_seed.append(row)
        path = os.path.join(args.out_dir, f"e13_fewshot_{ds}_per_seed.csv")
        write_csv(path, per_seed, E13_PER_SEED_COLS)
        _wrote(path, len(per_seed), "runs")
        cells = []
        references = OrderedDict()                     # method -> full-data reference (or None)
        for c in aggregate(mine, keys, metrics=_CELL_METRICS + ("til_fa",)):
            members = _members(mine, c, keys)
            ok = _ok(members)
            row = OrderedDict(dataset=ds, method=c["_method"], shots=c["_shots"], scenario=members[0]["_scenario"],
                              config_names=_names(members), train_prefix=members[0]["config"].get("train_prefix"))
            row.update(_std_metrics(c))
            row["n_failed"] = c["n_failed"]
            row["til_fa"], row["til_fa_sd"] = c.get("til_fa"), c.get("til_fa_sd")
            d = next((_data_counts(r) for r in ok if _data_counts(r)), None)
            row["train_n"] = d["train"]["n"] if d else None
            row["train_median"] = d["train"]["median"] if d else None
            row["train_min"] = d["train"]["min"] if d else None
            row["train_max"] = d["train"]["max"] if d else None
            if c["_method"] not in references:
                references[c["_method"]] = _full_data_reference(runs, ds, c["_method"], refs)
            ref = references[c["_method"]]
            row["ref_fa"] = ref["fa"] if ref else None
            row["ref_aa"] = ref["aa"] if ref else None
            row["ref_ffm"] = ref["ffm"] if ref else None
            row["ref_source"] = ref["source"] if ref else None
            row["degradation_fa"] = (ref["fa"] - row["fa"]) if (ref and ref["fa"] is not None and row["fa"] is not None) else None
            cells.append(row)
        # one shots='full' row per method with a reference (the figure's 'full data' group)
        for method, ref in references.items():
            if ref is not None and ref["fa"] is not None:
                cells.append(_full_row(ds, method, ref))
        cells.sort(key=lambda r: (E13_METHODS.index(r["method"]) if r["method"] in E13_METHODS else 9, str(r["shots"])))
        path = os.path.join(args.out_dir, f"e13_fewshot_{ds}.csv")
        write_csv(path, cells, E13_COLS)
        _wrote(path, len(cells), "method x shots cells")
        all_cells.extend(cells)
    cols = sorted({(c["dataset"], c["shots"]) for c in all_cells}, key=lambda k: (str(k[0]), str(k[1])))
    table = []
    for method in sorted({c["method"] for c in all_cells}, key=lambda m: E13_METHODS.index(m) if m in E13_METHODS else 9):
        row = OrderedDict(method=method)
        for ds, shots in cols:
            c = next((x for x in all_cells if x["method"] == method and x["dataset"] == ds and x["shots"] == shots), None)
            row[f"{ds} K={shots}"] = (f"{_fmt_cell(c['fa'])} / {_fmt_cell(c['aa'])} / {_fmt_cell(c['ffm'])} (n={c['n_seeds']})"
                                      if c else "-")
        for ds in sorted({d for d, _s in cols}, key=str):
            c = next((x for x in all_cells if x["method"] == method and x["dataset"] == ds and x["ref_fa"] is not None), None)
            row[f"{ds} full-data FA"] = f"{c['ref_fa']:.2f} [{c['ref_source']}]" if c else "-"
            degs = [x for x in all_cells if x["method"] == method and x["dataset"] == ds
                    and x["shots"] != "full" and x["degradation_fa"] is not None]
            row[f"{ds} drop"] = "; ".join(f"K={x['shots']}: {x['degradation_fa']:+.2f}" for x in degs) if degs else "-"
        table.append(row)
    print("\n### Few-shot per task (FA / AA / FF; K=full = the full-data reference; "
          "drop = full-data FA - few-shot FA)")
    print(markdown_table(table, list(table[0].keys())))
    med = [(c["dataset"], c["shots"], c["train_median"], c["train_min"], c["train_max"]) for c in all_cells
           if c["train_median"] is not None]
    if med:
        print("\nRealised training images per class ([Data] line; min/median/max): " + "; ".join(
            f"{ds} K={k}: {mn}/{m:g}/{mx}" for ds, k, m, mn, mx in sorted(set(med), key=str)))
    til = [c for c in all_cells if c["til_fa"] is not None]
    if til:
        print("\nTask-incremental FA ([Evaluation-til], eval_til): " + "; ".join(
            f"{c['dataset']} {c['method']} K={c['shots']}: {c['til_fa']:.2f}" for c in til))
    return 0


# --- E14: validation-split selection ----------------------------------------------------
E14_CELL_COLS = ["dataset", "alpha", "lambda", "robust", "config_names", "n_seeds", "n_failed", "faa_val", "faa_val_sd",
                 "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd", "asa_val", "ffm_val", "is_headline"]
E14_SEL_COLS = ["dataset", "val_frac", "train_prefix", "n_cells", "n_seeds", "sel_alpha_per_seed", "sel_lambda_per_seed",
                "sel_robust_per_seed", "test_fa_selected", "test_fa_selected_sd", "test_aa_selected", "test_ffm_selected",
                "pooled_alpha", "pooled_lambda", "pooled_robust", "test_fa_pooled", "test_fa_pooled_sd", "val_fa_pooled",
                "headline_alpha", "headline_lambda", "test_fa_headline_on_val_chain", "test_fa_headline_sd",
                "test_fa_oracle", "oracle_alpha", "oracle_lambda", "oracle_robust", "headline_full_data_fa",
                "headline_full_data_source", "spearman_val_test", "per_seed_hits_oracle"]


def _e14_cell_key(run):
    return (round(_cfgf(run, "train_merge_alpha"), 6), round(_cfgf(run, "train_drift_lambda"), 6),
            round(_cfgf(run, "train_ca_robust_weight", 0.0), 6))


@subcommand("e14", "validation-split selection: per-seed and pooled val-selected cells, oracle, headline on the "
                   "val chain and the full-data headline (e14_valsplit_<ds>.csv, e14_valsplit_selected.csv)")
def cmd_e14(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e14", description=cmd_e14.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--refs", default=None, help="typed full-data headline CSV (method=ism,dataset,fa,...)")
    args = p.parse_args(argv)
    runs = _load_tags(args, capture_tags=())
    refs = _read_refs(args.refs)
    val = [r for r in runs if _is_truthy(r["config"].get("data_val_frac")) and r["faa_val"] is not None]
    if args.datasets:
        val = [r for r in val if r["dataset"] in args.datasets]
    if not val:
        print("(no validation-split runs found; nothing written)")
        return 0
    for r in val:
        r["_alpha"], r["_lambda"], r["_robust"] = _e14_cell_key(r)
    keys = ("dataset", "_alpha", "_lambda", "_robust")
    os.makedirs(args.out_dir, exist_ok=True)
    selected = []
    for ds in sorted({r["dataset"] for r in val}, key=str):
        mine = [r for r in val if r["dataset"] == ds]
        head = HEADLINE.get(ds, {})
        cells = []
        for c in aggregate(mine, keys, metrics=_CELL_METRICS + ("faa_val", "asa_val", "ffm_val")):
            members = _members(mine, c, keys)
            row = OrderedDict(dataset=ds, alpha=c["_alpha"], **{"lambda": c["_lambda"]}, robust=c["_robust"],
                              config_names=_names(members), n_seeds=c["n_seeds"], n_failed=c["n_failed"],
                              faa_val=c.get("faa_val"), faa_val_sd=c.get("faa_val_sd"), fa=c["faa"], fa_sd=c["faa_sd"],
                              aa=c["asa"], aa_sd=c["asa_sd"], ffm=c["ffm"], ffm_sd=c["ffm_sd"], asa_val=c.get("asa_val"),
                              ffm_val=c.get("ffm_val"),
                              is_headline=_close(c["_alpha"], head.get("alpha")) and _close(c["_lambda"], head.get("lambda")))
            cells.append(row)
        cells.sort(key=lambda r: (r["alpha"], r["lambda"], r["robust"]))
        path = os.path.join(args.out_dir, f"e14_valsplit_{ds}.csv")
        write_csv(path, cells, E14_CELL_COLS)
        _wrote(path, len(cells), "cells")
        full = [c for c in cells if c["n_seeds"] >= 3 and c["n_failed"] == 0] or [c for c in cells if c["n_seeds"]]
        # per-seed selection: argmax val FA among the cells that seed completed (ties -> smaller alpha, lambda)
        seeds = sorted({r["seed"] for r in mine if r["status"] == "ok"})
        sel_a, sel_l, sel_w, sel_fa, sel_aa, sel_ff, hits = [], [], [], [], [], [], 0
        oracle = max(full, key=lambda c: (c["fa"], -c["alpha"], -c["lambda"])) if full else None
        for s in seeds:
            rs = [r for r in mine if r["seed"] == s and r["status"] == "ok"]
            best = max(rs, key=lambda r: (r["faa_val"], -r["_alpha"], -r["_lambda"]))
            sel_a.append(best["_alpha"])
            sel_l.append(best["_lambda"])
            sel_w.append(best["_robust"])
            sel_fa.append(best["faa"])
            sel_aa.append(best["asa"])
            sel_ff.append(best["ffm"])
            if oracle and _close(best["_alpha"], oracle["alpha"]) and _close(best["_lambda"], oracle["lambda"]) \
                    and _close(best["_robust"], oracle["robust"]):
                hits += 1
        pooled = max(full, key=lambda c: (c["faa_val"], -c["alpha"], -c["lambda"])) if full else None
        headc = next((c for c in cells if c["is_headline"]), None)
        full_ref = _full_data_reference(runs, ds, "ism", refs)
        sp = _spearman([c["faa_val"] for c in full], [c["fa"] for c in full]) if len(full) >= 3 else None
        prefix = mine[0]["config"].get("train_prefix")
        row = OrderedDict(dataset=ds, val_frac=mine[0]["config"].get("data_val_frac"), train_prefix=prefix,
                          n_cells=len(cells), n_seeds=len(seeds), sel_alpha_per_seed=sel_a, sel_lambda_per_seed=sel_l,
                          sel_robust_per_seed=sel_w, test_fa_selected=_mean(sel_fa), test_fa_selected_sd=_sd(sel_fa),
                          test_aa_selected=_mean(sel_aa), test_ffm_selected=_mean(sel_ff),
                          pooled_alpha=pooled["alpha"] if pooled else None, pooled_lambda=pooled["lambda"] if pooled else None,
                          pooled_robust=pooled["robust"] if pooled else None,
                          test_fa_pooled=pooled["fa"] if pooled else None, test_fa_pooled_sd=pooled["fa_sd"] if pooled else None,
                          val_fa_pooled=pooled["faa_val"] if pooled else None,
                          headline_alpha=head.get("alpha"), headline_lambda=head.get("lambda"),
                          test_fa_headline_on_val_chain=headc["fa"] if headc else None,
                          test_fa_headline_sd=headc["fa_sd"] if headc else None,
                          test_fa_oracle=oracle["fa"] if oracle else None,
                          oracle_alpha=oracle["alpha"] if oracle else None, oracle_lambda=oracle["lambda"] if oracle else None,
                          oracle_robust=oracle["robust"] if oracle else None,
                          headline_full_data_fa=full_ref["fa"] if full_ref else None,
                          headline_full_data_source=full_ref["source"] if full_ref else None,
                          spearman_val_test=sp, per_seed_hits_oracle=f"{hits}/{len(seeds)}")
        selected.append(row)
        print(f"\n### {ds}: validation grid (val_frac {row['val_frac']}, chain {prefix}); val FA vs test FA per cell")
        for c in cells:
            c["val FA"] = pm(c["faa_val"], c["faa_val_sd"])
            c["test FA"] = pm(c["fa"], c["fa_sd"])
            c["test AA"], c["test FF"] = pm(c["aa"], c["aa_sd"]), pm(c["ffm"], c["ffm_sd"])
            c["headline"] = "<-" if c["is_headline"] else ""
        print(markdown_table(cells, ["alpha", "lambda", "robust", "n_seeds", "val FA", "test FA", "test AA", "test FF", "headline"]))
    path = os.path.join(args.out_dir, "e14_valsplit_selected.csv")
    write_csv(path, selected, E14_SEL_COLS)
    _wrote(path, len(selected), "datasets")
    if selected:
        for row in selected:
            row["per-seed (alpha, lambda)"] = "; ".join(f"({a:g}, {l:g})" for a, l in zip(row["sel_alpha_per_seed"], row["sel_lambda_per_seed"]))
            row["test FA selected"] = pm(row["test_fa_selected"], row["test_fa_selected_sd"])
            row["pooled"] = (f"({row['pooled_alpha']:g}, {row['pooled_lambda']:g}) -> {_fmt_cell(row['test_fa_pooled'])}"
                             if row["pooled_alpha"] is not None else "-")
            row["headline on val chain"] = _fmt_cell(row["test_fa_headline_on_val_chain"], row["test_fa_headline_sd"])
            row["oracle"] = (f"({row['oracle_alpha']:g}, {row['oracle_lambda']:g}) -> {_fmt_cell(row['test_fa_oracle'])}"
                             if row["oracle_alpha"] is not None else "-")
            row["full-data headline"] = (f"{row['headline_full_data_fa']:.2f} [{row['headline_full_data_source']}]"
                                         if row["headline_full_data_fa"] is not None else "-")
            row["Spearman(val, test)"] = _fmt_cell(row["spearman_val_test"], None, 3)
        print("\n### Validation-split selection (alpha, lambda chosen on the held-out split; test FA reported)")
        print(markdown_table(selected, ["dataset", "n_cells", "n_seeds", "per-seed (alpha, lambda)", "test FA selected", "pooled",
                                        "headline on val chain", "oracle", "per_seed_hits_oracle", "full-data headline",
                                        "Spearman(val, test)"]))
    return 0


# --- E15: OPCM / NUFILT rows and the InfLoRA port ----------------------------------------
E15_RULES = ("opcm", "nufilt")
E15_COLS = ["rule", "setting", "dataset", "config_names", "n_seeds", "n_failed", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd",
            "old_mean", "new_acc", "first5_mean", "lam_t_last", "kept_frac_mean", "filtered_frac_mean", "n_adapted_mean",
            "adapt_loss_first_mean", "adapt_loss_last_mean", "flags"]
E15_INFLORA_COLS = ["dataset", "config_name", "backbone", "resolved_backbone", "lora_r", "inflora_lamb", "train_epochs",
                    "train_base_lr", "train_batch_size", "n_seeds", "n_failed", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd",
                    "micro_acc", "micro_acc_sd", "avg", "avg_sd", "sweep_fa", "sweep_micro_acc", "trainable_params",
                    "basis_cols_last_max", "basis_cols_monotone", "leak_max", "published_acc", "published_avg", "flags"]
# InfLoRA Tab. 1 (arXiv:2404.00228 v3, ViT-B/16 IN21K, fixed order, seeds 0-4): ACC / AVG, mean +- std.
INFLORA_PUBLISHED = {"cifar224": (86.51, 91.70), "imagenetr": (75.65, 80.82)}
E15_KEEPLORA_COLS = ["dataset", "config_name", "backbone", "resolved_backbone", "lora_r", "eps_w", "eps_f", "wp_side",
                     "feat_mode", "b_init", "head_init", "train_epochs", "train_base_lr", "train_batch_size", "n_seeds",
                     "n_failed", "fa", "fa_sd", "aa", "aa_sd", "ffm", "ffm_sd", "micro_acc", "micro_acc_sd", "avg", "avg_sd",
                     "sweep_fa", "sweep_micro_acc", "trainable_params", "basis_cols_task0_max", "basis_cols_last_max",
                     "basis_cols_last_mean", "basis_full_last", "leak_max", "skipped_batches", "flags"]


def e15_setting(cfg):
    rule = str(cfg.get("train_merge"))
    if rule == "opcm":
        return f"a{cfg.get('merge_opcm_alpha', 0.5)}_{cfg.get('merge_opcm_lam', 'adaptive')}"
    if rule == "nufilt":
        return f"rp{cfg.get('merge_nufilt_rp', 128)}_ad{int(bool(cfg.get('merge_nufilt_adapt', False)))}"
    return None


@subcommand("e15", "NUFILT (and OPCM) rows (e15_opcm_nufilt.csv; stage-1 grid + Tab. 6 layout), the KeepLoRA port "
                   "(e15_keeplora.csv from run_keeplora logs: FA/AA/FF, micro ACC/AVG, basis growth) and the InfLoRA "
                   "port (e15_inflora.csv, parity gate)")
def cmd_e15(argv):
    p = argparse.ArgumentParser(prog="parse_logs.py e15", description=cmd_e15.__doc__)
    _common_args(p)
    p.add_argument("--out-dir", default="figures_data")
    p.add_argument("--select-on", default="imagenetr", help="stage-1 dataset whose best cell per rule is the Tab. 6 row")
    args = p.parse_args(argv)
    runs = _load_tags(args, capture_tags=("InfLoRA", "KeepLoRA", "Freeze", "MergeStat"))
    sel = [r for r in runs if str(r["config"].get("train_merge")) in E15_RULES]
    infl = [r for r in runs if str(r["config"].get("train_method")) == "inflora"]
    keep = [r for r in runs if str(r["config"].get("train_method")) == "keeplora"]
    if not sel and not infl and not keep:
        print("(no opcm / nufilt / keeplora / inflora runs found; nothing written)")
        return 0
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- OPCM / NUFILT ----------------------------------------------------------------
    for r in sel:
        r["_rule"] = str(r["config"]["train_merge"])
        r["_setting"] = e15_setting(r["config"])
    keys = ("_rule", "_setting", "dataset")
    cells = []
    for c in aggregate(sel, keys, metrics=_CELL_METRICS):
        members = _members(sel, c, keys)
        ok = _ok(members)
        row = OrderedDict(rule=c["_rule"], setting=c["_setting"], dataset=c["dataset"], config_names=_names(members))
        row.update(_std_metrics(c))
        row["n_failed"] = c["n_failed"]
        states = [st for m in ok for st in (m.get("merge_state") or [])]
        row["lam_t_last"] = _mean([m["merge_state"][-1][1].get("lam_t") for m in ok if m.get("merge_state")])
        row["kept_frac_mean"] = _mean([kv.get("kept_frac") for _mth, kv in states])
        row["filtered_frac_mean"] = _mean([kv.get("filtered_frac") for _mth, kv in states])
        row["n_adapted_mean"] = _mean([kv.get("n_adapted") for _mth, kv in states])
        row["adapt_loss_first_mean"] = _mean([kv.get("adapt_loss_first") for _mth, kv in states])
        row["adapt_loss_last_mean"] = _mean([kv.get("adapt_loss_last") for _mth, kv in states])
        flags = []
        if c["n_failed"]:
            flags.append(f"failed seeds {c['failed_seeds']}")
        if any(m["merge_sota_method"] not in (None, c["_rule"]) for m in members):
            flags.append("SOTA line names another rule")
        row["flags"] = "; ".join(flags)
        cells.append(row)
    cells.sort(key=lambda r: (E15_RULES.index(r["rule"]), r["setting"], E12_ORDER.index(r["dataset"]) if r["dataset"] in E12_ORDER else 9))
    if cells:
        path = os.path.join(args.out_dir, "e15_opcm_nufilt.csv")
        write_csv(path, cells, E15_COLS)
        _wrote(path, len(cells), "rule x setting x dataset cells")
        grid = [c for c in cells if c["dataset"] == args.select_on]
        for c in grid:
            c["FA"], c["AA"], c["FF"] = pm(c["fa"], c["fa_sd"]), pm(c["aa"], c["aa_sd"]), pm(c["ffm"], c["ffm_sd"])
            c["state"] = (f"lam_t {c['lam_t_last']:.3f}, kept {c['kept_frac_mean']:.3f}" if c["rule"] == "opcm" and c["lam_t_last"] is not None
                          else (f"filtered {c['filtered_frac_mean']:.3f}" if c["filtered_frac_mean"] is not None else "-"))
        print(f"\n### Stage-1 grid on {args.select_on}")
        print(markdown_table(grid, ["rule", "setting", "n_seeds", "FA", "AA", "FF", "state", "flags"]))
        datasets = [d for d in E12_ORDER if any(c["dataset"] == d for c in cells)]
        ism = {ds: _reference_cell(runs, ds, prefer=("e15_ism", "e12_ism", "e8_", "mc3d_", "e10_trunc", "cov_class", "best_"))
               for ds in datasets}
        table = []
        for rule in E15_RULES:
            rc = [c for c in grid if c["rule"] == rule and c["fa"] is not None]
            best = max(rc, key=lambda c: c["fa"]) if rc else None
            settings = [best["setting"]] if best else sorted({c["setting"] for c in cells if c["rule"] == rule})
            for st in settings:
                row = OrderedDict(method=f"{rule.upper()} ({st})" + (" [best stage-1]" if best and st == best["setting"] else ""))
                for ds in datasets:
                    c = next((x for x in cells if x["rule"] == rule and x["setting"] == st and x["dataset"] == ds), None)
                    row[ds] = f"{_fmt_cell(c['fa'])} / {_fmt_cell(c['aa'])}" if c and c["fa"] is not None else "-"
                table.append(row)
        row = OrderedDict(method="ISM (ours), headline cell on the same chain")
        for ds in datasets:
            c = ism[ds]
            row[ds] = f"{_fmt_cell(c['faa'])} / {_fmt_cell(c['asa'])} [{c['config_name']}]" if c else "-"
        table.append(row)
        print("\n### Tab. 6 layout (FA / AA): OPCM and NUFILT on the sequential chain")
        print(markdown_table(table, ["method"] + datasets))
    else:
        print("(no opcm / nufilt runs found)")

    # ---- KeepLoRA port -----------------------------------------------------------------
    rows = []
    for c in aggregate(keep, ("dataset", "config_name"), metrics=_CELL_METRICS):
        members = _members(keep, c, ("dataset", "config_name"))
        ok = _ok(members)
        cfg = members[0]["config"]
        info = [parse_keeplora_lines(m) for m in ok]
        micro = [m["totals"][-1] for m in ok if m["totals"]]
        avg = [float(np.mean(m["totals"])) for m in ok if m["totals"]]
        sweep = next((m["sweep"] for m in members if m.get("sweep")), None)
        t0_max, last_max, last_mean, full_last, leaks, skipped = [], [], [], [], [], 0
        for i in info:
            if i["design"]:
                t0_max.append(i["design"][0]["cols_max"])
            if i["subspace"]:
                last_max.append(i["subspace"][-1]["cols_max"])
                last_mean.append(i["subspace"][-1]["cols_mean"])
                full_last.append(i["subspace"][-1]["full"])
                skipped += sum(s["skipped"] for s in i["subspace"])
            leaks.extend(d["leak"] for d in i["design"])
        params = sorted({m["trainable_params"] for m in members if m["trainable_params"] is not None})
        row = OrderedDict(dataset=c["dataset"], config_name=c["config_name"], backbone=cfg.get("keeplora_backbone"),
                          resolved_backbone=next((i["backbone"][1] for i in info if i["backbone"]), None),
                          lora_r=cfg.get("lora_r"), eps_w=cfg.get("keeplora_eps_w"), eps_f=cfg.get("keeplora_eps_f"),
                          wp_side=cfg.get("keeplora_wp_side"), feat_mode=cfg.get("keeplora_feat_mode"),
                          b_init=cfg.get("keeplora_b_init"), head_init=cfg.get("keeplora_head_init"),
                          train_epochs=cfg.get("train_epochs"), train_base_lr=cfg.get("train_base_lr"),
                          train_batch_size=cfg.get("train_batch_size"))
        row.update(_std_metrics(c))
        row["n_failed"] = c["n_failed"]
        row["micro_acc"], row["micro_acc_sd"] = _mean(micro), _sd(micro)
        row["avg"], row["avg_sd"] = _mean(avg), _sd(avg)
        row["sweep_fa"] = sweep["FA"][0] if sweep and "FA" in sweep else None
        row["sweep_micro_acc"] = sweep["micro ACC"][0] if sweep and "micro ACC" in sweep else None
        row["trainable_params"] = params[0] if len(params) == 1 else (params or None)
        row["basis_cols_task0_max"] = max(t0_max) if t0_max else None
        row["basis_cols_last_max"] = max(last_max) if last_max else None
        row["basis_cols_last_mean"] = _mean(last_mean)
        row["basis_full_last"] = max(full_last) if full_last else None
        row["leak_max"] = max(leaks) if leaks else None
        row["skipped_batches"] = skipped
        flags = []
        if c["n_failed"]:
            flags.append(f"failed seeds {c['failed_seeds']}")
        if row["basis_full_last"]:
            flags.append(f"{row['basis_full_last']} slices with a full (d x d) basis: no residual left to learn in")
        if row["leak_max"] is not None and row["leak_max"] > 1e-5:
            flags.append(f"max|U^T A| {row['leak_max']:.2e} > 1e-5")
        if skipped:
            flags.append(f"{skipped} feature batches skipped (unequal batch size)")
        if row["sweep_fa"] is not None and row["fa"] is not None and abs(row["sweep_fa"] - row["fa"]) > 0.0101:
            flags.append("SWEEP SUMMARY FA differs from the recomputed mean")
        row["flags"] = "; ".join(flags)
        rows.append(row)
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["config_name"])))
    if rows:
        path = os.path.join(args.out_dir, "e15_keeplora.csv")
        write_csv(path, rows, E15_KEEPLORA_COLS)
        _wrote(path, len(rows), "KeepLoRA cells")
        for r in rows:
            r["FA"], r["AA"], r["FF"] = pm(r["fa"], r["fa_sd"]), pm(r["aa"], r["aa_sd"]), pm(r["ffm"], r["ffm_sd"])
            r["micro ACC"], r["AVG"] = pm(r["micro_acc"], r["micro_acc_sd"]), pm(r["avg"], r["avg_sd"])
            r["params"] = (f"{r['trainable_params']:,}" if isinstance(r["trainable_params"], int)
                           else ("-" if r["trainable_params"] is None else str(r["trainable_params"])))
            r["basis cols (task0 max / last max / last mean)"] = (
                f"{r['basis_cols_task0_max']} / {r['basis_cols_last_max']} / "
                f"{r['basis_cols_last_mean']:.0f}" if r["basis_cols_last_mean"] is not None else "-")
            r["switches"] = f"w{r['eps_w']} f{r['eps_f']} {r['wp_side']} {r['feat_mode']} B={r['b_init']} head={r['head_init']}"
        print("\n### KeepLoRA port (run_keeplora.py): task-equal FA/AA/FF, micro ACC / AVG and basis growth")
        print(markdown_table(rows, ["dataset", "config_name", "backbone", "switches", "n_seeds", "n_failed", "FA", "AA", "FF",
                                    "micro ACC", "AVG", "params", "basis cols (task0 max / last max / last mean)", "flags"]))
    else:
        print("(no keeplora runs found)")

    # ---- InfLoRA port ------------------------------------------------------------------
    rows = []
    for c in aggregate(infl, ("dataset", "config_name"), metrics=_CELL_METRICS):
        members = _members(infl, c, ("dataset", "config_name"))
        ok = _ok(members)
        cfg = members[0]["config"]
        info = [parse_inflora_lines(m) for m in ok]
        micro = [m["totals"][-1] for m in ok if m["totals"]]
        avg = [float(np.mean(m["totals"])) for m in ok if m["totals"]]
        sweep = next((m["sweep"] for m in members if m.get("sweep")), None)
        cols_last, mono, leaks = [], True, []
        for i in info:
            for g in i["gpm"]:
                cols_last.append(max(g["cols"]) if g["cols"] else None)
            seq = [g["cols"] for g in i["gpm"]]
            for a, b in zip(seq, seq[1:]):
                if len(a) == len(b) and any(y < x for x, y in zip(a, b)):
                    mono = False
            leaks.extend(d["leak"] for d in i["design"] if d["task"] > 0)
        params = sorted({m["trainable_params"] for m in members if m["trainable_params"] is not None})
        row = OrderedDict(dataset=c["dataset"], config_name=c["config_name"], backbone=cfg.get("inflora_backbone"),
                          resolved_backbone=next((i["backbone"][1] for i in info if i["backbone"]), None),
                          lora_r=cfg.get("lora_r"), inflora_lamb=cfg.get("inflora_lamb"), train_epochs=cfg.get("train_epochs"),
                          train_base_lr=cfg.get("train_base_lr"), train_batch_size=cfg.get("train_batch_size"))
        row.update(_std_metrics(c))
        row["n_failed"] = c["n_failed"]
        row["micro_acc"], row["micro_acc_sd"] = _mean(micro), _sd(micro)
        row["avg"], row["avg_sd"] = _mean(avg), _sd(avg)
        row["sweep_fa"] = sweep["FA"][0] if sweep and "FA" in sweep else None
        row["sweep_micro_acc"] = sweep["micro ACC"][0] if sweep and "micro ACC" in sweep else None
        row["trainable_params"] = params[0] if len(params) == 1 else (params or None)
        row["basis_cols_last_max"] = max([v for v in cols_last if v is not None], default=None)
        row["basis_cols_monotone"] = mono if cols_last else None
        row["leak_max"] = max(leaks) if leaks else None
        pub = INFLORA_PUBLISHED.get(c["dataset"])
        row["published_acc"], row["published_avg"] = (pub if pub else (None, None))
        flags = []
        if c["n_failed"]:
            flags.append(f"failed seeds {c['failed_seeds']}")
        if row["basis_cols_last_max"] is not None and row["basis_cols_last_max"] > 384:
            flags.append("basis columns > 384 (d/2)")
        if cols_last and not mono:
            flags.append("basis columns not monotone")
        if row["leak_max"] is not None and row["leak_max"] > 1e-5:
            flags.append(f"max|A M| {row['leak_max']:.2e} > 1e-5")
        if row["sweep_fa"] is not None and row["fa"] is not None and abs(row["sweep_fa"] - row["fa"]) > 0.0101:
            flags.append("SWEEP SUMMARY FA differs from the recomputed mean")
        row["flags"] = "; ".join(flags)
        rows.append(row)
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["config_name"])))
    if rows:
        path = os.path.join(args.out_dir, "e15_inflora.csv")
        write_csv(path, rows, E15_INFLORA_COLS)
        _wrote(path, len(rows), "InfLoRA cells")
        for r in rows:
            r["FA"], r["AA"], r["FF"] = pm(r["fa"], r["fa_sd"]), pm(r["aa"], r["aa_sd"]), pm(r["ffm"], r["ffm_sd"])
            r["micro ACC"], r["AVG"] = pm(r["micro_acc"], r["micro_acc_sd"]), pm(r["avg"], r["avg_sd"])
            r["published ACC / AVG"] = (f"{r['published_acc']:.2f} / {r['published_avg']:.2f}" if r["published_acc"] else "-")
            r["params"] = (f"{r['trainable_params']:,}" if isinstance(r["trainable_params"], int)
                           else ("-" if r["trainable_params"] is None else str(r["trainable_params"])))
            r["basis cols (max, last)"] = r["basis_cols_last_max"] if r["basis_cols_last_max"] is not None else "-"
        print("\n### InfLoRA port (run_inflora.py): task-equal FA/AA/FF and InfLoRA's own micro ACC / AVG")
        print(markdown_table(rows, ["dataset", "config_name", "backbone", "resolved_backbone", "n_seeds", "n_failed", "FA", "AA",
                                    "FF", "micro ACC", "AVG", "published ACC / AVG", "params", "basis cols (max, last)", "flags"]))
    else:
        print("(no inflora runs found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
