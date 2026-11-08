"""Rebuttal figure (E13): final accuracy under K training images per class and task.

Data: figures_data/e13_fewshot_<ds>.csv for ds in cifar224, imagenetr, written by
`python parse_logs.py e13`. Columns read per row: method (ism | nomerge | avg |
e2lora | sgplora), shots (5 | 20 | full), n_seeds, and the metric pair fa / fa_sd
(aliases faa / faa_sd; aa <-> asa; ffm). When the method / shots columns are
absent they are parsed from config_name (e13_fs5_ism, e13_full_sgplora), so the
generic aggregate `python parse_logs.py --glob "logs_main/<ds>/e13_*.log" --agg
figures_data/e13_fewshot_<ds>.csv --keys dataset config_name` is accepted too.
Every drawn bar must rest on 3 seeds (--allow-partial downgrades that to a warning).

Layout: one panel per dataset; groups K = 5, K = 20 and the full-data reference
(drawn when any method has it); one bar per method with the seed std as whisker
and the value printed above. Two-ink print logic of the paper figures: cobalt
solid = ISM, cobalt screens / paper with a cobalt outline = controls and
baselines, Cool Gray substrate, STIX, orthogonal marks only.

Run:  python scripts/fig_e13_fewshot.py  ->  assets/figures/fig_e13_fewshot.{pdf,png}
      python scripts/fig_e13_fewshot.py --metric ffm --name fig_e13_fewshot_ff
      options: --data-dir figures_data --out-dir assets/figures
               --datasets cifar224 imagenetr --allow-partial
"""
import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Drawn at the ICLR text width (5.5 in) so the font sizes below are the printed sizes.
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 6.5})
FS_TICK, FS_VAL, FS_TITLE, FS_LEG, FS_LABEL = 6.0, 5.2, 6.8, 6.0, 6.5

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "figures_data")
OUT_DIR = os.path.join(ROOT, "assets", "figures")

PAPER, COBALT, TERRA = "#E9E9E5", "#2148B8", "#C65F38"
C60, C40, C25 = "#7A91D4", "#A6B5E3", "#C9D2EE"
INK = COBALT

# (csv method key, legend label, face, edge); ISM is the only solid-cobalt bar.
METHODS = [
    ("ism",     "ISM (ours)",                      COBALT, COBALT),
    ("nomerge", "No merge ($W_t$ deployed)",       PAPER,  COBALT),
    ("avg",     "Task Arithmetic (average)",       C60,    COBALT),
    ("e2lora",  "E$^2$-LoRA (matched backbone)",   C40,    COBALT),
    ("sgplora", "SGP-LoRA (ported) + Stage 2",     C25,    COBALT),
]
DATASETS = [("cifar224", "CIFAR-100 (10 x 10)"), ("imagenetr", "ImageNet-R (10 x 20)")]
GROUPS = ["5", "20", "full"]
GROUP_LABEL = {"5": "$K$ = 5", "20": "$K$ = 20", "full": "full data"}
# metric -> (value column aliases, axis label); the std column is <alias>_sd
METRICS = {
    "fa": (("fa", "faa"), "Final accuracy FA (%)"),
    "aa": (("aa", "asa"), "Average accuracy AA (%)"),
    "ffm": (("ffm", "ff"), "Forgetting FF (%), lower is better"),
}
NAME_RE = re.compile(r"^e13_(?:fs(\d+)|(full))_([A-Za-z0-9]+)(?:_smoke)?$")


def norm_shots(value):
    """'5' / '20' / 'full' from the shots column ('5', '5.0', 'K5', 'fs5', 'full', 'inf')."""
    s = str(value).strip().lower()
    if s in ("full", "inf", "all", "ref", "reference", "none", ""):
        return "full"
    m = re.match(r"^(?:k|fs|k=)?\s*(\d+)(?:\.0+)?$", s)
    if not m:
        raise ValueError(f"unreadable shots value {value!r}")
    return m.group(1)


def _pick(row, aliases):
    for a in aliases:
        if a in row and row[a] not in ("", None):
            return float(row[a])
    return None


def load(path, metric):
    """{(method, shots): (value, sd, n_seeds)} of one e13 CSV."""
    aliases, _ = METRICS[metric]
    data = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            method, shots = row.get("method"), row.get("shots")
            if not method or shots in (None, ""):
                m = NAME_RE.match(row.get("config_name", "") or row.get("name", ""))
                if not m:
                    continue                       # not an E13 cell (e.g. a headline row)
                method = m.group(3)
                shots = m.group(1) or "full"
            method = method.strip().lower()
            shots = norm_shots(shots)
            value = _pick(row, aliases)
            if value is None:
                continue
            sd = _pick(row, tuple(a + "_sd" for a in aliases)) or 0.0
            n = int(float(row.get("n_seeds", 0) or 0))
            data[(method, shots)] = (value, sd, n)
    return data


def check(data, label, allow_partial):
    """Every drawn bar rests on 3 seeds unless --allow-partial."""
    bad = {k: v[2] for k, v in data.items() if v[2] != 3}
    if bad:
        msg = f"{label}: cells without 3 seeds: {bad}"
        if not allow_partial:
            raise AssertionError(msg + " (pass --allow-partial to draw them anyway)")
        print("WARNING:", msg)


def panel(ax, data, title, groups, ylabel):
    n = len(METHODS)
    w = min(0.15, 0.8 / n)
    xs = list(range(len(groups)))
    lo, hi = float("inf"), float("-inf")
    ax.set_facecolor(PAPER)
    for i, (m, _label, fc, ec) in enumerate(METHODS):
        off = (i - (n - 1) / 2) * w
        for x, grp in zip(xs, groups):
            if (m, grp) not in data:
                continue
            v, sd, _n = data[(m, grp)]
            ax.bar(x + off, v, width=w, color=fc, edgecolor=ec, linewidth=0.5, zorder=3)
            if sd > 0:
                ax.errorbar(x + off, v, yerr=sd, fmt="none", ecolor=INK, elinewidth=0.6,
                            capsize=1.4, capthick=0.6, zorder=4)
            ax.text(x + off, v + sd, f"{v:.1f}", ha="center", va="bottom", fontsize=FS_VAL,
                    color=INK, rotation=90, zorder=5)
            lo, hi = min(lo, v - sd), max(hi, v + sd)
    if lo == float("inf"):
        lo, hi = 0.0, 100.0
    span = max(hi - lo, 5.0)
    ax.set_ylim(max(0.0, lo - 0.25 * span), hi + 0.42 * span)   # room for the rotated values
    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABEL[g] for g in groups], fontsize=FS_TICK, color=INK)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_title(title, fontsize=FS_TITLE, color=INK, loc="left", pad=3)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=FS_LABEL)
    ax.tick_params(colors=INK, length=2, labelsize=FS_TICK)
    ax.yaxis.grid(True, color="#FFFFFF", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK)
        ax.spines[sp].set_linewidth(0.8)


def make_figure(datasets, data_dir, out_dir, metric="fa", name="fig_e13_fewshot", allow_partial=False):
    _aliases, ylabel = METRICS[metric]
    panels = []
    for ds, title in datasets:
        path = os.path.join(data_dir, f"e13_fewshot_{ds}.csv")
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found; write it with `python parse_logs.py e13` first")
        data = load(path, metric)
        if not data:
            raise SystemExit(f"{path}: no E13 rows with a {metric} value")
        check(data, ds, allow_partial)
        panels.append((ds, title, data))
    groups = [g for g in GROUPS if any(any((m, g) in d for m, *_ in METHODS) for _, _, d in panels)]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5, 2.25), squeeze=False)
    fig.patch.set_facecolor(PAPER)
    for j, (ax, (ds, title, data)) in enumerate(zip(axes[0], panels)):
        panel(ax, data, title, groups, ylabel if j == 0 else None)
    present = {m for _, _, d in panels for (m, _g) in d}
    handles = [Patch(facecolor=fc, edgecolor=ec, linewidth=0.6, label=label)
               for m, label, fc, ec in METHODS if m in present]
    leg = fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)), frameon=False,
                     fontsize=FS_LEG, bbox_to_anchor=(0.5, 0.0), handlelength=1.0, columnspacing=0.9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.30, wspace=0.16)
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(pdf, facecolor=PAPER)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), facecolor=PAPER, dpi=200)
    plt.close(fig)

    # Text companion of the figure (same numbers, for RESULTS.md / the rebuttal).
    print(f"{metric.upper()} per method (mean +- sd over seeds); groups {groups}")
    for ds, _title, data in panels:
        for m, label, *_ in METHODS:
            cells = ["%s: %.2f +- %.2f" % (GROUP_LABEL[g].replace("$", ""), *data[(m, g)][:2])
                     for g in groups if (m, g) in data]
            if cells:
                print(f"  {ds:<10} {label:<32} " + " | ".join(cells))
    print("saved", os.path.abspath(pdf), "and .png")
    return pdf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR, help="directory of e13_fewshot_<ds>.csv")
    ap.add_argument("--out-dir", default=OUT_DIR, help="output directory (pdf + png)")
    ap.add_argument("--datasets", "--ds", nargs="+", default=[d for d, _ in DATASETS],
                    choices=[d for d, _ in DATASETS])
    ap.add_argument("--metric", choices=sorted(METRICS), default="fa")
    ap.add_argument("--name", default="fig_e13_fewshot", help="output file stem")
    ap.add_argument("--allow-partial", action="store_true",
                    help="draw cells with fewer than 3 seeds (warning instead of an error)")
    args = ap.parse_args(argv)
    datasets = [(d, t) for d, t in DATASETS if d in args.datasets]
    make_figure(datasets, args.data_dir, args.out_dir, metric=args.metric, name=args.name,
                allow_partial=args.allow_partial)
    return 0


if __name__ == "__main__":
    sys.exit(main())
