"""Figure E1: final accuracy vs sequence length (10 / 20 / 50 tasks) on ImageNet-R
and CIFAR-100, ISM against the strongest published baselines (rebuttal A7).

Data: figures_data/e1_long_agg.csv -- ISM, written by `python parse_logs.py e1`
(one row per dataset x split x config; the reference cell of a split is the
headline alpha / lambda cell, else the cell with most seeds, or --config) -- and
figures_data/e1_long_baselines.csv (E2-LoRA Table 2 via RESULTS.md section 4,
plus the 10-task main-table rows; columns method, dataset, split, last, last_sd,
inc, inc_sd, source).  The baselines report sample-weighted Last-Acc, so ISM is
drawn with its sample-weighted last Total Acc (column last_total) by default;
--metric faa draws the task-equal FA instead (file suffix _fa).  --from-results
takes the ISM points from the RESULTS.md 'Degradation profile (ours)' table
(task-equal FA; a preview before the logs are pulled; suffix _results).

Two panels (ImageNet-R, CIFAR-100), x = number of tasks, y = Last-Acc with
+- std error bars.  ISM solid cobalt; E2-LoRA, the strongest competitor,
terracotta; TUNA / MOS / LoRA-DRS cobalt screens; the E2-LoRA joint-training
bound as a dotted line.  Every ISM point must have 3 complete seeds
(--allow-incomplete to draw anyway).

Run:  python scripts/fig_e1_long.py  ->  assets/figures/fig_e1_long.{pdf,png}
"""
import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import rebuttal_common as rc  # noqa: E402  (torch-free: headline alpha / lambda per dataset)

PAPER, COBALT, TERRA = "#E9E9E5", "#2148B8", "#C65F38"
C60, C40, C25 = "#7A91D4", "#A6B5E3", "#C9D2EE"

PANELS = [("imagenetr", "ImageNet-R"), ("cifar224", "CIFAR-100")]
SPLITS = [("t10", "10"), ("t20", "20"), ("t50", "50")]
# baseline method (as in the CSV) -> (label, colour, linestyle, marker)
BASELINES = [
    ("E2-LoRA", "E$^2$-LoRA", TERRA, "-", "s"),
    ("TUNA", "TUNA", C60, "--", "^"),
    ("MOS", "MOS", C40, "-.", "v"),
    ("LoRA-DRS", "LoRA-DRS", C25, ":", "D"),
]
JOINT = "Joint training (E2-LoRA)"
RESULTS_ROW_RE = re.compile(r"^\|\s*(CIFAR-100|ImageNet-R)\s*\|\s*(\d+) tasks x \d+\s*\|\s*([\d.]+) ± ([\d.]+)\s*\|")
RESULTS_DS = {"CIFAR-100": "cifar224", "ImageNet-R": "imagenetr"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def ism_points(agg_rows, metric, config=None, allow_incomplete=False):
    """{(dataset, split): (mean, sd, label)} from the e1 agg CSV."""
    out = {}
    for ds, _ in PANELS:
        for split, _ in SPLITS:
            cells = [r for r in agg_rows if r["dataset"] == ds and r["split"] == split and _f(r.get(metric)) is not None]
            if config:
                cells = [r for r in cells if r["config_name"] == config]
            if not cells:
                continue
            head = [r for r in cells if _f(r["alpha"]) == rc.HEADLINE_ALPHA[ds] and _f(r["lambda"]) == rc.HEADLINE_LAMBDA[ds]]
            ref = head[0] if head else max(cells, key=lambda r: int(_f(r["n_seeds"]) or 0))
            n, complete = int(_f(ref["n_seeds"]) or 0), ref.get("complete") == "True"
            if not (n == 3 and complete):
                msg = f"{ds} {split} {ref['config_name']}: n_seeds={n} complete={ref.get('complete')}"
                if not allow_incomplete:
                    raise AssertionError("ISM point without 3 complete seeds: " + msg)
                print("WARNING (drawn anyway):", msg)
            out[(ds, split)] = (_f(ref[metric]), _f(ref.get(metric + "_sd")) or 0.0, ref["config_name"])
    return out


def ism_points_from_results(path):
    """ISM points from the RESULTS.md 'Degradation profile (ours)' table (task-equal FA)."""
    out, inside = {}, False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("### Degradation profile"):
                inside = True
                continue
            if inside and line.startswith("#"):
                break
            m = RESULTS_ROW_RE.match(line) if inside else None
            if m:
                out[(RESULTS_DS[m.group(1)], f"t{m.group(2)}")] = (float(m.group(3)), float(m.group(4)), "RESULTS.md")
    if not out:
        raise AssertionError(f"no 'Degradation profile (ours)' rows found in {path}")
    return out


def baseline_points(rows):
    """{(method, dataset, split): (last, last_sd)}"""
    out = {}
    for r in rows:
        last = _f(r["last"])
        if last is not None:
            out[(r["method"].strip(), r["dataset"].strip(), r["split"].strip())] = (last, _f(r["last_sd"]) or 0.0)
    return out


def style_axis(ax):
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=COBALT, length=2, labelsize=6)
    ax.yaxis.grid(True, color="#FFFFFF", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(COBALT)
        ax.spines[sp].set_linewidth(0.8)


def draw(ism, base, metric_label, ism_label, out_path):
    xs = list(range(len(SPLITS)))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05))
    fig.patch.set_facecolor(PAPER)
    for ax, (ds, title) in zip(axes, PANELS):
        style_axis(ax)
        joint = [base.get((JOINT, ds, s)) for s, _ in SPLITS]
        if all(j is not None for j in joint):
            ax.plot(xs, [j[0] for j in joint], color=C60, lw=0.8, ls=(0, (1, 1.5)), zorder=2)
        for method, _, colour, ls, marker in BASELINES:
            pts = [(x, base[(method, ds, s)]) for x, (s, _) in zip(xs, SPLITS) if (method, ds, s) in base]
            if not pts:
                continue
            ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts], yerr=[p[1][1] for p in pts], color=colour,
                        ls=ls, lw=0.9, marker=marker, ms=3.2, mfc=PAPER if colour != TERRA else TERRA, mec=colour,
                        mew=0.8, capsize=1.5, elinewidth=0.6, zorder=3)
        pts = [(x, ism[(ds, s)]) for x, (s, _) in zip(xs, SPLITS) if (ds, s) in ism]
        if pts:
            ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts], yerr=[p[1][1] for p in pts], color=COBALT,
                        lw=1.3, marker="o", ms=3.8, mfc=COBALT, mec=COBALT, capsize=1.5, elinewidth=0.7, zorder=5)
            for x, (m, _, _) in pts:
                ax.annotate(f"{m:.1f}", (x, m), xytext=(0, 4.5), textcoords="offset points", ha="center",
                            va="bottom", fontsize=5.6, color=COBALT, zorder=6)
        else:
            print(f"WARNING: no ISM point for {ds}")
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab in SPLITS])
        ax.set_xlim(-0.35, len(SPLITS) - 0.65)
        ax.set_xlabel("number of tasks", color=COBALT, fontsize=6.5, labelpad=2)
        ax.set_title(title, color=COBALT, fontsize=7, pad=3)
    axes[0].set_ylabel(metric_label, color=COBALT, fontsize=6.5, labelpad=2)
    handles = [Line2D([], [], color=COBALT, lw=1.3, marker="o", ms=3.8, label=ism_label)]
    handles += [Line2D([], [], color=c, ls=ls, lw=0.9, marker=mk, ms=3.2, mfc=PAPER if c != TERRA else TERRA, mec=c,
                       label=lab) for _, lab, c, ls, mk in BASELINES]
    handles.append(Line2D([], [], color=C60, lw=0.8, ls=(0, (1, 1.5)), label="Joint training (E$^2$-LoRA, full FT)"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6,
                     bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.0)
    for t in leg.get_texts():
        t.set_color(COBALT)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.3, wspace=0.22)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + ".pdf", facecolor=PAPER)
    fig.savefig(out_path + ".png", facecolor=PAPER, dpi=200)
    plt.close(fig)
    print("saved", out_path + ".{pdf,png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "figures_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "assets", "figures"))
    ap.add_argument("--metric", choices=("last_total", "faa"), default="last_total",
                    help="ISM column: sample-weighted last Total Acc (default, the baselines' metric) or task-equal FA")
    ap.add_argument("--config", default=None, help="ISM config_name to use for every split (default: headline cell)")
    ap.add_argument("--from-results", action="store_true",
                    help="ISM points from the RESULTS.md degradation-profile table (task-equal FA; preview)")
    ap.add_argument("--allow-incomplete", action="store_true", help="draw ISM cells with fewer than 3 seeds")
    args = ap.parse_args(argv)

    base = baseline_points(read_csv(os.path.join(args.data_dir, "e1_long_baselines.csv")))
    if args.from_results:
        ism = ism_points_from_results(os.path.join(ROOT, "RESULTS.md"))
        label, suffix = "ISM (ours; RESULTS.md, task-equal FA)", "_results"
        ylabel = "Last-Acc (%); ISM = task-equal FA"
        print("NOTE: ISM points are the task-equal FA of RESULTS.md, the baselines are sample-weighted Last-Acc")
    else:
        agg = read_csv(os.path.join(args.data_dir, "e1_long_agg.csv"))
        ism = ism_points(agg, args.metric, args.config, args.allow_incomplete)
        if args.metric == "faa":
            label, suffix, ylabel = "ISM (ours; task-equal FA)", "_fa", "Last-Acc (%); ISM = task-equal FA"
        else:
            label, suffix, ylabel = "ISM (ours; sample-weighted)", "", "Last-Acc (%, sample-weighted)"
    for key, (m, sd, src) in sorted(ism.items()):
        print(f"ISM {key[0]} {key[1]}: {m:.2f} +- {sd:.2f} ({src})")
    draw(ism, base, ylabel, label, os.path.join(args.out_dir, "fig_e1_long" + suffix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
