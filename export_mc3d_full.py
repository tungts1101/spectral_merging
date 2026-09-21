"""Export the FULL per-cell metrics of the 3-D merge ablation, not just
final accuracy: FA (last-acc), AA (inc-acc), FFM (forgetting), and the
old-vs-new task split of the final accuracy matrix.

For each (variant, p, alpha) cell and each seed the sweep logs contain, per
task, a "Grouped: [...]" line with the per-task accuracies of all tasks seen
so far. The LAST such line of a seed is the final accuracy vector; we split
it into:
  old_mean  = mean over tasks 0..T-2   (everything but the newest task)
  new_acc   = accuracy on task T-1     (the newest task)
  first5    = mean over tasks 0..4     (oldest half-decade)
which exposes the stability/plasticity trade-off that final accuracy hides.

Writes figures_data/mc3d_<ds>_full.csv (one row per cell, seed-averaged).
"""
import csv, glob, os, re, sys
import statistics as st

def parse(path, n_tasks=10):
    txt = open(path, errors="ignore").read()
    grouped = re.findall(r"Grouped: \[([0-9., ]+)\]", txt)
    faas = re.findall(r"Evaluation\] FAA: ([\d.]+), FFM: ([\d.]+)", txt)
    summ = re.search(r"ASA: ([\d.]+) ± ([\d.]+) \| FAA: ([\d.]+) ± ([\d.]+) \| FFM: ([\d.]+) ± ([\d.]+)", txt)
    finals = [g for i, g in enumerate(grouped, 1) if i % n_tasks == 0]
    old, new, first5 = [], [], []
    for g in finals:
        v = [float(x) for x in g.split(",")]
        if len(v) < n_tasks:
            continue
        old.append(st.mean(v[:-1])); new.append(v[-1]); first5.append(st.mean(v[:5]))
    row = {}
    if summ:
        row.update(aa=float(summ.group(1)), aa_sd=float(summ.group(2)),
                   fa=float(summ.group(3)), fa_sd=float(summ.group(4)),
                   ffm=float(summ.group(5)), ffm_sd=float(summ.group(6)))
    if old:
        row.update(old_mean=st.mean(old), new_acc=st.mean(new),
                   first5_mean=st.mean(first5), n_seeds=len(old))
    return row

def main(ds):
    rows = []
    for f in sorted(glob.glob(f"logs_exp21/{ds}/mc3d_*.log")):
        m = re.match(r"mc3d_(\w+?)_p([\d.]+)_a([\d.]+)$", os.path.basename(f)[:-4])
        if not m:
            continue
        r = parse(f)
        if "fa" not in r:
            continue
        rows.append(dict(variant=m.group(1), p=float(m.group(2)),
                         alpha=float(m.group(3)), **r))
    cols = ["variant", "p", "alpha", "n_seeds", "fa", "fa_sd", "aa", "aa_sd",
            "ffm", "ffm_sd", "old_mean", "new_acc", "first5_mean"]
    out = f"figures_data/mc3d_{ds}_full.csv"
    os.makedirs("figures_data", exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"wrote {out}: {len(rows)} cells")
    return rows

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "imagenetr")
