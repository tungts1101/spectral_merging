"""Table-6-format view of the E12 merge grids while they run.

Reads the combined output of scripts/e12p_status.sh from both machines (stdin or a
file), takes the best finished cell (highest 3-seed FAA) per method and dataset, and
prints one row per method in the order of the paper's Table 6 (plus NUFILT and
Breadcrumbs; '-' where no cell has finished): FA and AA as mean ± std over 3 seeds
per dataset, then the average over the datasets that have a value and their count.
Two tables: the proxy protocol (e12p_ cells, 2026-11-11 pass; ISM rank rows
'ISM r{r} (proxy)' from the e12p_ism_r{r}_* cells) and the full protocol (e12_ cells,
2026-11-12 pass; the ISM row is the e12_ism chain cell = the paper setting).  The
full-protocol rank chains (e12r_ism_r{r}) are printed as one line at the end.

    python scripts/e12p_table6.py status.txt
"""
import re
import sys
from collections import OrderedDict, defaultdict

ROWS = OrderedDict([("ism", "ISM (ours)"), ("modelstock", "Model Stock"), ("avg", "Task Arithmetic (average)"),
                    ("della", "DELLA"), ("knots", "KnOTS-TIES"), ("dare", "DARE"), ("ties", "TIES"),
                    ("max_abs", "Max-abs (MagMax)"), ("nufilt", "NUFILT"), ("breadcrumbs", "Breadcrumbs")])
DATASETS = OrderedDict([("cifar224", "CIFAR-100"), ("cub", "CUB-200"), ("imagenetr", "ImageNet-R"), ("cars", "Stanford Cars")])
CELL_RE = re.compile(r"^(\w+) e12(p?)_(ism|nufilt|ties|dare|della|knots|breadcrumbs|avg|max_abs|modelstock)"
                     r"(?:_r(\d+))?(?:_(\S*?))?(?:_c([0-9.]+))? ASA: ([0-9.]+) ± ([0-9.]+) \| FAA: ([0-9.]+) ± ([0-9.]+)"
                     r"(?:.*\| n=(\d+))?")
CHAIN_RE = re.compile(r"^(\w+) e12r_ism_r(\d+) ASA: ([0-9.]+) ± ([0-9.]+) \| FAA: ([0-9.]+) ± ([0-9.]+)")
ISM_RE = re.compile(r"p([0-9.]+)_a([0-9.]+)")


def knob_text(rule, mid, coef):
    if rule == "ism":
        m = ISM_RE.match(mid or "")
        return f"a{m.group(2)} p{m.group(1)}" if m else (mid or "paper setting")
    if rule == "modelstock":
        return ""
    return " ".join(x for x in ((mid or "").replace("_", " "), f"c{coef}" if coef else "") if x)


def collect(text, proto):
    """best cell per (rule, ds) for one protocol: proto=True -> e12p_ cells, False -> e12_ cells."""
    best, done, ranks = {}, defaultdict(int), set()
    for line in text.splitlines():
        m = CELL_RE.match(line.strip())
        if not m:
            continue
        ds, pflag, rule, rank, mid, coef, asa, asa_sd, faa, faa_sd, nseeds = m.groups()
        if (pflag == "p") != proto or (nseeds is not None and int(nseeds) < 3):
            continue   # single-seed timing cells are not comparable
        key_rule = rule if rank is None else f"ism_r{rank}"
        if rank is not None:
            ranks.add(int(rank))
        key = (key_rule, ds)
        done[key] += 1
        if key not in best or float(faa) > best[key][0]:
            best[key] = (float(faa), float(asa), knob_text(rule, mid, coef), float(faa_sd), float(asa_sd))
    return best, done, ranks


def print_table(best, done, ranks, title):
    rows = OrderedDict(ROWS)
    for r in sorted(ranks):
        rows[f"ism_r{r}"] = f"ISM r{r} (proxy)"
    head = "| Merge rule | " + " | ".join(f"{n} FA | AA" for n in DATASETS.values()) + " | Avg FA | Avg AA | n ds |"
    sep = "|" + "---|" * (2 + 2 * len(DATASETS) + 2)
    print(f"**{title}**\n")
    print(head)
    print(sep)
    for rule, label in rows.items():
        cols, fas, aas = [], [], []
        for ds in DATASETS:
            b = best.get((rule, ds))
            if b:
                cols += [f"{b[0]:.2f}±{b[3]:.2f}", f"{b[1]:.2f}±{b[4]:.2f}"]
                fas.append(b[0]); aas.append(b[1])
            else:
                cols += ["-", "-"]
        avg = [f"{sum(fas) / len(fas):.2f}", f"{sum(aas) / len(aas):.2f}", f"{len(fas)}/4"] if fas else ["-", "-", "0/4"]
        print(f"| {label} | " + " | ".join(cols + avg) + " |")
    sel = [f"{rows[r]} / {DATASETS[d]}: {b[2]} ({done[(r, d)]} cells)" for (r, d), b in sorted(best.items())
           if b[2] and r in rows]
    if sel:
        print("selected settings: " + "; ".join(sel))
    print()


def main(argv):
    text = open(argv[1], encoding="utf-8").read() if len(argv) > 1 else sys.stdin.read()
    best, done, ranks = collect(text, proto=True)
    print_table(best, done, ranks, "Proxy protocol (cosine-prototype head, no drift map, no alignment): "
                "mean ± std over 3 seeds of the best finished cell per method and dataset, selection on FA")
    bestf, donef, _ = collect(text, proto=False)
    if bestf:
        print_table(bestf, donef, set(), "Full protocol (paper: merge + drift map + alignment), same 4 cells per "
                    "method and dataset; the ISM row is the e12_ism chain cell (paper setting)")
    full = []
    for line in text.splitlines():
        m = CHAIN_RE.match(line.strip())
        if m:
            ds, rank, asa, asa_sd, faa, faa_sd = m.groups()
            full.append(f"{DATASETS[ds]} r{rank}: FA {float(faa):.2f}±{float(faa_sd):.2f} / AA {float(asa):.2f}±{float(asa_sd):.2f}")
    if full:
        print("full-protocol rank chains (paper protocol, headline cell): " + "; ".join(full))


if __name__ == "__main__":
    main(sys.argv)
