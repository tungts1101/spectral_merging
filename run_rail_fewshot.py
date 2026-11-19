"""RAIL-Adapter on frozen ViT-B/16 features in the few-shot protocol (rebuttal follow-up).

RAIL (Xu et al., NeurIPS 2024, arXiv:2406.18868; released code
github.com/linghan1997/Regression-based-Analytic-Incremental-Learning) keeps the
pre-trained encoder frozen and fits a closed-form ridge classifier on its image
features, re-solved after every task on all features seen so far. Its second
part, a zero-shot fusion with CLIP text embeddings, needs a text encoder that
our backbone does not have, so only the adapter is run here. Two forms, as in
the released code:
    dual     Gaussian-kernel ridge regression on L2-normalised features
             (utils.kernel: exp(-gamma ||x - x'||^2), gamma 0.1; dual_RAIL.py:
             lamda 1e-3; alpha = (K + lamda I)^-1 Y, one-hot Y zero-padded for
             the new classes; prediction = argmax of k(x, X) alpha)
    primal   a random, untrained expansion ReLU(W x) to hidden_dim 15,000
             (nn.Linear init, bias-free) and ridge regression with
             regularization 0.1 (configs/analytic_clip.yaml). The released
             code updates the ridge solution recursively per batch (RLS); the
             recursion is exact, so the solution is computed here in closed form
             on all accumulated features (identical up to float round-off).
Features: the frozen backbone our LoRA module sits on (timm
vit_base_patch16_224, AugReg ImageNet-21K -> ImageNet-1K), the class token after
the final norm, test transform, L2-normalised as in RAIL's encode_images.

Protocol: Appendix H / run_e13_fewshot: ImageNet-R, 10 tasks of 20 classes,
K training images per class and task through the same DataManager hook
(data_fewshot_k, drawn with the run seed, so the RAIL rows see exactly the
training images of the ISM rows), test set untouched. After every task the
adapter classifies the test images of all seen classes (argmax over the seen
columns); the accuracy matrix gives FA / AA / FF with helper.compute_metrics,
as every other row. Three seeds, K in {5, 10}.

Outputs: logs_main/imagenetr/e16_rail_fs<K>.log (one line per task and seed,
'[RAIL] ... FAA/FFM/ASA' per seed and form, a SUMMARY block per form) and
figures_data/appendix/e16_rail_fewshot.csv (form, K, seed, FA, AA, FF).
Cost: one feature pass over the K-shot train set and the 6,000 test images per
seed and K (about a minute), kernel solves on at most 2,000 points.

    python run_rail_fewshot.py --dry-run          # torch-free
    python run_rail_fewshot.py --smoke            # 2 tasks, seed 1993, K 5, no CSV
    python run_rail_fewshot.py                    # K 5 10, seeds 1993 1994 1995, both forms
"""
import argparse
import csv
import logging
import os
import sys
import time

import rebuttal_common as rc

# Streams: ImageNet-R few-shot (Appendix H) and RAIL's X-TAIL (utils.data.xtail, 16 shots).
# backbone, timm num_classes (None = the model's own head: CLIP's 512-d projection, RAIL's
# feature), class-order shuffle, default shots.
STREAMS = {
    "imagenetr": ("vit_base_patch16_224", 0, True, (5, 10)),
    "xtail": ("vit_base_patch16_clip_224.openai", None, False, (16,)),
}
DATASET = "imagenetr"
KS = (5, 10)
NUM_CLASSES = 0
SHUFFLE = True
FORMS = ("dual", "primal")
BACKBONE = "vit_base_patch16_224"      # rc.MODEL_BACKBONE without the '_lora' suffix
# Released defaults (configs/analytic_clip.yaml, dual_RAIL.py)
GAMMA = 0.1
LAMDA = 1e-3
HIDDEN_DIM = 15000
REGULARIZATION = 0.1
CSV_PATH = os.path.join("figures_data", "appendix", "e16_rail_fewshot.csv")


def cell_name(k):
    return f"e16_rail_fs{int(k)}" if DATASET == "imagenetr" else f"e16_rail_{DATASET}_fs{int(k)}"


# ----------------------------------------------------------------------------
# Adapters (torch, float64)
# ----------------------------------------------------------------------------

def rbf_kernel(a, b, gamma):
    """exp(-gamma ||a_i - b_j||^2) for row vectors (RAIL utils.kernel)."""
    import torch
    sq = (a * a).sum(1, keepdim=True) + (b * b).sum(1).unsqueeze(0) - 2.0 * a @ b.t()
    return torch.exp(-gamma * sq.clamp_min(0.0))


class DualRAIL:
    """Kernel ridge regression re-solved on all accumulated features."""

    def __init__(self, gamma=GAMMA, lamda=LAMDA):
        self.gamma, self.lamda = gamma, lamda
        self.X = None
        self.alpha = None

    def fit(self, X, Y):
        import torch
        self.X = X
        self.alpha = None                                 # free the previous solution first
        K = rbf_kernel(X, X, self.gamma)
        K.diagonal().add_(self.lamda)
        self.alpha = torch.linalg.solve(K, Y)
        del K

    def predict(self, Xq, chunk=1024):
        import torch
        return torch.cat([rbf_kernel(Xq[i:i + chunk], self.X, self.gamma) @ self.alpha
                          for i in range(0, Xq.shape[0], chunk)])


class PrimalRAIL:
    """Random ReLU expansion + ridge regression (closed form of the released RLS)."""

    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, reg=REGULARIZATION, seed=0, device="cpu"):
        import torch
        g = torch.Generator().manual_seed(int(seed))
        lin = torch.nn.Linear(in_dim, hidden_dim, bias=False)
        with torch.no_grad():
            bound = 1.0 / (in_dim ** 0.5)                 # nn.Linear default: U(-1/sqrt(in), 1/sqrt(in))
            lin.weight.uniform_(-bound, bound, generator=g)
        self.W = lin.weight.detach().to(device=device, dtype=torch.float64).t()   # (in, hidden)
        self.reg = reg
        self.w = None

    def expand(self, X):
        import torch
        return torch.relu(X @ self.W)

    def fit(self, X, Y):
        import torch
        Phi = self.expand(X)                              # (n, hidden)
        n = Phi.shape[0]
        # (Phi^T Phi + reg I)^-1 Phi^T Y == Phi^T (Phi Phi^T + reg I)^-1 Y  (n << hidden)
        self.w = None
        G = Phi @ Phi.t()
        G.diagonal().add_(self.reg)
        self.w = Phi.t() @ torch.linalg.solve(G, Y)      # (hidden, C)
        del G, Phi

    def predict(self, Xq, chunk=1024):
        import torch
        return torch.cat([self.expand(Xq[i:i + chunk]) @ self.w for i in range(0, Xq.shape[0], chunk)])


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------

def extract(model, dataset, device, batch_size=256, workers=4):
    """L2-normalised float64 features and labels of `dataset` (test transform)."""
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    feats, labels = [], []
    with torch.no_grad():
        for _, _, x, y in loader:
            f = model(x.to(device, non_blocking=True)).double()
            feats.append(torch.nn.functional.normalize(f, dim=1))
            labels.append(y.long().to(device))
    return torch.cat(feats), torch.cat(labels)


def one_hot(y, n_cls):
    import torch
    Y = torch.zeros(y.numel(), n_cls, dtype=torch.float64, device=y.device)
    Y[torch.arange(y.numel(), device=y.device), y] = 1.0
    return Y


# ----------------------------------------------------------------------------
# One seed
# ----------------------------------------------------------------------------

def run_seed(k, seed, forms, device, smoke=False):
    """{form: (faa, ffm, asa)} for one (K, seed)."""
    import numpy as np
    import timm
    import torch
    from utils.data_manager import DataManager
    from helper import compute_metrics

    T_all, init_cls, inc = rc.DATA_TABLE[DATASET]
    dm = DataManager(DATASET, SHUFFLE, seed, init_cls, inc, False, fewshot_k=int(k))
    logging.info(dm.describe())
    T = 2 if smoke else dm.nb_tasks
    kw = {} if NUM_CLASSES is None else {"num_classes": NUM_CLASSES}
    model = timm.create_model(BACKBONE, pretrained=True, **kw).to(device).eval()
    n_seen = sum(dm.get_task_size(t) for t in range(T))

    t0 = time.time()
    Xte, yte = extract(model, dm.get_dataset(np.arange(0, n_seen), source="test", mode="test"), device)
    Xtr, ytr = extract(model, dm.get_dataset(np.arange(0, n_seen), source="train", mode="test"), device)
    logging.info(f"[RAIL] seed {seed} K {k}: features of {Xtr.shape[0]} train and {Xte.shape[0]} test "
                 f"images in {time.time() - t0:.1f}s (d = {Xtr.shape[1]})")

    adapters = {}
    for form in forms:
        adapters[form] = (DualRAIL() if form == "dual"
                          else PrimalRAIL(Xtr.shape[1], seed=seed, device=device))
    mats = {form: np.zeros((T, T)) for form in forms}
    bounds = np.cumsum([0] + [dm.get_task_size(t) for t in range(T)])
    for t in range(T):
        hi = int(bounds[t + 1])
        tr_mask = ytr < hi
        X, Y = Xtr[tr_mask], one_hot(ytr[tr_mask], hi)
        te_mask = yte < hi
        Xq, yq = Xte[te_mask], yte[te_mask]
        for form in forms:
            t1 = time.time()
            adapters[form].fit(X, Y)
            pred = adapters[form].predict(Xq).argmax(1)
            accs = []
            for j in range(t + 1):
                m = (yq >= bounds[j]) & (yq < bounds[j + 1])
                accs.append(100.0 * (pred[m] == yq[m]).double().mean().item())
                mats[form][t, j] = accs[-1]
            logging.info(f"[RAIL-{form}] seed {seed} K {k} task {t}: n_train {X.shape[0]}, "
                         f"grouped {[round(a, 2) for a in accs]}, total {np.mean(accs):.2f}, "
                         f"{time.time() - t1:.2f}s")
    out = {}
    for form in forms:
        faa, ffm, ffd, asa = compute_metrics(mats[form])
        logging.info(f"[RAIL-{form}] seed {seed} K {k}: FAA: {faa:.2f}, FFM: {ffm:.2f}, ASA: {asa:.2f}")
        out[form] = (float(faa), float(ffm), float(asa))
    del model
    torch.cuda.empty_cache()
    return out


def run_cell(k, seeds, forms, device, smoke=False, log_root=None, write_csv=True):
    import numpy as np
    name = cell_name(k) + ("_smoke" if smoke else "")
    root = rc.SMOKE_LOG_ROOT if smoke else (log_root or rc.LOG_DIR)
    os.makedirs(os.path.join(root, DATASET), exist_ok=True)
    logfile = os.path.join(root, DATASET, name + ".log")
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s] => %(message)s",
                        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)], force=True)
    results = {form: [] for form in forms}
    for seed in seeds:
        logging.info("\n" + "=" * 80)
        logging.info(f"Experiment: {DATASET} — {name} — seed {seed}")
        logging.info(f"Configuration: backbone {BACKBONE} (frozen), forms {list(forms)}, K {k}, "
                     f"gamma {GAMMA}, lamda {LAMDA}, hidden_dim {HIDDEN_DIM}, regularization {REGULARIZATION}")
        t0 = time.time()
        res = run_seed(k, seed, forms, device, smoke=smoke)
        logging.info(f"Time: {time.time() - t0:.1f}s")
        for form in forms:
            results[form].append(res[form])
            if write_csv and not smoke:
                new = not os.path.isfile(CSV_PATH)
                os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
                with open(CSV_PATH, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["form", "k", "seed", "fa", "aa", "ff"])
                    faa, ffm, asa = res[form]
                    w.writerow([form, k, seed, f"{faa:.2f}", f"{asa:.2f}", f"{ffm:.2f}"])
    for form in forms:
        r = np.array(results[form])
        logging.info("\n" + "=" * 80)
        logging.info(f"SUMMARY  {DATASET.upper()} — {name.upper()} — RAIL-{form}")
        logging.info("=" * 80)
        logging.info(f"  ASA: {r[:, 2].mean():.2f} ± {r[:, 2].std():.2f} | "
                     f"FAA: {r[:, 0].mean():.2f} ± {r[:, 0].std():.2f} | "
                     f"FFM: {r[:, 1].mean():.2f} ± {r[:, 1].std():.2f}")
    return results


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(STREAMS), default="imagenetr")
    ap.add_argument("--k", nargs="+", type=int, default=None, help="shots per class (default: the stream's)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(rc.SEEDS))
    ap.add_argument("--forms", nargs="+", choices=FORMS, default=list(FORMS))
    ap.add_argument("--dry-run", action="store_true", help="print the cells; no torch")
    ap.add_argument("--smoke", action="store_true", help="2 tasks, first seed, K 5, no CSV")
    ap.add_argument("--log-root", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    global DATASET, BACKBONE, NUM_CLASSES, SHUFFLE
    DATASET = args.dataset
    BACKBONE, NUM_CLASSES, SHUFFLE, default_ks = STREAMS[DATASET]
    if args.k is None:
        args.k = list(default_ks)
    if any(k <= 0 for k in args.k):
        raise SystemExit(f"--k must be positive, got {args.k}")
    ks, seeds = (list(args.k), list(args.seeds))
    if args.smoke:
        ks, seeds = [ks[0]], seeds[:1]
    for k in ks:
        name = cell_name(k) + ("_smoke" if args.smoke else "")
        print(f"[dry-run] ds={DATASET} name={name} tag=rail_{'+'.join(args.forms)} seeds={seeds} "
              f"backbone={BACKBONE} log={os.path.join(args.log_root or rc.LOG_DIR, DATASET, name + '.log')}"
              if args.dry_run else f"[launch] {DATASET} | {name} | forms {args.forms} | seeds {seeds}")
        if not args.dry_run:
            run_cell(k, seeds, args.forms, args.device, smoke=args.smoke, log_root=args.log_root)
    if args.dry_run:
        rc.print_cells(len(ks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
