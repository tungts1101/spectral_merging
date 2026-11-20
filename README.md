# Spectral Merging for Class-Incremental Learning

Rehearsal-free class-incremental learning with a pre-trained ViT-B/16, built
around **incremental spectral merging** of sequentially fine-tuned LoRA
adapters. No exemplars are stored; old-class knowledge is carried by the
merged adapter, per-class feature Gaussians, and a drift map.

## Results

FA (final class-balanced accuracy) | AA (average incremental accuracy) |
FF (final forgetting), mean ± std over 3 seeds, 10-task splits:

| Benchmark | FA | AA | FF |
|---|---|---|---|
| CIFAR-100 | 91.95 ± 0.27 | 94.82 ± 0.14 | 3.05 ± 0.27 |
| CUB-200 | 88.90 ± 0.46 | 92.16 ± 0.63 | 4.74 ± 0.39 |
| ImageNet-R | 83.68 ± 0.15 | 87.54 ± 0.20 | 3.74 ± 0.39 |
| Stanford Cars | 83.72 ± 0.05 | 87.09 ± 0.47 | 5.71 ± 0.53 |

Joint-training upper bounds (same recipe, all classes in one task, 3 seeds;
AA = FA by definition for a single session — `run_joint.py`):

| Benchmark | FA | AA |
|---|---|---|
| CIFAR-100 | 93.48 ± 0.04 | 93.48 ± 0.04 |
| CUB-200 | 89.88 ± 0.33 | 89.88 ± 0.33 |
| ImageNet-R | 87.11 ± 0.25 | 87.11 ± 0.25 |
| Stanford Cars | 89.19 ± 0.29 | 89.19 ± 0.29 |

All per-dataset configurations use the uniform posture: no covariance
shrinkage (gamma = 0) and identity-ridge drift lambda = 100 (CUB: 10);
only weight decay, merge alpha and the LCA robust weight vary by dataset.

## Framework

Each task t is processed in four steps:

```
train task t                merge                    deploy
  LoRA seqft  ──w_t──►  M_t = M_{t-1} ⊕ w_t  ──►  drift map W_t + class stats
                        (spectral merging)          └─► LCA alignment ─► eval
```

### 1. Sequential LoRA fine-tuning

One persistent LoRA adapter (rank 64, scale 2r, on **qkv + fc1 + fc2** of
every block) is trained task after task: task t initializes from task t−1's
trained state and is fine-tuned with plain cross-entropy on task-t data only
(SGD 0.9, lr 1e-2, cosine, 10 epochs). Adapting the MLP (fc1/fc2) alongside
attention is a significant component: it improves ImageNet-R and Cars by
+0.6–0.7 FA over qkv-only at equal rank, while extending coverage further
(attn.proj) gives the gain back — coverage has an interior optimum.

The per-task trained states w_0 … w_t are snapshotted; the *deployed* feature
extractor is never the raw last state but the merged state below.

### 2. Spectral merging (core contribution)

Sequential fine-tuning alone drifts the extractor toward the newest task and
forgets. Instead, we maintain a merged adapter M_t built **continuously** —
each task merges only the previous merged state with the newly trained state:

```
M_0 = w_0
M_t = M_{t-1} + α · trunc_p(w_t − M_{t-1})        (t ≥ 1)
```

`trunc_p` is an **energy-percentile SVD truncation** of the update
Δ = w_t − M_{t-1}: compute the SVD of Δ, keep the smallest set of leading
singular directions whose cumulative squared singular values reach a fraction
p of the update's total spectral energy, and discard the rest:

```
Δ = U S Vᵀ,   k* = min{ k : Σ_{i≤k} s_i² ≥ p · Σ_i s_i² },
trunc_p(Δ) = U[:, :k*] S[:k*] Vᵀ[:k*, :]
```

Because LoRA states factor as B·A, truncation can act at different levels
(`spectral_variant`):

| Variant | What is truncated |
|---|---|
| `only_A` (default) | the update of the down-projection A; B interpolated plainly |
| `only_B` | the update of B; A interpolated plainly |
| `separate` | both factor updates, independently |
| `combine` | the effective update B·A, then re-factored to rank r |

Two scalars control the merge: **p** (energy percentile, how many directions
survive) and **α** (damping of the surviving update). Tuned operating point:
`only_A`, p = 0.9, α = 0.15–0.3 (per dataset).

Why it works — findings from ablations in this project:

- **The merge is the single largest component** of the framework: removing it
  (deploying the raw sequential state) costs ~2–3 FA and roughly doubles
  forgetting on every benchmark tested.
- Its benefit is **direction selection, not norm control**: training-time
  spectral-norm regularization or hard σ-caps reproduce the merged chain's
  norm trajectory but not its accuracy; proportional rescaling (division by
  σ) is harmful at every rank. Truncating *which* directions of the update
  survive is what suppresses interference.
- It **outperforms task-vector merges** (avg / max / min / max-abs / TIES) in
  both the sequential regime and an independent-training regime; sign-based
  merges degrade most when task solutions disagree.
- It requires **sequential initialization**: training each task from a common
  fresh init (instead of the previous solution) and merging identically costs
  ~2.5 FA — the merge exploits the compatibility of updates that sequential
  training produces.

Implementation: `pspectral_merging` in `helper.py` (factor-pair detection is
name-generic, so any adapted module merges the same way); the incremental
dispatch is `Learner.merge()` in `main.py`.

### 3. Drift compensation (identity-ridge map)

Class statistics estimated under older extractors go stale as M changes. After
each merge, a closed-form ridge map W (768×768) is fit on paired features of
current-task data — f_{t−1}(x) under the previous merged extractor vs f_t(x)
under the current one — with the residual anchored to identity:

```
min_W ‖F_old Wᵀ − F_new‖² + λ‖W − I‖²        (λ = 10–100)
```

Old-class Gaussians are transported through W (μ ← Wμ, Σ ← WΣWᵀ). The
identity anchor makes the extrapolation from current-task features to
old-class regions safe: dimensions without evidence of drift stay untouched.

### 4. Classifier alignment (N×K LCA)

After each task, per-class feature Gaussians (empirical mean + covariance with
a small jitter) are estimated for the new classes, old ones are transported as
above, and the linear head is aligned by sampling pseudo-features: N classes ×
K samples per batch, cross-entropy plus a robustness term (weight 0.1) over
resampled draws, 10 epochs. Only the head trains here; the backbone is frozen.

## Reproduction

```bash
pip install -r requirements.txt
python run_best.py          # all 4 benchmarks x 3 seeds
```

- Datasets resolve under `DATA_ROOT` in `utils/data.py` (default
  `/home/lis/data`; set the environment variable `DATA_ROOT` to your own
  folder). The main-table datasets are fetched on first use; the table below
  gives the sources and the on-disk layout.
- Backbone: timm `vit_base_patch16_224.augreg2_in21k_ft_in1k` (downloaded from
  the HF hub on first run).
- First run trains all backbones (~2 h/benchmark on a single 24 GB GPU);
  states are cached under `checkpoints_main/`, after which Stage-2 re-runs
  take ~10 min/seed.
- Per-dataset hyperparameters (weight decay, adapter coverage, merge α/p,
  drift λ, alignment settings) are in `run_best.py`; everything else is shared
  across benchmarks. `RESULTS.md` lists every number of the paper with the
  configuration that produced it (see "Where each number comes from" below).
- Checkpoints: none are needed. Every number is regenerated from scratch by
  the scripts below; cached chains only save training time. `RESULTS.md`,
  Appendix C, lists which chains the authors keep and which were lost with the
  original server. The Stage-2 drivers (`run_sota_merge.py`, the `run_e*.py`
  and `analyze_*.py` files) look for the chain under the prefix
  `main_<ds>_r64_qkvfc` (`rebuttal_common.prefix`) and train it on first use;
  where a driver offers `--prefix-family best`, it reuses `run_best.py`'s
  `best_<ds>_r64` chain instead.

### Datasets

| Dataset | Used for | How it is obtained | Layout under `DATA_ROOT` |
|---|---|---|---|
| CIFAR-100 | main table (10 / 20 / 50 tasks), robustness | `torchvision.datasets.CIFAR100(download=True)`, automatic | `cifar-100-python/` |
| ImageNet-R | main table, long tasks, appendix experiments | automatic: `gdown` from the Google-Drive archive of Zhou et al. ([RevisitingCIL](https://github.com/zhoudw-zdw/RevisitingCIL)), file id `1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR`; OneDrive mirror linked in `utils/data.py` | `imagenet-r/{train,test}/<class>/` |
| CUB-200 | main table | automatic: `gdown`, RevisitingCIL file id `1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb` | `cub/{train,test}/<class>/` |
| Stanford Cars | main table | automatic through `kagglehub` (`rickyyyyyyy/torchvision-stanford-cars`); needs a Kaggle API token in `~/.kaggle/kaggle.json` | `stanford_cars/` (torchvision `StanfordCars` layout) |
| CIFAR-100-C / -P | robustness tables (`RESULTS.md`, Section 2) | the releases of Hendrycks & Dietterich (Zenodo records 3555552 and 3555553); the per-corruption results are in `figures_data/*_per_category.csv` | `CIFAR-100-C/`, `CIFAR-100-P/` |
| Office-Home | domain-incremental table | manual download of the official release; the loader writes its own 70/30 split files (seed 1993) | `OfficeHome/{Art,Clipart,Product,Real World}/<class>/` |
| DomainNet | domain-incremental table | manual download of the *cleaned* version from http://ai.bu.edu/M3SDA/ together with the official `<domain>_{train,test}.txt` split files | `DomainNet/<domain>/...` and the split files |
| X-TAIL (RAIL stream) | rebuttal only | `scripts/xtail_download.py`, `scripts/xtail_splits.py` | see `utils/data.py` (`xtail`) |

If `gdown` is rate-limited, download the archive by hand from the link in
`utils/data.py` and unzip it under `DATA_ROOT` (the archives already contain
the `train/` and `test/` folders). Nothing else is needed for the main table:
the backbone comes from the HF hub through timm, and every checkpoint is
trained from scratch by `run_best.py`.

### Long task sequences

```bash
python run_long.py --ds cifar224 imagenetr --split t20   # 20 tasks, 3 seeds
python run_long.py --ds cifar224 imagenetr --split t50   # 50 tasks
```

Splits: CIFAR-100 20 x 5 and 50 x 2 classes, ImageNet-R 20 x 10 and 50 x 4.
The 50-task chains use 5 epochs per task (`RESULTS.md`, Section 4). Stage-2
options on a cached chain (`--cells ALPHA:P`, `--lambda-rule gcv`,
`--set KEY=VALUE`, `--tag`) are documented in the file header; the 20-task
CIFAR-100 tuning campaign that used them is recorded in `RESULTS.md`,
Section 5.

### Domain-incremental learning

```bash
python run_dil.py                     # Office-Home and DomainNet, 3 seeds
```

Office-Home and DomainNet are manual downloads (table above). The configuration
(Section 1 posture, α 0.15, λ 100, local-term weight 0.1; DomainNet 5 epochs per
session) is in the file; results in `RESULTS.md`, Section 3.

### Robustness: CIFAR-100-C and CIFAR-100-P

```bash
python run_robustness_models.py                                             # exports final_models/cifar224_rob_ours_s{1993,1994,1995}.pt
python robustness/eval_corruption.py   final_models/cifar224_rob_ours_s*.pt  # -> robustness/results/corruption.csv
python robustness/eval_perturbation.py final_models/cifar224_rob_ours_s*.pt  # -> robustness/results/perturbation.csv
python robustness/eval_adversarial.py  final_models/cifar224_rob_ours_s*.pt  # FGSM / PGD (RESULTS.md, "Known weaknesses")
```

`run_robustness_models.py` re-runs the main-table CIFAR-100 cell on the cached
chain of `run_best.py` (10 min per seed; it trains the chain first if it is
missing) and writes the deployed model (LoRA trainables, aligned head, class
Gaussians) through `main.py`'s `final_model_dir` hook. `--arms merge_avg
merge_ties merge_max_abs nomerge` export the merge-rule ablation of Table 8 from
the same chain, `--arms e2lora` the matched-backbone E2-LoRA row. The data sit
under `DATA_ROOT/CIFAR-100-C/` and `DATA_ROOT/CIFAR-100-P/` (Zenodo records
3555552 and 3555553). The corruption script prints the mean over all 19
corruptions (`RESULTS.md`, Section 2) and over the 15 standard corruptions (the
paper's Table 3 and Figure 3); the perturbation script reports the mean flip
probability over the 10 official perturbations on the first 2,500 sequences.
One CIFAR-100-C evaluation takes about 1.5 h per model on an RTX 4090, one
CIFAR-100-P evaluation about 1.1 h. The SLCA / EASE / MOS / TUNA rows were
exported from a LAMDA-PILOT checkout with a `save_final_model` hook
(`robustness/model_zoo.py` loads them); that patched copy is not part of this
repository. The per-corruption and per-perturbation numbers of every published
model are in `figures_data/*cifar_{c,p}_per_category.csv`.

### Appendix experiments (`RESULTS.md`, Section 6)

All on the cached chain of each dataset (see the checkpoint note above), 3
seeds, ImageNet-R unless stated; logs go to `logs_main/<ds>/<cell>.log` and
`python parse_logs.py <experiment>` turns them into the `figures_data/` CSVs.

| Paper | Command |
|---|---|
| Table 6 / 7, merge rules under an equal budget | `python run_e12_merge_grid.py --budget 4 --rules <rule> --ds <ds>` for every rule (`scripts/e12f_machine.sh` runs a machine's share); `python scripts/e12p_table6.py` builds the table |
| Table 1, KeepLoRA row | `python run_keeplora.py --stage main` (official recipe, rank 16) |
| Table 11, LoRA rank | `python run_e12_ism_ext.py --rank 16 --stage chain` (also 32 and 128; each trains its own chain) |
| Table 13, few-shot tasks | `python run_e13_fewshot.py --k 5 10 --rows ism e2lora lca`; `python run_rail_fewshot.py --dataset imagenetr` |
| Table 14, adaptive damping | `python run_e7_adaptive_alpha.py --rules specnorm featshift --tau 0.1 --kappa 0.15 --lambda-rule val` (constants from `--calibrate`, `figures_data/e7_calibration.json`) |
| Table 15, discarded directions | `python analyze_subspace.py --ds imagenetr --p 0.1` and the default p 0.9; `scripts/fig8_subspace.py` |
| Table 16, cheaper class statistics | `python run_covmode.py --modes cov_class cov_diag cov_mean` |
| Table 17, cost of the drift solve | `python run_covmode.py --timing --solver dense` and `--solver woodbury`; few-shot rows from `run_e13_fewshot.py --timing` |
| Table 17, ViT-L/16 row | `python run_vitl.py` (3 seeds, lr 5e-3, two micro-batches) then `python run_vitl.py --controls` |
| Table 18, drift map on stored samples | `python run_e9_buffer.py --n 5 10 20 --with-reference` |
| Appendix G, long task sequences | `python run_long.py --ds cifar224 imagenetr --split t20 t50`; `scripts/fig_e1_long.py` |
| Spectrum of the updates (calibration input) | `python analyze_spectrum.py --ds <ds>` |

### Where each number comes from

| Paper item | `RESULTS.md` | Produced by |
|---|---|---|
| Table 1 (main table), joint bounds | Section 1 | `run_best.py`, `run_joint.py`; published rows quoted from the papers; KeepLoRA row `run_keeplora.py` (Section 6.3) |
| Table 2 (domain-incremental) | Section 3 | `run_dil.py` |
| Table 3, Figures 3–4 (robustness) | Section 2 | `run_robustness_models.py`, `robustness/eval_*.py`, `scripts/fig3_robustness.py` |
| Table 4 (damping and energy fraction, p = 1 rows) | Section 2, "Merge 3-D ablation" | `main.run_config_sweep` cells over variant × p × α on the cached chain, exported by `export_mc3d_full.py` to `figures_data/mc3d_<ds>_full.csv`; `run_e8_trunc_vs_damp.py` for the p = 1 rows |
| Table 5 (per-dataset settings) | Section 1, "Running configuration" | `run_best.py` (`CONFIGS`) |
| Tables 6–7 (merge rules, equal budget) | Section 6.2 | `run_e12_merge_grid.py --budget 4` |
| Table 8 (robustness of the merge rules) | Section 2 | `run_robustness_models.py --arms merge_*` and the same export with the rules of `merging_sota.py` |
| Table 9 (component contributions) | "Component contributions" | `main.run_config_sweep` lattice cells (`train_merge`, `train_ca`, `train_drift` toggled) on the cached chain |
| Table 10 (sequential vs independent initialisation) | "Sequential versus independent task initialisation" | `main.run_config_sweep` with `seqft_independent` |
| Tables 11–18 (appendices F–M) | Sections 6.4–6.10 | the commands of the table above |
| Figure 2 (merge sweep) | Section 2, "Merge 3-D ablation" | `scripts/fig5_merge_sweep.py` from `figures_data/mc3d_imagenetr.csv` |
| Appendix G, Table 12 (long task sequences) | Sections 4 and 5 | `run_long.py`, `scripts/fig_e1_long.py` |

`PAPER_TRACKER.md` (authors' copy) maps every figure and table to its script
and log file.

## Files

| File | Contents |
|---|---|
| `main.py` | learner: seqft training, incremental merge dispatch, drift map, class stats, LCA alignment, eval + sweep harness |
| `helper.py` | `pspectral_merging` / `truncate_energy` (the spectral merge), task-vector merges, PEFT backbone builder, metrics |
| `backbones.py` | reference-recipe ports and adapter utilities used by main.py |
| `utils/` | dataset definitions and the incremental `DataManager` |
| `run_best.py` | headline-result reproduction script |
| `run_joint.py` | joint-training upper bound (all classes in one task, same recipe) |
| `run_dil.py` | domain-incremental Office-Home / DomainNet (Table 2) |
| `run_robustness_models.py`, `robustness/` | export of the deployed CIFAR-100 models and the CIFAR-100-C / -P / adversarial evaluators (Table 3, Figures 3–4, Table 8) |
| `run_long.py` | 20- and 50-task splits and the Stage-2 options used in the 20-task campaign (`RESULTS.md`, Sections 4–5) |
| `run_sota_merge.py`, `merging_sota.py` | published merge rules (DARE, DELLA, Breadcrumbs, KnOTS, Model Stock, NUFILT, OPCM) in the sequential protocol |
| `rebuttal_common.py`, `run_e*.py`, `run_keeplora.py`, `run_vitl.py`, `run_rail_fewshot.py`, `analyze_*.py`, `run_covmode.py` | the appendix experiments (`RESULTS.md`, Section 6) |
| `parse_logs.py`, `export_mc3d_full.py` | log parsers that write the `figures_data/` CSVs |
| `scripts/` | figure scripts (`fig*.py`), the Table 6 builder, the machine queues that ran the appendix experiments, X-TAIL data tools |
| `figures_data/` | the CSVs behind every figure and the appendix tables |
| `RESULTS.md` | every number of the paper with its configuration and log; `PAPER_TRACKER.md` (authors' copy) maps tables and figures to scripts |
