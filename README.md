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
| CIFAR-100 | 91.97 ± 0.32 | 94.80 ± 0.14 | 2.94 ± 0.32 |
| CUB-200 | 88.90 ± 0.46 | 92.16 ± 0.63 | 4.74 ± 0.39 |
| ImageNet-R | 83.68 ± 0.15 | 87.54 ± 0.20 | 3.74 ± 0.39 |
| Stanford Cars | 83.72 ± 0.05 | 87.09 ± 0.47 | 5.71 ± 0.53 |

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
dispatch is `Learner.merge()` in `exp21.py`.

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
  `/home/lis/data`); CIFAR-100 downloads via torchvision, ImageNet-R / CUB /
  Cars are fetched automatically on first use.
- Backbone: timm `vit_base_patch16_224.augreg2_in21k_ft_in1k` (downloaded from
  the HF hub on first run).
- First run trains all backbones (~2 h/benchmark on a single 24 GB GPU);
  states are cached under `checkpoints_exp21/`, after which Stage-2 re-runs
  take ~10 min/seed.
- Per-dataset hyperparameters (weight decay, adapter coverage, merge α/p,
  drift λ, alignment settings) are in `run_best.py`; everything else is shared
  across benchmarks.

## Files

| File | Contents |
|---|---|
| `exp21.py` | learner: seqft training, incremental merge dispatch, drift map, class stats, LCA alignment, eval + sweep harness |
| `helper.py` | `pspectral_merging` / `truncate_energy` (the spectral merge), task-vector merges, PEFT backbone builder, metrics |
| `backbones_exp21.py` | reference-recipe ports and adapter utilities used by exp21 |
| `utils/` | dataset definitions and the incremental `DataManager` |
| `run_best.py` | headline-result reproduction script |
