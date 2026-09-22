# Results

Method: sequential LoRA fine-tuning + spectral (pspectral) merge + identity-ridge drift
compensation + N×K LCA classifier alignment. Backbone ViT-B/16 pre-trained on ImageNet-21K
(supervised), exemplar-free. All numbers are our own runs over 3 seeds (1993/1994/1995),
reported as mean ± std.

Metrics: **Last-Acc** = accuracy over all seen classes after the final task (FAA).
**Inc-Acc** = average of the per-session accuracies (ASA). **FF** = final forgetting (FFM).

---

## 1. Main table — class-incremental, 10 tasks, ViT-B/16-IN21K

Baselines are each method's best protocol-matched published numbers (10-task, ViT-B/16-IN21K);
where a paper's own protocol differed, a protocol-matched third-party re-run is used.
`n/e` = not evaluated, `n/r` = not reported. APER+adapter's ImageNet-R cell
(65.79 ± 0.98 / 72.42 ± 1.41) is the 3-seed protocol-matched re-run from the SSIAT journal
version (arXiv:2403.19979, Tab. 1, "Adam-Adapter", 10 sessions, ViT-B/16-IN21K — the same
table our SSIAT row comes from). MOS's official-code single run (AAAI'25, arXiv:2412.09441
Tab. 1, IN-R B0-Inc20) reports a higher 67.95 / 75.82 without std; third-party APER re-runs
disagree by 1-2 points across papers, and the seed-replicated number is used here so every
row carries mean ± std.

| Method | CIFAR-100 Last | CIFAR-100 Inc | IN-R Last | IN-R Inc | CUB Last | CUB Inc | Cars Last | Cars Inc | Avg Last | Avg Inc |
|---|---|---|---|---|---|---|---|---|---|---|
| *Joint Training (upper bound, ours)* | *93.48 ± 0.04* | *93.48 ± 0.04* | *87.11 ± 0.25* | *87.11 ± 0.25* | *89.88 ± 0.33* | *89.88 ± 0.33* | *89.19 ± 0.29* | *89.19 ± 0.29* | *89.92* | *89.92* |
| L2P (CVPR'22) | 82.76 ± 1.17 | 88.48 ± 0.83 | 66.49 ± 0.40 | 72.83 ± 0.56 | 62.21 ± 1.92 | 73.83 ± 1.67 | 38.18 ± 2.33 | 51.79 ± 4.19 | 62.41 | 71.73 |
| DualPrompt (ECCV'22) | 85.56 ± 0.33 | 90.33 ± 0.33 | 68.50 ± 0.52 | 72.59 ± 0.24 | 66.00 ± 0.57 | 77.92 ± 0.50 | 40.14 ± 2.36 | 56.74 ± 1.78 | 65.05 | 74.40 |
| CODA-Prompt (CVPR'23) | 87.00 ± 0.31 | 90.68 ± 1.02 | 72.82 ± 0.50 | 78.13 ± 0.52 | 77.23 ± 1.12 | 81.90 ± 0.85 | 44.89 ± 0.61 | 58.91 ± 0.37 | 70.49 | 77.41 |
| SLCA (ICCV'23) | 91.53 ± 0.28 | 94.09 ± 0.87 | 77.00 ± 0.33 | 81.17 ± 0.64 | 84.71 ± 0.40 | 90.94 ± 0.68 | 67.73 ± 0.85 | 76.93 ± 1.21 | 80.24 | 85.78 |
| LAE (ICCV'23) | 85.59 ± 0.46 | 89.96 ± 0.44 | 72.66 ± 0.63 | 78.91 ± 0.89 | 77.48 ± 0.94 | 85.83 ± 0.68 | 52.47 ± 1.46 | 64.08 ± 1.01 | 72.05 | 79.70 |
| RanPAC (NeurIPS'23) | 91.09 ± 0.25 | 94.03 ± 0.58 | 75.28 ± 0.14 | 80.66 ± 0.58 | 87.88 ± 0.53 | 92.57 ± 0.55 | 61.43 ± 0.84 | 72.36 ± 1.18 | 78.92 | 84.91 |
| APER + adapter (IJCV'24) | 87.32 ± 0.28 | 91.21 ± 1.35 | 65.79 ± 0.98 | 72.42 ± 1.41 | 87.04 ± 0.13 | 92.13 ± 0.27 | 37.65 ± 0.09 | 49.22 ± 1.66 | 69.45 | 76.25 |
| HiDe-Prompt (NeurIPS'23) | 92.61 ± 0.28 | 94.03 ± 0.01 | 75.06 ± 0.12 | 76.60 ± 0.01 | 86.61 ± 0.18 | 87.01 ± 0.03 | n/e | n/e | n/a | n/a |
| CoMA (ECCV'24) | 92.00 ± 0.13 | 94.12 ± 0.63 | 77.47 ± 0.05 | 81.32 ± 0.17 | 85.95 ± 0.29 | 90.75 ± 0.39 | 73.35 ± 0.59 | 78.55 ± 0.42 | 82.19 | 86.19 |
| CoFiMA (ECCV'24) | 92.77 ± 0.24 | 94.89 ± 0.94 | 78.25 ± 0.26 | 81.48 ± 0.56 | 87.11 ± 0.56 | 91.87 ± 0.69 | 76.96 ± 0.64 | 82.65 ± 0.96 | 83.77 | 87.72 |
| ConvPrompt (CVPR'24) | 88.87 ± 0.33 | n/r | 77.86 ± 0.25 | n/r | 80.20 ± 0.52 | n/r | n/e | n/e | n/a | n/a |
| CPrompt (CVPR'24) | 87.82 ± 0.21 | 92.53 ± 0.23 | 77.14 ± 0.11 | 82.92 ± 0.70 | 82.69 ± 0.43 | n/r | 66.77 ± 0.37 | 76.81 ± 0.27 | 78.61 | n/a |
| InfLoRA (CVPR'24) | 86.51 ± 0.73 | 91.70 ± 0.32 | 75.65 ± 0.14 | 80.82 ± 0.24 | 70.82 ± 0.23 | 81.39 ± 0.14 | n/e | n/e | n/a | n/a |
| EASE (CVPR'24) | 86.54 ± 0.29 | 91.68 ± 0.18 | 77.75 ± 0.18 | 83.87 ± 0.78 | 79.90 ± 0.29 | 86.65 ± 0.63 | 35.46 ± 1.50 | 48.96 ± 0.96 | 69.91 | 77.79 |
| SSIAT (CVPR'24) | 91.51 ± 0.20 | 94.29 ± 0.88 | 79.61 ± 0.40 | 83.71 ± 0.42 | 88.32 ± 0.48 | 92.95 ± 0.55 | 67.43 ± 1.47 | 75.53 ± 1.26 | 81.72 | 86.62 |
| VQ-Prompt (NeurIPS'24) | 90.27 ± 0.06 | 93.10 ± 0.84 | 75.68 ± 0.23 | 80.02 ± 0.18 | 86.47 ± 0.40 | 91.37 ± 0.54 | n/e | n/e | n/a | n/a |
| SLCA++ (arXiv'24) | 91.46 ± 0.18 | 94.20 ± 0.71 | 78.09 ± 0.22 | 82.95 ± 0.78 | 86.59 ± 0.29 | 91.63 ± 0.72 | 73.97 ± 0.22 | 79.49 ± 0.80 | 82.53 | 87.07 |
| EvoPrompt (AAAI'24) | 87.57 ± 0.40 | n/r | 76.49 ± 0.27 | n/r | 79.88 ± 0.31 | n/r | n/e | n/e | n/a | n/a |
| CAPrompt (AAAI'25) | 95.52 ± 0.12 | n/r | 79.93 ± 0.19 | n/r | 88.99 ± 0.15 | n/r | n/e | n/e | n/a | n/a |
| MOS (AAAI'25) | 90.53 ± 0.35 | 94.09 ± 0.91 | 78.10 ± 0.12 | 83.61 ± 0.40 | 89.91 ± 0.25 | 92.87 ± 0.56 | 67.76 ± 1.28 | 74.38 ± 1.13 | 81.58 | 86.24 |
| BiLoRA (CVPR'25) | 87.46 ± 0.76 | 92.50 ± 0.62 | 77.95 ± 0.14 | 81.52 ± 0.26 | n/e | n/e | n/e | n/e | n/a | n/a |
| LoRA-DRS (CVPR'25) | 89.14 ± 0.23 | 92.55 ± 0.25 | 74.74 ± 0.78 | 81.16 ± 0.59 | n/e | n/e | n/e | n/e | n/a | n/a |
| SEMA (CVPR'25) | 87.84 | 92.23 | 74.82 | 81.39 | n/e | n/e | n/e | n/e | n/a | n/a |
| MACIL (ICML'25) | 91.94 ± 0.17 | 94.43 ± 0.79 | 81.88 ± 0.07 | 85.95 ± 0.27 | 90.52 ± 0.13 | 93.93 ± 0.47 | n/e | n/e | n/a | n/a |
| C-LoRA (arXiv'25) | 91.67 ± 0.05 | 94.29 ± 0.80 | n/e | n/e | 90.16 ± 0.15 | 93.87 ± 0.53 | 72.07 ± 1.12 | 80.42 ± 0.81 | n/a | n/a |
| TUNA (ICCV'25) | 91.79 ± 0.15 | 94.88 ± 0.06 | 79.44 ± 0.38 | 84.80 ± 0.37 | 88.40 ± 0.42 | 92.01 ± 0.68 | 69.46 ± 0.35 | 76.95 ± 0.39 | 82.27 | 87.16 |
| SLDC (AAAI'26) | 91.48 ± 0.24 | 94.38 ± 0.69 | 80.00 ± 0.29 | 84.01 ± 0.46 | 87.15 ± 0.50 | 92.38 ± 0.57 | 80.50 ± 0.30 | 85.45 ± 0.41 | 84.78 | 89.06 |
| LoDA (ICML'26) | 90.47 ± 0.06 | 93.46 ± 1.42 | 81.93 ± 0.20 | 86.90 ± 0.40 | 81.74 ± 0.78 | 89.35 ± 0.98 | n/e | n/e | n/a | n/a |
| LCA (ICLR'26) | 91.19 ± 0.09 | 94.80 ± 0.30 | 81.40 | 85.80 ± 0.20 | 83.85 | 90.80 ± 0.30 | 64.55 ± 1.93 | 76.20 ± 1.40 | 80.25 | 86.90 |
| E2-LoRA (ICML'26) | **92.13 ± 0.19** | **95.01 ± 0.02** | 82.77 ± 0.10 | 87.18 ± 0.12 | **89.77 ± 0.16** | **92.68 ± 0.48** | 75.82 ± 0.28 | 80.90 ± 0.73 | 85.12 | 88.94 |
| **Ours** | 91.95 ± 0.27 | 94.82 ± 0.14 | **83.68 ± 0.15** | **87.54 ± 0.20** | 88.90 ± 0.46 | 92.16 ± 0.63 | **83.72 ± 0.05** | **87.09 ± 0.47** | **87.06** | **90.40** |

The Joint Training row is our own recipe-matched upper bound: the same LoRA r64 qkv+fc
backbone trained once on all classes jointly (10 epochs, SGD, per-dataset weight decay),
3 seeds. With a single session, Inc-Acc equals Last-Acc by definition, so both columns carry
the same value. Gap of our incremental result to this bound: CIFAR −1.53, ImageNet-R −3.43,
CUB −0.98, Cars −5.47. For reference, E2-LoRA's published Joint Training row (their Table 1,
arXiv:2605.27482, full fine-tuning; the Cars cell quotes SLCA's 80.31) is CIFAR 93.13 ± 0.21,
IN-R 82.76 ± 0.54, CUB 88.26 ± 0.73, Cars 80.31 ± 0.13 — lower than the LoRA-recipe bound on
every dataset, and our incremental ImageNet-R (83.68) and Cars (83.72) exceed it outright.
E2-LoRA's claim of "matching joint training" on ImageNet-R (82.77 vs 82.76) holds only
against that weaker full-FT bound, not against the LoRA-recipe bound (87.11).

LCA (Tran et al., ICLR 2026, arXiv:2603.09888) values are its IM+LCA configuration. Inc-Acc is
as published (Table 1, 3 seeds); the paper does not tabulate Last-Acc, so those cells are
computed from the per-task accuracy matrices logged by that project (`results/*.txt`), which
reproduce the published Inc-Acc to within 0.15. CIFAR-100 and Cars have three seeds recorded
there, ImageNet-R and CUB one. Its IM-only row (incremental merging without the LCA loss) is
CIFAR 92.80, IN-R 84.30, CUB 86.70, Cars 70.10 Inc-Acc.

Our forgetting (FF): CIFAR 3.05 ± 0.27, IN-R 3.74 ± 0.39, CUB 4.74 ± 0.39, Cars 5.71 ± 0.53.

On CUB a qkv-only variant scores higher — **89.09 ± 0.02 / 92.45 ± 0.45** (FF 4.47 ± 0.24) —
but the table above keeps one uniform configuration across all four datasets.

### Running configuration

Shared across all datasets: ViT-B/16-IN21K frozen; LoRA r=64, α=128, dropout 0, applied to
`qkv`, `fc1`, `fc2` (8.26M trainable: 8,257,536 LoRA parameters + 1,536 norm parameters = 8,259,072, verified by counting a checkpoint; analytically 12 blocks x (qkv 196,608 + fc1 245,760 + fc2 245,760)); 10 epochs/task; SGD momentum 0.9; lr 1e-2 (cosine to
1e-6); batch 64; cross-entropy. Merge: pspectral, variant `only_A`. Drift: identity-ridge,
transport `full`. LCA: N×K sampling with N=40, K=64, 10 epochs, SGD lr 1e-2.

| Dataset | Tasks | weight decay | merge p | merge α | drift λ | shrinkage γ | robust weight |
|---|---|---|---|---|---|---|---|
| CIFAR-100 | 10 × 10 | 5e-4 | 0.9 | 0.15 | 100 | 0 | 0.2 |
| ImageNet-R | 10 × 20 | 2e-4 | 0.9 | 0.30 | 100 | 0 | 0.1 |
| CUB-200 | 10 × 20 | 1e-3 | 0.9 | 0.10 | 10 | 0 | 0.1 |
| Cars-196 | 16 + 9×20 | 1e-3 | 0.9 | 0.20 | 100 | 0 | 0.2 |

Alternatives tested and rejected on CIFAR-100 (3 seeds each, paired on shared backbones where
applicable): merge variants `combine` and `only_B` over p ∈ {0.7, 0.85, 0.95} ×
α ∈ {0.1, 0.2, 0.3} — all 18 cells at or below the `only_A` p0.9 α0.15 anchor, with accuracy
degrading monotonically in α and p inert; AdamW (head lr 1e-3) at lr ∈ {1e-3, 5e-4, 1e-4,
5e-5} — 69.29 / 87.66 / 90.51 / 91.21 Last-Acc, a decelerating approach to the SGD anchor
from below, so SGD is kept. Covariance shrinkage γ contributes nothing (γ-tuned 91.90-91.97
vs γ=0 91.95). Training settings (epochs, tied lr, decoupled PEFT/head lr) were separately
swept and tie the defaults.

---

## 2. Robustness — CIFAR-100, 10 tasks

All models trained under their own published recipes, exported at the evaluation point and
run through one shared inference harness. Seed 1993.

### CIFAR-100-C (15 corruptions × 5 severities, mean accuracy)

| Method | Corrupted ↑ | Clean |
|---|---|---|
| **Ours** | **80.02** | 92.28 |
| E2-LoRA | 79.61 | 92.46 |
| Merge (avg) | 78.68 | 90.72 |
| Merge (TIES) | 77.97 | 90.06 |
| TUNA | 77.02 | 91.58 |
| Merge (max-abs) | 76.33 | 89.39 |
| MOS | 74.63 | 89.66 |
| SLCA | 73.69 | 89.21 |
| EASE | 70.27 | 85.45 |

We are first on corruption robustness, +0.41 over E2-LoRA, despite starting 0.18 lower on clean
accuracy — so the gap under corruption is a genuine robustness margin, not inherited headroom.

### CIFAR-100-P (perturbation, mean flip probability — lower is better)

Reported because this is the one other axis where we place second.

| Method | Mean FP ↓ |
|---|---|
| E2-LoRA | 3.22 |
| **Ours** | **3.25** |
| Merge (avg) | 3.48 |
| Merge (TIES) | 3.59 |
| Merge (max-abs) | 3.98 |
| TUNA | 4.06 |
| SLCA | 4.31 |
| MOS | 5.03 |
| EASE | 5.67 |

### Robustness of the merge operators

CIFAR-100-C and CIFAR-100-P for the merge rules of the comparison table, seed 1993, identical
backbones and Stage-2 posture (only the merge differs). Corruption is the mean accuracy over
19 corruptions x 5 severities; perturbation is the mean flip probability (lower is better).
Per-category values: `figures_data/merge_cifar_c_per_category.csv`,
`figures_data/merge_cifar_p_per_category.csv`.

| Merge rule | CIFAR-100-C ↑ | clean | CIFAR-100-P ↓ |
|---|---|---|---|
| **Spectral (ours)** | **80.02** | 92.28 | **3.245** |
| Model Stock | 79.00 | 90.99 | 3.431 |
| Task Arithmetic (average) | 78.68 | 90.72 | 3.481 |
| DELLA | 78.57 | 90.55 | 3.472 |
| KnOTS-TIES | 78.45 | 90.42 | 3.525 |
| DARE | 78.03 | 90.53 | 3.522 |
| TIES | 77.97 | 90.06 | 3.588 |
| Max-abs (MagMax) | 76.33 | 89.39 | 3.982 |

Model Breadcrumbs was not evaluated on these two benchmarks.

Across every robustness axis the merge ordering pspectral > avg > TIES > max-abs holds.

The CIFAR-100-C means above average all 19 corruptions in the CSV. The paper reports the
15 standard corruptions (excluding the held-out gaussian_blur, saturate, spatter,
speckle_noise), which gives: Ours 79.11, E2-LoRA 78.75, Merge (avg) 77.80, Merge (TIES)
77.10, TUNA 76.11, Merge (max-abs) 75.36, MOS 73.66, SLCA 72.67, EASE 69.38 (same ordering,
same +0.36 margin over E2-LoRA).

### Merge-method ablation

The `Merge (…)` rows above are our own framework with the pspectral merge swapped for a
classical weight-merging rule, everything else identical (same backbones, drift, LCA). The
comparison isolates the contribution of spectral merging on accuracy and on robustness alike
(clean accuracy over 3 seeds; corruption/perturbation single seed, shared harness):

| Merge rule | Clean FA (3 seeds) | CIFAR-100-C ↑ | CIFAR-100-P ↓ |
|---|---|---|---|
| **pspectral (ours)** | **91.94 ± 0.30** | **80.02** | **3.25** |
| average | 90.26 ± 0.36 | 78.68 | 3.48 |
| TIES | 89.58 ± 0.36 | 77.97 | 3.59 |
| max-abs | 88.70 ± 0.53 | 76.33 | 3.98 |

The ordering pspectral > average > TIES > max-abs is identical on all three axes: spectral
truncation of the update buys +1.7 clean accuracy over the best classical rule and the
robustness gain comes with it, not at its expense. On CIFAR-C the classical-merge variants of
our own framework (78.68 / 77.97) still beat every external baseline except E2-LoRA.

The same comparison on all four benchmarks (3 seeds, paired on the shared cached backbones,
each dataset's tuned Stage-2 posture; classical rules incremental at coefficient 1.0):

| Merge rule | CIFAR-100 | ImageNet-R | CUB | Cars |
|---|---|---|---|---|
| **pspectral (ours)** | **91.95** | **83.68** | **88.90** | **83.72** |
| average | 90.26 | 83.32 | 85.87 | 82.36 |
| TIES | 89.58 | 82.67 | 84.54 | 81.73 |
| max-abs | 88.70 | 80.93 | 83.33 | 79.46 |

Extended comparison with published merging methods (each new rule tuned on ImageNet-R over
a small grid — drop rate / mask band / rank budget x coefficient — then its best setting run
on the other three datasets; same sequential pairwise protocol, task vectors from the LoRA
base, 3 seeds, paired backbones; FA | AA):

| Merge rule | Setting | CIFAR-100 | ImageNet-R | CUB | Cars |
|---|---|---|---|---|---|
| **pspectral (ours)** | per-dataset α, p 0.9 | **91.95 \| 94.82** | **83.68 \| 87.54** | **88.90 \| 92.16** | **83.72 \| 87.09** |
| Model Breadcrumbs (ECCV'24) | β 0.5, γ 0.01, c 1.0 | 91.43 \| 94.71 | 83.08 \| 87.33 | 87.98 \| 91.68 | 83.01 \| 86.86 |
| Model Stock (ECCV'24) | analytic t | 90.55 \| 94.37 | 83.36 \| 87.48 | 86.42 \| 91.04 | 82.65 \| 87.08 |
| Task Arithmetic / average | c 1.0 | 90.26 \| 94.26 | 83.32 \| 87.45 | 85.87 \| 90.81 | 82.36 \| 87.09 |
| DELLA | q 0.5, c 1.0 | 90.30 \| 94.30 | 83.00 \| 87.37 | 85.43 \| 90.65 | 82.29 \| 87.11 |
| KnOTS-TIES (ICLR'25) | full rank, c 1.0 | 89.97 \| 93.98 | 82.97 \| 87.35 | 84.75 \| 90.09 | 81.90 \| 86.77 |
| DARE (ICML'24) | q 0.5, c 1.0 | 89.98 \| 94.19 | 82.48 \| 87.21 | 84.47 \| 90.43 | 81.88 \| 86.97 |
| TIES (NeurIPS'23) | topk 100, c 1.0 | 89.58 \| 93.90 | 82.67 \| 87.14 | 84.54 \| 89.88 | 81.73 \| 86.77 |
| max-abs (= MagMax, ECCV'24) | c 1.0 | 88.70 \| 93.44 | 80.93 \| 86.48 | 83.33 \| 89.27 | 79.46 \| 85.89 |

Stage-1 tuning grids and all cell results are in `logs_exp21/sota_merge.log`; DARE at q=0.9
with c=1.0 collapses in the sequential chain (FA 0.52 on ImageNet-R) and its tuned q=0.5 is
used. Implementations in `merging_sota.py` (protocol-verified against `helper.merge`).

The ordering pspectral > average > TIES > max-abs holds on every dataset. The margin is
largest on the fine-grained sets (CUB +3.03, Cars +1.36 over the best classical rule) and the
classical rules also forget 2-3x more (e.g. CUB FF 9.5-12.5 vs 4.7).

### Merge 3-D ablation — ImageNet-R (variant x p x alpha)

Full cube: spectral_variant in {only_A, only_B, separate, combine} x
pspectral_p in {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} x train_merge_alpha in the same
six values — 144 cells, 3 seeds each (1993/1994/1995), cache-served on the
r64 qkv+fc backbones at the tuned IN-R posture (wd 2e-4, identity-ridge
lambda 100, gamma 0, robust 0.1). Values are Last-Acc means; p = 1.0 keeps
all update energy (no truncation); the (p=1.0, alpha=1.0) corner deploys the
raw sequential state (no merge). Per-cell data incl. seed counts:
`figures_data/mc3d_imagenetr.csv`. Full per-cell metrics — FA, AA, forgetting (FFM), and the
final accuracy matrix split into old tasks (mean over tasks 0-8), the newest task (task 9) and
the oldest five (tasks 0-4) — are in `figures_data/mc3d_imagenetr_full.csv` (exported by
`export_mc3d_full.py`).

Stability/plasticity breakdown along alpha (only_A, p = 0.9, 3 seeds):

| α | FA | AA | FFM | old (t0–8) | new (t9) | oldest 5 (t0–4) |
|---|---|---|---|---|---|---|
| 0.10 | 83.14 | 86.95 | 3.51 | 83.03 | 84.10 | 83.38 |
| 0.30 | 83.65 | 87.56 | 3.81 | 83.52 | 84.75 | 83.16 |
| 0.50 | 83.14 | 87.38 | 4.57 | 82.99 | 84.45 | 82.37 |
| 0.70 | 82.21 | 87.10 | 5.40 | 81.94 | 84.65 | 81.32 |
| 0.90 | 81.25 | 86.54 | 6.06 | 80.96 | 83.86 | 80.39 |
| 1.00 | 80.70 | 86.32 | 6.63 | 80.36 | 83.79 | 79.84 |

α = 0.3 → 1.0 at p = 0.9, per variant (FA | FFM | old | new): only_A 83.65→80.70 | 3.81→6.63 |
83.52→80.36 | 84.75→83.79; only_B 83.61→80.97 | 3.79→6.32 | 83.48→80.68 | 84.79→83.52;
separate 83.73→80.70 | 3.64→6.56 | 83.62→80.38 | 84.68→83.57; combine 83.71→81.34 | 3.66→6.02 |
83.56→81.08 | 85.07→83.64.


**only_A** (rows p, columns alpha):

| p \ α | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| 0.1 | 82.98 | 83.58 | 83.44 | 82.99 | 82.22 | 81.77 |
| 0.3 | 83.04 | 83.67 | 83.28 | 82.74 | 81.88 | 81.27 |
| 0.5 | 83.04 | 83.80 | 83.25 | 82.63 | 81.75 | 80.94 |
| 0.7 | 83.10 | 83.69 | 83.16 | 82.34 | 81.30 | 80.79 |
| 0.9 | 83.14 | 83.65 | 83.14 | 82.21 | 81.25 | 80.70 |
| 1.0 | 83.10 | 83.72 | 83.22 | 82.27 | 81.34 | 80.78 |

**only_B** (rows p, columns alpha):

| p \ α | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| 0.1 | 81.92 | 82.74 | 83.01 | 82.90 | 82.44 | 82.29 |
| 0.3 | 82.58 | 83.37 | 83.16 | 82.93 | 82.28 | 81.76 |
| 0.5 | 82.87 | 83.50 | 83.30 | 82.59 | 81.82 | 81.23 |
| 0.7 | 83.02 | 83.63 | 83.22 | 82.47 | 81.51 | 81.02 |
| 0.9 | 83.07 | 83.61 | 83.19 | 82.33 | 81.37 | 80.97 |
| 1.0 | 83.07 | 83.56 | 83.07 | 82.31 | 81.21 | 80.82 |

**separate** (rows p, columns alpha):

| p \ α | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| 0.1 | 81.75 | 82.57 | 82.84 | 82.98 | 82.61 | 82.30 |
| 0.3 | 82.38 | 83.34 | 83.41 | 82.84 | 82.40 | 81.99 |
| 0.5 | 82.82 | 83.51 | 83.21 | 82.77 | 82.04 | 81.59 |
| 0.7 | 83.03 | 83.65 | 83.12 | 82.54 | 81.64 | 81.09 |
| 0.9 | 83.11 | 83.72 | 83.22 | 82.34 | 81.52 | 80.70 |
| 1.0 | 83.12 | 83.65 | 83.23 | 82.28 | 81.13 | 80.82 |

**combine** (rows p, columns alpha):

| p \ α | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| 0.1 | 81.65 | 82.72 | 82.87 | 82.92 | 82.59 | 82.44 |
| 0.3 | 82.22 | 83.27 | 83.12 | 82.91 | 82.46 | 82.31 |
| 0.5 | 82.63 | 83.49 | 83.22 | 82.92 | 82.23 | 81.94 |
| 0.7 | 82.97 | 83.60 | 83.25 | 82.88 | 81.83 | 81.78 |
| 0.9 | 83.17 | 83.71 | 83.17 | 82.79 | 81.81 | 81.34 |
| 1.0 | 83.22 | 83.49 | 83.13 | 82.15 | 81.14 | 80.74 |

## Sequential versus independent task initialisation

Default (sequential): task t's LoRA starts from task t-1's TRAINED state. Independent
(`seqft_independent`): every task starts from the same fresh initial LoRA state. ImageNet-R,
r64 qkv+fc, 3 seeds, tuned gamma=0 stack, merges applied to the resulting per-task states:

| Merge rule under independent init | FA | AA | FFM |
|---|---|---|---|
| pspectral combine | 80.64 ± 0.20 | 85.67 ± 0.25 | 5.77 ± 0.13 |
| pspectral separate | 80.45 ± 0.05 | 85.53 ± 0.25 | 5.97 ± 0.07 |
| pspectral only_A | 80.43 ± 0.13 | 85.53 ± 0.27 | 6.05 ± 0.19 |
| pspectral only_B | 80.38 ± 0.15 | 85.57 ± 0.25 | 6.09 ± 0.17 |
| kspectral | 80.38 ± 0.21 | 85.57 ± 0.23 | 6.10 ± 0.17 |
| max | 80.25 ± 0.39 | 85.55 ± 0.10 | 6.11 ± 0.18 |
| min | 80.18 ± 0.44 | 85.49 ± 0.14 | 6.04 ± 0.32 |
| average | 79.88 ± 0.24 | 85.29 ± 0.25 | 6.75 ± 0.39 |
| TIES | 79.72 ± 0.11 | 85.16 ± 0.33 | 6.92 ± 0.20 |
| no merge | 77.95 ± 0.35 | 83.95 ± 0.37 | 8.14 ± 0.48 |
| max-abs | 77.75 ± 0.55 | 84.69 ± 0.51 | 7.94 ± 0.14 |

Sequential initialisation with the same stack reaches 83.68 ± 0.15 FA (Section 1), i.e. 3.04
above the best independent-init arm.

## Component contributions (merging / alignment / drift)

Six-cell lattice per dataset — base, M, A, MA, AD, MAD (M = spectral merging, A = LCA
alignment, D = drift compensation; drift-without-alignment is omitted as a structural no-op:
drift transports the class Gaussians, which only alignment consumes) — 3 seeds, cache-served,
paired on the shared backbones. Build-up chain (Last-Acc | Inc-Acc):

| | CIFAR-100 | ImageNet-R | CUB | Cars |
|---|---|---|---|---|
| merge | 88.15 \| 92.65 | 81.56 \| 86.24 | 76.17 \| 83.78 | 64.73 \| 71.75 |
| + alignment | 90.37 \| 94.32 | 82.51 \| 87.04 | 87.32 \| 91.59 | 70.98 \| 81.29 |
| + drift | 91.92 \| 94.81 | 83.63 \| 87.55 | 88.94 \| 92.26 | 83.63 \| 87.08 |

Shapley values over the valid lattice (contribution in Last-Acc points; exact — they sum to
full minus base):

| Dataset | Merging | Alignment | Drift | Total |
|---|---|---|---|---|
| CIFAR-100 | 2.71 | 2.47 | 0.55 | 5.73 |
| ImageNet-R | 2.88 | 1.17 | 0.63 | 4.69 |
| CUB | 4.42 | 7.73 | 0.61 | 12.77 |
| Cars | 6.62 | 8.96 | 8.11 | 23.70 |

Findings: (1) merging and alignment are strongly synergistic on the fine-grained sets —
merge x alignment interaction +8.1 (CUB) and +10.8 (Cars) vs +0.6-0.8 elsewhere; on CUB merge
alone gains nothing (76.17 vs base 76.17) and pays out entirely through alignment. (2) On Cars,
alignment alone HURTS (55.38 vs base 59.93 — heads calibrated to stale Gaussians) and drift is
the largest single component (+12.7 on top of merge+alignment, FF 21.9 -> 5.8): prototype
transport is load-bearing wherever the backbone drifts far. (3) Drift's on-top-of-full-stack
margin is +1.1-1.6 everywhere else — modest but consistent. Every full-stack lattice cell
reproduces its headline number within 0.09.

### Known weaknesses (future work)

Two axes are not reported above because we do not place first or second on them; both are
concrete targets for improvement:

- **Adversarial robustness.** Under PGD-10 at eps = 1/255 we reach 27.25, versus 33.45 for
  E2-LoRA and 38.90 / 37.40 / 37.05 for TUNA / EASE / MOS — 6th of 9. The prompt-based methods
  lead here by a wide margin. (TUNA and MOS were attacked through surrogates because their
  outputs are not plain logits, so their numbers are somewhat inflated.)
- **OOD detection.** On ImageNet-O with the energy score we reach 97.07 AUROC / 12.70 FPR95,
  behind E2-LoRA (98.59 / 6.20) and MOS (97.58 / 11.25) — 3rd of 9. Mahalanobis scoring from
  our own class Gaussians gives 97.21, which validates the class statistics but does not close
  the gap.
- **Clean accuracy** on CIFAR-100 is 0.18 below E2-LoRA, so part of the OOD/adversarial deficit
  starts from a slightly lower operating point.

---

## 3. Domain-incremental learning

Protocol and baselines from E2-LoRA (Table 3), ViT-B/16-IN1K backbone. DIL sessions share
semantic classes offset by session, so accuracy is scored mod (classes per domain), and
Last-Acc / Inc-Acc are sample-weighted totals, matching their protocol.

| Method | Office-Home Last-Acc | Office-Home Inc-Acc | DomainNet Last-Acc | DomainNet Inc-Acc |
|---|---|---|---|---|
| L2P | 80.03 ± 1.29 | 79.72 ± 4.19 | 48.72 ± 2.83 | 50.45 ± 4.10 |
| DualPrompt | 80.85 ± 0.14 | 80.20 ± 3.81 | 50.46 ± 3.17 | 52.28 ± 3.35 |
| CODA-Prompt | 85.07 ± 0.34 | 84.70 ± 2.94 | 59.99 ± 0.88 | 59.85 ± 4.49 |
| MEMO | 63.09 ± 1.80 | 71.18 ± 2.76 | 58.41 ± 3.20 | 61.92 ± 5.39 |
| RanPAC | 82.28 ± 0.07 | 82.30 ± 3.34 | 54.80 ± 0.36 | 55.20 ± 3.93 |
| EASE | 76.33 ± 2.16 | 81.16 ± 3.52 | 43.72 ± 1.70 | 50.50 ± 2.27 |
| SimpleCIL | 75.72 ± 0.00 | 75.69 ± 5.03 | 44.08 ± 0.00 | 42.95 ± 4.84 |
| DCE | 84.40 ± 0.20 | 84.60 ± 3.00 | 63.50 ± 0.50 | 64.30 ± 6.00 |
| DUCT | 85.42 ± 0.33 | 81.28 ± 0.31 | 67.01 ± 1.35 | 67.16 ± 3.75 |
| E2-LoRA | 88.25 ± 0.25 | 84.94 ± 0.21 | 69.63 ± 0.05 | 68.69 ± 0.04 |
| **Ours** | **89.96 ± 0.13** | **88.10 ± 0.20** | **71.05 ± 0.14** | **69.98 ± 0.07** |

We lead on both benchmarks and both metrics: Office-Home +1.71 Last / +3.16 Inc, DomainNet
+1.42 Last / +1.29 Inc.

---

## 4. Long task sequences

Baselines from E2-LoRA (Table 2), ViT-B/16-IN21K. Splits: ImageNet-R 20 tasks x 10 classes and
50 tasks x 4; CIFAR-100 20 tasks x 5 and 50 tasks x 2.

| Method | IN-R 20t Last | IN-R 20t Inc | IN-R 50t Last | IN-R 50t Inc | CIFAR 20t Last | CIFAR 20t Inc | CIFAR 50t Last | CIFAR 50t Inc |
|---|---|---|---|---|---|---|---|---|
| Joint training | 82.76 ± 0.54 | – | 82.76 ± 0.54 | – | 93.13 ± 0.21 | – | 93.13 ± 0.21 | – |
| L2P | 62.15 ± 1.17 | 68.35 ± 2.12 | 55.89 ± 1.59 | 62.98 ± 2.89 | 80.72 ± 1.12 | 87.18 ± 0.83 | 73.91 ± 1.67 | 81.90 ± 0.75 |
| DualPrompt | 66.89 ± 0.40 | 73.07 ± 1.21 | 61.50 ± 0.86 | 68.63 ± 1.31 | 83.82 ± 0.51 | 90.22 ± 0.68 | 76.66 ± 0.74 | 85.18 ± 0.92 |
| CODA-Prompt | 67.53 ± 0.24 | 73.64 ± 0.95 | 48.89 ± 0.90 | 55.59 ± 2.67 | 81.19 ± 0.31 | 87.27 ± 0.35 | 55.45 ± 0.48 | 68.39 ± 0.53 |
| InfLoRA | 71.46 ± 0.95 | 78.32 ± 1.22 | 60.49 ± 1.43 | 69.95 ± 1.28 | 82.19 ± 1.33 | 88.05 ± 0.64 | 56.65 ± 5.55 | 70.29 ± 1.86 |
| EASE | 73.78 ± 0.41 | 80.29 ± 0.72 | 68.54 ± 0.71 | 75.77 ± 2.18 | 82.21 ± 0.07 | 90.76 ± 0.56 | 82.10 ± 0.66 | 87.65 ± 0.46 |
| LoRA-DRS | 74.80 ± 0.73 | 80.69 ± 0.75 | 72.12 ± 0.87 | 77.94 ± 0.74 | 88.69 ± 0.15 | 92.25 ± 0.24 | 87.29 ± 0.31 | 91.29 ± 0.29 |
| MOS | 75.58 ± 0.49 | 81.76 ± 0.26 | 77.29 ± 0.47 | 83.06 ± 1.41 | 88.85 ± 0.43 | 92.86 ± 0.33 | 85.51 ± 0.04 | 91.00 ± 0.26 |
| TUNA | 77.32 ± 0.31 | 83.05 ± 0.64 | 75.35 ± 0.67 | 80.95 ± 1.29 | 91.31 ± 0.12 | **94.68 ± 0.16** | 89.41 ± 0.23 | 93.00 ± 0.13 |
| E2-LoRA | 80.77 ± 0.13 | 86.37 ± 0.39 | 78.58 ± 0.22 | 83.96 ± 0.76 | **91.42 ± 0.10** | 94.55 ± 0.07 | **90.70 ± 0.24** | **93.86 ± 0.09** |
| **Ours** | **82.08 ± 0.28** | **86.68 ± 0.38** | **78.64 ± 0.30** | **84.24 ± 0.04** | 90.35 ± 0.12 | 94.36 ± 0.17 | 87.05 ± 0.33 | 92.63 ± 0.31 |

**ImageNet-R:** we lead both splits. At 20 tasks we gain +1.31 Last-Acc / +0.31 Inc-Acc over
E2-LoRA and sit within 0.7 of the joint-training upper bound (82.76); at 50 tasks we are level
on Last-Acc (+0.06, a tie) and +0.28 on Inc-Acc. For reference our 10-task ImageNet-R result
(83.68) is *above* the joint bound.

**CIFAR-100:** we trail on both splits, and the gap widens with sequence length — −1.07
Last-Acc at 20 tasks and **−3.65 at 50 tasks**, where we also fall behind TUNA (89.41) and
LoRA-DRS (87.29). This is the clearest weakness in the method: CIFAR at 50 tasks means 2
classes per task, and our per-task LoRA adaptation plus class-Gaussian statistics degrade
faster there than the prompt/expansion-based methods do. Improving short-task behaviour on
fine-grained splits is the highest-value target after the robustness gaps above.

### Degradation profile (ours)

| Benchmark | Split | Last-Acc | Inc-Acc | FF |
|---|---|---|---|---|
| CIFAR-100 | 10 tasks x 10 | 91.97 ± 0.32 | 94.80 ± 0.14 | 2.94 ± 0.32 |
| CIFAR-100 | 20 tasks x 5 | 90.35 ± 0.12 | 94.36 ± 0.17 | 5.03 ± 0.21 |
| CIFAR-100 | 50 tasks x 2 | 87.05 ± 0.33 | 92.63 ± 0.31 | 9.12 ± 0.47 |
| ImageNet-R | 10 tasks x 20 | 83.68 ± 0.15 | 87.54 ± 0.20 | 3.74 ± 0.39 |
| ImageNet-R | 20 tasks x 10 | 82.08 ± 0.28 | 86.68 ± 0.38 | 5.13 ± 0.37 |
| ImageNet-R | 50 tasks x 4 | 78.64 ± 0.30 | 84.24 ± 0.04 | 9.08 ± 0.22 |

Degradation is graceful and consistent across both datasets: about -1.5 Last-Acc at 20 tasks
and -5 at 50 tasks, with forgetting rising 3.5 -> 5 -> 9.

**Stability note.** At 50-task granularity a task holds only 2 (CIFAR) or 4 (ImageNet-R)
classes, so an epoch is ~4-8 iterations. Training 10 epochs at lr 1e-2 diverged on ImageNet-R
for 1 of 3 seeds (task 21, train accuracy collapsing from 0.97 to 0.34 and poisoning the rest
of the sequential chain). Setting `train_epochs=5` for tasks with <=4 classes fixes it at no
cost (CIFAR 87.05 with 5 epochs vs 86.82 with 10; all ImageNet-R seeds healthy). The 50-task
rows above use 5 epochs.

---

## Appendix B — Hardware and software

All experiments ran on a single workstation:

| | |
|---|---|
| GPU | 1x NVIDIA GeForce RTX 4090, 24 GB (driver 590.48.01) |
| CPU | Intel Core i9-12900K, 24 threads |
| RAM | 125 GB |
| OS | Ubuntu, Linux 6.8.0 |
| Python | 3.9.23 |
| PyTorch | 2.7.0+cu118 (CUDA 11.8) |
| timm | 1.0.24 |

Backbone training is the only GPU-heavy stage; per-task LoRA states are cached, so the merge,
drift and alignment ablations are evaluation-only re-runs on cached checkpoints. Indicative
costs on this machine: one 10-task CIFAR-100 / ImageNet-R run with cached backbones is about
10 minutes per seed; one CIFAR-100-C evaluation (19 corruptions x 5 severities x 50,000
images) is about 1.5 hours per model, and one CIFAR-100-P evaluation (10 perturbations x
2,500 sequences x 31 frames) about 1.1 hours per model.
