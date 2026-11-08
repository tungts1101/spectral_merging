"""main: unified backbone-training recipe comparison.

Compares the current exp19 recipe ("seqft": PEFT LoRA r=64/qkv, init each task
from the previous TRAINED LoRA, spectral-family merge for feature extraction)
against the recipes ported from https://github.com/raoxuan98-hash/lr_rgda_hopdc
(backbones.py): basic_lora, sgp_lora, nsp_lora, full, full_nsp,
joint_lora/joint_full, first_task_lora — with Stage 2 (class Gaussians +
shrinkage -> identity-ridge drift transport -> NxK LCA alignment) held
IDENTICAL across recipes (ported verbatim from exp19).

Merge axis (seqft only): train_merge in {ties, max, min, max_abs, avg,
kspectral, pspectral}. kspectral = fixed top-k singular values, pspectral =
energy percentile; both support spectral_variant in {only_A, only_B, separate,
combine}. Merge-family recipes fold weights into the backbone instead.

Feature-KD (gamma_kd > 0) from the previous-task feature extractor is
composable with every recipe.

Usage:
    python main.py --smoke              # tiny 2-task run of every recipe
    python main.py --smoke-inflora      # the same plus the InfLoRA port (E15b)
    python main.py                      # controlled comparison sweep (default)
"""

from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import math
import re
from utils.data_manager import DataManager, DummyDataset
import gc
import time
from helper import (
    compute_metrics,
    accuracy,
    set_random,
    merge,
    pspectral_merging,
    truncated_svd,
    count_parameters,
    seed_worker,
    get_backbone,
    ContinualLinear,
)
from merge_ops import lora_scale, step_norms, alpha_from_stat
from driftsolve import (
    GCV_GRID,
    gcv_lambda,
    ridge_identity_woodbury,
    ridge_zero_woodbury,
    solve_drift,
)
from covstats import (
    COV_MODES,
    JITTER as COV_JITTER,
    fit_full,
    fit_diag,
    fit_lowrank,
    scatter as cov_scatter,
    shared_cov,
    shared_scale_tril,
    transport_diag,
    transport_lowrank_U,
    transport_full,
    to_full_cov,
    make_dist,
    storage_floats,
)
from diagnostics import gaussian_errors, transport_gaussian, csv_append
from backbones import (
    MERGE_FAMILY,
    build_recipe_backbone,
    build_projection,
    compute_covariances,
    peft_qkv_modules,
    peft_lora_A_params,
    Distiller,
    symmetric_cross_entropy_loss,
    trainable_state,
    load_trainable_state,
)
import inflora_gpm   # InfLoRA port (E15b): threshold schedule, lr factor (NumPy only)
from torch.distributions import MultivariateNormal
import logging
import sys
import copy


CHECKPOINT_ROOT = "checkpoints_main"
LOG_DIR = "logs_main"
os.makedirs(LOG_DIR, exist_ok=True)

g = torch.Generator()
g.manual_seed(0)

# Rebuttal instrumentation (every key defaults to off): CSV schemas of the
# stage timers (E3, config stage_timing_csv) and of the drift-map diagnostics
# (E9, config drift_diag_csv). Rows are appended by diagnostics.csv_append.
TIMING_CSV_HEADER = ("dataset", "seed", "task", "stage", "seconds", "extra")
DRIFTDIAG_CSV_HEADER = (
    "dataset", "seed", "task", "group", "n_classes", "n_img",
    "mean_rel_raw", "mean_rel_before", "mean_rel_after",
    "mean_cos_raw", "mean_cos_before", "mean_cos_after",
    "cov_relF_raw", "cov_relF_before", "cov_relF_after",
    "symkl_raw", "symkl_before", "symkl_after",
    "os_mean_rel_I", "os_mean_rel_W", "os_mean_rel_oracle",
    "os_cov_relF_I", "os_cov_relF_W", "os_cov_relF_oracle",
    "W_minus_I_F", "Wor_minus_I_F", "W_minus_Wor_F", "n_fit_new", "n_fit_old",
)
# Rebuttal data hook (E13 few-shot per task, E14 validation carve): the config
# keys run_single_experiment forwards to DataManager. None of them lives in
# BASE_CONFIG, so a key is present in a run's config only when its experiment
# set it; the '[Data]' line is logged exactly then.
DATA_HOOK_KEYS = ("data_val_frac", "data_val_seed", "data_fewshot_k", "data_subsample_seed")


# ------------------------------------------------------------------
# kspectral merge: fixed top-k analogue of helper.pspectral_merging
# ------------------------------------------------------------------

def kspectral_merging(w_prev, w_new, k=16, alpha=1.0, variant="separate"):
    """Top-k spectral merge for LoRA states (4 variants, mirroring
    helper.pspectral_merging but truncating to a FIXED rank k instead of an
    energy percentile).

      only_A   : truncate the lora_A update; lora_B merged by interpolation
      only_B   : truncate the lora_B update; lora_A merged plain
      separate : truncate both factor updates (== old helper.spectral_merging)
      combine  : truncate the effective update W = B@A, re-factor to rank r
                 (== old helper.spectral_merging_lora)
    Non-LoRA params (norms/biases) are merged by plain interpolation.
    """
    FACTOR_SUFFIXES = [
        ("lora_A.default.weight", "lora_B.default.weight"),
        ("down_proj.weight", "up_proj.weight"),
    ]
    pairs = {}
    for name in w_new:
        for a_suf, b_suf in FACTOR_SUFFIXES:
            if name.endswith(a_suf):
                pairs.setdefault(name[: -len(a_suf)], {})["A"] = name
            elif name.endswith(b_suf):
                pairs.setdefault(name[: -len(b_suf)], {})["B"] = name

    merged, lora_names = {}, set()
    for _, d in pairs.items():
        if "A" not in d or "B" not in d:
            continue
        An, Bn = d["A"], d["B"]
        lora_names.update((An, Bn))
        A_prev, B_prev = w_prev[An].float(), w_prev[Bn].float()
        A_new, B_new = w_new[An].float(), w_new[Bn].float()
        dA, dB = A_new - A_prev, B_new - B_prev

        if variant == "only_A":
            merged[An] = (A_prev + alpha * truncated_svd(dA, k)).to(w_new[An].dtype)
            merged[Bn] = (B_prev + alpha * dB).to(w_new[Bn].dtype)
        elif variant == "only_B":
            merged[An] = (A_prev + alpha * dA).to(w_new[An].dtype)
            merged[Bn] = (B_prev + alpha * truncated_svd(dB, k)).to(w_new[Bn].dtype)
        elif variant == "separate":
            merged[An] = (A_prev + alpha * truncated_svd(dA, k)).to(w_new[An].dtype)
            merged[Bn] = (B_prev + alpha * truncated_svd(dB, k)).to(w_new[Bn].dtype)
        elif variant == "combine":
            r = A_new.shape[0]
            Wp, Wn = B_prev @ A_prev, B_new @ A_new
            C = Wp + alpha * truncated_svd(Wn - Wp, k)
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
            rr = min(r, S.shape[0])
            s = S[:rr].clamp(min=0).sqrt()
            merged[Bn] = (U[:, :rr] * s).to(w_new[Bn].dtype)
            merged[An] = (s.unsqueeze(1) * Vh[:rr, :]).to(w_new[An].dtype)
        else:
            raise ValueError(f"unknown kspectral variant {variant!r}")

    for name in w_new:
        if name in lora_names:
            continue
        w_t = w_prev[name]
        merged[name] = w_t + alpha * (w_new[name] - w_t)
    return merged


def generic_incremental_merge(w_prev, w_new, method="pspectral", k=16, p=0.9, alpha=1.0):
    """Incremental spectral merge for FULL weight-state snapshots (merge-family
    recipes fold their adapters, so no LoRA factors survive -- truncate the
    update of every 2D float tensor directly instead).

      C = W_prev + alpha * truncate(W_new - W_prev)
    truncate = energy-percentile p ("pspectral") or top-k ("kspectral").
    1D float tensors (biases, norms, DoRA magnitudes): plain interpolation.
    Non-float tensors (e.g. lora_active flags): taken from w_new.
    """
    from helper import truncate_energy
    merged = {}
    for name in w_new:
        v_new = w_new[name]
        if not torch.is_floating_point(v_new):
            merged[name] = v_new.clone()
            continue
        v_prev = w_prev[name].float()
        d = v_new.float() - v_prev
        if d.dim() == 2:
            if method == "pspectral":
                d = truncate_energy(d, p)
            elif method == "kspectral":
                d = truncated_svd(d, k)
            else:
                raise ValueError(f"unknown state-merge method {method!r}")
        merged[name] = (v_prev + alpha * d).to(v_new.dtype)
    return merged


# ------------------------------------------------------------------
# Model: helper.Model with an injected backbone
# ------------------------------------------------------------------

class Model21(nn.Module):
    def __init__(self, config, backbone):
        super().__init__()
        self._config = config
        self.backbone = backbone
        self.norm = (
            nn.LayerNorm(self.feature_dim)
            if config.get("model_use_norm", False) else None
        )
        self.classifier = None

    @property
    def feature_dim(self):
        dim = getattr(self.backbone, "num_features", None)
        if dim is None:
            dim = getattr(self.backbone, "feature_dim", 768)
        return dim

    def update_classifier(self, num_classes, with_norm=False, with_bias=False,
                          freeze_old=True, norm_layer=None):
        if self.classifier is None:
            self.classifier = ContinualLinear(
                self.feature_dim, num_classes,
                with_norm=with_norm, with_bias=with_bias, norm_layer=norm_layer,
            )
        else:
            self.classifier.update(num_classes, freeze_old=freeze_old)

    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params[name] = param
        if self.norm is not None:
            for name, param in self.norm.named_parameters():
                if param.requires_grad:
                    params[f"norm.{name}"] = param
        return params

    def get_features(self, x):
        z = self.backbone(x)
        feature_norm = self._config.get("model_feature_norm", "ln")
        if feature_norm == "l2":
            # Hyperspherical feature normalization (SimbaV2, Lee et al. 2025):
            # project features onto the sqrt(d)-radius sphere. sqrt(d) keeps
            # per-dim magnitudes ~1 (LayerNorm convention) so covariance
            # jitter and LCA hyperparameters stay calibrated.
            z = z * (z.shape[-1] ** 0.5) / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        elif self.norm is not None:
            z = self.norm(z)
        return z

    def forward(self, x):
        return self.classifier(self.get_features(x))

    def __repr__(self):
        trainable = count_parameters(self, trainable=True)
        total = count_parameters(self)
        return (f"Model21(trainable_params={trainable:,}, total_params={total:,}, "
                f"percentage={trainable * 100 / total:.2f})")


# ------------------------------------------------------------------
# Learner
# ------------------------------------------------------------------

# Hybrids: seqft training + spectral merge, with the SGP/NSP projection applied
# to the lora_A GRADIENT (grad <- grad @ P, the GPM formulation). Update
# subspace matches their forward-side A@P; everything else (init, scale,
# states, merge) stays byte-identical to seqft, so the projection is the only
# difference vs. the seqft row.
HYBRID_METHODS = ("sgp_seqft", "nsp_seqft")


def merge_tag(cfg):
    """Tag that names a config's merged checkpoints ({prefix}_merged_{tag}_{t}.pt).

    The one rule shared by Learner.merged_checkpoint and the cleanup block of
    run_config_sweep. Existing configs keep their byte-identical legacy strings:
        pspectral -> pspec_{variant}_p{p}_a{alpha}
        kspectral -> kspec_{variant}_k{k}_a{alpha}
        otherwise -> {method}_c{coef}
    Rebuttal keys extend the pspectral tag only when they are set: an '{op}_'
    token BEFORE the alpha token for merge_update_op != 'trunc'
    (pspec_only_A_p0.9_clip_a0.3), and 'a{rule}{param}' in place of 'a{alpha}'
    for merge_alpha_rule != 'fixed' (train_merge_alpha is never mutated, so an
    adaptive chain resolves its own merged files). The op token sits before the
    alpha token so that the cleanup glob of the plain cell, '*_merged_{tag}_*.pt'
    with tag 'pspec_only_A_p0.9_a0.3', can never match the files of a clip or
    scale_fro cell at the same alpha in a shared directory. The OPCM / NUFILT
    rules of merging_sota (E15a) are named by their own parameters:
        opcm_a{alpha}_{lam}  and  nufilt_rp{rp}_rv{rv}_rl{rl}_ad{0|1}.
    Missing merge keys fall back to BASE_CONFIG (the cleanup block passes the
    experiment overrides merged into a BASE_CONFIG copy).
    """
    def get(key):
        return cfg[key] if key in cfg else BASE_CONFIG[key]

    method = get("train_merge")
    variant = cfg.get("spectral_variant", cfg.get("pspectral_variant", "only_A"))
    if method == "pspectral":
        rule = cfg.get("merge_alpha_rule", "fixed")
        if rule == "fixed":
            alpha_part = f"a{get('train_merge_alpha')}"
        else:
            alpha_part = f"a{rule}{cfg.get('merge_alpha_param', 1.0)}"
        op = cfg.get("merge_update_op", "trunc")
        op_part = "" if op == "trunc" else f"{op}_"
        return f"pspec_{variant}_p{get('pspectral_p')}_{op_part}{alpha_part}"
    if method == "kspectral":
        return f"kspec_{variant}_k{get('train_merge_k')}_a{get('train_merge_alpha')}"
    if method == "opcm":
        return f"opcm_a{cfg.get('merge_opcm_alpha', 0.5)}_{cfg.get('merge_opcm_lam', 'adaptive')}"
    if method == "nufilt":
        return (f"nufilt_rp{cfg.get('merge_nufilt_rp', 128)}_rv{cfg.get('merge_nufilt_rv', 8)}"
                f"_rl{cfg.get('merge_nufilt_rl', 64)}"
                f"_ad{int(bool(cfg.get('merge_nufilt_adapt', False)))}")
    return f"{method}_c{get('train_merge_coef')}"


class Learner:
    def __init__(self, config):
        self._config = config
        self._method = config.get("train_method", "seqft")
        assert self._method == "seqft" or self._method in HYBRID_METHODS \
            or self._method in MERGE_FAMILY, f"Unknown train_method {self._method!r}"
        # peft family: seqft lifecycle (LoRA states + spectral merge)
        self._is_seqft = self._method == "seqft" or self._method in HYBRID_METHODS
        # E2-LoRA: growing per-task pair lists, custom lifecycle
        self._is_e2 = self._method == "e2lora"
        # InfLoRA (rebuttal E15b): merge-family path (fold, state cache) plus
        # the two Gram hooks _inflora_before_train / _inflora_after_train.
        self._is_inflora = self._method == "inflora"
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self._mlp_matrix = []
        self._cls_to_task_idx = {}
        self._acc = 0.0
        self._acc_history = []
        # Rebuttal E14: per-task grouped accuracies on the validation carve
        # ([Evaluation-val] rows); stays empty without data_val_frac > 0.
        self._mlp_matrix_val = []
        # Rebuttal E15a: cross-task state of a stateful SOTA merge rule
        # (merging_sota OPCM keeps lam_t and the task-vector norms here; NUFILT
        # writes its per-step statistics). One dict per Learner, i.e. per seed.
        self._merge_rule_state = {}

        backbone = get_backbone(config) if self._is_seqft else build_recipe_backbone(config)
        self.model = Model21(config, backbone)
        self.model.cuda()
        if self._is_seqft:
            # Merge base (-1): pretrained-init LoRA state (exp19 behavior).
            torch.save(self.model.get_backbone_trainable_params(), self.backbone_checkpoint())

        self._covariances = None   # EMA input covariances (sgp/nsp/full_nsp/hybrids)
        self._prev_model = None    # post-task-(t-1) snapshot (merge family)
        # merge_states: apply the user's incremental spectral merge to the
        # merge-family per-task weight snapshots; the merged state M_t becomes
        # the feature extractor while task t+1 trains from the unmerged s_t.
        self._merge_states = (not self._is_seqft) and config.get("merge_states", False)
        self._merged_state = None
        if self._merge_states:
            assert self._cache_enabled(), "merge_states requires cache_backbone=True"

        # Hybrid gradient projection: P per qkv block, applied via grad hooks
        # on the PEFT lora_A weights. Empty dict (task 0) -> hooks are no-ops.
        # load_state_dict copies into the same Parameter objects, so the hooks
        # survive checkpoint loads across tasks.
        self._proj_P = {}
        if self._method in HYBRID_METHODS:
            def make_hook(name):
                def hook(grad):
                    P = self._proj_P.get(name)
                    if P is None:
                        return grad
                    with torch.no_grad():
                        return grad @ P.to(device=grad.device, dtype=grad.dtype)
                return hook

            for name, p in peft_lora_A_params(self.model.backbone).items():
                p.register_hook(make_hook(name))

        gamma_kd = config.get("gamma_kd", 0.0)
        self.distiller = None
        if gamma_kd > 0:
            self.distiller = Distiller(
                kd_type=config.get("kd_type", "feat"),
                gamma_kd=gamma_kd,
                feat_dim=self.model.feature_dim,
                transform=config.get("kd_transform", "weaknonlinear"),
            )

    # ------------------------------------------------------------------
    # seqft LoRA freeze helper (exp19)
    # ------------------------------------------------------------------

    def _freeze_non_lora(self):
        peft_key = self._config.get("peft_key", "lora_")
        n_frozen = n_trainable = 0
        for name, p in self.model.backbone.named_parameters():
            if peft_key in name:
                p.requires_grad = True
                n_trainable += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        logging.info(
            f"[Freeze] Non-PEFT frozen ({n_frozen:,} params), "
            f"PEFT '{peft_key}' trainable ({n_trainable:,} params)"
        )

    # ------------------------------------------------------------------
    # Continual learning lifecycle
    # ------------------------------------------------------------------

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks
        self.model.cuda()

        stop_at_task = self._config.get("train_stop_at_task", None)
        if stop_at_task == -1:
            stop_at_task = None

        es_ref = self._config.get("early_stop_ref", None)
        es_margin = self._config.get("early_stop_margin", 0.5)
        es_from = self._config.get("early_stop_from_task", 2)
        self._pruned_at = None

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()
            t0 = self._tic()                     # stage_timing (E3); None when off
            self.eval()
            self._toc(t0, "eval", classes=self._total_classes)
            self.after_task()
            # Early pruning: abandon runs that fall > margin below the best
            # run's per-task trajectory (search-time cutoff; from task es_from
            # to avoid killing trials on early-task noise).
            if (es_ref is not None and task >= es_from and task < len(es_ref)
                    and self._last_total_acc < es_ref[task] - es_margin):
                logging.info(
                    f"[EarlyStop] Pruned at task {task}: acc "
                    f"{self._last_total_acc:.2f} < ref {es_ref[task]:.2f} - {es_margin}")
                self._pruned_at = task
                break
            if stop_at_task is not None and task >= stop_at_task:
                break

    def before_task(self, task, data_manager):
        task_size = data_manager.get_task_size(task)
        self._total_classes = self._known_classes + task_size
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

        for clz in range(self._known_classes, self._total_classes):
            self._cls_to_task_idx[clz] = self._cur_task

        if self._is_e2:
            # E2-LoRA lifecycle (their trainer): add_task (prune old pairs,
            # allocate the new one) BEFORE training. If merged-A is deployed
            # from the previous task, snapshot it first (drift's f_{t-1}) and
            # restore raw pairs so pruning/training see their own trajectory.
            if task > 0 and self._config.get("train_drift", False):
                self._snapshot_prev()
            self.model.backbone.restore_raw_A()
            self.model.backbone.add_task()
            n_tr = sum(p.numel() for p in self.model.backbone.get_param_groups())
            logging.info(f"[E2] add_task -> task {task}, trainable {n_tr:,}")
            return

        if not self._is_seqft:
            # Merge family: weights live in self.model across tasks; the
            # adapter wrapper already configured requires_grad.
            return

        # seqft task-init rule. Default (sequential): init from the previous
        # TRAINED LoRA. seqft_independent: every task starts from the SAME
        # fresh initial LoRA state (saved as the base checkpoint at __init__,
        # never overwritten in this mode) -- separate per-task training.
        model_use_norm = self._config.get("model_use_norm", False)
        if self._config.get("seqft_independent", False):
            if task > 0:
                logging.info(f"[Before Task {task}] Independent mode: reloading initial LoRA state")
                self.load_backbone(torch.load(self.backbone_checkpoint()), load_norm=model_use_norm)
            self._freeze_non_lora()
            return

        if task == 0:
            self._freeze_non_lora()
        else:
            if task == 1:
                task0_ckpt = self.backbone_checkpoint(0)
                if os.path.exists(task0_ckpt):
                    logging.info("[Before Task 1] Loading task-0 trained checkpoint as backbone foundation")
                    self.load_backbone(torch.load(task0_ckpt), load_norm=model_use_norm)
            else:
                prev_ckpt = self.backbone_checkpoint(task - 1)
                if os.path.exists(prev_ckpt):
                    logging.info(f"[Before Task {task}] Loading task {task - 1} trained checkpoint")
                    self.load_backbone(torch.load(prev_ckpt), load_norm=model_use_norm)
            self._freeze_non_lora()

    def after_task(self):
        self._known_classes = self._total_classes

    def train(self):
        class_range = np.arange(self._known_classes, self._total_classes)
        train_set = self.data_manager.get_dataset(class_range, source="train", mode="train")
        train_loader = DataLoader(
            train_set,
            batch_size=self._config["train_batch_size"],
            shuffle=True, num_workers=4,
            worker_init_fn=seed_worker, generator=g,
        )

        # 1. Snapshot the previous feature extractor (merge family): the model
        #    currently holds the state after task t-1; drift pairs and the KD
        #    teacher both need it after this task's training mutates it.
        #    (e2lora snapshots in before_task, while merged-A is still deployed.)
        if (not self._is_seqft and not self._is_e2 and not self._is_inflora
                and self._cur_task > 0):
            # InfLoRA has no drift map and no teacher: skip the deep copy of the
            # model (incl. the 12 x 768^2 DualGPM bases) that nothing consumes.
            self._snapshot_prev()

        # 2. KD teacher = previous-task feature extractor.
        if self.distiller is not None and self._cur_task > 0:
            self.distiller.update_teacher(self._get_prev_feature_model())

        # merge_states: the model currently holds the merged extractor M_{t-1}
        # (snapshotted above for drift/KD); training must continue from the
        # UNMERGED post-task state s_{t-1} of their lifecycle.
        if self._merge_states and self._cur_task > 0:
            load_trainable_state(
                self.model.backbone,
                torch.load(self.state_checkpoint(self._cur_task - 1)),
            )

        # 3+4. Train backbone + head, then finalize the feature extractor.
        t0 = self._tic()                         # stage_timing (E3); None when off
        trained = self.train_backbone(train_loader)
        self._toc(t0, "train", cached=int(not trained))

        if self._is_e2:
            if trained:
                # Energy-structured transformation on one proxy batch
                # (their _pca_lora), then cache the raw structural state.
                x_proxy = next(iter(train_loader))[-2].cuda()
                self.model.backbone.energy_transform(x_proxy)
                logging.info("[E2] Energy transform applied (one proxy batch)")
                if self._cache_enabled():
                    torch.save(self.model.backbone.e2_state(),
                               self.e2_checkpoint(self._cur_task))
            if self._config.get("e2_merge", False):
                # Deploy A-merged extractor: A_k <- alpha*trunc(A_k, p) for all
                # pairs (B orthonormal => damps each task's effective update).
                self.model.backbone.apply_merged_A(
                    self._config["pspectral_p"], self._config["train_merge_alpha"])
                logging.info(
                    f"[E2 Merge] merged-A deployed "
                    f"(p={self._config['pspectral_p']}, "
                    f"alpha={self._config['train_merge_alpha']})")
        elif self._is_seqft:
            if self._cur_task == 0:
                # exp19: overwrite the merge base with the task-0 LoRA state so
                # task vectors are relative to the trained foundation. In
                # independent mode the base must STAY the fresh init (it is
                # reloaded before every task), so only the task-0 ckpt is saved.
                self._freeze_non_lora()
                lora_params = self.model.get_backbone_trainable_params()
                if not self._config.get("seqft_independent", False):
                    torch.save(lora_params, self.backbone_checkpoint())   # base (-1)
                torch.save(lora_params, self.backbone_checkpoint(0))  # task-0
                logging.info("[Task 0] Base and task-0 checkpoints updated with LoRA-only state")
                if self._config.get("infer_speccap"):
                    self._apply_inference_cap()
            else:
                if self._config.get("train_merge", "none") != "none":
                    # Rebuttal E7: an adaptive damping alpha replaces the fixed
                    # train_merge_alpha for this task only (config untouched).
                    alpha_override = None
                    if self._config.get("merge_alpha_rule", "fixed") != "fixed":
                        alpha_override = self._adaptive_alpha(class_range)
                    elif self._config.get("merge_log_featshift", False):
                        self._feature_shift(class_range)   # '[FeatShift]' line only
                    t0 = self._tic()
                    self.merge(alpha_override=alpha_override)
                    self._toc(t0, "merge", rule=self._config.get("train_merge"))
                elif self._config.get("merge_log_stats", False):
                    # Merge-less run: the deployed step is the raw update
                    # W_t - W_{t-1}; log its norms ('[StepNorm]', E10 L2M arm).
                    self._log_step_norm()
                if self._config.get("infer_speccap"):
                    self._apply_inference_cap()
        else:
            if trained:
                # Fold this task's adapter into the backbone (no-op for
                # full/full_nsp). Upstream folds at the START of the next task
                # (subspace_lora.py:357); no training happens in between, so
                # folding here is equivalent and makes the post-task state
                # well-defined for caching and Stage 2.
                self.model.backbone.merge_lora_weights()
                if self._is_inflora:
                    # InfLoRA hook 2: post-fold Gram pass + DualGPM memory
                    # update BEFORE the state save, so the cached state (and
                    # the cache-hit path) carries the updated basis.
                    self._inflora_after_train(train_loader)
                if self._cache_enabled():
                    dtype = {"fp16": torch.float16, "fp32": None}[self._config.get("cache_dtype", "fp32")]
                    torch.save(trainable_state(self.model.backbone, dtype=dtype),
                               self.state_checkpoint(self._cur_task))
            if self._merge_states:
                # M_t = incremental merge of (M_{t-1}, s_t); becomes the
                # feature extractor for Stage 2 + eval.
                s_t = trainable_state(self.model.backbone)
                if self._merged_state is None:
                    self._merged_state = s_t
                else:
                    self._merged_state = generic_incremental_merge(
                        self._merged_state, s_t,
                        method=self._config.get("train_merge", "pspectral"),
                        k=self._config.get("train_merge_k", 16),
                        p=self._config.get("pspectral_p", 0.9),
                        alpha=self._config.get("train_merge_alpha", 0.3),
                    )
                    load_trainable_state(self.model.backbone, self._merged_state)
                    logging.info(f"[StateMerge] Loaded merged extractor M_{self._cur_task}")

        # 5. Projection update (sgp/nsp/full_nsp/hybrids): EMA input covariances
        #    of the CURRENT task through the post-task backbone; P applies
        #    during the NEXT task's training (subspace_lora.py:729).
        #    skip_projection_update: for fully-cached Stage-2/eval-only runs --
        #    P is consumed only by training, so rebuilding it (and re-saving
        #    ~0.5 GB/task covariance caches) is pure waste. Never set it on a
        #    config that may actually train.
        if not self._config.get("skip_projection_update", False) and (
                (self._method in HYBRID_METHODS)
                or (not self._is_seqft and self.model.backbone.use_projection)):
            self.update_projections(class_range, trained)

        # 6. Stage 2 (identical for every recipe, ported from exp19).
        train_ca = self._config.get("train_ca", False)
        train_drift = self._config.get("train_drift", False)
        if train_ca and self._config.get("train_ca_impl", "lca") == "e2":
            # E2-LoRA's native classifier stage (their _compute_class_mean +
            # _stage2_compact_classifier) -- used by the reproduction config.
            self._e2_compute_class_stats()
            if self._cur_task > 0:
                task_size = self._total_classes - self._known_classes
                self._e2_native_ca(task_size)
            return
        # Rebuttal instrumentation, both default off: stage_timing (E3) times
        # every Stage-2 call; drift_diag (E9) measures the stored old-class
        # statistics before / after the transport and feeds nothing back.
        diag_on = self._config.get("drift_diag", False)
        if self._cur_task == 0:
            if train_ca or train_drift:
                t0 = self._tic()
                self.compute_multivariate_normal()
                self._toc(t0, "stats", classes=self._total_classes - self._known_classes)
                if diag_on:
                    self._diag_store_raw()
        else:
            if train_ca or train_drift:
                if train_drift:
                    t0 = self._tic()
                    W = self.fit_drift_map(train_loader)
                    self._toc(t0, "drift")
                    before = self._diag_snapshot() if diag_on else None
                    self.transport_prototypes(W)
                    if diag_on:
                        t0 = self._tic()
                        self.drift_diagnostics(W, before)
                        self._toc(t0, "driftdiag")
                t0 = self._tic()
                self.compute_multivariate_normal()
                self._toc(t0, "stats", classes=self._total_classes - self._known_classes)
                if diag_on:
                    self._diag_store_raw()
                t0 = self._tic()
                self.align(self.model.classifier)
                self._toc(t0, "align", classes=self._total_classes,
                          epochs=self._config.get("train_ca_epochs", 10))

    # ------------------------------------------------------------------
    # Merge-family plumbing
    # ------------------------------------------------------------------

    def _cache_enabled(self):
        cache = self._config.get("cache_backbone", None)
        if cache is None:
            # Full-weight states are ~230 MB/task -- default off for full FT.
            cache = self._method not in ("full", "full_nsp", "joint_full")
        return cache

    def _snapshot_prev(self):
        self._prev_model = copy.deepcopy(self.model).cpu()
        self._prev_model.backbone.finalize_without_lora()
        self._prev_model.eval()

    def _get_prev_feature_model(self):
        """Feature extractor of the previous task's post-task state, on GPU."""
        if self._is_seqft:
            model_use_norm = self._config.get("model_use_norm", False)
            if self._cur_task == 1 or self._config.get("train_merge", "none") == "none":
                # Task 1's prev state is the task-0 ckpt (no merged ckpt for
                # task 0); with no merge at all, the deployed extractor after
                # task t-1 IS the task-(t-1) trained state.
                prev_ckpt = self.backbone_checkpoint(self._cur_task - 1)
            else:
                prev_ckpt = self.merged_checkpoint(self._cur_task - 1)
            old_model = copy.deepcopy(self.model)
            self.load_backbone(torch.load(prev_ckpt), load_norm=model_use_norm, target=old_model)
            if self._config.get("infer_speccap"):
                # Drift must map capped space -> capped space: the previous
                # merged checkpoint is stored uncapped, so re-apply the
                # deployment cap to the reloaded extractor.
                self._cap_model(old_model)
            return old_model.cuda().eval()
        assert self._prev_model is not None, "prev snapshot missing"
        return self._prev_model.cuda().eval()

    def _release_prev_feature_model(self, old_model):
        if self._is_seqft:
            del old_model
        else:
            self._prev_model.cpu()
        torch.cuda.empty_cache()

    def update_projections(self, class_range, trained):
        covs_ckpt = self.covs_checkpoint(self._cur_task)
        hybrid = self._method in HYBRID_METHODS
        if not trained and os.path.exists(covs_ckpt):
            logging.info(f"[Projection] Loading cached covariance EMA for task {self._cur_task}")
            self._covariances = {k: v.float() for k, v in torch.load(covs_ckpt).items()}
        else:
            cov_set = self.data_manager.get_dataset(class_range, source="train", mode="test")
            cov_loader = DataLoader(
                cov_set, batch_size=128, shuffle=False,
                num_workers=4, worker_init_fn=seed_worker, generator=g,
            )
            modules = peft_qkv_modules(self.model.backbone) if hybrid else None
            new_covs = compute_covariances(self.model.backbone, cov_loader, modules=modules)
            new_covs = {k: v.cpu() for k, v in new_covs.items() if v is not None}
            decay = self._config.get("cov_ema_decay", 0.9)
            if self._covariances is None:
                self._covariances = new_covs
            else:
                for k in self._covariances:
                    self._covariances[k] = (
                        decay * self._covariances[k] + new_covs[k]
                        + 1e-7 * torch.eye(self._covariances[k].size(0))
                    )
            if self._cache_enabled():
                # Follows cache_dtype: fp32 keeps resumed runs identical to
                # fresh ones (P is rebuilt from these); fp16 halves disk.
                if self._config.get("cache_dtype", "fp32") == "fp16":
                    torch.save({k: v.half() for k, v in self._covariances.items()}, covs_ckpt)
                else:
                    torch.save(self._covariances, covs_ckpt)
        if hybrid:
            soft = self._method == "sgp_seqft"
            self._proj_P = {
                name: build_projection(
                    cov,
                    soft_projection=soft,
                    weight_temp=self._config.get("weight_temp", 2.0),
                    weight_kind=self._config.get("weight_kind", "log1p"),
                    weight_p=self._config.get("weight_p", 1.0),
                    nsp_eps=self._config.get("nsp_eps", 0.05),
                    nsp_weight=self._config.get("nsp_weight", 0.0),
                ).cuda()
                for name, cov in self._covariances.items()
            }
        else:
            covs_gpu = {k: v.cuda() for k, v in self._covariances.items()}
            self.model.backbone.update_projection_matrices(covs_gpu)
            del covs_gpu
            torch.cuda.empty_cache()
        logging.info(f"[Projection] Updated projection matrices after task {self._cur_task}")

    # ------------------------------------------------------------------
    # InfLoRA hooks (rebuttal E15b; train_method "inflora" only)
    # ------------------------------------------------------------------
    # Upstream methods/inflora.py (InfLoRA._train) runs two no-grad passes over
    # the task's shuffled, augmented train loader: pass 1 BEFORE training gives
    # the per-block input Gram G_t = X^T X / n (their cur_matrix) from which the
    # fixed down-projection A_t is designed; pass 2 AFTER training, with the
    # trained branch active (here: folded, which is equivalent), feeds the
    # DualGPM memory update. Both reuse compute_covariances on the qkv wrappers
    # (lora_modules) and bypass update_projections, whose covariance EMA
    # (cov_ema_decay) is the wrong statistic for DualGPM.

    def _inflora_gram_loader(self, train_loader):
        """Loader of the Gram passes. inflora_gram_mode 'train' (upstream): the
        task's shuffled, augmented train loader; 'test': the same images with
        the deterministic test transform (built like update_projections)."""
        mode = self._config.get("inflora_gram_mode", "train")
        if mode == "train":
            return train_loader
        if mode != "test":
            raise ValueError(f"inflora_gram_mode must be 'train' or 'test', got {mode!r}")
        class_range = np.arange(self._known_classes, self._total_classes)
        gram_set = self.data_manager.get_dataset(class_range, source="train", mode="test")
        return DataLoader(
            gram_set, batch_size=128, shuffle=False,
            num_workers=4, worker_init_fn=seed_worker, generator=g,
        )

    @torch.no_grad()
    def _inflora_grams(self, loader):
        """{block_{i}_attn_qkv: float64 (d, d) Gram X^T X / n} of the qkv inputs
        (norm1(x), all tokens) -- upstream's cur_matrix; the SVDs run in NumPy
        on the CPU (inflora_gpm), so torch's deterministic mode is untouched."""
        covs = compute_covariances(self.model.backbone, loader)
        return {k: v.detach().cpu().double().numpy() for k, v in covs.items() if v is not None}

    def _inflora_before_train(self, train_loader):
        """Hook 1 (train_backbone, right after the cache check): design A_t."""
        rows = self.model.backbone.design_lora_A(
            self._inflora_grams(self._inflora_gram_loader(train_loader)))
        n_retain = sum(1 for r in rows if r[2] == "retain")
        leak = max([r[3] for r in rows], default=0.0)
        logging.info(
            f"[InfLoRA] Task {self._cur_task}: A_t designed for {len(rows)} blocks "
            f"(remove {len(rows) - n_retain} / retain {n_retain}) | max|A M| {leak:.3e}")

    def _inflora_after_train(self, train_loader):
        """Hook 2 (train, after the fold and before the state save): DualGPM
        update at threshold lamb + (lame - lamb) t / T (upstream schedule;
        T = number of sessions, 10 on every benchmark here)."""
        thr = inflora_gpm.threshold_schedule(
            self._config.get("inflora_lamb", 0.95), self._config.get("inflora_lame", 1.0),
            self._cur_task, self.data_manager.nb_tasks)
        rows = self.model.backbone.update_memory(
            self._inflora_grams(self._inflora_gram_loader(train_loader)),
            thr, dual=bool(self._config.get("inflora_dual", True)))
        logging.info(
            f"[InfLoRA] DualGPM task {self._cur_task} threshold {thr:.4f} | "
            f"cols {[r[1] for r in rows]} | types {[r[2] for r in rows]} | "
            f"changed {[r[3] for r in rows]}")

    # ------------------------------------------------------------------
    # Training (unified loop; seqft epoch-mode == exp19.train_mlp)
    # ------------------------------------------------------------------

    def train_backbone(self, train_loader):
        """Returns True if training ran, False on a cache hit."""
        logging.info(f"[Training] Task {self._cur_task} ({self._method})")

        model_use_norm = self._config.get("model_use_norm", False)
        model_classifier_use_norm = self._config.get(
            "model_classifier_use_norm", not model_use_norm)
        model_classifier_norm_layer = self._config.get("model_classifier_norm_layer", "ln")

        self.model.update_classifier(
            self._total_classes - self._known_classes,
            with_norm=model_classifier_use_norm,
            with_bias=self._config.get("model_classifier_bias", False),
            freeze_old=True,
            norm_layer=model_classifier_norm_layer,
        )
        self.model.cuda()

        # Training cache
        reset_train = self._config.get("reset_train", False)
        if self._is_seqft:
            bb_ckpt = self.backbone_checkpoint(self._cur_task)
        elif self._is_e2:
            bb_ckpt = self.e2_checkpoint(self._cur_task)
        else:
            bb_ckpt = self.state_checkpoint(self._cur_task)
        cls_ckpt = self.classifier_checkpoint(self._cur_task)
        cache_ok = (self._is_seqft or self._cache_enabled())
        if not reset_train and cache_ok and os.path.exists(bb_ckpt) and os.path.exists(cls_ckpt):
            logging.info(f"[Training] Reusing cached trained checkpoint for task {self._cur_task}")
            if self._is_seqft:
                self.load_backbone(torch.load(bb_ckpt), load_norm=model_use_norm)
            elif self._is_e2:
                # Raw structural state, saved post-energy-transform.
                self.model.backbone.load_e2_state(torch.load(bb_ckpt))
            else:
                # Post-task state: adapter already folded.
                load_trainable_state(self.model.backbone, torch.load(bb_ckpt))
            cls_sd = torch.load(cls_ckpt)
            align_on = self._config.get("train_ca", False) or self._config.get("train_drift", False)
            if align_on and self._cur_task > 0:
                # Alignment retrains old heads every task; keep the aligned
                # in-memory old heads, load only the CURRENT head (exp19).
                cls_sd = {k: v for k, v in cls_sd.items()
                          if k.startswith(f"heads.{self._cur_task}.")}
                self.model.classifier.load_state_dict(cls_sd, strict=False)
                logging.info(f"[Training] Loaded only head {self._cur_task} from cache "
                             "(preserving aligned old heads)")
            else:
                self.model.classifier.load_state_dict(cls_sd)
            self.model.cuda()
            return False

        if self._is_inflora:
            # InfLoRA hook 1: design the fixed down-projection A_t from the
            # current task's input Grams. A cache hit returned above, so the
            # Gram pass runs only when the task really trains.
            self._inflora_before_train(train_loader)

        # first_task_lora: backbone adapted on task 0 only; later tasks train
        # the new head only (backbone frozen).
        head_only = (self._method == "first_task_lora" and self._cur_task > 0)

        self.model.train()
        logging.info(f"[Training] {self.model}")

        epochs = self._config["train_epochs"]
        # Rebuttal E1 (train_epochs_small_task, default None): on mixed long
        # splits a task with few classes trains for a different number of
        # epochs ("5 epochs when a task has <= 4 classes"); unset -> unchanged.
        small_epochs = self._config.get("train_epochs_small_task", None)
        if small_epochs is not None:
            task_size = self._total_classes - self._known_classes
            if task_size <= int(self._config.get("train_small_task_max_classes", 4)):
                epochs = int(small_epochs)
                logging.info(f"[Training] small task ({task_size} classes): epochs={epochs}")
        base_lr = self._config["train_base_lr"]
        weight_decay = self._config["train_weight_decay"]
        head_lr = self._config.get("train_head_lr", None) or base_lr
        iterations = self._config.get("train_iterations", None)

        if head_only:
            backbone_params = []
            for p in self.model.backbone.parameters():
                p.requires_grad = False
        elif self._is_seqft:
            backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        else:
            backbone_params = self.model.backbone.get_param_groups()

        parameters = []
        if backbone_params:
            parameters.append({"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay})
        parameters.append({
            "params": [
                p for p in self.model.classifier.heads[self._cur_task].parameters()
                if p.requires_grad
            ],
            "lr": head_lr, "weight_decay": weight_decay,
        })
        if self.distiller is not None and self._cur_task > 0:
            # Trainable KD transform head at 10x lr (subspace_lora.py:449).
            self.distiller.head.train()
            kd_params = [p for p in self.distiller.head.parameters() if p.requires_grad]
            if kd_params:
                parameters.append({"params": kd_params, "lr": 10 * base_lr, "weight_decay": weight_decay})

        opt_name = self._config.get("train_optimizer", "sgd").lower()
        if opt_name == "sgd":
            optimizer = optim.SGD(parameters, lr=base_lr, momentum=0.9, weight_decay=weight_decay)
        elif opt_name == "adamw":
            optimizer = optim.AdamW(parameters, lr=base_lr, weight_decay=weight_decay)
        elif opt_name == "adam":
            optimizer = optim.Adam(parameters, lr=base_lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown train_optimizer: {opt_name!r}")
        logging.info(f"[Optimizer] {opt_name} (lr={base_lr}, head_lr={head_lr}, "
                     f"weight_decay={weight_decay})")

        loss_name = self._config.get("train_loss", "ce")
        if loss_name == "ce":
            loss_fn = F.cross_entropy
        elif loss_name == "sce":
            loss_fn = symmetric_cross_entropy_loss
        else:
            raise ValueError(f"Unknown train_loss: {loss_name!r}")

        sched_name = self._config.get("train_scheduler", "cosine")

        e2_sd_weight = self._config.get("e2_sd_weight", 0.25)

        # Spectral-norm control of the LoRA update (seqft only).
        #   specreg_weight > 0 : soft penalty  loss += l_s * mean_layers
        #       sigma_max(scale*(B@A) - W_start), W_start = task-start state.
        #       sigma via 1-step warm-started power iteration; grad = u v^T.
        #   speccap = c : hard bound -- after each optimizer step, rescale B so
        #       sigma_max(scale*(B@A)) <= c (SN-style projection on the TOTAL
        #       accumulated update).
        specreg = self._config.get("specreg_weight", 0.0)
        # Task-0 exemption: the foundation task adapts freely (merge/E2 lesson:
        # never constrain the base).
        if self._config.get("specreg_exempt_task0", False) and self._cur_task == 0:
            specreg = 0.0
        speccap = self._config.get("speccap", None)
        speccap_anchor = self._config.get("speccap_anchor", "total")  # total | task
        spec_layers = None
        if self._method == "seqft" and (specreg > 0 or speccap):
            from backbones import peft_qkv_modules
            spec_scale = self._config["model_lora_alpha"] / self._config["model_lora_r"]
            spec_layers = {}
            need_W0 = specreg > 0 or (speccap and speccap_anchor == "task")
            for lname, qkv in peft_qkv_modules(self.model.backbone).items():
                A = qkv.lora_A["default"].weight
                B = qkv.lora_B["default"].weight
                # Separate power-iteration warm starts: the penalty iterates on
                # the task delta, the cap on its own target matrix.
                st = {"A": A, "B": B,
                      "u_pen": torch.nn.functional.normalize(torch.randn(B.shape[0], device=A.device), dim=0),
                      "u_cap": torch.nn.functional.normalize(torch.randn(B.shape[0], device=A.device), dim=0)}
                if need_W0:
                    st["W0"] = (spec_scale * (B @ A)).detach().clone()
                    st["A0"] = A.detach().clone()
                spec_layers[lname] = st
            logging.info(f"[SpecNorm] {len(spec_layers)} layers | "
                         f"penalty l_s={specreg} | cap={speccap} "
                         f"({self._config.get('speccap_mode', 'rescale')}, anchor={speccap_anchor})")

        # L2 penalty toward the previously deployed module (rebuttal E10,
        # training-time arm):  loss += l2m_weight * sum ||p - p_anchor||^2 / numel
        # over every trainable LoRA tensor present in the anchor state (qkv,
        # fc1, fc2 -- unlike specreg, which is qkv-only); numel is the total
        # element count of those tensors, so the weight is a mean squared
        # per-element deviation. The anchor follows the _get_prev_feature_model
        # rule: the task-(t-1) trained state at t == 1 or in merge-less runs
        # (where the penalty replaces merging), else the merged state M_{t-1}.
        # Task 0 is never penalised; the default l2m_weight = 0 adds nothing.
        l2m_weight = float(self._config.get("l2m_weight", 0.0))
        l2m_pairs, l2m_numel = None, 0
        if self._is_seqft and l2m_weight > 0 and self._cur_task > 0:
            if self._cur_task == 1 or self._config.get("train_merge", "none") == "none":
                anchor_ckpt = self.backbone_checkpoint(self._cur_task - 1)
            else:
                anchor_ckpt = self.merged_checkpoint(self._cur_task - 1)
            anchor = torch.load(anchor_ckpt, map_location="cuda")
            l2m_pairs = [
                (p, anchor[name].detach().to(device=p.device, dtype=p.dtype))
                for name, p in self.model.backbone.named_parameters()
                if p.requires_grad and name in anchor
            ]
            l2m_numel = sum(p.numel() for p, _ in l2m_pairs)
            del anchor
            logging.info(f"[L2M] anchor={os.path.basename(anchor_ckpt)} "
                         f"weight={l2m_weight} tensors={len(l2m_pairs)}")

        def _l2m_penalty():
            pen = 0.0
            for p, a in l2m_pairs:
                pen = pen + (p - a).pow(2).sum()
            return pen / max(1, l2m_numel)

        def _spec_penalty():
            pen = 0.0
            for st in spec_layers.values():
                D = spec_scale * (st["B"] @ st["A"]) - st["W0"]
                with torch.no_grad():
                    v = torch.nn.functional.normalize(D.t() @ st["u_pen"], dim=0)
                    u = torch.nn.functional.normalize(D @ v, dim=0)
                    st["u_pen"] = u
                pen = pen + torch.dot(u, D @ v)
            return pen / len(spec_layers)

        speccap_mode = self._config.get("speccap_mode", "rescale")

        @torch.no_grad()
        def _spec_cap():
            for st in spec_layers.values():
                A, B = st["A"], st["B"]
                if speccap_mode == "clip" and speccap_anchor == "task":
                    # Clip the CURRENT TASK's update D = W - W0 at c. D's row
                    # space spans rowspace(A) U rowspace(A0) (rank <= 2r), so
                    # compute its SVD via that small joint basis. The write-back
                    # is B-only, i.e. the correction is PROJECTED onto
                    # rowspace(A) -- approximate; enforcement error is measured
                    # exactly at end of task and logged.
                    D = spec_scale * (B @ A) - st["W0"]
                    K = torch.cat([A, st["A0"]], dim=0)              # (2r, in)
                    Qr, _ = torch.linalg.qr(K.t())                   # (in, 2r)
                    Ds = D @ Qr                                      # (out, 2r)
                    U, S, Vh = torch.linalg.svd(Ds, full_matrices=False)
                    over = (S - speccap).clamp_min(0.0)
                    if over.max() <= 0:
                        continue
                    Apinv = A.t() @ torch.linalg.inv(A @ A.t())      # (in, r)
                    corr = (U * over.unsqueeze(0)) @ (Vh @ Qr.t() @ Apinv)
                    B.data.sub_(corr / spec_scale)
                elif speccap_mode == "clip":
                    # Exact projection on the TOTAL update: clip singular
                    # values at c via the rank-r core; B-only write-back is
                    # exact here (clipped row space stays inside rowspace(A)).
                    Q, R = torch.linalg.qr(B)               # (out,r),(r,r)
                    M = spec_scale * (R @ A)                # (r,in) core
                    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
                    over = (S - speccap).clamp_min(0.0)
                    if over.max() <= 0:
                        continue
                    Apinv = A.t() @ torch.linalg.inv(A @ A.t())     # (in,r)
                    corr = (Q @ (U * over.unsqueeze(0))) @ (Vh @ Apinv)
                    B.data.sub_(corr / spec_scale)
                else:
                    W = spec_scale * (B @ A)
                    u = st["u_cap"]
                    for _ in range(2):
                        v = torch.nn.functional.normalize(W.t() @ u, dim=0)
                        Wv = W @ v
                        sigma = Wv.norm()
                        u = Wv / sigma.clamp_min(1e-12)
                    st["u_cap"] = u
                    if sigma > speccap:
                        st["B"].data.mul_(speccap / sigma)

        def run_batch(x, y):
            x = x.cuda()
            y = (y - self._known_classes).cuda()
            z = self.model.get_features(x)
            if self._is_e2:
                # Their loss (e2lora.py:_run): CE on new-class logits of the
                # FULL head + T=2 logit self-distillation on old classes, the
                # teacher being the same net with the current pair excluded.
                logits_full = self.model.classifier(z)
                loss = loss_fn(logits_full[:, self._known_classes:], y)
                if self._cur_task > 0 and e2_sd_weight > 0:
                    self.model.backbone.set_role("teacher")
                    with torch.no_grad():
                        z_tea = self.model.get_features(x)
                        tea_logits = self.model.classifier(z_tea)
                    self.model.backbone.set_role("student")
                    T = 2.0
                    distill = F.kl_div(
                        F.log_softmax(logits_full[:, :self._known_classes] / T, dim=1),
                        F.softmax(tea_logits[:, :self._known_classes] / T, dim=1),
                        reduction="batchmean",
                    ) * (T * T)
                    loss = loss + e2_sd_weight * distill
                logits = logits_full[:, self._known_classes:]
            else:
                logits = self.model.classifier.heads[-1](z)
                loss = loss_fn(logits, y)
                if self.distiller is not None and self._cur_task > 0:
                    loss = loss + self.distiller(x, z)
                if spec_layers is not None and specreg > 0:
                    loss = loss + specreg * _spec_penalty()
                if l2m_pairs is not None:
                    loss = loss + l2m_weight * _l2m_penalty()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if spec_layers is not None and speccap:
                _spec_cap()
            bs = len(y)
            acc = (logits.argmax(dim=1) == y).sum().item()
            return loss.item() * bs, acc, bs

        if iterations is None:
            # Epoch mode (exp19): per-epoch cosine schedule; "constant" keeps
            # the lr fixed (E2-LoRA's IN-R config: milestones never fire).
            if sched_name == "constant":
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0)
            elif sched_name == "multistep":
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=self._config.get("train_milestones", [40]),
                    gamma=self._config.get("train_lr_gamma", 0.1))
            elif sched_name == "inflora_cosine":
                # InfLoRA's per-epoch quarter cosine (upstream utils/schedulers.py
                # CosineSchedule): lr * cos(99 pi e / (200 (K - 1))), i.e. the base
                # lr at epoch 0 and 0.0157 lr at the last epoch; K <= 1 keeps the
                # lr constant (the smoke configs train one epoch).
                n_epochs = epochs
                scheduler = optim.lr_scheduler.LambdaLR(
                    optimizer, lambda e: inflora_gpm.cosine_lr_factor(e, n_epochs))
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            for epoch in range(epochs):
                total_loss, total_acc, total = 0.0, 0, 0
                epoch_lr = optimizer.param_groups[0]["lr"]   # lr applied in this epoch
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
                for _, (_, _, x, y) in enumerate(pbar):
                    l, a, b = run_batch(x, y)
                    total_loss += l
                    total_acc += a
                    total += b
                scheduler.step()
                logging.info(
                    f"[Training] Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {total_loss / total:.4f}, Acc: {total_acc / total:.4f}"
                    # InfLoRA only: the realised per-epoch lr (quarter-cosine check)
                    + (f", lr: {epoch_lr:.3e}" if self._is_inflora else "")
                )
        else:
            # Step mode (reference repo): per-step warmup-cosine schedule.
            warmup_ratio = self._config.get("train_warmup_ratio", 0.1)
            warmup_steps = int(warmup_ratio * iterations)
            if sched_name == "warmup_cosine":
                eta_min_ratio = (base_lr / 3) / base_lr

                def lr_lambda(step):
                    if step < warmup_steps:
                        return step / max(1, warmup_steps)
                    progress = (step - warmup_steps) / max(1, iterations - warmup_steps)
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return eta_min_ratio + cosine_decay * (1.0 - eta_min_ratio)

                scheduler = optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda] * len(parameters))
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-6)

            step, done = 0, False
            total_loss, total_acc, total = 0.0, 0, 0
            while not done:
                for _, (_, _, x, y) in enumerate(train_loader):
                    l, a, b = run_batch(x, y)
                    total_loss += l
                    total_acc += a
                    total += b
                    scheduler.step()
                    step += 1
                    if step % 200 == 0:
                        logging.info(
                            f"[Training] Step {step}/{iterations}, "
                            f"Loss: {total_loss / total:.4f}, Acc: {total_acc / total:.4f}"
                        )
                        total_loss, total_acc, total = 0.0, 0, 0
                    if step == iterations:
                        done = True
                        break

        if spec_layers is not None:
            with torch.no_grad():
                sigmas = [torch.linalg.matrix_norm(
                    spec_scale * (st["B"] @ st["A"]), ord=2).item()
                    for st in spec_layers.values()]
            logging.info(f"[SpecNorm] end-of-task sigma_max(total update): "
                         f"mean {np.mean(sigmas):.3f}, max {np.max(sigmas):.3f}")
            if speccap and speccap_anchor == "task":
                # EXACT enforcement check for the projected task-anchor clip.
                with torch.no_grad():
                    dsig = [torch.linalg.matrix_norm(
                        spec_scale * (st["B"] @ st["A"]) - st["W0"], ord=2).item()
                        for st in spec_layers.values()]
                logging.info(f"[SpecNorm] end-of-task sigma_max(task delta): "
                             f"mean {np.mean(dsig):.3f}, max {np.max(dsig):.3f} "
                             f"(cap {speccap} -- excess = projection error)")

        if l2m_pairs is not None:
            # Realised distance of the trained module from its anchor.
            with torch.no_grad():
                dist = math.sqrt(sum(float((p - a).pow(2).sum()) for p, a in l2m_pairs))
            logging.info(f"[L2M] end-of-task ||W_t-anchor||_F={dist:.4f}")
            l2m_pairs = None

        # Save training cache. Merge-family backbone state is saved post-fold
        # in train(); here we save the seqft LoRA state + the classifier.
        if self._is_seqft:
            torch.save(
                self.model.get_backbone_trainable_params(),
                self.backbone_checkpoint(self._cur_task),
            )
        cls_sd = self.model.classifier.state_dict()
        if (self._config.get("train_ca", False) or self._config.get("train_drift", False)) \
                and self._cur_task > 0:
            # Keep the shared classifier cache pristine (exp19): restore cached
            # (unaligned) old heads before saving.
            prev_ckpt = self.classifier_checkpoint(self._cur_task - 1)
            if os.path.exists(prev_ckpt):
                cls_sd = {**cls_sd, **torch.load(prev_ckpt)}
        if self._is_seqft or self._cache_enabled():
            torch.save(cls_sd, self.classifier_checkpoint(self._cur_task))
        return True

    # ------------------------------------------------------------------
    # Evaluation (exp19)
    # ------------------------------------------------------------------

    def _predict(self, source, generator, til=False):
        """Forward pass of the deployed model over classes [0, total) of `source`.

        Returns (y_pred, y_true, y_pred_til) as numpy arrays. The test pass of
        eval() calls it with the module generator g and batch 256: that is the
        former inline loop (byte-identical; a DataLoader draws its worker base
        seed from its generator, so the stream of g is consumed exactly as
        before). The validation pass (rebuttal E14) passes a private generator,
        so g, and with it the training of later tasks, is untouched.
        til=True (eval_til) also returns the task-masked prediction: the argmax
        restricted to the columns of each sample's own task (_class_increments),
        accumulated per batch so no logits are kept; otherwise y_pred_til is None.
        """
        dataset = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source=source, mode="test"
        )
        loader = DataLoader(
            dataset, batch_size=256, shuffle=False,
            num_workers=4, worker_init_fn=seed_worker, generator=generator,
        )

        self.model.eval()
        y_true, y_pred, y_pred_til = [], [], []
        if til:
            # Column range (lo, hi) of the task that owns each class.
            lo = torch.empty(self._total_classes, dtype=torch.long)
            hi = torch.empty(self._total_classes, dtype=torch.long)
            for a, b in self._class_increments:
                lo[a:b + 1] = a
                hi[a:b + 1] = b
            lo, hi = lo.cuda(), hi.cuda()

        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(loader):
                x, y = x.cuda(), y.cuda()
                logits = self.model(x)
                y_pred.append(logits.argmax(dim=1).cpu().numpy())
                y_true.append(y.cpu().numpy())
                if til:
                    cols = torch.arange(logits.size(1), device=logits.device)
                    mask = ((cols[None, :] >= lo[y][:, None])
                            & (cols[None, :] <= hi[y][:, None]))
                    y_pred_til.append(
                        logits.masked_fill(~mask, float("-inf")).argmax(dim=1).cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        return y_pred, y_true, (np.concatenate(y_pred_til) if til else None)

    def _score(self, y_pred, y_true):
        """(total acc, per-task grouped accs as floats) of one prediction vector.

        The former inline block of eval(). Domain-incremental scoring
        (E2-LoRA/DUCT convention): every session holds the SAME semantic
        classes, offset by session in the label space, so credit is given for
        the semantic class regardless of which domain block it was predicted
        in (pred % C == true % C). Exact-match scoring would additionally
        require identifying the domain, which is not the DIL task.
        """
        dil_classes = self._config.get("dil_classes_per_domain", None)
        if dil_classes:
            correct = (y_pred % dil_classes) == (y_true % dil_classes)
            acc_total = np.around(correct.sum() * 100 / len(y_true), decimals=2)
            grouped = []
            for lo, hi in self._class_increments:
                idxes = np.where((y_true >= lo) & (y_true <= hi))[0]
                grouped.append(np.around(
                    correct[idxes].sum() * 100 / len(idxes), decimals=2))
        else:
            acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)
        return acc_total, [float(a) for a in grouped]

    def eval(self):
        # Test pass: module generator g, batch 256 (the unchanged default path).
        # eval_til (default False) additionally keeps the task-masked argmax.
        y_pred, y_true, y_pred_til = self._predict(
            "test", g, til=bool(self._config.get("eval_til", False)))

        logging.info(f"[Evaluation] Task {self._cur_task}")
        num_tasks = self._cur_task + 1

        dil_classes = self._config.get("dil_classes_per_domain", None)
        acc_total, grouped = self._score(y_pred, y_true)
        self._mlp_matrix.append(grouped)
        self._last_total_acc = float(acc_total)
        logging.info(f"[Evaluation] Total Acc: {acc_total:.2f}, Grouped: {grouped}")

        # Sample-weighted curve. For DIL this is the directly comparable pair
        # to the E2-LoRA/DUCT tables: Last-Acc = final total acc, Inc-Acc =
        # mean of the per-session totals. (FA/ASA below are the class/domain-
        # EQUAL averages used elsewhere in this project — different numbers.)
        if not hasattr(self, "_total_acc_curve"):
            self._total_acc_curve = []
        self._total_acc_curve.append(float(acc_total))
        if dil_classes:
            logging.info(
                f"[Evaluation] DIL Last-Acc: {self._total_acc_curve[-1]:.2f}, "
                f"Inc-Acc: {np.mean(self._total_acc_curve):.2f} "
                f"(sample-weighted, mod-{dil_classes} scoring)")

        mat = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                mat[i, j] = self._mlp_matrix[i][j]
        faa, ffm, ffd, asa = compute_metrics(mat)
        logging.info(f"[Evaluation] FAA: {faa:.2f}, FFM: {ffm:.2f}, "
                     f"FFD: {ffd:.2f}, ASA: {asa:.2f}")

        self._faa = faa
        self._ffm = ffm
        self._asa = asa
        self._acc = asa
        self._acc_history.append(float(np.round(self._acc, 2)))

        # Rebuttal (q6ZZ-W4) eval_til: task-incremental accuracy from the same
        # logits, argmax restricted to each sample's own task columns.
        if y_pred_til is not None:
            acc_til, grouped_til = self._score(y_pred_til, y_true)
            logging.info(f"[Evaluation-til] Total Acc: {acc_til:.2f}, Grouped: {grouped_til}")

        # Rebuttal E14: the same evaluation on the validation carve (private
        # generator, so the stream of g is untouched); selection statistics only.
        if self._config.get("data_val_frac", 0.0) and self.data_manager.has_val:
            y_pred_v, y_true_v, _ = self._predict("val", torch.Generator().manual_seed(0))
            acc_v, grouped_v = self._score(y_pred_v, y_true_v)
            self._mlp_matrix_val.append(grouped_v)
            logging.info(f"[Evaluation-val] Total Acc: {acc_v:.2f}, Grouped: {grouped_v}")
            mat_v = np.zeros((num_tasks, num_tasks))
            for i in range(num_tasks):
                for j in range(i + 1):
                    mat_v[i, j] = self._mlp_matrix_val[i][j]
            faa_v, ffm_v, ffd_v, asa_v = compute_metrics(mat_v)
            logging.info(f"[Evaluation-val] FAA: {faa_v:.2f}, FFM: {ffm_v:.2f}, "
                         f"FFD: {ffd_v:.2f}, ASA: {asa_v:.2f}")
            self._faa_val, self._ffm_val, self._asa_val = faa_v, ffm_v, asa_v

    # ------------------------------------------------------------------
    # Stage 2: class Gaussians (exp19, verbatim)
    # ------------------------------------------------------------------

    def compute_multivariate_normal(self):
        """Compute per-class Gaussian (mean + full covariance) from training features.

        train_ca_cov_mode (rebuttal E5). 'class' (default: one full covariance
        per class) and 'shared_corr' are unchanged and materialise
        _class_covs (C, d, d). The cheap modes never allocate that tensor:
          'diag'    : _class_vars (C, d)                      covstats.fit_diag
          'lowrank' : _class_lr_U (C, d, train_ca_cov_rank) + residual
                      _class_vars                             covstats.fit_lowrank
          'shared'  : _shared_scatter (d, d) pooled within-class scatter,
                      _shared_n = sum_c (n_c - 1)             covstats.scatter
        Any other string raises ValueError (it used to run the full model).
        """
        logging.info(
            f"[Alignment] Computing class stats for classes "
            f"{self._known_classes}–{self._total_classes - 1}"
        )
        total_class = self._total_classes
        feature_dim = self.model.feature_dim
        cov_mode = self._config.get("train_ca_cov_mode", "class")
        if cov_mode not in COV_MODES:
            raise ValueError(
                f"unknown train_ca_cov_mode {cov_mode!r}; expected one of {COV_MODES}")
        cov_rank = int(self._config.get("train_ca_cov_rank", 64))
        cheap = cov_mode in ("diag", "lowrank", "shared")

        if not hasattr(self, "_class_means") or not hasattr(self, "_class_covs"):
            self._class_means = torch.zeros((total_class, feature_dim))
            # Cheap modes keep no (C, d, d) tensor at all (None, never grown).
            self._class_covs = (None if cheap else
                                torch.zeros((total_class, feature_dim, feature_dim)))
            self._class_vars = torch.zeros((total_class, feature_dim))
        else:
            new_means = torch.zeros((total_class, feature_dim))
            new_means[: self._known_classes] = self._class_means
            self._class_means = new_means
            if self._class_covs is not None:
                new_covs = torch.zeros((total_class, feature_dim, feature_dim))
                new_covs[: self._known_classes] = self._class_covs
                self._class_covs = new_covs
            new_vars = torch.zeros((total_class, feature_dim))
            new_vars[: self._known_classes] = self._class_vars
            self._class_vars = new_vars
        if cov_mode == "lowrank":
            new_U = torch.zeros((total_class, feature_dim, cov_rank))
            if hasattr(self, "_class_lr_U"):
                new_U[: self._known_classes] = self._class_lr_U
            self._class_lr_U = new_U
        if cov_mode == "shared" and not hasattr(self, "_shared_scatter"):
            self._shared_scatter = torch.zeros((feature_dim, feature_dim), dtype=torch.double)
            self._shared_n = 0
        if not hasattr(self, "_corr_accum"):
            self._corr_accum = torch.zeros((feature_dim, feature_dim), dtype=torch.double)
            self._corr_count = 0

        for cls_idx in range(self._known_classes, self._total_classes):
            proto_set = self.data_manager.get_dataset(
                np.arange(cls_idx, cls_idx + 1), source="train", mode="test"
            )
            proto_loader = DataLoader(
                proto_set, batch_size=512, shuffle=False,
                num_workers=4, worker_init_fn=seed_worker, generator=g,
            )
            feats = []
            self.model.eval()
            with torch.no_grad():
                for _, (_, _, x, _) in enumerate(proto_loader):
                    feats.append(self.model.get_features(x.cuda()).cpu())
            if not feats:
                # Ghost class (present in the label space but absent from this
                # split — e.g. DomainNet's one missing train class). Neutral
                # stats: zero mean, jitter cov; contributes ~nothing to LCA.
                logging.warning(f"[Stats] class {cls_idx} has no train samples; "
                                f"using neutral stats")
                self._class_means[cls_idx] = torch.zeros(feature_dim)
                if self._class_covs is not None:
                    self._class_covs[cls_idx] = torch.eye(feature_dim) * 1e-4
                else:
                    # Cheap modes: jitter variance, zero factor, nothing pooled.
                    self._class_vars[cls_idx] = torch.full((feature_dim,), 1e-4)
                continue
            feats = torch.cat(feats, dim=0)
            n_feats = feats.size(0)                      # before the duplication below
            if feats.size(0) == 1:
                # torch.cov needs n>=2; duplicate -> zero cov + jitter below.
                feats = torch.cat([feats, feats], dim=0)
            self._class_means[cls_idx] = feats.mean(dim=0)
            if cov_mode == "shared_corr":
                var = (feats.var(dim=0, unbiased=True)
                       if feats.size(0) > 1 else torch.ones(feature_dim))
                self._class_vars[cls_idx] = var
                z = (feats - self._class_means[cls_idx]) / (var + 1e-8).sqrt()
                self._corr_accum += (z.T @ z).double()
                self._corr_count += feats.size(0)
            elif cov_mode == "diag":
                self._class_vars[cls_idx] = fit_diag(feats)
            elif cov_mode == "lowrank":
                U, var_res = fit_lowrank(feats, cov_rank)
                self._class_lr_U[cls_idx] = U
                self._class_vars[cls_idx] = var_res
            elif cov_mode == "shared":
                self._shared_scatter += cov_scatter(feats)
                self._shared_n += max(0, n_feats - 1)    # a duplicated single sample adds no dof
            else:
                cov = torch.cov(feats.T)
                gamma = self._config.get("train_ca_cov_shrinkage", 0.0)
                if gamma > 0:
                    iso = torch.eye(feature_dim) * (torch.trace(cov) / feature_dim)
                    cov = (1.0 - gamma) * cov + gamma * iso
                self._class_covs[cls_idx] = cov + torch.eye(feature_dim) * 1e-4

        if self._config.get("train_ca_cov_mode", "class") == "shared_corr":
            gamma = self._config.get("train_ca_cov_shrinkage", 0.1)
            R = (self._corr_accum / max(1, self._corr_count - 1)).float()
            dsqrt = R.diagonal().clamp_min(1e-8).sqrt()
            R = R / (dsqrt[:, None] * dsqrt[None, :])
            R = 0.5 * (R + R.T)
            R = (1.0 - gamma) * R + gamma * torch.eye(feature_dim)
            for c in range(self._total_classes):
                s = (self._class_vars[c] + 1e-8).sqrt()
                self._class_covs[c] = (s[:, None] * R * s[None, :]) \
                    + torch.eye(feature_dim) * 1e-4
            logging.info(
                f"[Alignment] shared_corr covs rebuilt for {self._total_classes} "
                f"classes (gamma={gamma}, pooled n={self._corr_count})"
            )
        if cheap:
            # One line per task in the cheap modes only (default log unchanged).
            logging.info(
                f"[Alignment] cov_mode={cov_mode} "
                f"rank={cov_rank if cov_mode == 'lowrank' else '-'} "
                f"stats_floats={storage_floats(cov_mode, feature_dim, total_class, cov_rank)}")

    # ------------------------------------------------------------------
    # Stage 2: drift compensation (exp19; prev extractor is recipe-aware)
    # ------------------------------------------------------------------

    def fit_drift_map(self, train_loader):
        """Closed-form ridge drift map W (d, d) mapping the previous feature
        space f_{t-1} to the current space f_t, fit on paired current-task
        features (exp19.fit_drift_map; the previous extractor comes from
        _get_prev_feature_model, which abstracts seqft vs merge family)."""
        logging.info(f"[Drift] Fitting ridge drift map for task {self._cur_task}")
        t_feat = self._tic()      # stage_timing: previous-extractor reload + two feature passes
        old_model = self._get_prev_feature_model()
        self.model.eval()

        F_old, F_new = [], []
        with torch.no_grad():
            for _, (_, _, x, _) in enumerate(train_loader):
                x = x.cuda()
                F_old.append(old_model.get_features(x).cpu())
                F_new.append(self.model.get_features(x).cpu())
        self._release_prev_feature_model(old_model)

        F_old = torch.cat(F_old, dim=0).double()
        F_new = torch.cat(F_new, dim=0).double()
        d = F_old.size(1)
        self._toc(t_feat, "drift_features", n=F_old.size(0), d=d)
        lam = self._config.get("train_drift_lambda", 1e2)
        ridge_mode = self._config.get("train_drift_ridge", "zero")
        lambda_rule = self._config.get("train_drift_lambda_rule", None)
        if lambda_rule == "gcv":
            # Rebuttal E7b: data-free ridge lambda by closed-form generalised
            # cross-validation on this task's paired features
            # (driftsolve.gcv_lambda: one thin SVD of F_old serves the grid).
            # No old data and no validation split; the chosen value feeds the
            # unchanged solve below and appears as lambda in the '[Drift] W
            # shape' line.
            grid = [float(v) for v in self._config.get("train_drift_lambda_grid", list(GCV_GRID))]
            lam, scores = gcv_lambda(F_old, F_new, grid, ridge=ridge_mode)
            logging.info(
                f"[Drift] lambda_rule=gcv chosen lambda={lam} | gcv "
                + " ".join(f"{v:g}:{scores[v]:.6g}" for v in grid)
            )
        elif lambda_rule not in (None, "fixed"):
            raise ValueError(f"unknown train_drift_lambda_rule {lambda_rule!r}")
        solver = self._config.get("train_drift_solver", "dense")
        t_solve = self._tic()     # stage_timing: the ridge solve alone
        if solver == "dense":
            A = F_old.T @ F_old + lam * torch.eye(d, dtype=torch.double)
            if ridge_mode == "identity":
                Cd = (F_new - F_old).T @ F_old
                Delta = torch.linalg.solve(A, Cd.T).T
                W = torch.eye(d, dtype=torch.double) + Delta
            else:
                C = F_new.T @ F_old
                W = torch.linalg.solve(A, C.T).T
        elif solver == "woodbury":
            # Rebuttal E3: push-through form of the same ridge, an (n, n)
            # solve instead of (d, d) (driftsolve; equal to the dense solve
            # up to float64 round-off).
            if ridge_mode == "identity":
                W = ridge_identity_woodbury(F_old, F_new, lam)
            else:
                W = ridge_zero_woodbury(F_old, F_new, lam)
        else:
            raise ValueError(f"unknown train_drift_solver {solver!r} (dense | woodbury)")
        self._toc(t_solve, "drift_solve", n=F_old.size(0), d=d, solver=solver)
        # Realised fit parameters, reused by the E9 oracle map and its CSV.
        self._drift_lambda_used = lam
        self._drift_fit_n = int(F_old.size(0))
        pred = F_old @ W.T
        cos = F.cosine_similarity(pred, F_new, dim=1).mean().item()
        dI = (W - torch.eye(d, dtype=torch.double)).norm().item()
        solver_note = "" if solver == "dense" else f", solver={solver}"
        logging.info(
            f"[Drift] W shape {tuple(W.shape)}, ||W||={W.norm().item():.3f}, "
            f"||W-I||={dI:.3f}, mean cos(W·f_(t-1), f_t)={cos:.4f} "
            f"over {F_old.size(0)} samples (λ={lam}, ridge={ridge_mode}{solver_note})"
        )
        return W.float()

    def transport_prototypes(self, W):
        K = self._known_classes
        if K == 0 or not hasattr(self, "_class_means"):
            return
        mode = self._config.get("train_drift_transport", "full")
        logging.info(f"[Drift] Transporting {K} old-class prototypes (mode={mode})")
        t0 = self._tic()
        cov_mode = self._config.get("train_ca_cov_mode", "class")
        Wc = W.cpu().double()
        feature_dim = Wc.size(0)
        means = self._class_means[:K].double()
        self._class_means[:K] = (means @ Wc.T).float()
        if mode == "full":
            if cov_mode in ("class", "shared_corr"):
                for c in range(K):
                    S = Wc @ self._class_covs[c].double() @ Wc.T
                    S = 0.5 * (S + S.T) + torch.eye(feature_dim, dtype=torch.double) * 1e-4
                    self._class_covs[c] = S.float()
            elif cov_mode == "diag":
                # Rebuttal E5: v' = (W o W) v, the exact diagonal of W diag(v) W^T,
                # plus the 1e-4 jitter the full path adds at every transport.
                self._class_vars[:K] = transport_diag(self._class_vars[:K], Wc, jitter=COV_JITTER)
            elif cov_mode == "lowrank":
                # (W U)(W U)^T + W diag(v) W^T; the residual keeps its exact diagonal.
                self._class_lr_U[:K] = transport_lowrank_U(self._class_lr_U[:K], Wc)
                self._class_vars[:K] = transport_diag(self._class_vars[:K], Wc, jitter=COV_JITTER)
            elif cov_mode == "shared":
                # One W S W^T on the pooled old-class scatter (linear in S, so
                # equal to transporting every class's scatter); the new task's
                # scatter is pooled afterwards in compute_multivariate_normal.
                self._shared_scatter = transport_full(self._shared_scatter, Wc, jitter=0.0)
        self._toc(t0, "transport", K=K, mode=mode, cov_mode=cov_mode)

    # ------------------------------------------------------------------
    # Stage 2: LCA alignment (exp19, verbatim)
    # ------------------------------------------------------------------

    def align(self, classifier):
        """Retrain classifier heads on Gaussian-sampled features."""
        logging.info(f"[Alignment] Task {self._cur_task}")
        samples_per_cls = self._config.get("train_ca_samples_per_cls", 256)
        mc_z_per_class = self._config.get("train_ca_mc_z_per_class", 64)
        epochs = self._config.get("train_ca_epochs", 10)
        batch_size = self._config.get("train_ca_batch_size", 64)
        robust_weight = self._config.get("train_ca_robust_weight", 0.0)
        robust_impl = self._config.get("train_ca_robust_impl", "fresh_z")
        classes_per_batch = self._config.get("train_ca_classes_per_batch", 16)
        sampling = self._config.get("train_ca_sampling", "full")
        n_classes_nk = self._config.get("train_ca_n_classes", 5)
        k_samples_nk = self._config.get("train_ca_k_samples", 64)
        steps_cfg = self._config.get("train_ca_steps_per_epoch", None)
        if robust_weight > 0 or sampling != "full":
            logging.info(
                f"[Alignment] Robust impl: {robust_impl}, sampling: {sampling}"
                + (f" (N={n_classes_nk}, K={k_samples_nk})" if sampling == "nk" else "")
            )

        device = next(classifier.parameters()).device

        for p in classifier.parameters():
            p.requires_grad = True
        logging.info(f"[Alignment] Trainable: {count_parameters(classifier, trainable=True):,}")

        optimizer = optim.SGD(classifier.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

        cov_mode = self._config.get("train_ca_cov_mode", "class")
        dists = []
        if cov_mode in ("class", "shared_corr"):
            for cls_idx in range(self._total_classes):
                mu = self._class_means[cls_idx].to(device).float()
                cov = self._class_covs[cls_idx].to(device).float()
                dists.append(MultivariateNormal(mu, cov))
        else:
            # Rebuttal E5: diag / lowrank / shared sampling distributions.
            dists = self._cheap_dists(cov_mode, device)

        sampled_label = torch.cat([
            torch.full((samples_per_cls,), c, dtype=torch.long, device=device)
            for c in range(self._total_classes)
        ])
        n_total = sampled_label.size(0)
        means_dev = self._class_means[: self._total_classes].to(device).float()

        for epoch in range(epochs):
            total_loss = total_ce_loss = total_rb_loss = total_acc = total = 0

            if sampling == "nk":
                C = self._total_classes
                N = min(n_classes_nk, C)
                K = k_samples_nk
                if steps_cfg is not None:
                    steps = int(steps_cfg)
                else:
                    steps = max(1, -(-(C * samples_per_cls) // (N * K)))

                def _nk_batches():
                    for _ in range(steps):
                        sel = torch.randperm(C, device=device)[:N]
                        xb = torch.cat(
                            [dists[int(c)].sample((K,)) for c in sel], dim=0
                        )
                        yb = sel.repeat_interleave(K)
                        yield xb, yb

                batches = _nk_batches()
            else:
                sampled_data = torch.cat(
                    [d.sample((samples_per_cls,)) for d in dists], dim=0
                )

                if robust_impl == "batch_pairs":
                    C = self._total_classes
                    P = min(classes_per_batch, C)
                    K = max(2, batch_size // P)
                    data_by_cls = sampled_data.view(C, samples_per_cls, -1)
                    batches = []
                    for r in range(samples_per_cls // K):
                        cls_order = torch.randperm(C, device=device)
                        for c0 in range(0, C, P):
                            sel = cls_order[c0:c0 + P]
                            xb = data_by_cls[sel, r * K:(r + 1) * K].reshape(-1, data_by_cls.size(-1))
                            yb = sel.repeat_interleave(K)
                            batches.append((xb, yb))
                else:
                    perm = torch.randperm(n_total, device=device)
                    epoch_data = sampled_data[perm]
                    epoch_label = sampled_label[perm]
                    num_iters = (n_total + batch_size - 1) // batch_size
                    batches = [
                        (epoch_data[i * batch_size:(i + 1) * batch_size],
                         epoch_label[i * batch_size:(i + 1) * batch_size])
                        for i in range(num_iters)
                    ]

            for x, y in batches:
                n_batch = y.size(0)

                logits = classifier(x)
                loss_vec = F.cross_entropy(logits, y, reduction="none")
                base_loss = loss_vec.mean()

                if torch.isnan(base_loss):
                    continue

                reg_loss = torch.tensor(0.0, device=device)
                if robust_weight > 0 and robust_impl == "batch_pairs":
                    unique_classes = torch.unique(y).tolist()
                    nearest = torch.cdist(x, means_dev).argmin(dim=1)
                    for class_i in unique_classes:
                        label_mask = (y == class_i)
                        class_mask = label_mask & (nearest == class_i)
                        if class_mask.any():
                            class_losses = loss_vec[class_mask]
                        else:
                            class_losses = loss_vec[label_mask]
                        n_cl = class_losses.numel()
                        if n_cl >= 2:
                            diffs = torch.abs(
                                class_losses[:, None] - class_losses[None, :]
                            )
                            off_diag = ~torch.eye(n_cl, dtype=torch.bool, device=device)
                            reg_loss = reg_loss + diffs[off_diag].mean()
                    reg_loss = reg_loss / max(1, len(unique_classes))
                elif robust_weight > 0:
                    if sampling == "nk":
                        region_weight = 1.0 / max(1, len(torch.unique(y)))
                    else:
                        region_weight = float(samples_per_cls) / float(n_total)
                    for class_i in torch.unique(y).tolist():
                        x_i = x[y == class_i]
                        y_i = y[y == class_i]
                        if x_i.size(0) == 0:
                            continue
                        loss_s = F.cross_entropy(classifier(x_i), y_i, reduction="none")
                        z_i = dists[class_i].sample((mc_z_per_class,))
                        y_z = torch.full((mc_z_per_class,), class_i, dtype=torch.long, device=device)
                        loss_z = F.cross_entropy(classifier(z_i), y_z, reduction="none")
                        reg_loss = reg_loss + region_weight * torch.abs(
                            loss_z[:, None] - loss_s[None, :]
                        ).mean()

                loss = base_loss + robust_weight * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * n_batch
                total_ce_loss += base_loss.item() * n_batch
                total_rb_loss += reg_loss.item() * n_batch
                total_acc += (logits.argmax(dim=1) == y).sum().item()
                total += n_batch

            if epoch % 5 == 4 or epoch == epochs - 1:
                logging.info(
                    f"[Alignment] Epoch {epoch + 1}/{epochs}, "
                    f"Base Loss: {total_ce_loss / max(total, 1):.4f}, "
                    f"Robust Term: {total_rb_loss / max(total, 1):.4f}, "
                    f"Total Loss: {total_loss / max(total, 1):.4f}, "
                    f"Accuracy: {total_acc / max(total, 1):.4f}"
                )

    # ------------------------------------------------------------------
    # Rebuttal instrumentation: stage timers (E3), cheap class statistics
    # (E5) and drift-map diagnostics (E9). Every entry point is gated by a
    # config key that defaults to the current behaviour.
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_now():
        '''perf_counter after a CUDA synchronize: the one clock behind every
        timer in this file (stage timers and the [MergeStat] merge_time).'''
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _tic(self):
        '''Start of a stage timer: _sync_now() when config stage_timing is on,
        else None (then the matching _toc is a no-op), so a timed call site is
        `t0 = self._tic(); ...; self._toc(t0, "stage", k=v)` with no branch.'''
        if self._config.get("stage_timing", False):
            return self._sync_now()
        return None

    def _toc(self, t0, stage, **extra):
        if t0 is None:
            return
        self._log_timing(stage, self._sync_now() - t0, **extra)

    def _log_timing(self, stage, sec, **extra):
        '''One '[Timing] task {t} {stage}={sec:.3f}s k=v ...' line and, when
        config stage_timing_csv is set, one appended CSV row
        (TIMING_CSV_HEADER: dataset, seed, task, stage, seconds, extra).
        Stages: train (cached=0|1), merge (rule), drift (whole fit_drift_map),
        drift_features (n, d), drift_solve (n, d, solver), transport (K, mode,
        cov_mode), stats (classes), align (classes, epochs), eval (classes),
        driftdiag (E9 overhead).'''
        extras = " ".join(f"{k}={v}" for k, v in extra.items())
        logging.info(f"[Timing] task {self._cur_task} {stage}={sec:.3f}s"
                     + (f" {extras}" if extras else ""))
        path = self._config.get("stage_timing_csv", None)
        if path:
            csv_append(path, TIMING_CSV_HEADER, [{
                "dataset": self._config.get("dataset_name", ""),
                "seed": self._config.get("seed", ""),
                "task": self._cur_task, "stage": stage,
                "seconds": f"{sec:.6f}", "extra": extras,
            }])

    def _cheap_dists(self, cov_mode, device):
        '''Sampling distributions of align() for the cheap statistics
        (covstats.make_dist): 'diag' -> Independent(Normal(mu, sqrt(var))),
        'lowrank' -> LowRankMultivariateNormal(mu, U, var), 'shared' ->
        MultivariateNormal(mu, scale_tril=L) with ONE Cholesky factor of the
        pooled covariance per align() call. align() only calls .sample() on
        them, so the rest of the alignment loop is untouched.'''
        shared_L = None
        if cov_mode == "shared":
            shared_L = shared_scale_tril(self._shared_scatter, self._shared_n).to(device)
        dists = []
        for c in range(self._total_classes):
            mu = self._class_means[c]
            if cov_mode == "diag":
                dists.append(make_dist("diag", mu, var=self._class_vars[c], device=device))
            elif cov_mode == "lowrank":
                dists.append(make_dist("lowrank", mu, var=self._class_vars[c],
                                       U=self._class_lr_U[c], device=device))
            else:
                dists.append(make_dist("shared", mu, shared_L=shared_L, device=device))
        return dists

    # -- E9 drift-map diagnostics (config drift_diag; evaluation only) --------

    def _diag_snapshot(self, n=None, clone=True):
        '''The stored statistics of classes [0, n) (default: the old classes)
        in the current covariance mode as a dict {mode, means, covs | vars |
        vars + U | shared}; clones by default (diagnostic storage only).'''
        if n is None:
            n = self._known_classes
        mode = self._config.get("train_ca_cov_mode", "class")
        take = (lambda x: x.clone()) if clone else (lambda x: x)
        snap = {"mode": mode, "means": take(self._class_means[:n])}
        if mode in ("class", "shared_corr"):
            snap["covs"] = take(self._class_covs[:n])
        elif mode == "diag":
            snap["vars"] = take(self._class_vars[:n])
        elif mode == "lowrank":
            snap["vars"] = take(self._class_vars[:n])
            snap["U"] = take(self._class_lr_U[:n])
        else:
            snap["shared"] = shared_cov(self._shared_scatter, self._shared_n)
        return snap

    @staticmethod
    def _diag_cov_full(snap, c):
        '''(d, d) covariance of class c implied by a _diag_snapshot
        (covstats.to_full_cov on the mode's representation).'''
        mode = snap["mode"]
        if mode in ("class", "shared_corr"):
            return snap["covs"][c]
        if mode == "diag":
            return to_full_cov("diag", var=snap["vars"][c])
        if mode == "lowrank":
            return to_full_cov("lowrank", U=snap["U"][c], var=snap["vars"][c])
        return snap["shared"]

    def _diag_store_raw(self):
        '''Never-transported copy of every class's statistics as fitted at its
        own task (the 'raw' = no-correction-ever curve of the E9 diagnostics):
        grows _diag_raw_means (C, d) / _diag_raw_covs (C, d, d) with the rows
        [known, total) of the statistics just computed. Diagnostic storage
        only (a second (C, d, d) CPU tensor, not part of the method's memory
        accounting).'''
        total, d = self._total_classes, self.model.feature_dim
        if not hasattr(self, "_diag_raw_means"):
            self._diag_raw_means = torch.zeros((total, d))
            self._diag_raw_covs = torch.zeros((total, d, d))
        else:
            new_means = torch.zeros((total, d))
            new_means[: self._known_classes] = self._diag_raw_means
            self._diag_raw_means = new_means
            new_covs = torch.zeros((total, d, d))
            new_covs[: self._known_classes] = self._diag_raw_covs
            self._diag_raw_covs = new_covs
        live = self._diag_snapshot(total, clone=False)
        for c in range(self._known_classes, total):
            self._diag_raw_means[c] = live["means"][c]
            self._diag_raw_covs[c] = self._diag_cov_full(live, c)

    @torch.no_grad()
    def drift_diagnostics(self, W, before):
        '''Rebuttal E9: how well the stored old-class Gaussians match the
        deployed extractor M_t, before and after the drift transport, next to
        the never-corrected statistics ('raw') and a one-step comparison
        (identity / the fitted W / an oracle W fitted on the old-class pairs
        themselves with the same ridge anchor and lambda, driftsolve).

        Evaluation only: the old-class images are never stored by the method,
        nothing is written back to the class statistics, and the global RNG
        (CPU + CUDA) is restored afterwards (torch.random.fork_rng, a private
        np.random.RandomState for the subsample, a private torch.Generator for
        the loader), so the accuracy trajectory equals a run without it.

        Keys (read only when drift_diag is True): drift_diag_source ('train' =
        the same images and deterministic transform as
        compute_multivariate_normal, so est-vs-ref isolates the transport
        error; 'test' = held-out images), drift_diag_max_per_class (100; 0 =
        all), drift_diag_seed (0), drift_diag_onestep (True), drift_diag_kl
        (False), drift_diag_csv (None). Logs one '[DriftDiag]' line per task
        and appends one CSV row per (task, old-task group) plus 'all'
        (DRIFTDIAG_CSV_HEADER). Errors: diagnostics.gaussian_errors.'''
        K = self._known_classes
        if before is None or K == 0:
            return
        if not hasattr(self, "_diag_raw_means") or self._diag_raw_means.size(0) < K:
            logging.warning("[DriftDiag] raw statistics missing; skipping")
            return
        cfg = self._config
        src = cfg.get("drift_diag_source", "train")
        max_pc = int(cfg.get("drift_diag_max_per_class", 100))
        seed = int(cfg.get("drift_diag_seed", 0))
        onestep = bool(cfg.get("drift_diag_onestep", True))
        use_kl = bool(cfg.get("drift_diag_kl", False))
        csv_path = cfg.get("drift_diag_csv", None)
        t = self._cur_task
        after = self._diag_snapshot()
        raw = {"mode": "class", "means": self._diag_raw_means[:K],
               "covs": self._diag_raw_covs[:K]}

        # Features of the old classes under M_t (and M_{t-1} for the one-step
        # view) from a seeded per-class subsample; private RNG everywhere.
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            ds = self.data_manager.get_dataset(np.arange(0, K), source=src, mode="test")
            labels = np.asarray(ds.labels)
            if max_pc > 0:
                rs = np.random.RandomState(seed)
                keep = []
                for c in range(K):
                    idx = np.where(labels == c)[0]
                    if len(idx) > max_pc:
                        idx = np.sort(rs.choice(idx, max_pc, replace=False))
                    keep.append(idx)
                keep = np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)
                ds = DummyDataset(ds.images[keep], labels[keep], ds.trsf, ds.use_path)
            gen = torch.Generator()
            gen.manual_seed(seed)
            loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4,
                                worker_init_fn=seed_worker, generator=gen)
            old_model = self._get_prev_feature_model() if onestep else None
            self.model.eval()
            F_cur, F_prev, ys = [], [], []
            for _, (_, _, x, y) in enumerate(loader):
                x = x.cuda()
                F_cur.append(self.model.get_features(x).cpu())
                if old_model is not None:
                    F_prev.append(old_model.get_features(x).cpu())
                ys.append(torch.as_tensor(y))
            if old_model is not None:
                self._release_prev_feature_model(old_model)
        if not ys:
            logging.warning(f"[DriftDiag] task {t}: no old-class images in source {src!r}")
            return
        F_cur = torch.cat(F_cur, dim=0).float()
        F_prev = torch.cat(F_prev, dim=0).float() if onestep else None
        y = torch.cat(ys).numpy()
        n_img = int(len(y))

        # One-step maps: the fitted W of this task and the oracle fitted on the
        # old-class pairs with the same anchor and lambda (dense solve).
        Wd = W.detach().cpu().double()
        d = Wd.size(0)
        eye_d = torch.eye(d, dtype=torch.double)
        lam = getattr(self, "_drift_lambda_used", cfg.get("train_drift_lambda", 1e2))
        ridge_mode = cfg.get("train_drift_ridge", "zero")
        W_or = None
        if onestep:
            W_or = solve_drift(F_prev.double(), F_cur.double(), lam,
                               ridge=ridge_mode, solver="dense")

        rows = []
        for c in range(K):
            idx = torch.from_numpy(np.where(y == c)[0])
            if idx.numel() < 2:
                continue
            mu_ref, cov_ref = fit_full(F_cur[idx])
            row = {"group": int(self._cls_to_task_idx.get(c, -1)), "n_img": int(idx.numel())}
            for name, snap in (("raw", raw), ("before", before), ("after", after)):
                err = gaussian_errors(snap["means"][c], self._diag_cov_full(snap, c),
                                      mu_ref, cov_ref, kl=use_kl)
                row[f"mean_rel_{name}"] = err["mean_rel"]
                row[f"mean_cos_{name}"] = err["mean_cos"]
                row[f"cov_relF_{name}"] = err["cov_relF"]
                if use_kl:
                    row[f"symkl_{name}"] = err["symkl"]
            if onestep:
                mu_p, cov_p = fit_full(F_prev[idx])          # true stats under M_{t-1}
                e_i = gaussian_errors(mu_p, cov_p, mu_ref, cov_ref)
                mu_w, cov_w = transport_gaussian(mu_p, cov_p, Wd)
                e_w = gaussian_errors(mu_w, cov_w, mu_ref, cov_ref)
                mu_o, cov_o = transport_gaussian(mu_p, cov_p, W_or)
                e_o = gaussian_errors(mu_o, cov_o, mu_ref, cov_ref)
                row.update({
                    "os_mean_rel_I": e_i["mean_rel"], "os_mean_rel_W": e_w["mean_rel"],
                    "os_mean_rel_oracle": e_o["mean_rel"],
                    "os_cov_relF_I": e_i["cov_relF"], "os_cov_relF_W": e_w["cov_relF"],
                    "os_cov_relF_oracle": e_o["cov_relF"],
                })
            rows.append(row)
        if not rows:
            logging.warning(f"[DriftDiag] task {t}: no old class with >= 2 images")
            return

        # Means over classes per old-task group and overall ('all').
        metric_keys = [k for k in DRIFTDIAG_CSV_HEADER
                       if k.startswith(("mean_", "cov_", "symkl_", "os_"))]

        def _agg(rs):
            out = {"n_classes": len(rs), "n_img": sum(r["n_img"] for r in rs)}
            for k in metric_keys:
                vals = [r[k] for r in rs if k in r]
                if vals:
                    out[k] = sum(vals) / len(vals)
            return out

        groups = [("all", rows)] + [
            (str(gid), [r for r in rows if r["group"] == gid])
            for gid in sorted({r["group"] for r in rows})]
        agg = {name: _agg(rs) for name, rs in groups}
        a = agg["all"]
        msg = (f"[DriftDiag] task {t} K={K} n_img={n_img} src={src} | "
               f"mean_rel raw={a['mean_rel_raw']:.4f} before={a['mean_rel_before']:.4f} "
               f"after={a['mean_rel_after']:.4f} | "
               f"cov_relF raw={a['cov_relF_raw']:.4f} before={a['cov_relF_before']:.4f} "
               f"after={a['cov_relF_after']:.4f}")
        if onestep:
            msg += (f" | onestep mean_rel I={a['os_mean_rel_I']:.4f} W={a['os_mean_rel_W']:.4f} "
                    f"oracle={a['os_mean_rel_oracle']:.4f} "
                    f"cov_relF I={a['os_cov_relF_I']:.4f} W={a['os_cov_relF_W']:.4f} "
                    f"oracle={a['os_cov_relF_oracle']:.4f}")
        logging.info(msg)

        if csv_path:
            base = {
                "dataset": cfg.get("dataset_name", ""), "seed": cfg.get("seed", ""),
                "task": t, "W_minus_I_F": (Wd - eye_d).norm().item(),
                "n_fit_new": getattr(self, "_drift_fit_n", ""), "n_fit_old": n_img,
            }
            if W_or is not None:
                base["Wor_minus_I_F"] = (W_or - eye_d).norm().item()
                base["W_minus_Wor_F"] = (Wd - W_or).norm().item()
            csv_append(csv_path, DRIFTDIAG_CSV_HEADER,
                       [dict(base, group=name, **vals) for name, vals in agg.items()])

    # ------------------------------------------------------------------
    # E2-LoRA native classifier stage (port of e2lora.py:_compute_class_mean
    # + _stage2_compact_classifier; logit_norm disabled as in their IN-R cfg)
    # ------------------------------------------------------------------

    def _e2_compute_class_stats(self):
        total, d = self._total_classes, self.model.feature_dim
        if not hasattr(self, "_e2_means"):
            self._e2_means = torch.zeros((total, d), dtype=torch.double)
            self._e2_covs = torch.zeros((total, d, d))
        else:
            new_means = torch.zeros((total, d), dtype=torch.double)
            new_means[: self._known_classes] = self._e2_means
            self._e2_means = new_means
            new_covs = torch.zeros((total, d, d))
            new_covs[: self._known_classes] = self._e2_covs
            self._e2_covs = new_covs

        self.model.eval()
        for cls_idx in range(self._known_classes, self._total_classes):
            proto_set = self.data_manager.get_dataset(
                np.arange(cls_idx, cls_idx + 1), source="train", mode="test")
            loader = DataLoader(proto_set, batch_size=256, shuffle=False,
                                num_workers=4, worker_init_fn=seed_worker, generator=g)
            feats = []
            with torch.no_grad():
                for _, (_, _, x, _) in enumerate(loader):
                    feats.append(self.model.get_features(x.cuda()).cpu())
            feats = torch.cat(feats).double()
            self._e2_means[cls_idx] = feats.mean(dim=0)
            self._e2_covs[cls_idx] = (
                torch.cov(feats.T) + torch.eye(d, dtype=torch.double) * 2e-4
            ).float()
        logging.info(f"[E2 CA] Class stats up to class {self._total_classes - 1}")

    def _e2_native_ca(self, task_size):
        logging.info(f"[E2 CA] Compact classifier, task {self._cur_task}")
        classifier = self.model.classifier
        for p in classifier.parameters():
            p.requires_grad = True
        ca_epochs = self._config.get("e2_ca_epochs", 3)
        lr = self._config.get("e2_ca_lr", 1e-2)
        wd = self._config.get("train_weight_decay", 5e-4)
        optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ca_epochs)

        device = next(classifier.parameters()).device
        crct_num = self._total_classes
        n_per_cls = 256
        self.model.eval()
        for epoch in range(ca_epochs):
            sampled_data, sampled_label = [], []
            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                mean = (self._e2_means[c_id].to(device) * (0.9 + decay)).float()
                cov = self._e2_covs[c_id].to(device)
                m = MultivariateNormal(mean, cov)
                sampled_data.append(m.sample(sample_shape=(n_per_cls,)))
                sampled_label.extend([c_id] * n_per_cls)
            inputs = torch.cat(sampled_data).float().to(device)
            targets = torch.tensor(sampled_label).long().to(device)
            perm = torch.randperm(inputs.size(0), device=device)
            inputs, targets = inputs[perm], targets[perm]

            losses = 0.0
            for it in range(crct_num):
                inp = inputs[it * n_per_cls:(it + 1) * n_per_cls]
                tgt = targets[it * n_per_cls:(it + 1) * n_per_cls]
                logits = classifier(inp)
                loss = F.cross_entropy(logits[:, :crct_num], tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            scheduler.step()
            logging.info(f"[E2 CA] Epoch {epoch + 1}/{ca_epochs}, Loss {losses / crct_num:.4f}")

    def _apply_inference_cap(self):
        """SN-GAN-style deployment normalization: cap sigma_max of the DEPLOYED
        (merged) LoRA product at infer_speccap, in memory only -- stored
        checkpoints stay unconstrained; training never sees the cap. Applied
        after every merge (and after task-0 training), so all downstream
        consumers (stats, drift, align, eval) use the normalized extractor.
        Division rescale if sigma > c (divide-always would AMPLIFY early
        states whose natural sigma < 1)."""
        self._cap_model(self.model)

    def _cap_model(self, target):
        from backbones import peft_qkv_modules
        scale = self._config["model_lora_alpha"] / self._config["model_lora_r"]
        cap = float(self._config["infer_speccap"])
        capped = 0
        with torch.no_grad():
            for name, qkv in peft_qkv_modules(target.backbone).items():
                A = qkv.lora_A["default"].weight
                B = qkv.lora_B["default"].weight
                sigma = torch.linalg.matrix_norm(scale * (B @ A), ord=2)
                if sigma > cap:
                    B.data.mul_(cap / sigma)
                    capped += 1
        logging.info(f"[InferCap] c={cap}: capped {capped}/12 layers "
                     f"(deployment-only normalization)")

    # ------------------------------------------------------------------
    # Merge (seqft only; exp19.merge with kspectral/pspectral)
    # ------------------------------------------------------------------

    # -- rebuttal helpers (default-off keys merge_log_stats / merge_alpha_rule) --

    @staticmethod
    def _merge_tic(enabled):
        '''perf_counter after a CUDA sync when merge_log_stats is on (the
        spectral merge runs its SVDs on the GPU); None otherwise, so the
        default path neither syncs nor times anything. Same synchronised
        clock as the stage timers (_sync_now; one timing helper, plan rule 1).'''
        return Learner._sync_now() if enabled else None

    @staticmethod
    def _merge_toc(t0):
        return Learner._sync_now() - t0

    def _prev_deployed_checkpoint(self):
        '''Checkpoint of the module deployed after task t-1 for a merging run:
        the task-0 state at t == 1 (no merged file exists for task 0), else
        M_{t-1}. Same rule as the merge branches and _get_prev_feature_model
        (merge-less runs use backbone_checkpoint(t-1) instead).'''
        if self._cur_task == 1:
            return self.backbone_checkpoint(0)
        return self.merged_checkpoint(self._cur_task - 1)

    def _lora_scale(self):
        '''PEFT scaling of the effective update, alpha / r (2 at r=64, alpha=128).'''
        return lora_scale(self._config["model_lora_alpha"], self._config["model_lora_r"])

    @staticmethod
    def _norm_fields(agg):
        '''Shared field block of the [MergeStat] / [StepNorm] lines from a
        merge_ops.step_norms aggregate: sigma_max of the lora_A step and of the
        effective-weight step scale * (B A - B' A'), plus its Frobenius norm.'''
        return (f"A_sigma_max_mean={agg['sigma1_A_mean']:.5g} "
                f"A_sigma_max_max={agg['sigma1_A_max']:.5g} | "
                f"eff_sigma_max_mean={agg['sigma1_eff_mean']:.5g} "
                f"eff_sigma_max_max={agg['sigma1_eff_max']:.5g} "
                f"eff_fro_mean={agg['fro_eff_mean']:.5g}")

    @torch.no_grad()
    def _log_merge_stat(self, w_prev, w_new, merged, merge_time, alpha=None, op=None):
        '''One '[MergeStat]' line per merge (config merge_log_stats; plan S4):
        realised step norms w_prev -> merged, the raw-update norms
        w_prev -> w_new, the kept rank k(p) of the raw lora_A updates at
        pspectral_p (spectral branches only, with the realised alpha and the
        update op) and the merge wall-clock. Statistics: merge_ops.step_norms
        (k follows the exact helper.truncate_energy rule; k = 0 marks a
        degenerate pair). The SOTA / classical branches log the reduced line
        without the alpha / k fields.'''
        p = self._config.get("pspectral_p", 0.9)
        scale = self._lora_scale()
        real = step_norms(w_prev, merged, scale, p)["agg"]
        raw = step_norms(w_prev, w_new, scale, p)["agg"]
        head = f"[MergeStat] task {self._cur_task}"
        if real.get("n_pairs", 0) == 0:
            logging.info(f"{head} no LoRA factor pairs | merge_time={merge_time:.3f}s")
            return
        if alpha is not None:
            head += f" alpha={alpha}"
            if op is not None:
                head += f" op={op}"
            head += f" k_mean={raw['k_mean']:.2f} k_min={raw['k_min']} k_max={raw['k_max']}"
        logging.info(
            f"{head} | {self._norm_fields(real)} | "
            f"raw_sigma_max_mean={raw['sigma1_A_mean']:.5g} "
            f"raw_eff_sigma_max_mean={raw['sigma1_eff_mean']:.5g} | "
            f"merge_time={merge_time:.3f}s")

    @torch.no_grad()
    def _log_step_norm(self):
        '''One '[StepNorm]' line per task for merge-less seqft runs
        (train_merge == 'none' with merge_log_stats): norms of the raw update
        W_t - W_{t-1}, which IS the deployed step there. The E10 L2M pilot reads
        eff_sigma_max from it next to the [MergeStat] line of the truncation
        run (same fields, same merge_ops.step_norms arithmetic).'''
        w_prev = torch.load(self.backbone_checkpoint(self._cur_task - 1))
        w_new = torch.load(self.backbone_checkpoint(self._cur_task))
        agg = step_norms(w_prev, w_new, self._lora_scale(),
                         self._config.get("pspectral_p", 0.9))["agg"]
        if agg.get("n_pairs", 0) == 0:
            logging.info(f"[StepNorm] task {self._cur_task} no LoRA factor pairs")
            return
        logging.info(
            f"[StepNorm] task {self._cur_task} k_mean={agg['k_mean']:.2f} "
            f"k_min={agg['k_min']} k_max={agg['k_max']} | {self._norm_fields(agg)}")

    def _adaptive_alpha(self, class_range):
        '''Per-task damping alpha for the pspectral merge (rebuttal E7).

        merge_alpha_rule:
          'specnorm'  : s_t = mean over the LoRA pairs of sigma_max of the raw
                        update W_t - M_{t-1}, in the effective-weight space
                        (merge_alpha_stat 'eff', default) or the lora_A space
                        ('A'); needs no data (merge_ops.step_norms on the
                        cached states).
          'featshift' : s_t = relative feature shift of the current task's
                        images between M_{t-1} and W_t (_feature_shift).
        alpha_t = clip(merge_alpha_param / s_t, merge_alpha_min, 1)
        (merge_ops.alpha_from_stat). Logs the realised value; never writes
        train_merge_alpha, so merged_checkpoint(t-1) stays resolvable.
        '''
        rule = self._config.get("merge_alpha_rule", "fixed")
        if rule == "fixed":
            return None
        if self._config.get("train_merge") != "pspectral":
            raise ValueError("merge_alpha_rule requires train_merge='pspectral'")
        param = float(self._config.get("merge_alpha_param", 1.0))
        alpha_min = float(self._config.get("merge_alpha_min", 0.05))
        if rule == "specnorm":
            stat = self._config.get("merge_alpha_stat", "eff")
            if stat not in ("eff", "A"):
                raise ValueError(f"unknown merge_alpha_stat {stat!r} (expected 'eff' or 'A')")
            with torch.no_grad():
                w_prev = torch.load(self._prev_deployed_checkpoint())
                w_new = torch.load(self.backbone_checkpoint(self._cur_task))
                agg = step_norms(w_prev, w_new, self._lora_scale(),
                                 self._config.get("pspectral_p", 0.9))["agg"]
            key = "sigma1_eff_mean" if stat == "eff" else "sigma1_A_mean"
            s_t = float(agg[key]) if agg.get("n_pairs", 0) else 0.0
        elif rule == "featshift":
            stat = "shift"
            s_t = self._feature_shift(class_range)
        else:
            raise ValueError(f"unknown merge_alpha_rule {rule!r}")
        alpha = alpha_from_stat(s_t, param, alpha_min, 1.0)
        logging.info(f"[Merging] adaptive alpha rule={rule} stat={stat} "
                     f"s_t={s_t:.6g} -> alpha={alpha:.4f}")
        return alpha

    @torch.no_grad()
    def _feature_shift(self, class_range):
        '''Relative feature shift of the current task's training images
        between the previously deployed extractor M_{t-1} (reloaded through
        _get_prev_feature_model) and the freshly trained W_t held by
        self.model:
            s_t = mean_i ||f_t(x_i) - f_{t-1}(x_i)|| / mean_i ||f_{t-1}(x_i)||.
        Deterministic transform (mode='test'), no shuffling and a private
        torch.Generator, so neither the global RNG nor the module generator g
        is touched. Logs '[FeatShift] task t shift=.. n=..' and returns s_t.'''
        shift_set = self.data_manager.get_dataset(class_range, source="train", mode="test")
        gen = torch.Generator()
        gen.manual_seed(0)
        loader = DataLoader(shift_set, batch_size=256, shuffle=False,
                            num_workers=4, worker_init_fn=seed_worker, generator=gen)
        old_model = self._get_prev_feature_model()
        self.model.eval()
        num = den = 0.0
        n = 0
        for _, (_, _, x, _) in enumerate(loader):
            x = x.cuda()
            f_old = old_model.get_features(x).float()
            f_new = self.model.get_features(x).float()
            num += (f_new - f_old).norm(dim=1).sum().item()
            den += f_old.norm(dim=1).sum().item()
            n += x.size(0)
        self._release_prev_feature_model(old_model)
        s_t = num / den if den > 0.0 else 0.0
        logging.info(f"[FeatShift] task {self._cur_task} shift={s_t:.6g} n={n}")
        return s_t

    def _log_merge_rule_state(self, method):
        """One '[Merging] opcm ...' or '[Merging] nufilt ...' line per merge (E15a).

        Reads the per-step statistics that merging_sota.opcm_pair / nufilt_pair
        leave in self._merge_rule_state (lam_t, kept_frac, ...; filtered_frac,
        n_adapted, ...). Called only for those two rules; the five older SOTA
        rules write nothing there and log nothing here.
        """
        st = self._merge_rule_state

        def fmt(v):
            return f"{v}" if isinstance(v, (int, np.integer)) else f"{float(v):.4g}"

        if method == "opcm":
            fields = [f"alpha={self._config.get('merge_opcm_alpha', 0.5)}",
                      f"lam_t={float(st.get('lam', float('nan'))):.4f}",
                      f"lam_rule={self._config.get('merge_opcm_lam', 'adaptive')}"]
            optional = ("lam_prev", "kept_frac", "r_alpha_mean", "norm_mean")
        else:
            fields = [f"rp={self._config.get('merge_nufilt_rp', 128)}",
                      f"adapt={int(bool(self._config.get('merge_nufilt_adapt', False)))}"]
            optional = ("filtered_frac", "rp_eff_mean", "n_adapted",
                        "adapt_loss_first", "adapt_loss_last")
        fields += [f"{k}={fmt(st[k])}" for k in optional if st.get(k) is not None]
        logging.info(f"[Merging] {method} " + " ".join(fields))

    def _spectral_variant(self):
        return self._config.get(
            "spectral_variant", self._config.get("pspectral_variant", "only_A")
        )

    def merge(self, alpha_override=None):
        '''Merge the task-t LoRA state into the deployed module (M_t).

        alpha_override (rebuttal E7): realised damping alpha for THIS task in
        place of config train_merge_alpha (pspectral branch only); the config
        is never mutated, so merged_checkpoint(t-1) stays resolvable.
        merge_log_stats (default False) adds one '[MergeStat]' line per task.
        '''
        logging.info(f"[Merging] Task {self._cur_task}")
        if alpha_override is not None and self._config["train_merge"] != "pspectral":
            raise ValueError("alpha_override is supported by the pspectral merge only")

        reset_merge = self._config.get("reset_merge", False)
        if not reset_merge:
            saved = self.merged_checkpoint(self._cur_task)
            if os.path.exists(saved):
                logging.info(f"[Merging] Load merged checkpoint for task {self._cur_task}")
                self.load_backbone(torch.load(saved))
                return

        if self._cur_task > 0:
            model_use_norm = self._config.get("model_use_norm", False)
            method = self._config["train_merge"]

            if method in ("kspectral", "pspectral"):
                # Incremental spectral merge of two consecutive LoRA states.
                w_prev_ckpt = (
                    self.backbone_checkpoint(0)
                    if self._cur_task == 1
                    else self.merged_checkpoint(self._cur_task - 1)
                )
                w_prev = torch.load(w_prev_ckpt)
                w_new = torch.load(self.backbone_checkpoint(self._cur_task))
                variant = self._spectral_variant()
                # Realised alpha: the config value (same object, so the log line
                # below is unchanged) unless an adaptive rule supplied one.
                alpha = (self._config["train_merge_alpha"] if alpha_override is None
                         else float(alpha_override))
                update_op = self._config.get("merge_update_op", "trunc")
                log_stats = self._config.get("merge_log_stats", False)
                if method == "pspectral":
                    logging.info(
                        f"[Merging] pspectral {variant} "
                        f"(p={self._config['pspectral_p']}, "
                        f"alpha={alpha}) of "
                        f"{os.path.basename(w_prev_ckpt)} + task {self._cur_task}"
                    )
                    t0 = self._merge_tic(log_stats)
                    backbone_params = pspectral_merging(
                        w_prev, w_new,
                        p=self._config["pspectral_p"],
                        alpha=alpha,
                        variant=variant,
                        update_op=update_op,
                    )
                else:
                    logging.info(
                        f"[Merging] kspectral {variant} "
                        f"(k={self._config['train_merge_k']}, "
                        f"alpha={self._config['train_merge_alpha']}) of "
                        f"{os.path.basename(w_prev_ckpt)} + task {self._cur_task}"
                    )
                    t0 = self._merge_tic(log_stats)
                    backbone_params = kspectral_merging(
                        w_prev, w_new,
                        k=self._config["train_merge_k"],
                        alpha=self._config["train_merge_alpha"],
                        variant=variant,
                    )
                if log_stats:
                    self._log_merge_stat(
                        w_prev, w_new, backbone_params, self._merge_toc(t0),
                        alpha=alpha, op=update_op if method == "pspectral" else None)
                self.load_backbone(backbone_params, load_norm=model_use_norm)
                logging.info(f"[Merging] Saving merged checkpoint for task {self._cur_task}")
                torch.save(
                    self.model.get_backbone_trainable_params(),
                    self.merged_checkpoint(self._cur_task),
                )
                return

            # SOTA merging baselines (merging_sota.py): sequential pairwise
            # merge of prev deployed state + current task state, same
            # protocol as the spectral/classical incremental branches.
            from merging_sota import SOTA_METHODS, sota_merge
            if method in SOTA_METHODS:
                base_params = torch.load(self.backbone_checkpoint(-1), map_location="cpu")
                prev_ckpt = (
                    self.backbone_checkpoint(0)
                    if self._cur_task == 1
                    else self.merged_checkpoint(self._cur_task - 1)
                )
                w_prev = torch.load(prev_ckpt, map_location="cpu")
                w_new = torch.load(self.backbone_checkpoint(self._cur_task), map_location="cpu")
                logging.info(
                    f"[Merging] SOTA {method} (lamb={self._config['train_merge_coef']}) "
                    f"of {os.path.basename(prev_ckpt)} + task {self._cur_task}")
                log_stats = self._config.get("merge_log_stats", False)
                t0 = self._merge_tic(log_stats)
                backbone_params = sota_merge(
                    base_params, w_prev, w_new, method,
                    lamb=self._config["train_merge_coef"],
                    dare_q=self._config.get("merge_dare_q", 0.9),
                    della_q=self._config.get("merge_della_q", 0.5),
                    bc_beta=self._config.get("merge_bc_beta", 0.85),
                    bc_gamma=self._config.get("merge_bc_gamma", 0.01),
                    knots_topk=self._config.get("merge_knots_topk", 50.0),
                    # Rebuttal E15a (OPCM / NUFILT; ignored by the five older
                    # rules): their parameters, the cross-task state dict and
                    # the step index t that opcm_pair's lam_t schedule needs.
                    opcm_alpha=self._config.get("merge_opcm_alpha", 0.5),
                    opcm_lam=self._config.get("merge_opcm_lam", "adaptive"),
                    nufilt_rp=self._config.get("merge_nufilt_rp", 128),
                    nufilt_rv=self._config.get("merge_nufilt_rv", 8),
                    nufilt_rl=self._config.get("merge_nufilt_rl", 64),
                    nufilt_adapt=self._config.get("merge_nufilt_adapt", False),
                    nufilt_iters=self._config.get("merge_nufilt_iters", 50),
                    nufilt_lr=self._config.get("merge_nufilt_lr", 1e-3),
                    state=self._merge_rule_state,
                    task=self._cur_task,
                )
                if log_stats:
                    self._log_merge_stat(w_prev, w_new, backbone_params, self._merge_toc(t0))
                if method in ("opcm", "nufilt"):
                    self._log_merge_rule_state(method)
                self.load_backbone(backbone_params, load_norm=model_use_norm)
                logging.info(f"[Merging] Saving merged checkpoint for task {self._cur_task}")
                torch.save(
                    self.model.get_backbone_trainable_params(),
                    self.merged_checkpoint(self._cur_task),
                )
                return

            # Task-vector merge (helper.merge) on CPU.
            base_params = torch.load(self.backbone_checkpoint(-1), map_location="cpu")
            logging.info(
                f"[Merging] {sum(p.numel() for p in base_params.values()):,} total parameters"
            )

            if self._config.get("train_merge_incremental", False):
                # Continuous merge: prev MERGED state + current task ckpt.
                # No merged ckpt exists for task 0 (task-0 state IS the
                # deployed state), so task 1 merges against the task-0 ckpt —
                # same convention as the spectral branch above.
                prev_ckpt = (
                    self.backbone_checkpoint(0)
                    if self._cur_task == 1
                    else self.merged_checkpoint(self._cur_task - 1)
                )
                task_params = [
                    torch.load(prev_ckpt, map_location="cpu"),
                    torch.load(self.backbone_checkpoint(self._cur_task), map_location="cpu"),
                ]
            else:
                task_params = [
                    torch.load(self.backbone_checkpoint(t), map_location="cpu")
                    for t in range(self._cur_task + 1)
                ]
            logging.info(f"[Merging] Merging {len(task_params)} task checkpoints")

            log_stats = self._config.get("merge_log_stats", False)
            t0 = self._merge_tic(log_stats)
            backbone_params = merge(
                base_params, task_params,
                method=method,
                lamb=self._config["train_merge_coef"],
                topk=self._config["train_merge_topk"],
            )
            if log_stats:
                # Step from the previously deployed state (task-0 state at
                # t = 1, else M_{t-1}) to the merged state; raw update = W_t.
                merge_time = self._merge_toc(t0)
                w_prev = torch.load(self._prev_deployed_checkpoint(), map_location="cpu")
                self._log_merge_stat(w_prev, task_params[-1], backbone_params, merge_time)

            self.load_backbone(backbone_params, load_norm=model_use_norm)

        logging.info(f"[Merging] Saving merged checkpoint for task {self._cur_task}")
        torch.save(
            self.model.get_backbone_trainable_params(),
            self.merged_checkpoint(self._cur_task),
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _ckpt_dir(self):
        d = self._config.get("checkpoint_dir", None)
        if d is None:
            d = os.path.join(CHECKPOINT_ROOT, self._method)
        os.makedirs(d, exist_ok=True)
        return d

    def prefix(self):
        # seqft keeps the exp19-compatible prefix so existing checkpoints_lora_*
        # training caches can be reused by pointing checkpoint_dir at them.
        parts = [
            str(self._config["seed"]),
            self._config["dataset_name"],
            self._config["model_backbone"] if self._is_seqft else self._method,
        ]
        train_prefix = self._config.get("train_prefix", "")
        if train_prefix:
            parts.append(train_prefix)
        return "_".join(parts)

    def backbone_checkpoint(self, task=-1):
        filename = f"{self.prefix()}_backbone" + (
            f"_{task}.pt" if task >= 0 else "_base.pt"
        )
        return os.path.join(self._ckpt_dir(), filename)

    def state_checkpoint(self, task):
        """Merge-family post-task trainable state (folded)."""
        return os.path.join(self._ckpt_dir(), f"{self.prefix()}_state_{task}.pt")

    def e2_checkpoint(self, task):
        """E2-LoRA raw structural state (pairs + eigenvalues, fp16)."""
        return os.path.join(self._ckpt_dir(), f"{self.prefix()}_e2state_{task}.pt")

    def covs_checkpoint(self, task):
        return os.path.join(self._ckpt_dir(), f"{self.prefix()}_covs_{task}.pt")

    def classifier_checkpoint(self, task):
        return os.path.join(self._ckpt_dir(), f"{self.prefix()}_classifier_{task}.pt")

    def merged_checkpoint(self, task):
        # Tag rule shared with the run_config_sweep cleanup block (merge_tag).
        return os.path.join(
            self._ckpt_dir(), f"{self.prefix()}_merged_{merge_tag(self._config)}_{task}.pt"
        )

    def load_backbone(self, backbone_params, load_norm=True, target=None):
        tgt = target if target is not None else self.model
        peft_params, norm_params = {}, {}
        for name, param in backbone_params.items():
            if name.startswith("norm."):
                norm_params[name[5:]] = param
            else:
                peft_params[name] = param
        tgt.backbone.load_state_dict(peft_params, strict=False)
        if norm_params and load_norm and tgt.norm is not None:
            tgt.norm.load_state_dict(norm_params, strict=True)


# ------------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------------

DATA_TABLE = {
    "cifar224": [(10, 10, 10)],   # CIFAR-100, 10 tasks x 10
    "imageneta": [(10, 20, 20)],
    "imagenetr": [(10, 20, 20)],
    "cub": [(10, 20, 20)],
    "cars": [(10, 16, 20)],
    # Domain-incremental (E2-LoRA/DUCT protocol): session = domain, labels
    # offset by session -> structurally identical to CIL for the learner.
    "officehome_dil": [(4, 65, 65)],
    "dndil": [(6, 345, 345)],
}

TOTAL_CLASSES = {"cifar224": 100, "imageneta": 200, "imagenetr": 200,
                 "cub": 200, "cars": 196,
                 "officehome_dil": 260, "dndil": 2070}

BASE_CONFIG = {
    "seed": 1993,
    "train_method": "seqft",
    "reset_train": False,
    "reset_merge": True,
    "train_stop_at_task": -1,
    # unified loop (defaults = our recipe)
    "train_epochs": 10,
    "train_iterations": None,
    "train_batch_size": 64,
    "train_optimizer": "sgd",
    "train_base_lr": 1e-2,
    "train_head_lr": None,
    "train_weight_decay": 5e-4,
    "train_scheduler": "cosine",
    "train_warmup_ratio": 0.1,
    "train_loss": "ce",
    # KD (composable with every recipe)
    "gamma_kd": 0.0,
    "kd_type": "feat",
    "kd_transform": "weaknonlinear",
    # backbone: seqft (PEFT)
    "model_backbone": "vit_base_patch16_224_lora",
    "model_outdim": 768,
    "model_use_norm": True,
    "model_lora_r": 64,
    "model_lora_alpha": 128,
    "model_lora_dropout": 0.0,
    "model_lora_target_modules": ["qkv"],
    "model_classifier": ["mlp"],
    "model_classifier_norm_layer": "ln",
    # backbone: merge family (reference-repo defaults)
    "lora_r": 4,
    "lora_scale": 1.0,
    "replace_final_norm": False,
    "weight_temp": 2.0,
    "weight_kind": "log1p",
    "weight_p": 1.0,
    "sgp_module": "dora",
    "nsp_eps": 0.05,
    "nsp_weight": 0.0,
    "cov_ema_decay": 0.9,
    # merge axis (seqft): our tuned recipe
    "train_merge": "pspectral",
    "spectral_variant": "only_A",
    "pspectral_p": 0.9,
    "train_merge_alpha": 0.3,
    "train_merge_k": 16,
    "train_merge_coef": 1.0,
    "train_merge_topk": 100,
    "train_merge_incremental": False,
    "cleanup_merged": True,
    # Stage 2: FIXED across recipes (our tuned values)
    "train_ca": True,
    "train_ca_epochs": 10,
    "train_ca_cov_mode": "class",
    # Shrinkage removed from the method (user decision 2026-08-27): class covs
    # are plain empirical + 1e-4 I jitter. Key kept for ablation rows only.
    "train_ca_cov_shrinkage": 0.0,
    "train_ca_robust_weight": 0.1,
    "train_ca_robust_impl": "batch_pairs",
    "train_ca_sampling": "nk",
    "train_ca_n_classes": 40,
    "train_ca_k_samples": 64,
    "train_ca_steps_per_epoch": 100,
    "train_drift": True,
    "train_drift_ridge": "identity",
    "train_drift_lambda": 10.0,
    "train_drift_transport": "full",
    # caching. fp32 keeps the cache byte-exact (cache-hit == fresh run, the
    # property exp19 sweeps rely on); "fp16" halves disk at ~1% feature error.
    "cache_backbone": None,   # None -> per-recipe default (off for full FT)
    "cache_dtype": "fp32",
}

# Per-recipe overrides applied on top of BASE_CONFIG. The default sweep keeps
# our training loop for every recipe (only the adaptation mechanism varies);
# lr is overridden only where required (full FT at lr 1e-2 collapses the
# pretrained features -- exp19 calibrated 3e-4 SGD; full_nsp uses the native
# 5e-6 AdamW because the gradient-projection recipe was tuned there).
RECIPE_PRESETS = {
    "seqft": {},
    "sgp_seqft": {},
    "nsp_seqft": {},
    "basic_lora": {},
    "sgp_lora": {},
    "nsp_lora": {},
    "e2lora": {
        # Their e2lora_inr_lora.json: lrate 0.01 * bcb_lrscale 0.05 for the
        # backbone (current pair's A), head 0.01, epochs 5, batch 64, wd 5e-4,
        # constant lr (milestones never fire), plain biased heads, no extra
        # feature norm (final norm affine reset to identity in the wrapper).
        "train_base_lr": 5e-4,
        "train_head_lr": 1e-2,
        "train_epochs": 5,
        "train_batch_size": 64,
        "train_weight_decay": 5e-4,
        "train_scheduler": "constant",
        "model_use_norm": False,
        "model_classifier_use_norm": False,
        "model_classifier_bias": True,
        "train_merge": "none",
        "e2_sd_weight": 0.25,
        "e2_ca_epochs": 3,
    },
    "inflora": {
        # github.com/liangyanshuo/InfLoRA configs/cifar100_inflora.json and
        # mimg10_inflora.json: Adam (0.9, 0.999) lr 5e-4 for LoRA-B and the
        # current head (one optimizer, tied lr), weight decay 0, batch 128,
        # 20 (CIFAR) / 50 (IN-R) epochs, per-epoch CosineSchedule
        # lr*cos(99*pi*e/(200*(K-1))), rank 10 on K and V of all 12 blocks,
        # DualGPM threshold eps -> 1.0 over the 10 sessions, plain biased
        # per-task nn.Linear heads trained with local CE and concatenated at
        # test, CLS after the final LayerNorm with no extra norm, no Stage 2.
        "train_optimizer": "adam",
        "train_base_lr": 5e-4,
        "train_head_lr": None,           # -> base lr (train_backbone)
        "train_weight_decay": 0.0,
        "train_epochs": 20,
        "train_batch_size": 128,
        "train_scheduler": "inflora_cosine",
        "model_use_norm": False,
        "model_classifier_use_norm": False,
        "model_classifier_bias": True,
        "train_merge": "none",
        "train_ca": False,               # native classifier (no LCA)
        "train_drift": False,            # no drift map
        "lora_r": 10,
        "inflora_lamb": 0.95,            # eps (CIFAR default; IN-R 0.98)
        "inflora_lame": 1.0,
        "inflora_backbone": "vit_base_patch16_224_in21k",   # = augreg_in21k (published)
        "inflora_gram_mode": "train",    # upstream: shuffled, augmented train loader
        "inflora_dual": True,
        "inflora_layers": None,          # all 12 blocks ([0..4] = InfLoRA-b5 layers)
    },
    "full": {"train_base_lr": 3e-4},
    "full_nsp": {"train_base_lr": 5e-6, "train_optimizer": "adamw", "nsp_weight": 0.02,
                 "train_head_lr": 1e-3},
    "joint_lora": {"joint": True, "train_iterations": 6000, "train_scheduler": "warmup_cosine",
                   "train_drift": False, "train_ca": False},
    "joint_full": {"joint": True, "train_iterations": 6000, "train_base_lr": 1e-5,
                   "train_optimizer": "adamw", "train_scheduler": "warmup_cosine",
                   "train_drift": False, "train_ca": False},
    "first_task_lora": {},
}

# Reference-repo native settings (second sweep to reproduce their numbers).
NATIVE_PRESET = {
    "train_optimizer": "adamw",
    "train_base_lr": 1e-4,
    "train_head_lr": 1e-3,
    "train_weight_decay": 3e-5,
    "train_loss": "sce",
    "train_scheduler": "warmup_cosine",
    "train_iterations": 1000,
    "train_batch_size": 16,
}


def run_single_experiment(dataset_name, config_name, experiment_config, seed):
    config = copy.deepcopy(BASE_CONFIG)
    method = experiment_config.get("train_method", config["train_method"])
    config.update(RECIPE_PRESETS.get(method, {}))
    config["train_method"] = method
    config.update(experiment_config)
    config["seed"] = seed
    set_random(config["seed"])

    dataset_num_task, dataset_init_cls, dataset_increment = DATA_TABLE[dataset_name][0]
    if config.get("joint", False):
        total = TOTAL_CLASSES[dataset_name]
        dataset_num_task, dataset_init_cls, dataset_increment = 1, total, total
    config.update({
        "dataset_name": dataset_name,
        "dataset_num_task": dataset_num_task,
        "dataset_init_cls": dataset_init_cls,
        "dataset_increment": dataset_increment,
    })
    # Long-term / custom splits: an experiment config may override the split
    # (e.g. IN-R 20x10 or 50x4 per the E2-LoRA Table-2 protocol).
    for k in ("dataset_num_task", "dataset_init_cls", "dataset_increment"):
        if k in experiment_config:
            config[k] = experiment_config[k]

    # DIL datasets carry domain structure in the label offsets — the class
    # order must NOT be shuffled (their protocol uses a fixed domain order).
    shuffle_order = not dataset_name.endswith("dil")
    data_manager = DataManager(
        config["dataset_name"], shuffle_order, config["seed"],
        config["dataset_init_cls"], config["dataset_increment"], False,
        # Rebuttal data hook (E13 few-shot per task, E14 validation carve); all
        # four default to the no-op and a None seed means the run seed. The
        # carve precedes the few-shot draw; test set and class order untouched.
        val_frac=config.get("data_val_frac", 0.0),
        val_seed=config.get("data_val_seed"),
        fewshot_k=config.get("data_fewshot_k"),
        subsample_seed=config.get("data_subsample_seed"),
    )

    if dataset_name == "imageneta" and config.get("train_iterations") is None:
        config["train_batch_size"] = min(config["train_batch_size"], 48)

    experiment_name = f"{dataset_name}_{config_name}"
    result = {}
    try:
        logging.info("Configuration:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")
        if any(k in config for k in DATA_HOOK_KEYS):
            # Realised per-class train / val / test counts of the data hook;
            # describe() carries the '[Data] ' prefix itself. Logged only when
            # a data_* key is set, so default logs stay byte-identical.
            logging.info(data_manager.describe())

        learner = Learner(config)
        learner.learn(data_manager)

        # Robustness-suite export: persist the DEPLOYED final model (merged
        # backbone LoRA + aligned head + class Gaussians) in one small file.
        if config.get("final_model_dir"):
            os.makedirs(config["final_model_dir"], exist_ok=True)
            out = os.path.join(
                config["final_model_dir"],
                f"{dataset_name}_{config_name}_s{seed}.pt")
            if getattr(learner, "_is_e2", False) or getattr(learner, "_is_inflora", False):
                # E2's older pairs are frozen (not in trainable params) and its
                # structure grows per task -> pickle the whole module instead.
                # InfLoRA folds its branches into the frozen qkv weights, which
                # the 'main' payload (requires_grad params only) would drop.
                payload = {
                    "kind": "main_pickle",
                    "config": {k: v for k, v in config.items()},
                    "model": learner.model.cpu().eval(),
                    "known_classes": learner._total_classes,
                }
            else:
                payload = {
                    "kind": "main",
                    "config": {k: v for k, v in config.items()},
                    "backbone": {k: v.cpu() for k, v in
                                 learner.model.get_backbone_trainable_params().items()},
                    "classifier": learner.model.classifier.state_dict(),
                    "known_classes": learner._total_classes,
                }
            if hasattr(learner, "_class_means"):
                payload["class_means"] = learner._class_means.cpu()
                if getattr(learner, "_class_covs", None) is not None:
                    payload["class_covs"] = learner._class_covs.cpu()
                else:
                    # Rebuttal E5 cheap statistics (train_ca_cov_mode diag |
                    # lowrank | shared): keys present in those modes only, so
                    # the default payload is unchanged.
                    cov_mode = config.get("train_ca_cov_mode", "class")
                    if cov_mode in ("diag", "lowrank"):
                        payload["class_vars"] = learner._class_vars.cpu()
                    if cov_mode == "lowrank":
                        payload["class_lr_U"] = learner._class_lr_U.cpu()
                    if cov_mode == "shared":
                        payload["class_cov_shared"] = shared_cov(
                            learner._shared_scatter, learner._shared_n).cpu()
            torch.save(payload, out)
            logging.info(f"[Export] Final model saved to {out}")

        result = {
            "faa": learner._faa, "ffm": learner._ffm, "asa": learner._asa,
            "pruned_at": getattr(learner, "_pruned_at", None),
            # Per-task total-accuracy trajectory (IN-R task groups are equal-
            # sized, so total acc == mean of grouped accs). Used as pruning
            # references for later searches.
            "traj": [float(np.mean(row)) for row in learner._mlp_matrix],
            # Sample-weighted per-task total accuracy (InfLoRA's ACC_t, E15b);
            # the exception path returns an empty list under the same key.
            "total_curve": list(getattr(learner, "_total_acc_curve", [])),
        }
        if hasattr(learner, "_faa_val"):
            # Rebuttal E14: metrics on the validation carve (data_val_frac > 0).
            result.update({"faa_val": learner._faa_val, "ffm_val": learner._ffm_val,
                           "asa_val": learner._asa_val})

        del learner
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        import traceback
        logging.error(f"[Experiment {experiment_name}] {type(e).__name__}: {e}")
        logging.error(traceback.format_exc())
        result = {k: 0.0 for k in ["faa", "ffm", "asa"]}
        result["total_curve"] = []   # same key as the success path (run_inflora)
    return result


def run_config_sweep(datasets, experiment_configs, seeds, log_root=LOG_DIR):
    """Per-(config x seed) runner with ASA/FAA/FFM summaries (exp19)."""
    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        dataset_results = {}

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        for config_name, config in experiment_configs.items():
            dir_path = os.path.join(log_root, dataset_name)
            os.makedirs(dir_path, exist_ok=True)
            logfile = os.path.join(dir_path, config_name + ".log")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(filename)s] => %(message)s",
                handlers=[
                    logging.FileHandler(filename=logfile),
                    logging.StreamHandler(sys.stdout),
                ],
                force=True,
            )

            for seed in seeds:
                logging.info("\n" + "=" * 80)
                logging.info(f"Experiment: {dataset_name} — {config_name} — seed {seed}")
                t0 = time.time()
                result = run_single_experiment(dataset_name, config_name, config, seed)
                logging.info(f"Time: {time.time() - t0:.1f}s")

                if config_name not in dataset_results:
                    dataset_results[config_name] = {k: [] for k in ["faa", "ffm", "asa"]}
                for k in ("faa", "ffm", "asa"):
                    dataset_results[config_name][k].append(result[k])
                # Rebuttal E14: validation-split metrics exist only for seeds
                # that ran with data_val_frac > 0 and finished (a crashed seed
                # returns none), so they are collected where present.
                for k in ("faa_val", "ffm_val", "asa_val"):
                    if k in result:
                        dataset_results[config_name].setdefault(k, []).append(result[k])

            logging.info("\n" + "=" * 80)
            logging.info(f"SUMMARY  {dataset_name.upper()} — {config_name.upper()}")
            logging.info("=" * 80)
            r = dataset_results[config_name]
            logging.info(
                f"  ASA: {np.mean(r['asa']):.2f} ± {np.std(r['asa']):.2f} | "
                f"FAA: {np.mean(r['faa']):.2f} ± {np.std(r['faa']):.2f} | "
                f"FFM: {np.mean(r['ffm']):.2f} ± {np.std(r['ffm']):.2f}"
            )
            if r.get("faa_val"):
                # Rebuttal E14: one extra line after the unchanged SUMMARY line;
                # mean and std (ddof 0) over the seeds that returned val metrics,
                # with the count appended when a seed is missing (crashed).
                n_val = len(r["faa_val"])
                logging.info(
                    f"  [val] ASA: {np.mean(r['asa_val']):.2f} ± {np.std(r['asa_val']):.2f} | "
                    f"FAA: {np.mean(r['faa_val']):.2f} ± {np.std(r['faa_val']):.2f} | "
                    f"FFM: {np.mean(r['ffm_val']):.2f} ± {np.std(r['ffm_val']):.2f}"
                    + (f" | n={n_val}/{len(seeds)}" if n_val != len(seeds) else "")
                )

            method = config.get("train_method", "seqft")
            if config.get("cleanup_merged", False) and method in ("seqft",) + HYBRID_METHODS:
                # Tag-specific (exp19): a shared checkpoint_dir may hold merged
                # checkpoints of other configs -- never glob them all.
                merged = BASE_CONFIG.copy()
                merged.update(config)
                tag = merge_tag(merged)
                ckpt_dir = config.get("checkpoint_dir", os.path.join(CHECKPOINT_ROOT, method))
                # Only files whose suffix after the tag is a bare task index:
                # a tag that is a prefix of another config's tag must not
                # claim that config's files (legacy files always end in _<t>.pt,
                # so the filter leaves the legacy behaviour unchanged).
                task_file = re.compile(rf"_merged_{re.escape(tag)}_\d+\.pt$")
                stale = [f for f in glob.glob(os.path.join(ckpt_dir, f"*_merged_{tag}_*.pt"))
                         if task_file.search(os.path.basename(f))]
                for f in stale:
                    os.remove(f)
                logging.info(f"[Cleanup] Removed {len(stale)} merged checkpoints for {config_name}")
        logging.info("=" * 80 + "\n")


def run_recipe_comparison(datasets=None, recipes=None, seeds=(1993,),
                          native=False, extra=None, log_root=None):
    """The controlled comparison: every recipe under our loop (or their native
    loop with native=True), Stage 2 fixed."""
    datasets = datasets or ["imagenetr"]
    recipes = recipes or ["seqft", "basic_lora", "sgp_lora", "nsp_lora", "full", "full_nsp"]
    configs = {}
    for method in recipes:
        cfg = {"train_method": method, "train_prefix": "main_v1"}
        if native and method != "seqft":
            cfg.update(NATIVE_PRESET)
        if extra:
            cfg.update(extra)
        name = f"{method}_native" if (native and method != "seqft") else method
        configs[name] = cfg
    run_config_sweep(datasets, configs, list(seeds),
                     log_root=log_root or LOG_DIR)


def run_smoke_tests(include_inflora=False):
    """Tiny 2-task run of every recipe on imagenetr: catches shape/lifecycle
    bugs, not accuracy. No caching, 1 epoch / 50 iterations.

    The legacy recipe list is unchanged by default. `include_inflora` (CLI
    `--smoke-inflora`) appends the InfLoRA port (E15b), which downloads its own
    published backbone (vit_base_patch16_224_in21k) and writes its own
    checkpoints_main/inflora states; it runs last, so the earlier cells keep
    their RNG stream either way. run_inflora.py --smoke covers the same cell."""
    recipes = ["seqft", "basic_lora", "sgp_lora", "nsp_lora", "full",
               "full_nsp", "first_task_lora", "joint_lora"]
    if include_inflora:
        recipes.append("inflora")
    configs = {}
    for method in recipes:
        cfg = {
            "train_method": method,
            "train_prefix": "main_smoke",
            "train_stop_at_task": 1,
            "train_epochs": 1,
            "reset_train": True,
            "reset_merge": True,
            "cache_backbone": False,
            "train_ca_epochs": 2,
        }
        if method.startswith("joint"):
            cfg["train_iterations"] = 50
        configs[f"smoke_{method}"] = cfg
    run_config_sweep(["imagenetr"], configs, [1993],
                     log_root=os.path.join(LOG_DIR, "smoke"))


if __name__ == "__main__":
    if "--smoke" in sys.argv or "--smoke-inflora" in sys.argv:
        run_smoke_tests(include_inflora="--smoke-inflora" in sys.argv)
    else:
        run_recipe_comparison()
