"""Backbone adapters for main.py: ports of the training recipes from
https://github.com/raoxuan98-hash/lr_rgda_hopdc (models/basic_lora.py,
models/sgp_lora.py, models/full_finetune.py, models/distillator.py, lora.py).

InfLoRA (Liang & Li, CVPR 2024) is ported from github.com/liangyanshuo/InfLoRA
(section "InfLoRA" below; its projection math is the NumPy module
inflora_gpm.py). KeepLoRA (Luo et al., ICLR 2026) is ported from
github.com/MaolinLuo/KeepLoRA (section "KeepLoRA" below; its subspace math is
the NumPy module keeplora_subspace.py).

Self-contained (torch + timm, plus NumPy for the InfLoRA / KeepLoRA ports). All wrappers
share one interface driven by main.Learner:
    forward(x) -> features            (num_classes=0 ViT, pooled CLS)
    get_param_groups()                -> trainable params for the optimizer
    merge_lora_weights()              fold adapter into backbone, re-init adapter
    finalize_without_lora()           fold adapter, deactivate (for snapshots)
    update_projection_matrices(covs)  rebuild P from covariances (sgp/nsp only)
    use_projection                    bool
    get_module_names() / lora_modules for covariance hooks
"""

import math
import copy
import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from inflora_gpm import (
    basis_leakage,
    design_down_projection,
    dualgpm_step,
    pack_basis,
    unpack_basis,
)
from keeplora_subspace import (
    basis_leakage as keeplora_basis_leakage,
    extend_subspace as keeplora_extend_subspace,
    gradient_init as keeplora_gradient_init,
    pack_basis as keeplora_pack_basis,
    unpack_basis as keeplora_unpack_basis,
    weight_subspace as keeplora_weight_subspace,
)


# LoRA / DoRA modules =========================================================
class OriginalLoRA(nn.Module):
    """Plain LoRA on one nn.Linear; merge folds (B@A)*scale into the weight."""

    def __init__(self, linear: nn.Linear, r: int, lora_scale: float = 1.0):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.lora_scale = lora_scale
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)
        self.register_buffer("lora_active", torch.tensor(True))

    def forward(self, x):
        if self.lora_active:
            lora_w = torch.matmul(self.lora_B, self.lora_A) * self.lora_scale
            return F.linear(x, lora_w + self.linear.weight, self.bias)
        return self.linear(x)

    def merge_lora_weights(self, lora_active: bool = True) -> None:
        with torch.no_grad():
            delta = self.lora_B @ self.lora_A * self.lora_scale
            self.linear.weight.data.add_(delta.to(self.linear.weight.device))
            self.lora_active.fill_(lora_active)
            if lora_active:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_B.data.zero_()


class FixedProjection(nn.Module):
    def __init__(self, P: torch.Tensor):
        super().__init__()
        self.register_buffer("P", P)

    def forward(self) -> torch.Tensor:
        return self.P


class SGPBaseLoRA(nn.Module):
    """Projected LoRA: delta = B @ (A @ P). Upstream sgp_lora.py:17 (their
    forward dropped x -- fixed here)."""

    def __init__(self, linear: nn.Linear, r: int, proj: nn.Module):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj

        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)
        self.register_buffer("lora_active", torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_active:
            lora_delta = self.B @ (self.A @ self.P())
            return F.linear(x, self.linear.weight + lora_delta, self.bias)
        return self.linear(x)

    def merge_lora_weights(self, lora_active: bool = True) -> None:
        with torch.no_grad():
            delta = self.B @ self.A @ self.P()
            self.linear.weight.data.add_(delta.to(self.linear.weight.device))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B.data.zero_()
            self.lora_active.fill_(lora_active)


class SGPBaseDoRA(nn.Module):
    """Projected DoRA: W = (directions + B@(A@P)) * magnitude, magnitude
    trainable. Upstream default for sgp/nsp (sgp_lora.py:61)."""

    def __init__(self, linear: nn.Linear, r: int, proj: nn.Module):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj

        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        with torch.no_grad():
            weight = linear.weight.data
            weight_norm = weight.norm(p=2, dim=1, keepdim=True) + 1e-8
            self.weight_directions = nn.Parameter(weight / weight_norm, requires_grad=False)
            self.magnitude = nn.Parameter(weight_norm.clone(), requires_grad=True)

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)
        self.register_buffer("lora_active", torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_active:
            lora_delta = self.B @ (self.A @ self.P())
            adapted_weight = (self.weight_directions + lora_delta) * self.magnitude
        else:
            adapted_weight = self.weight_directions * self.magnitude
        return F.linear(x, adapted_weight, self.bias)

    def merge_lora_weights(self, lora_active: bool = True) -> None:
        with torch.no_grad():
            lora_delta = self.B @ self.A @ self.P()
            self.weight_directions.data.add_(lora_delta)
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B.data.zero_()
            self.lora_active.fill_(lora_active)


# Projection builder ==========================================================
def compute_weights(x, weight_kind="log1p", beta=1.0, weight_p=1.0,
                    weight_alpha=0.5, weight_kappa=2.0):
    if weight_kind == "exp":
        return torch.exp(-beta * x)
    elif weight_kind == "rational1":
        return 1.0 / (1.0 + beta * x)
    elif weight_kind == "rational2":
        return 1.0 / (1.0 + beta * (x ** 2))
    elif weight_kind == "sqrt_rational2":
        return 1.0 / torch.sqrt(1.0 + beta * (x ** 2))
    elif weight_kind == "log1p":
        return 1.0 / (1.0 + beta * torch.log1p(x ** weight_p))
    elif weight_kind == "power_family":
        return (1.0 + beta * (x ** weight_p)) ** (-weight_alpha)
    elif weight_kind == "stretched_exp":
        return torch.exp(-((beta * x) ** weight_kappa))
    raise ValueError(f"Unknown weight_kind={weight_kind!r}")


def build_projection(cov, soft_projection=True, weight_temp=5.0, nsp_eps=0.05,
                     nsp_weight=0.0, *, weight_kind="log1p", weight_alpha=0.5,
                     weight_p=2.0, weight_kappa=2.0):
    """Soft (eigenvalue-weighted) or hard (null-space) projection from an input
    covariance. Runs on CPU: torch.use_deterministic_algorithms(True) is active
    (helper.set_random) and CUDA eigh is not deterministic."""
    out_device, out_dtype = cov.device, cov.dtype
    cov = cov.detach().float().cpu()
    eps = 1e-6
    cov = cov + eps * torch.eye(cov.size(0))
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
    eigvals = torch.abs(eigvals)
    d = cov.size(0)
    eigvals = eigvals * (d / (eigvals.sum() + eps))

    if soft_projection:
        weights = compute_weights(eigvals, weight_kind, weight_temp, weight_p, weight_alpha, weight_kappa)
        weights = weights / weights.max()
        P = eigvecs @ torch.diag(weights) @ eigvecs.t()
    else:
        # Keep the low-energy eigen-subspace holding <= nsp_eps of total energy.
        ratio = torch.cumsum(eigvals, dim=0) / (eigvals.sum() + 1e-12)
        idx = (ratio >= nsp_eps).nonzero(as_tuple=False)
        m = idx[0].item() if idx.numel() > 0 else eigvals.numel()
        V_keep = eigvecs[:, :m]
        P = V_keep @ V_keep.t()
        P = (1 - nsp_weight) * P + nsp_weight * torch.eye(P.size(0))
    return P.to(device=out_device, dtype=out_dtype)


# ViT wrappers ================================================================
class BaseAdapterViT(nn.Module):
    """Common plumbing: wraps a timm ViT, tracks adapted modules in
    self.lora_modules, exposes feature_dim/num_features."""

    def __init__(self, vit_model: nn.Module):
        super().__init__()
        self.lora_vit = vit_model
        self.feature_dim = getattr(vit_model, "embed_dim", 768)
        self.num_features = self.feature_dim
        self.lora_modules = nn.ModuleDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_vit(x)

    def get_module_names(self) -> List[str]:
        return list(self.lora_modules.keys())

    def merge_lora_weights(self) -> None:
        for mod in self.lora_modules.values():
            if hasattr(mod, "merge_lora_weights"):
                mod.merge_lora_weights(lora_active=True)

    def finalize_without_lora(self) -> None:
        self.eval()
        for mod in self.lora_modules.values():
            if hasattr(mod, "merge_lora_weights"):
                mod.merge_lora_weights(lora_active=False)

    def update_projection_matrices(self, covariances) -> None:
        pass

    @property
    def use_projection(self) -> bool:
        return False

    def get_param_groups(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


class PlainLoRAViT(BaseAdapterViT):
    """basic_lora: rank-r LoRA on qkv + mlp.fc1 + mlp.fc2 of every block,
    folded into the backbone at each task boundary (basic_lora.py:219)."""

    def __init__(self, vit_model, r: int, lora_layer: Optional[Iterable[int]] = None,
                 include_norm: bool = False, lora_scale: float = 1.0):
        super().__init__(vit_model)
        assert r > 0
        self.r = r
        self.lora_scale = lora_scale
        layers = list(lora_layer) if lora_layer is not None else list(range(len(vit_model.blocks)))

        for p in vit_model.parameters():
            p.requires_grad_(False)
        if include_norm:
            for n, p in vit_model.named_parameters():
                if ("norm" in n) or ("cls_token" in n):
                    p.requires_grad_(True)

        for idx, blk in enumerate(vit_model.blocks):
            if idx not in layers:
                continue
            blk.attn.qkv = OriginalLoRA(blk.attn.qkv, r, lora_scale)
            self.lora_modules[f"block_{idx}_attn_qkv"] = blk.attn.qkv
            blk.mlp.fc1 = OriginalLoRA(blk.mlp.fc1, r, lora_scale)
            self.lora_modules[f"block_{idx}_mlp_fc1"] = blk.mlp.fc1
            blk.mlp.fc2 = OriginalLoRA(blk.mlp.fc2, r, lora_scale)
            self.lora_modules[f"block_{idx}_mlp_fc2"] = blk.mlp.fc2

    def get_param_groups(self):
        params, seen = [], set()
        for mod in self.lora_modules.values():
            for pname, p in mod.named_parameters(recurse=False):
                if p.requires_grad and pname in ("lora_A", "lora_B", "magnitude"):
                    params.append(p)
                    seen.add(id(p))
        for name, p in self.lora_vit.named_parameters():
            if p.requires_grad and id(p) not in seen and (("norm" in name) or ("cls_token" in name)):
                params.append(p)
        return params


class SGPLoRAViT(BaseAdapterViT):
    """sgp_lora / nsp_lora: projected LoRA/DoRA on qkv + attn.proj + fc1 + fc2.
    use_soft_projection=True -> SGP (covariance-weighted), False -> NSP (hard
    null-space). P placeholders start as identity; update_projection_matrices
    rebuilds them from EMA'd input covariances (sgp_lora.py:233)."""

    def __init__(self, vit_model, r: int, lora_layer: Optional[Iterable[int]] = None,
                 use_soft_projection: bool = True, weight_temp: float = 1.0,
                 weight_kind: str = "log1p", weight_p: float = 1.0,
                 nsp_eps: float = 0.05, nsp_weight: float = 0.0,
                 sgp_module: str = "dora", include_norm: bool = False):
        super().__init__(vit_model)
        assert r > 0
        self.r = r
        self.use_soft_projection = use_soft_projection
        self.weight_temp = weight_temp
        self.weight_kind = weight_kind
        self.weight_p = weight_p
        self.nsp_eps = nsp_eps
        self.nsp_weight = nsp_weight

        lora_class = {"dora": SGPBaseDoRA, "lora": SGPBaseLoRA}[sgp_module]
        layers = list(lora_layer) if lora_layer is not None else list(range(len(vit_model.blocks)))

        for n, p in vit_model.named_parameters():
            p.requires_grad_(include_norm and "norm" in n)

        dev = vit_model.patch_embed.proj.weight.device

        def wrap(linear: nn.Linear):
            proj = FixedProjection(torch.eye(linear.in_features, device=dev, dtype=linear.weight.dtype))
            return lora_class(linear, r, proj)

        for idx, blk in enumerate(vit_model.blocks):
            if idx not in layers:
                continue
            blk.attn.qkv = wrap(blk.attn.qkv)
            self.lora_modules[f"block_{idx}_attn_qkv"] = blk.attn.qkv
            blk.attn.proj = wrap(blk.attn.proj)
            self.lora_modules[f"block_{idx}_attn_proj"] = blk.attn.proj
            blk.mlp.fc1 = wrap(blk.mlp.fc1)
            self.lora_modules[f"block_{idx}_mlp_fc1"] = blk.mlp.fc1
            blk.mlp.fc2 = wrap(blk.mlp.fc2)
            self.lora_modules[f"block_{idx}_mlp_fc2"] = blk.mlp.fc2

    @property
    def use_projection(self) -> bool:
        return True

    @torch.no_grad()
    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        # Fold any pending LoRA before swapping P (a no-op right after the
        # end-of-task fold: B is already zero) -- upstream sgp_lora.py:260.
        self.merge_lora_weights()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            mod = self.lora_modules[name]
            P = build_projection(
                cov,
                soft_projection=self.use_soft_projection,
                weight_temp=self.weight_temp,
                weight_kind=self.weight_kind,
                weight_p=self.weight_p,
                nsp_eps=self.nsp_eps,
                nsp_weight=self.nsp_weight,
            )
            ref = mod.magnitude if hasattr(mod, "magnitude") else mod.linear.weight
            mod.P = FixedProjection(P.to(device=ref.device, dtype=ref.dtype))


class FullFinetuneViT(BaseAdapterViT):
    """full: unfreeze only the MLP fc1/fc2 weights (biases frozen) of every
    block; no adapters, weights carried forward task-to-task
    (full_finetune.py:241)."""

    def __init__(self, vit_model, include_norm: bool = False,
                 finetune_layers: Optional[Iterable[int]] = None):
        super().__init__(vit_model)
        layers = set(finetune_layers) if finetune_layers is not None else set(range(len(vit_model.blocks)))

        for p in vit_model.parameters():
            p.requires_grad = False
        for idx, blk in enumerate(vit_model.blocks):
            if idx not in layers:
                continue
            for n, p in blk.mlp.named_parameters():
                if "bias" not in n:
                    p.requires_grad = True
            self.lora_modules[f"block_{idx}_mlp_fc1"] = blk.mlp.fc1
            self.lora_modules[f"block_{idx}_mlp_fc2"] = blk.mlp.fc2
        if include_norm:
            for name, p in vit_model.named_parameters():
                if "norm" in name:
                    p.requires_grad = True


class NullSpaceViT(BaseAdapterViT):
    """full_nsp: full fc1/fc2 finetuning with a backward hook projecting each
    weight gradient into the null-space: grad <- grad @ P
    (full_finetune.py:10)."""

    def __init__(self, vit_model, nsp_eps: float = 0.05, nsp_weight: float = 0.02,
                 soft_projection: bool = False, weight_temp: float = 5.0,
                 weight_kind: str = "log1p", weight_p: float = 1.0):
        super().__init__(vit_model)
        self.nsp_eps = nsp_eps
        self.nsp_weight = nsp_weight
        self.soft_projection = soft_projection
        self.weight_temp = weight_temp
        self.weight_kind = weight_kind
        self.weight_p = weight_p
        self._projection_enabled = True
        self.projection_matrices: Dict[str, torch.Tensor] = {}
        self._param_to_name: Dict[nn.Parameter, str] = {}

        for p in vit_model.parameters():
            p.requires_grad = False
        for idx, blk in enumerate(vit_model.blocks):
            self.lora_modules[f"block_{idx}_mlp_fc1"] = blk.mlp.fc1
            self.lora_modules[f"block_{idx}_mlp_fc2"] = blk.mlp.fc2
        for name, module in self.lora_modules.items():
            module.weight.requires_grad = True
            self._param_to_name[module.weight] = name
            module.weight.register_hook(self._make_grad_projection_hook(module.weight))

    def _make_grad_projection_hook(self, param: nn.Parameter):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self._projection_enabled:
                return grad
            name = self._param_to_name.get(param)
            proj = self.projection_matrices.get(name)
            if proj is None:
                return grad
            if proj.device != grad.device or proj.dtype != grad.dtype:
                proj = proj.to(device=grad.device, dtype=grad.dtype)
            with torch.no_grad():
                return torch.matmul(grad, proj)
        return hook

    @property
    def use_projection(self) -> bool:
        return True

    def get_param_groups(self):
        return [m.weight for m in self.lora_modules.values() if m.weight.requires_grad]

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        self.projection_matrices = {}
        for name, module in self.lora_modules.items():
            if name not in covariances:
                continue
            proj = build_projection(
                covariances[name],
                soft_projection=self.soft_projection,
                weight_temp=self.weight_temp,
                weight_kind=self.weight_kind,
                weight_p=self.weight_p,
                nsp_eps=self.nsp_eps,
                nsp_weight=self.nsp_weight,
            )
            w = module.weight
            self.projection_matrices[name] = proj.to(device=w.device, dtype=w.dtype)

    def finalize_without_lora(self) -> None:
        self.eval()
        self.projection_matrices.clear()


# Covariance collection =======================================================
class FeatureCovarianceCalculator:
    """Uncentered input covariances X^T X / n per adapted module, accumulated
    over all tokens via forward hooks (upstream lora.py:399)."""

    def __init__(self, modules: Dict[str, nn.Module], device="cuda"):
        self.module_names = list(modules.keys())
        self.device = device
        self.covariances = {name: None for name in self.module_names}
        self.counts = {name: 0 for name in self.module_names}
        self.hooks = []
        for name, module in modules.items():

            def hook_fn(module, inputs, output, name=name):
                self._update_covariance(name, inputs[0])

            self.hooks.append(module.register_forward_hook(hook_fn))

    def _update_covariance(self, name, features):
        features = features.detach().to(self.device)
        if features.dim() == 3:
            B, N, D = features.size()
            features = features.reshape(B * N, D)
        cov_batch = features.t() @ features
        if self.covariances[name] is None:
            self.covariances[name] = cov_batch
        else:
            self.covariances[name] += cov_batch
        self.counts[name] += features.size(0)

    def compute_final_covariances(self):
        return {
            name: (self.covariances[name] / self.counts[name]) if self.counts[name] > 0 else None
            for name in self.module_names
        }

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


@torch.no_grad()
def compute_covariances(backbone, data_loader, device="cuda", modules=None):
    """Batches follow this repo's loader format: (idx, path, x, y). `modules`
    overrides the hooked module dict (default: backbone.lora_modules)."""
    if modules is None:
        modules = {name: backbone.lora_modules[name] for name in backbone.get_module_names()}
    calc = FeatureCovarianceCalculator(modules, device)
    backbone.to(device)
    backbone.eval()
    for batch in data_loader:
        x = batch[-2].to(device)
        backbone(x)
    covariances = calc.compute_final_covariances()
    calc.remove_hooks()
    return covariances


def peft_qkv_modules(peft_model) -> Dict[str, nn.Module]:
    """The adapted qkv LoraLayer per block of a PEFT-wrapped timm ViT, named
    like the ported wrappers (block_<i>_attn_qkv). Input hooks on these see the
    same activations that feed lora_A."""
    blocks = peft_model.base_model.model.blocks
    return {f"block_{idx}_attn_qkv": blk.attn.qkv for idx, blk in enumerate(blocks)}


def peft_adapted_modules(peft_model, targets=("qkv", "fc1", "fc2")) -> Dict[str, nn.Module]:
    """Every adapted linear layer per block of a PEFT-wrapped timm ViT, keyed
    block_<i>_attn_qkv / block_<i>_mlp_fc1 / block_<i>_mlp_fc2 (rebuttal E6).

    Generalises peft_qkv_modules (left untouched: the hybrids stay qkv-only) to
    the headline recipe's target modules ['qkv', 'fc1', 'fc2'] so that
    compute_covariances(backbone, loader, modules=peft_adapted_modules(backbone))
    gives the layer-INPUT covariances in the row space of each lora_A (768 for
    qkv/fc1, 3072 for fc2). `targets` selects the module names; unknown names
    raise ValueError. Blocks are visited in order, qkv before fc1 before fc2.
    """
    known = {"qkv": ("attn_qkv", lambda blk: blk.attn.qkv),
             "fc1": ("mlp_fc1", lambda blk: blk.mlp.fc1),
             "fc2": ("mlp_fc2", lambda blk: blk.mlp.fc2)}
    unknown = [t for t in targets if t not in known]
    if unknown:
        raise ValueError(f"peft_adapted_modules: unknown target module(s) {unknown}; known {sorted(known)}")
    blocks = peft_model.base_model.model.blocks
    out = {}
    for idx, blk in enumerate(blocks):
        for t in targets:
            suffix, getter = known[t]
            out[f"block_{idx}_{suffix}"] = getter(blk)
    return out


def peft_lora_A_params(peft_model) -> Dict[str, nn.Parameter]:
    """lora_A weight per adapted qkv, keyed like peft_qkv_modules."""
    out = {}
    for name, qkv in peft_qkv_modules(peft_model).items():
        if hasattr(qkv, "lora_A"):
            out[name] = qkv.lora_A["default"].weight
    return out


# Distillation ================================================================
def cosine_similarity_loss(x1, x2):
    cos_sim = F.cosine_similarity(x1.flatten(1), x2.flatten(1), dim=1)
    return (1.0 - cos_sim).mean()


def feature_distillation_loss(x1, x2):
    return torch.pow(x1 - x2, 2).mean()


class ResidMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x):
        return x + self.mlp(x)


class Distiller(nn.Module):
    """Feature KD from the previous-task feature extractor (distillator.py).
    The teacher is any module exposing get_features(x); the transform head is
    trainable and re-initialized on every teacher update."""

    def __init__(self, kd_type="feat", gamma_kd=0.0, feat_dim=768,
                 transform="weaknonlinear", mlp_ratio=4, device="cuda"):
        super().__init__()
        self.kd_type = kd_type
        self.gamma_kd = gamma_kd
        self.feat_dim = feat_dim
        self.transform = transform
        self.mlp_ratio = mlp_ratio
        self.device = device
        self.loss_fn = {"feat": feature_distillation_loss, "cos": cosine_similarity_loss}[kd_type]
        self.head = self._build_head()
        self.teacher = None

    def _build_head(self):
        if self.transform == "identity":
            head = nn.Identity()
        elif self.transform == "linear":
            head = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
            nn.init.eye_(head.weight)
        elif self.transform == "weaknonlinear":
            head = ResidMLP(self.feat_dim, self.mlp_ratio)
        else:
            raise ValueError(f"Unsupported kd transform {self.transform!r}")
        return head.to(self.device)

    @torch.no_grad()
    def update_teacher(self, teacher_model: nn.Module):
        """teacher_model: a frozen feature extractor (already snapshotted)."""
        if self.gamma_kd <= 0:
            return
        self.teacher = teacher_model.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.head = self._build_head()
        logging.info(f"[KD] Teacher updated, head reinitialized ({self.transform})")

    def forward(self, inputs, student_features):
        if self.gamma_kd <= 0.0 or self.teacher is None:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            teacher_features = self.teacher.get_features(inputs)
        return self.gamma_kd * self.loss_fn(self.head(student_features), teacher_features)


# Loss ========================================================================
def symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5):
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    label_one_hot = F.one_hot(targets, pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    ce_loss = -torch.sum(label_one_hot * torch.log(pred), dim=1).mean()
    rce_loss = -torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
    return sce_a * ce_loss + sce_b * rce_loss


# E2-LoRA (Li et al., ICML 2026) ==============================================
# Port of github.com/kiddo127/E2-LoRA (class_incremental_learning/utils/
# inc_net.py + trainer.py). Per-task LoRA pairs on qkv and mlp.fc1, kept as
# growing ModuleLists (never folded); dynamic rank allocation with energy
# pruning; post-task energy-structured transformation (PCA of output drift).

E2_INIT_THRESHOLD = 0.99999
E2_THRESHOLD_STEP = 1.0 - E2_INIT_THRESHOLD


def e2_principal_direction_svd(act_data):
    """PCA basis + eigenvalues of (flattened, centered) activations
    (upstream inc_net.py:16)."""
    feature_dim = act_data.shape[-1]
    if act_data.dim() > 2:
        act_data = act_data.reshape(-1, feature_dim)
    n_samples = act_data.shape[0]
    act_data = act_data - torch.mean(act_data, dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(act_data, full_matrices=False)
    eigenvectors = Vh.T
    eigenvalues = S ** 2 / (n_samples - 1)
    if eigenvectors.shape[1] < feature_dim:
        full_eig = torch.eye(feature_dim, device=act_data.device, dtype=act_data.dtype)
        full_eig[:, :eigenvectors.shape[1]] = eigenvectors
        eigenvectors = full_eig
        full_vals = torch.zeros(feature_dim, device=eigenvalues.device, dtype=eigenvalues.dtype)
        full_vals[:len(eigenvalues)] = eigenvalues
        eigenvalues = full_vals
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvalues[order].detach().cpu(), eigenvectors[:, order]


def e2_energy_threshold(eigenvalues, threshold=0.999):
    total = eigenvalues.sum()
    cumulative = torch.cumsum(eigenvalues, dim=0)
    mask = cumulative <= (threshold * total)
    return max(1, mask.sum().item() + 1)


class _E2PairMixin:
    """Shared add_task / energy-transform / merged-A logic for the two patched
    module types. Subclass must define: _in_dim (A input), _out_dim (budget =
    B output), _A/_B ModuleList attrs via self.A_list()/B_list()."""

    def _init_e2(self, budget=None):
        self._cur_task = 0
        self.pca_lora = False
        self.role = "student"
        self.eigenvalues = []
        self._merged_backup = None
        # Total rank budget shared across tasks. Their design: the full output
        # dimension (task 1 effectively full-rank). A smaller budget turns the
        # method into "E2-allocation at LoRA scale" (budgeted variant).
        self._budget = budget if budget is not None else self._out_dim

    def add_task(self):
        self._cur_task += 1
        A_list, B_list = self.A_list(), self.B_list()
        out_dim, in_dim = self._out_dim, self._in_dim
        budget = self._budget
        dev = next(self.parameters()).device

        old_rank = 0
        if self._cur_task > 1:
            saved_B_cols = []
            threshold = E2_INIT_THRESHOLD
            for task in range(self._cur_task - 1):
                old_rank += e2_energy_threshold(self.eigenvalues[task], threshold)
            while old_rank > (self._cur_task - 1) * budget // self._cur_task:
                old_rank = 0
                threshold -= E2_THRESHOLD_STEP
                for task in range(self._cur_task - 1):
                    old_rank += e2_energy_threshold(self.eigenvalues[task], threshold)

            old_rank = 0
            for task in range(self._cur_task - 1):
                A_w = A_list[task].weight.data.clone()
                B_w = B_list[task].weight.data.clone()
                keep = e2_energy_threshold(self.eigenvalues[task], threshold)
                old_rank += keep
                A_list[task] = nn.Linear(in_dim, keep, bias=False, device=dev)
                B_list[task] = nn.Linear(keep, out_dim, bias=False, device=dev)
                A_list[task].weight.data.copy_(A_w[:keep, :])
                B_list[task].weight.data.copy_(B_w[:, :keep])
                saved_B_cols.append(B_w[:, keep:])

        r = budget - old_rank
        A_new = nn.Linear(in_dim, r, bias=False, device=dev)
        B_new = nn.Linear(r, out_dim, bias=False, device=dev)
        if self._cur_task == 1:
            nn.init.kaiming_uniform_(A_new.weight, a=math.sqrt(5))
            nn.init.zeros_(B_new.weight)
        else:
            nn.init.zeros_(A_new.weight)
            B_new.weight.data.copy_(torch.cat(saved_B_cols, dim=1))
        A_list.append(A_new)
        B_list.append(B_new)

    def _lora_weight(self):
        tasks = self._cur_task - 1 if self.role == "teacher" else self._cur_task
        A_list, B_list = self.A_list(), self.B_list()
        if tasks == 0:
            return None
        return torch.stack([
            torch.mm(B_list[t].weight, A_list[t].weight) for t in range(tasks)
        ], dim=0).sum(dim=0)

    def _maybe_energy_transform(self, x):
        """During the post-task proxy pass (pca_lora=True): PCA the current
        pair's output drift, rotate B<-U, A<-U^T B A (product unchanged),
        store eigenvalues for future pruning (upstream inc_net.py:128,232)."""
        if not self.pca_lora:
            return
        A_list, B_list = self.A_list(), self.B_list()
        r = B_list[-1].weight.shape[1]
        cur_weight = torch.mm(B_list[-1].weight, A_list[-1].weight)
        eigenvalues, eigenvectors = e2_principal_direction_svd(F.linear(x, cur_weight))
        principal = eigenvectors[:, :r]
        self.eigenvalues.append(eigenvalues)
        A_list[-1].weight.data = principal.T @ B_list[-1].weight.data @ A_list[-1].weight.data
        B_list[-1].weight.data = principal

    def apply_merged_A(self, p, alpha):
        """Deploy mode: A_k <- alpha * truncate_energy(A_k, p) for task pairs
        k >= 1. The FIRST pair is the foundation and stays untouched -- the
        analogue of seqft's task-0 LoRA state, which is the merge BASE, never
        a truncated update (exp19: w_prev at the first merge is the task-0
        checkpoint). B orthonormal => damping A damps the effective update
        exactly. Originals backed up for restore."""
        from helper import truncate_energy
        if self._merged_backup is not None:
            return
        self._merged_backup = []
        for k, A in enumerate(self.A_list()):
            self._merged_backup.append(A.weight.data.clone())
            if k == 0:
                continue
            A.weight.data = (alpha * truncate_energy(A.weight.data.float(), p)).to(A.weight.dtype)

    def restore_raw_A(self):
        if self._merged_backup is None:
            return
        for A, raw in zip(self.A_list(), self._merged_backup):
            A.weight.data.copy_(raw)
        self._merged_backup = None


class E2LoRAAttention(_E2PairMixin, nn.Module):
    """timm Attention with per-task LoRA on qkv (upstream LoRAAttention)."""

    def __init__(self, attn, budget=None):
        nn.Module.__init__(self)
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.attn_drop = attn.attn_drop
        self.proj_drop = attn.proj_drop
        dim = attn.qkv.in_features
        self._in_dim = dim
        self._out_dim = dim * 3
        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()
        self._init_e2(budget)

    def A_list(self):
        return self.lora_A

    def B_list(self):
        return self.lora_B

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        if self._cur_task > 0:
            w = self._lora_weight()
            if w is not None:
                qkv = qkv + F.linear(x, w)
            self._maybe_energy_transform(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class E2LoRAMlp(_E2PairMixin, nn.Module):
    """timm Mlp with per-task LoRA on fc1 (upstream LoRAMlp)."""

    def __init__(self, mlp, budget=None):
        nn.Module.__init__(self)
        self.fc1 = mlp.fc1
        self.act = mlp.act
        self.drop1 = mlp.drop1
        self.norm = getattr(mlp, "norm", nn.Identity())
        self.fc2 = mlp.fc2
        self.drop2 = mlp.drop2
        self._in_dim = mlp.fc1.in_features
        self._out_dim = mlp.fc1.out_features
        self.lora_A1 = nn.ModuleList()
        self.lora_B1 = nn.ModuleList()
        self._init_e2(budget)

    def A_list(self):
        return self.lora_A1

    def B_list(self):
        return self.lora_B1

    def forward(self, x):
        h = self.fc1(x)
        if self._cur_task > 0:
            w = self._lora_weight()
            if w is not None:
                h = h + F.linear(x, w)
            self._maybe_energy_transform(x)
        x = self.act(h)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class E2LoRAViT(BaseAdapterViT):
    """E2-LoRA wrapper: patches every block's attn and mlp; final norm affine
    reset to identity (their trainer.py:64-65). lora_modules stays empty --
    the main.py fold/covariance paths must not touch the pair lists."""

    def __init__(self, vit_model, budget=None):
        super().__init__(vit_model)
        for blk in vit_model.blocks:
            blk.attn = E2LoRAAttention(blk.attn, budget=budget)
            blk.mlp = E2LoRAMlp(blk.mlp, budget=budget)
        nn.init.ones_(vit_model.norm.weight)
        nn.init.zeros_(vit_model.norm.bias)
        self._cur_task = 0

    def _e2_modules(self):
        out = {}
        for i, blk in enumerate(self.lora_vit.blocks):
            out[f"block_{i}_attn"] = blk.attn
            out[f"block_{i}_mlp"] = blk.mlp
        return out

    def add_task(self):
        """Their trainer loop: add_task on every block, then freeze everything
        except the new pair's A (and B at task 0)."""
        for mod in self._e2_modules().values():
            mod.add_task()
        self._cur_task += 1
        task = self._cur_task - 1  # 0-based, matches their name matching
        for name, p in self.named_parameters():
            if f"lora_A.{task}" in name or f"lora_A1.{task}" in name:
                p.requires_grad = True
            elif task == 0 and (f"lora_B.{task}" in name or f"lora_B1.{task}" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def set_role(self, role):
        for mod in self._e2_modules().values():
            mod.role = role

    @torch.no_grad()
    def energy_transform(self, x):
        """One proxy batch with pca_lora on (their _pca_lora)."""
        self.eval()
        for mod in self._e2_modules().values():
            mod.pca_lora = True
        self.lora_vit(x)
        for mod in self._e2_modules().values():
            mod.pca_lora = False

    def apply_merged_A(self, p, alpha):
        for mod in self._e2_modules().values():
            mod.apply_merged_A(p, alpha)

    def restore_raw_A(self):
        for mod in self._e2_modules().values():
            mod.restore_raw_A()

    def get_param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- raw structural state (shapes vary per task) ----
    def e2_state(self, dtype=torch.float16):
        state = {"cur_task": self._cur_task, "modules": {}}
        for name, mod in self._e2_modules().items():
            state["modules"][name] = {
                "A": [a.weight.data.cpu().to(dtype) for a in mod.A_list()],
                "B": [b.weight.data.cpu().to(dtype) for b in mod.B_list()],
                "eig": [e.clone() for e in mod.eigenvalues],
            }
        return state

    def load_e2_state(self, state):
        device = next(self.parameters()).device
        self._cur_task = state["cur_task"]
        for name, mod in self._e2_modules().items():
            saved = state["modules"][name]
            A_list, B_list = mod.A_list(), mod.B_list()
            for lst in (A_list, B_list):
                while len(lst) > 0:
                    del lst[0]
            for A_w, B_w in zip(saved["A"], saved["B"]):
                r, in_dim = A_w.shape
                out_dim = B_w.shape[0]
                A = nn.Linear(in_dim, r, bias=False)
                B = nn.Linear(r, out_dim, bias=False)
                A.weight.data.copy_(A_w.float())
                B.weight.data.copy_(B_w.float())
                A_list.append(A.to(device))
                B_list.append(B.to(device))
            mod.eigenvalues = [e.clone() for e in saved["eig"]]
            mod._cur_task = len(saved["A"])
            mod._merged_backup = None
        # freeze per the current task's rule
        task = self._cur_task - 1
        for name, p in self.named_parameters():
            if f"lora_A.{task}" in name or f"lora_A1.{task}" in name:
                p.requires_grad = True
            elif task == 0 and (f"lora_B.{task}" in name or f"lora_B1.{task}" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False


# InfLoRA (Liang & Li, CVPR 2024) =============================================
# Port of github.com/liangyanshuo/InfLoRA (MIT): models/vit_inflora.py
# (Attention_LoRA) and methods/inflora.py (InfLoRA._train, update_DualGPM).
# Upstream keeps one frozen LoRA branch per task on the K and V rows of every
# qkv projection -- lora_A_k/v (768 -> r; the paper's B_t, FIXED, designed by
# an SVD of the task's input Gram matrix) and lora_B_k/v (r -> 768; the paper's
# A_t, TRAINED, zero-init) -- and sums all branches on every forward with no
# alpha/r scaling, dropout or bias. Old branches never change, so folding
# dW_t = B_t A_t into the K/V rows of qkv.weight after task t is exactly
# equivalent to summing the frozen branches. The port therefore follows the
# PlainLoRA merge-family path of main.py (fold via merge_lora_weights,
# trainable_state caching, compute_covariances hooks on lora_modules) and
# needs only two extra calls from main.Learner: design_lora_A before a task
# trains and update_memory after it (both on Gram matrices from
# compute_covariances). The projection math is transcribed in inflora_gpm.py
# (NumPy on the CPU, as upstream); the per-block DualGPM basis lives in
# fixed-size buffers so trainable_state / load_trainable_state round-trip it.


class InfLoRAQKV(nn.Module):
    """InfLoRA branch on one timm qkv Linear (upstream Attention_LoRA, folded).

    Parameters: lora_A_k / lora_A_v (r x d, requires_grad False; designed by
    InfLoRAViT.design_lora_A before every task) and lora_B_k / lora_B_v (d x r,
    trained, zero-init). Buffers: lora_active, and this block's DualGPM memory
    as a zero-padded d x d ``basis`` with ``basis_cols`` used columns and the
    ``basis_retain`` flag (False: the protected subspace M of upstream's
    'remove' type, True: its complement M_perp of the 'retain' type).
    forward adds [0; B_k A_k; B_v A_v] to the weight -- rows 0:d are q, d:2d
    k, 2d:3d v in timm's (B, N, 3, heads, hd) reshape of the qkv output -- i.e.
    upstream's ``k += x (B_k A_k)^T`` and ``v += x (B_v A_v)^T``.
    """

    def __init__(self, linear: nn.Linear, r: int):
        super().__init__()
        assert linear.out_features == 3 * linear.in_features, "InfLoRAQKV expects a fused qkv Linear"
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        d = self.in_features
        w = linear.weight
        # Paper's B_t (code lora_A_k/v): fixed, overwritten by the SVD design.
        self.lora_A_k = nn.Parameter(torch.zeros(r, d, device=w.device, dtype=w.dtype), requires_grad=False)
        self.lora_A_v = nn.Parameter(torch.zeros(r, d, device=w.device, dtype=w.dtype), requires_grad=False)
        # Paper's A_t (code lora_B_k/v): trained, zero-init (branch starts at 0).
        self.lora_B_k = nn.Parameter(torch.zeros(d, r, device=w.device, dtype=w.dtype))
        self.lora_B_v = nn.Parameter(torch.zeros(d, r, device=w.device, dtype=w.dtype))
        self.register_buffer("lora_active", torch.tensor(True))
        # DualGPM memory (upstream feature_list[i] / project_type[i]) in fixed
        # shapes: float32 like upstream's feature_list; cols 0 = empty.
        self.register_buffer("basis", torch.zeros(d, d, dtype=torch.float32))
        self.register_buffer("basis_cols", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("basis_retain", torch.tensor(False))

    def _delta(self) -> torch.Tensor:
        """[0; B_k A_k; B_v A_v] (3d x d): the branch on the K and V rows."""
        dk = self.lora_B_k @ self.lora_A_k
        dv = self.lora_B_v @ self.lora_A_v
        return torch.cat([dk.new_zeros(self.in_features, self.in_features), dk, dv], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_active:
            return F.linear(x, self.linear.weight + self._delta(), self.linear.bias)
        return self.linear(x)

    @torch.no_grad()
    def merge_lora_weights(self, lora_active: bool = True) -> None:
        """Fold the trained branch into qkv.weight and zero B. Unlike
        OriginalLoRA, A is NOT re-initialised: the next task designs it."""
        self.linear.weight.data.add_(self._delta().to(self.linear.weight.dtype))
        self.lora_B_k.data.zero_()
        self.lora_B_v.data.zero_()
        self.lora_active.fill_(lora_active)

    # ---- DualGPM memory / down-projection access (NumPy side) ----
    def get_basis(self):
        """(M as a float64 (d, cols) array, or None when empty; retain flag)."""
        cols = int(self.basis_cols.item())
        M = unpack_basis(self.basis.detach().cpu().numpy(), cols)
        return M, bool(self.basis_retain.item())

    @torch.no_grad()
    def set_basis(self, M, retain: bool) -> None:
        """Store M (d x cols, orthonormal columns) zero-padded; None = empty."""
        self.basis.copy_(torch.from_numpy(pack_basis(M, self.in_features, dtype=np.float32)))
        cols = 0 if M is None else int(np.asarray(M).shape[1])
        self.basis_cols.fill_(cols)
        self.basis_retain.fill_(bool(retain))

    @torch.no_grad()
    def set_lora_A(self, A) -> None:
        """Copy the designed (r x d) matrix into BOTH lora_A_k and lora_A_v
        (upstream assigns the same U[:, :r]^T / sqrt(3) to K and V)."""
        t = torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32))
        self.lora_A_k.data.copy_(t)
        self.lora_A_v.data.copy_(t)


class InfLoRAViT(BaseAdapterViT):
    """InfLoRA: designed-A / trained-B branches on the K,V rows of qkv in the
    chosen blocks (default all 12; [0..4] with inflora_dual False and its own
    lr / eps is upstream's InfLoRA-b5) and a DualGPM input-subspace memory per
    block (upstream models/vit_inflora.py + methods/inflora.py). The wrappers
    are registered in lora_modules as block_{i}_attn_qkv so compute_covariances
    (the Gram passes), merge_lora_weights (the fold), finalize_without_lora
    (snapshots) and trainable_state (caching, basis buffers included) work
    unchanged. use_projection stays False: main.update_projections (an EMA of
    covariances) is the wrong statistic for DualGPM and never runs. Trainable
    parameters per task: 2 x 768 x r per block = 184,320 at r = 10 on 12
    blocks (+ head: upstream's logged 192,010 on CIFAR, 199,700 on IN-R)."""

    def __init__(self, vit_model, r: int = 10, lora_layer: Optional[Iterable[int]] = None):
        super().__init__(vit_model)
        assert r > 0
        self.r = r
        layers = list(lora_layer) if lora_layer is not None else list(range(len(vit_model.blocks)))

        for p in vit_model.parameters():
            p.requires_grad_(False)
        for idx, blk in enumerate(vit_model.blocks):
            if idx not in layers:
                continue
            blk.attn.qkv = InfLoRAQKV(blk.attn.qkv, r)
            self.lora_modules[f"block_{idx}_attn_qkv"] = blk.attn.qkv

    def get_param_groups(self):
        # Upstream _train freezes everything but lora_B_k[t], lora_B_v[t] (and
        # the current head, which main.train_backbone adds separately).
        params = []
        for mod in self.lora_modules.values():
            params.extend([mod.lora_B_k, mod.lora_B_v])
        return params

    @torch.no_grad()
    def design_lora_A(self, grams: Dict[str, np.ndarray]):
        """Upstream InfLoRA._train, 'fix the down-projection': per block,
        A_t = top-r left singular vectors of the current task's Gram after the
        one-sided removal ('remove') or retention ('retain') of the stored
        subspace, transposed and divided by sqrt(3); the raw Gram at task 0.
        `grams` = {lora_modules name: (d, d) Gram}. Returns one row per block
        (name, basis_cols, 'remove' | 'retain', leakage) with leakage =
        inflora_gpm.basis_leakage (max |A M|; ~0 when the design is right)."""
        rows = []
        for name, mod in self.lora_modules.items():
            M, retain = mod.get_basis()
            A = design_down_projection(grams[name], M, retain, mod.r)
            mod.set_lora_A(A)
            cols = 0 if M is None else int(M.shape[1])
            rows.append((name, cols, "retain" if retain else "remove", basis_leakage(A, M, retain)))
        return rows

    @torch.no_grad()
    def update_memory(self, grams: Dict[str, np.ndarray], threshold: float, dual: bool = True):
        """Upstream update_DualGPM (update_GPM when `dual` is False) on the
        post-task Grams: grow / shrink / switch every block's basis at
        `threshold`. Returns (name, basis_cols, 'remove' | 'retain', changed)
        per block; `changed` = columns stored / appended / removed."""
        rows = []
        for name, mod in self.lora_modules.items():
            M, retain = mod.get_basis()
            M2, retain2, changed = dualgpm_step(grams[name], M, retain, threshold, dual)
            mod.set_basis(M2, retain2)
            rows.append((name, int(M2.shape[1]), "retain" if retain2 else "remove", int(changed)))
        return rows


# KeepLoRA ====================================================================
# KeepLoRA (Luo et al., "KeepLoRA: Continual Learning with Residual Gradient
# Adaptation", ICLR 2026, arXiv:2601.19659) ported from the official MTIL code
# (github.com/MaolinLuo/KeepLoRA, branch v1: MTIL/model/peft_modules.py,
# peft_vit.py, prin_subspace.py, trainer/trainer.py, customClip.py). Upstream
# adapts q, k, v (the three row blocks of the fused in_proj) and the attention
# out-projection of every CLIP block with rank-16 LoRA (lora_alpha 1, scaling
# 1/r) and, per task, (1) averages the gradient of the FROZEN weights over one
# pass of the task loader with the adapters off, (2) projects it away from the
# module's protected basis and initialises A = U_r (frozen) and B = S_r V_r^T
# (trained) from the SVD of the residual, (3) subtracts scaling * (A B)^T from
# the weight so the network is unchanged at the start, (4) trains B (+ head)
# with AdamW 1e-3, OneCycleLR, 10 epochs, (5) merges the adapter into the
# weight, (6) runs a no-grad pass with the merged weights to accumulate the
# module inputs and extends the basis at energy 0.99. The basis starts (task 0,
# before any training) from the principal subspace of the pre-trained weight at
# energy 0.85. Every step maps onto the merge-family path of main.py (fold via
# merge_lora_weights, trainable_state caching with the basis in fixed-size
# buffers, compute_covariances-style input hooks for the statistics); the
# linear algebra is keeplora_subspace.py (NumPy, float64, unit-tested here).


class KeepLoRALinear(nn.Module):
    """KeepLoRA on one nn.Linear whose output rows split into ``n_slices``
    equal blocks with their own (A, B, basis): 3 slices for timm's fused qkv
    (q, k, v -- upstream's in_proj_weight rows), 1 for attn.proj (upstream's
    out_proj). Upstream module ``KeepLoRA``: ``lora_A`` (in, r) is a plain
    tensor overwritten by the gradient design before every task, ``lora_B``
    (r, out) the trained parameter, forward adds ``scaling * x A B``.

    Buffers: ``lora_active``; ``basis`` (n_slices, in, in) float32 zero-padded
    and ``basis_cols`` (n_slices,) -- the protected subspace U of every slice,
    carried in trainable_state so a cached task state restores it."""

    def __init__(self, linear: nn.Linear, r: int, n_slices: int, scaling: float):
        super().__init__()
        assert linear.out_features % n_slices == 0, "output rows must split into equal slices"
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = int(r)
        self.n_slices = int(n_slices)
        self.out_slice = linear.out_features // n_slices
        self.scaling = float(scaling)
        d, w = self.in_features, linear.weight
        self.lora_A = nn.Parameter(torch.zeros(n_slices, d, r, device=w.device, dtype=w.dtype),
                                   requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(n_slices, r, self.out_slice, device=w.device, dtype=w.dtype))
        self.register_buffer("lora_active", torch.tensor(True))
        self.register_buffer("basis", torch.zeros(n_slices, d, d, dtype=torch.float32))
        self.register_buffer("basis_cols", torch.zeros(n_slices, dtype=torch.int64))

    def _delta(self) -> torch.Tensor:
        """scaling * (A_s B_s)^T stacked over the slices -> (out, in) in the
        weight's layout (upstream get_delta_weight per slice, rows q | k | v)."""
        d = torch.einsum("sir,sro->soi", self.lora_A, self.lora_B)   # (s, out_slice, in)
        return self.scaling * d.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_active:
            return F.linear(x, self.linear.weight + self._delta(), self.linear.bias)
        return self.linear(x)

    @torch.no_grad()
    def merge_lora_weights(self, lora_active: bool = True) -> None:
        """Fold the adapter into the weight and zero B (upstream
        merge_keeplora_weights). A is kept: the next task redesigns it."""
        self.linear.weight.data.add_(self._delta().to(self.linear.weight.dtype))
        self.lora_B.data.zero_()
        self.lora_active.fill_(lora_active)

    @torch.no_grad()
    def subtract_init(self) -> None:
        """Upstream subtract_keeplora_weights: W <- W - scaling (A B)^T right
        after the SVD initialisation, so W + delta equals the old W."""
        self.linear.weight.data.sub_(self._delta().to(self.linear.weight.dtype))

    def slice_rows(self, s: int):
        return slice(s * self.out_slice, (s + 1) * self.out_slice)

    def get_basis(self, s: int):
        """float64 (in, cols) basis of slice s, or None when empty."""
        return keeplora_unpack_basis(self.basis[s].detach().cpu().numpy(), int(self.basis_cols[s].item()))

    @torch.no_grad()
    def set_basis(self, s: int, M) -> None:
        self.basis[s].copy_(torch.from_numpy(keeplora_pack_basis(M, self.in_features, dtype=np.float32)))
        self.basis_cols[s] = 0 if M is None else int(np.asarray(M).shape[1])

    @torch.no_grad()
    def set_init(self, s: int, A, B) -> None:
        """A (in, r), B (r, out_slice) from keeplora_subspace.gradient_init."""
        self.lora_A.data[s].copy_(torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32)))
        if B is None:
            self.lora_B.data[s].zero_()
        else:
            self.lora_B.data[s].copy_(torch.from_numpy(np.ascontiguousarray(B, dtype=np.float32)))

    def cols(self):
        return [int(c) for c in self.basis_cols.tolist()]


class KeepLoRAViT(BaseAdapterViT):
    """KeepLoRA on the attention of the chosen blocks: ``block_{i}_attn_qkv``
    (3 slices q / k / v) and, with ``use_proj``, ``block_{i}_attn_proj`` (1
    slice), registered in lora_modules so compute_covariances-style hooks,
    merge_lora_weights, finalize_without_lora and trainable_state (which
    carries the basis buffers) work unchanged. Trainable parameters per task:
    B only -- 4 x 768 x r per block = 589,824 at r = 16 on 12 blocks (+ head).
    use_projection stays False (nothing to rebuild from covariances)."""

    def __init__(self, vit_model, r: int = 16, lora_alpha: float = 1.0,
                 lora_layer: Optional[Iterable[int]] = None, use_proj: bool = True):
        super().__init__(vit_model)
        assert r > 0
        self.r = int(r)
        self.scaling = float(lora_alpha) / float(r)        # upstream lora_alpha / r (rsLoRA off)
        layers = list(lora_layer) if lora_layer is not None else list(range(len(vit_model.blocks)))
        for p in vit_model.parameters():
            p.requires_grad_(False)
        for idx, blk in enumerate(vit_model.blocks):
            if idx not in layers:
                continue
            assert blk.attn.qkv.out_features == 3 * blk.attn.qkv.in_features, "expects timm's fused qkv"
            blk.attn.qkv = KeepLoRALinear(blk.attn.qkv, self.r, 3, self.scaling)
            self.lora_modules[f"block_{idx}_attn_qkv"] = blk.attn.qkv
            if use_proj:
                blk.attn.proj = KeepLoRALinear(blk.attn.proj, self.r, 1, self.scaling)
                self.lora_modules[f"block_{idx}_attn_proj"] = blk.attn.proj

    def get_param_groups(self):
        # Upstream build_optimizer: only the tuner's parameters (lora_B; lora_A
        # is a plain tensor) plus the head, which main.train_backbone adds.
        return [mod.lora_B for mod in self.lora_modules.values()]

    def set_active(self, flag: bool) -> None:
        for mod in self.lora_modules.values():
            mod.lora_active.fill_(bool(flag))

    def base_weights(self) -> Dict[str, torch.Tensor]:
        """{lora_modules name: the frozen nn.Linear weight} -- the tensors
        whose task gradient initialises the adapter."""
        return {name: mod.linear.weight for name, mod in self.lora_modules.items()}

    def basis_cols(self) -> Dict[str, List[int]]:
        return {name: mod.cols() for name, mod in self.lora_modules.items()}

    @torch.no_grad()
    def init_weight_subspace(self, eps_w: float, side: str = "output"):
        """Task-0 step of upstream Framework4MTIL.train_and_evaluate: the
        principal subspace of every pre-trained slice weight at energy eps_w
        becomes the first block of its basis (skipped when eps_w <= 1e-8).
        Returns [(name, slice, cols)]."""
        rows = []
        for name, mod in self.lora_modules.items():
            for s in range(mod.n_slices):
                W = mod.linear.weight.data[mod.slice_rows(s)].detach().cpu().double().numpy()
                M = keeplora_weight_subspace(W, eps_w, side)
                mod.set_basis(s, M)
                rows.append((name, s, 0 if M is None else int(M.shape[1])))
        return rows

    @torch.no_grad()
    def init_from_gradients(self, grads: Dict[str, torch.Tensor], b_init: str = "svd"):
        """Upstream init_keeplora_with_grad + subtract_keeplora_weights.
        ``grads``: {lora_modules name: averaged weight.grad (out, in)}. Per
        slice: G = grad[rows]^T (paper layout), projected away from the basis,
        A = U_r, B = S_r V_r^T (``b_init='svd'``, branch v1) or B = 0
        (``'zero'``, the KeepLoRA++ branch); with 'svd' the weight is then
        shifted by -scaling (A B)^T so the forward pass starts unchanged.
        Returns [(name, slice, cols, leak = max|U^T A|)]."""
        if b_init not in ("svd", "zero"):
            raise ValueError(f"keeplora_b_init must be 'svd' or 'zero', got {b_init!r}")
        rows = []
        for name, mod in self.lora_modules.items():
            g = grads[name].detach().cpu().double()
            for s in range(mod.n_slices):
                G = g[mod.slice_rows(s)].t().numpy()                 # (in, out_slice)
                M = mod.get_basis(s)
                A, B = keeplora_gradient_init(G, M, mod.r)
                mod.set_init(s, A, B if b_init == "svd" else None)
                rows.append((name, s, 0 if M is None else int(M.shape[1]), keeplora_basis_leakage(A, M)))
            if b_init == "svd":
                mod.subtract_init()
        return rows

    @torch.no_grad()
    def update_feature_subspace(self, grams: Dict[str, np.ndarray], eps_f: float):
        """Upstream PrinSubspace.update_prin_subspace on the accumulated
        features: extend every slice's basis with the residual directions of
        the module input Gram at energy eps_f (the slices of one module share
        the input, so they see the same Gram but start from their own W_p).
        Returns [(name, slice, cols, added)]."""
        rows = []
        for name, mod in self.lora_modules.items():
            K = grams[name]
            for s in range(mod.n_slices):
                M, added = keeplora_extend_subspace(mod.get_basis(s), K, eps_f)
                mod.set_basis(s, M)
                rows.append((name, s, int(M.shape[1]), int(added)))
        return rows


@torch.no_grad()
def compute_keeplora_grams(backbone, data_loader, mode="code", device="cuda"):
    """Input statistics of every KeepLoRA module over ``data_loader`` (this
    repo's (idx, path, x, y) batches), as Grams {name: float64 (in, in)}.

    ``mode='code'``: upstream accumulate_features -- the element-wise mean over
    batches of the token matrix X_b^T (in, B*L), kept as a running sum on the
    device; its Gram Xbar Xbar^T is what keeplora_subspace.extend_subspace
    decomposes. Batches whose token count differs from the first one are
    skipped (upstream needs drop_last=True; the caller's loader sets it) and
    counted in the returned ``info``. ``mode='gram'``: the uncentred Gram over
    all tokens X^T X / n (plain GPM; paper Eq. 3-4 taken literally).
    Returns (grams, info) with info = {name: (n_batches_used, n_skipped)}."""
    if mode not in ("code", "gram"):
        raise ValueError(f"keeplora_feat_mode must be 'code' or 'gram', got {mode!r}")
    modules = {name: backbone.lora_modules[name] for name in backbone.get_module_names()}
    acc = {name: None for name in modules}
    used = {name: 0 for name in modules}
    skipped = {name: 0 for name in modules}
    hooks = []
    for name, module in modules.items():
        def hook_fn(module, inputs, output, name=name):
            x = inputs[0].detach().to(device)
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.float()
            if mode == "gram":
                stat = x.t() @ x
                acc[name] = stat if acc[name] is None else acc[name] + stat
                used[name] += x.shape[0]
                return
            xt = x.t()                                   # (in, tokens)
            if acc[name] is None:
                acc[name] = xt.clone()
                used[name] = 1
            elif xt.shape != acc[name].shape:
                skipped[name] += 1
            else:
                acc[name] += xt
                used[name] += 1
        hooks.append(module.register_forward_hook(hook_fn))
    backbone.to(device)
    backbone.eval()
    for batch in data_loader:
        backbone(batch[-2].to(device))
    for h in hooks:
        h.remove()
    grams = {}
    for name in modules:
        if acc[name] is None or used[name] == 0:
            raise RuntimeError(f"compute_keeplora_grams: no features collected for {name}")
        if mode == "gram":
            grams[name] = (acc[name] / used[name]).double().cpu().numpy()
        else:
            xbar = acc[name] / used[name]
            grams[name] = (xbar @ xbar.t()).double().cpu().numpy()
    info = {name: (used[name], skipped[name]) for name in modules}
    return grams, info


# Factory =====================================================================
MERGE_FAMILY = (
    "basic_lora", "sgp_lora", "nsp_lora", "full", "full_nsp",
    "joint_lora", "joint_full", "first_task_lora", "e2lora", "inflora", "keeplora",
)


def build_recipe_backbone(config):
    """Build the wrapped ViT for a merge-family train_method (seqft uses
    helper.get_backbone instead)."""
    method = config["train_method"]
    if method == "e2lora":
        # Their native backbone (e2lora_inr_lora.json): ViT-B/16 IN21K.
        # e2lora_budget: total rank budget per layer (None = their design,
        # the full output dimension; small values = budgeted variant).
        vit = timm.create_model(
            config.get("e2lora_backbone", "vit_base_patch16_224_in21k"),
            pretrained=True, num_classes=0)
        return E2LoRAViT(vit, budget=config.get("e2lora_budget", None))
    if method == "inflora":
        # InfLoRA's published backbone (sinet_inflora.py with timm 0.6.7):
        # 'vit_base_patch16_224_in21k' = AugReg ViT-B/16 pre-trained on
        # ImageNet-21k only; in timm 1.0.x that name is a deprecation alias
        # of 'vit_base_patch16_224.augreg_in21k' (the tag e2lora loads too).
        # inflora_backbone 'vit_base_patch16_224' gives our matched backbone
        # (augreg2_in21k_ft_in1k). inflora_layers: blocks with a branch
        # (None = all 12); lora_r: the branch rank (upstream 10).
        name = config.get("inflora_backbone", "vit_base_patch16_224_in21k")
        vit = timm.create_model(name, pretrained=True, num_classes=0)
        pcfg = getattr(vit, "pretrained_cfg", None) or {}
        logging.info(f"[InfLoRA] backbone {name} -> {pcfg.get('architecture')}.{pcfg.get('tag')}")
        return InfLoRAViT(vit, r=config.get("lora_r", 10),
                          lora_layer=config.get("inflora_layers", None))
    if method == "keeplora":
        # KeepLoRA has no published class-incremental ViT setting, so the
        # default is our matched backbone (vit_base_patch16_224 = augreg2
        # IN21k -> IN1k, as every other recipe); keeplora_backbone
        # 'vit_base_patch16_224_in21k' gives the IN21k-only weights (the
        # E2-LoRA / InfLoRA published backbone). lora_r 16, keeplora_alpha 1
        # (scaling 1/r), q/k/v + proj on all 12 blocks follow the released
        # MTIL config (keeplora_order1.yaml, v_adapter_dim 16).
        name = config.get("keeplora_backbone", "vit_base_patch16_224")
        vit = timm.create_model(name, pretrained=True, num_classes=0)
        pcfg = getattr(vit, "pretrained_cfg", None) or {}
        logging.info(f"[KeepLoRA] backbone {name} -> {pcfg.get('architecture')}.{pcfg.get('tag')}")
        return KeepLoRAViT(vit, r=config.get("lora_r", 16),
                           lora_alpha=config.get("keeplora_alpha", 1.0),
                           lora_layer=config.get("keeplora_layers", None),
                           use_proj=bool(config.get("keeplora_proj", True)))
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    if config.get("replace_final_norm", False):
        # Reference featurization (inc_net.py:36); default off to match ours.
        vit.norm = nn.LayerNorm(vit.embed_dim, elementwise_affine=False)

    r = config.get("lora_r", 4)
    if method in ("basic_lora", "joint_lora", "first_task_lora"):
        return PlainLoRAViT(vit, r=r, lora_scale=config.get("lora_scale", 1.0))
    if method in ("sgp_lora", "nsp_lora"):
        return SGPLoRAViT(
            vit, r=r,
            use_soft_projection=(method == "sgp_lora"),
            weight_temp=config.get("weight_temp", 2.0),
            weight_kind=config.get("weight_kind", "log1p"),
            weight_p=config.get("weight_p", 1.0),
            nsp_eps=config.get("nsp_eps", 0.05),
            nsp_weight=config.get("nsp_weight", 0.0),
            sgp_module=config.get("sgp_module", "dora"),
        )
    if method in ("full", "joint_full"):
        return FullFinetuneViT(vit)
    if method == "full_nsp":
        return NullSpaceViT(
            vit,
            nsp_eps=config.get("nsp_eps", 0.05),
            nsp_weight=config.get("nsp_weight", 0.02),
        )
    raise ValueError(f"Unknown merge-family train_method {method!r}")


# Checkpoint helpers ==========================================================
def trainable_state(adapter: BaseAdapterViT, dtype=None) -> Dict[str, torch.Tensor]:
    """Everything a recipe mutates across tasks: the adapted modules (folded
    linear weights, A/B, DoRA directions/magnitudes) plus the final norm.
    Projection buffers (.P.P) are excluded -- they are rebuilt from the cached
    covariance EMA and would triple the checkpoint size."""
    state = {}
    for key, tensor in adapter.state_dict().items():
        if (key.startswith("lora_modules.") or key.startswith("lora_vit.norm.")) \
                and not key.endswith(".P.P"):
            state[key] = tensor.cpu() if dtype is None else tensor.cpu().to(dtype)
    return state


def load_trainable_state(adapter: BaseAdapterViT, state: Dict[str, torch.Tensor]) -> None:
    device = next(adapter.parameters()).device
    ref = adapter.state_dict()
    cast = {k: v.to(device=device, dtype=ref[k].dtype) for k, v in state.items() if k in ref}
    missing_ok = adapter.load_state_dict(cast, strict=False)
    unexpected = [k for k in state if k not in ref]
    if unexpected:
        logging.warning(f"[Checkpoint] {len(unexpected)} unexpected keys ignored: {unexpected[:3]}...")
