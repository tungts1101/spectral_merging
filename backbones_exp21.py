"""Backbone adapters for exp21: ports of the training recipes from
https://github.com/raoxuan98-hash/lr_rgda_hopdc (models/basic_lora.py,
models/sgp_lora.py, models/full_finetune.py, models/distillator.py, lora.py).

Self-contained (torch + timm only). All wrappers share one interface driven by
exp21.Learner:
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

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    the exp21 fold/covariance paths must not touch the pair lists."""

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


# Factory =====================================================================
MERGE_FAMILY = (
    "basic_lora", "sgp_lora", "nsp_lora", "full", "full_nsp",
    "joint_lora", "joint_full", "first_task_lora", "e2lora",
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
