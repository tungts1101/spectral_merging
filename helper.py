from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
import timm
from timm.models.layers import trunc_normal_
from peft import get_peft_model, LoraConfig
import numpy as np
import logging
import random
import math


# Helper functions ============================================================
def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_metrics(accuracy_matrix):
    faa = np.mean(accuracy_matrix[-1])

    session_averages = []
    for i in range(accuracy_matrix.shape[0]):
        session_avg = np.mean(
            accuracy_matrix[i, : i + 1]
        )  # calculate total accuracy per session
        session_averages.append(session_avg)
    asa = np.mean(session_averages)

    if accuracy_matrix.shape[0] == 1:
        return faa, 0.0, 0.0, asa

    final_acc_per_task = accuracy_matrix[-1]
    max_acc_per_task = np.max(accuracy_matrix, axis=0)
    ffm = np.mean(max_acc_per_task[:-1] - final_acc_per_task[:-1])
    ffd = np.max(max_acc_per_task[:-1] - final_acc_per_task[:-1]) - np.min(
        max_acc_per_task[:-1] - final_acc_per_task[:-1]
    )

    return faa, ffm, ffd, asa


def setup_logger(log_file=f"logs/default.log", logger_name=None):
    if logger_name is None:
        logger_name = f"logger_{log_file.replace('/', '_').replace('.', '_')}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_formatter = logging.Formatter("%(asctime)s [%(filename)s] => %(message)s")

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def trim(tensor, topk=100):
    flattened = tensor.view(-1)
    magnitudes = torch.abs(flattened)
    num_keep = max(1, int(len(flattened) * topk / 100))
    threshold = torch.topk(magnitudes, num_keep, largest=True, sorted=True).values[-1]
    mask = magnitudes >= threshold
    trimmed = torch.where(mask, flattened, torch.tensor(0.0, dtype=tensor.dtype))

    gamma = torch.sign(trimmed)
    mu = torch.abs(trimmed)

    return (trimmed.view_as(tensor), gamma.view_as(tensor), mu.view_as(tensor))


def merge_task_vectors(trimmed_task_vectors):
    gamma_tvs = torch.stack([tv[1] for tv in trimmed_task_vectors], dim=0)
    gamma = torch.sign(gamma_tvs.sum(dim=0))
    mask = gamma_tvs == gamma
    tau_tvs = torch.stack([tv[0] for tv in trimmed_task_vectors], dim=0)
    mean_tvs = torch.where(mask, tau_tvs, torch.tensor(0.0, dtype=tau_tvs.dtype)).sum(
        dim=0
    ) / mask.sum(dim=0).clamp(min=1)

    return mean_tvs


def truncated_svd(matrix, k):
    """Rank-k approximation of a 2D matrix via SVD: A_k = U_k Σ_k V_k^T."""
    U, S, Vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    k = max(1, min(int(k), S.shape[0]))
    A_k = (U[:, :k] * S[:k]) @ Vh[:k, :]
    return A_k.to(matrix.dtype)


def spectral_merging(w_prev, w_new, k=16, alpha=1.0):
    """Spectral-norm-aware merge of two consecutive weight states.

    Given the previously accumulated weights W_t (`w_prev`) and the newly trained
    weights W_{t+1} (`w_new`), form the task update A = W_{t+1} - W_t, retain only
    its top-k singular components A_k = U_k Σ_k V_k^T, and merge as

        C = W_t + alpha * A_k,   0 < alpha <= 1.

    A smaller `alpha` or `k` yields a more conservative merge while preserving the
    principal update directions of task t+1. 1D parameters (biases, norms) have no
    low-rank structure and are merged directly.
    """
    merged = {}
    for name in w_new:
        w_t = w_prev[name]
        A = w_new[name] - w_t
        A_k = truncated_svd(A, k) if A.dim() == 2 else A
        merged[name] = w_t + alpha * A_k
    return merged


def truncate_energy(A, p):
    """Rank-k SVD approximation of A keeping the SMALLEST set of top singular
    values whose cumulative energy (sum of squares) reaches fraction p of the
    total. p in (0,1]; p<1 keeps k < full rank."""
    U, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)
    if S.numel() == 0 or float(S[0]) == 0.0:
        return A.clone()
    e = S ** 2
    cum = torch.cumsum(e, 0) / e.sum()
    k = int((cum < p).sum().item()) + 1          # smallest k with cum >= p
    k = max(1, min(k, S.numel()))
    return ((U[:, :k] * S[:k]) @ Vh[:k, :]).to(A.dtype)


def pspectral_merging(w_prev, w_new, p=0.5, alpha=1.0, variant="combine"):
    """Energy-percentile spectral merge for LoRA adapters (4 variants).

    Incremental merge  C = W_prev + alpha * truncate_energy(W_new - W_prev, p),
    where truncation keeps the smallest set of top singular values reaching p of
    the update's energy. `variant` selects WHAT is truncated:
      only_A   : truncate the lora_A update; lora_B merged by plain interpolation
      only_B   : truncate the lora_B update; lora_A merged plain
      separate : truncate lora_A and lora_B updates separately (before multiply)
      combine  : truncate the effective update W = B@A, then re-factor to rank r
    Non-LoRA params (norms/biases) are merged by plain interpolation.
    """
    # Detect down/up (A/B) factor pairs by suffix, supporting multiple PEFT types:
    #   LoRA        : lora_A.default.weight (A=down [r,in]),  lora_B.default.weight (B=up [out,r])
    #   AdaptFormer : down_proj.weight      (A=down [r,in]),  up_proj.weight        (B=up [out,r])
    FACTOR_SUFFIXES = [
        ("lora_A.default.weight", "lora_B.default.weight"),
        ("down_proj.weight",      "up_proj.weight"),
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
            merged[An] = (A_prev + alpha * truncate_energy(dA, p)).to(w_new[An].dtype)
            merged[Bn] = (B_prev + alpha * dB).to(w_new[Bn].dtype)
        elif variant == "only_B":
            merged[An] = (A_prev + alpha * dA).to(w_new[An].dtype)
            merged[Bn] = (B_prev + alpha * truncate_energy(dB, p)).to(w_new[Bn].dtype)
        elif variant == "separate":
            merged[An] = (A_prev + alpha * truncate_energy(dA, p)).to(w_new[An].dtype)
            merged[Bn] = (B_prev + alpha * truncate_energy(dB, p)).to(w_new[Bn].dtype)
        elif variant == "combine":
            r = A_new.shape[0]
            Wp, Wn = B_prev @ A_prev, B_new @ A_new
            C = Wp + alpha * truncate_energy(Wn - Wp, p)
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
            rr = min(r, S.shape[0])
            s = S[:rr].clamp(min=0).sqrt()
            merged[Bn] = (U[:, :rr] * s).to(w_new[Bn].dtype)
            merged[An] = (s.unsqueeze(1) * Vh[:rr, :]).to(w_new[An].dtype)
        else:
            raise ValueError(f"unknown pspectral variant {variant!r}")

    for name in w_new:
        if name in lora_names:
            continue
        w_t = w_prev[name]
        merged[name] = w_t + alpha * (w_new[name] - w_t)
    return merged


def spectral_merging_lora(w_prev, w_new, k=16, alpha=1.0):
    """Effective-weight spectral merge for LoRA adapters (option B).

    Truncating the LoRA factors lora_A / lora_B separately is a no-op once
    k >= lora_rank. Instead reconstruct the EFFECTIVE weight update W = B @ A per
    adapted layer, truncate the incremental weight-space update to rank-k, merge,
    then re-factor the merged update back into rank-r LoRA factors so the result
    is still a valid adapter the model can load:

        W_prev = B_prev @ A_prev,   W_new = B_new @ A_new
        C      = W_prev + alpha * truncated_svd(W_new - W_prev, k)
        C = U S V^T  ->  lora_B = U_r * sqrt(S_r),  lora_A = sqrt(S_r) * V_r^T

    With r = lora rank, k < r truncates the update meaningfully (the update
    W_new - W_prev has rank up to 2r, so k up to ~2r carries information).
    Non-LoRA params (norms/biases) are merged by direct interpolation.
    """
    # Discover lora_A / lora_B pairs by shared layer prefix.
    pairs = {}
    for name in w_new:
        if name.endswith("lora_A.default.weight"):
            pairs.setdefault(name[: -len("lora_A.default.weight")], {})["A"] = name
        elif name.endswith("lora_B.default.weight"):
            pairs.setdefault(name[: -len("lora_B.default.weight")], {})["B"] = name

    merged = {}
    lora_names = set()
    for _, d in pairs.items():
        if "A" not in d or "B" not in d:
            continue
        An, Bn = d["A"], d["B"]
        lora_names.update((An, Bn))
        A_prev, B_prev = w_prev[An].float(), w_prev[Bn].float()
        A_new, B_new = w_new[An].float(), w_new[Bn].float()
        r = A_new.shape[0]                          # lora rank (lora_A is [r, in])
        Wp = B_prev @ A_prev                        # [out, in] effective prev update
        Wn = B_new @ A_new                          # [out, in] effective new update
        C = Wp + alpha * truncated_svd(Wn - Wp, k)  # merged effective update
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        rr = min(r, S.shape[0])
        s = S[:rr].clamp(min=0).sqrt()
        merged[Bn] = (U[:, :rr] * s).to(w_new[Bn].dtype)          # [out, r]
        merged[An] = (s.unsqueeze(1) * Vh[:rr, :]).to(w_new[An].dtype)  # [r, in]

    # Non-LoRA params (norms/biases): direct interpolation.
    for name in w_new:
        if name in lora_names:
            continue
        w_t = w_prev[name]
        merged[name] = w_t + alpha * (w_new[name] - w_t)
    return merged


def merge(base_params, tasks_params, method="ties", lamb=1.0, topk=100):
    params = {}
    for name in tasks_params[0]:
        base_tv = base_params[name].clone()
        task_vectors = [task_params[name] for task_params in tasks_params]

        tvs = [task_vectors[i] - base_tv for i in range(len(task_vectors))]

        if method == "ties":
            tvs = [trim(tv, topk) for tv in tvs]
            merged_tv = merge_task_vectors(tvs)
        elif method == "max":
            merged_tv = torch.max(torch.stack(tvs, dim=0), dim=0)[0]
        elif method == "min":
            merged_tv = torch.min(torch.stack(tvs, dim=0), dim=0)[0]
        elif method == "max_abs":
            stacked = torch.stack(tvs, dim=0)
            abs_stacked = torch.abs(stacked)
            max_idx = torch.argmax(abs_stacked, dim=0)
            merged_tv = torch.gather(stacked, 0, max_idx.unsqueeze(0)).squeeze(0)
        elif method == "avg":
            merged_tv = torch.mean(torch.stack(tvs, dim=0), dim=0)
        else:
            raise ValueError(f"Unknown merge method: {method!r}")

        params[name] = base_tv + lamb * merged_tv

    return params


def set_random(seed):
    import os, random, torch, numpy as np

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def accuracy(y_pred, y_true, class_increments):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = []
    acc_total = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

    for task_id, classes in enumerate(class_increments):
        idxes = np.where(np.logical_and(y_true >= classes[0], y_true <= classes[1]))[0]
        all_acc.append(
            np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )
        )

    return acc_total, all_acc


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
# =============================================================================


# Backbone ====================================================================
def get_backbone(args):
    name = args["model_backbone"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif "lora" in name:
        model = timm.create_model(name[:-5], pretrained=True, num_classes=0)
        model.requires_grad_(False)
        outdim = args.get("model_outdim", 768)
        model.out_dim = outdim
        lora_config = LoraConfig(
            r=args["model_lora_r"],
            lora_alpha=args["model_lora_alpha"],
            target_modules=args["model_lora_target_modules"],
            lora_dropout=args["model_lora_dropout"],
            bias="none",
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        return model
    elif '_ssf' in name:
        from backbone import vit_ssf
        if name == "pretrained_vit_b16_224_ssf":
            model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
            model.out_dim = 768
        elif name == "pretrained_vit_b16_224_in21k_ssf":
            model = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
            model.out_dim = 768
        
        for name, param in model.named_parameters():
            if "ssf_" not in name and "ssf_" not in name: 
                param.requires_grad = False
        return model.eval()
    elif '_vpt' in name:
        from backbone.vpt import build_promptmodel
        if name == "pretrained_vit_b16_224_vpt":
            basicmodelname = "vit_base_patch16_224" 
        elif name == "pretrained_vit_b16_224_in21k_vpt":
            basicmodelname = "vit_base_patch16_224_in21k"
        
        print("modelname,", name, "basicmodelname", basicmodelname)
        VPT_type = "Deep"
        if args["vpt_type"] == 'shallow':
            VPT_type = "Shallow"
        Prompt_Token_num = args["prompt_token_num"]

        model = build_promptmodel(modelname=basicmodelname, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
        prompt_state_dict = model.obtain_prompt()
        model.load_prompt(prompt_state_dict)
        model.out_dim = 768
        
        for param_name, param in model.named_parameters():
            if "prompt" not in param_name.lower():
                param.requires_grad = False
        return model.eval()
    elif '_adapter' in name:
        ffn_num = args["ffn_num"]
        from backbone import vit_adapter
        from easydict import EasyDict
        tuning_config = EasyDict(
            # AdaptFormer
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            # VPT related
            vpt_on=False,
            vpt_num=0,
        )
        if name == "pretrained_vit_b16_224_adapter":
            model = vit_adapter.vit_base_patch16_224_adapter(num_classes=0,
                global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            model.out_dim=768
        elif name == "pretrained_vit_b16_224_in21k_adapter":
            model = vit_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            model.out_dim=768
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        for param_name, param in model.named_parameters():
            if "adapt" not in param_name.lower():
                param.requires_grad = False
            # else:
            #     print(f"Trainable params: {param_name}")
        return model.eval()
    else:
        raise NotImplementedError("Unknown type {}".format(name))


# =============================================================================


# Classifier ==================================================================
class ContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, with_norm=False, with_bias=False, norm_layer=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.with_norm = with_norm
        self.with_bias = with_bias
        self.norm_layer = norm_layer

        if with_norm and norm_layer is None:
            self.norm_layer = "ln"

        self.heads = nn.ModuleList([])
        self.update(nb_classes)

    def create_head(self, nb_classes):
        # single_head = []
        # if self.with_norm:
        #     single_head.append(nn.LayerNorm(self.embed_dim))
        # fc = nn.Linear(self.embed_dim, self.embed_dim * 2, bias=self.with_bias)
        # trunc_normal_(fc.weight, std=0.02)
        # if self.with_bias:
        #     nn.init.constant_(fc.bias, 0)
        # single_head.append(fc)

        # # combine layers
        # single_head.append(nn.GELU())
        # fc2 = nn.Linear(self.embed_dim * 2, nb_classes, bias=self.with_bias)
        # trunc_normal_(fc2.weight, std=0.02)
        # if self.with_bias:
        #     nn.init.constant_(fc2.bias, 0)
        # single_head.append(fc2)

        # head = nn.Sequential(*single_head)
        # return head

        single_head = []
        if self.with_norm:
            if self.norm_layer == "bn":
                single_head.append(nn.BatchNorm1d(self.embed_dim))
            elif self.norm_layer == "ln":
                single_head.append(nn.LayerNorm(self.embed_dim))
        fc = nn.Linear(self.embed_dim, nb_classes, bias=self.with_bias)
        trunc_normal_(fc.weight, std=0.02)
        if self.with_bias:
            nn.init.constant_(fc.bias, 0)
        single_head.append(fc)

        head = nn.Sequential(*single_head)
        return head

    def update(self, nb_classes, freeze_old=True):
        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad = False
        single_head = self.create_head(nb_classes)
        self.heads.append(single_head)

    def forward(self, x, return_dict=False):
        out = []
        for i, head in enumerate(self.heads):
            logits_i = head(x)

            out.append(logits_i)

        if return_dict:
            return {"logits": torch.cat(out, dim=1)}
        return torch.cat(out, dim=1)


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, return_dict=False):
        out = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if return_dict:
            return {'logits': out}
        return out
# =============================================================================


# Model =======================================================================
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.backbone = get_backbone(config)
        self.norm = None if not config.get("model_use_norm", False) else nn.LayerNorm(self.backbone.num_features)
        self.classifier = None

    @property
    def feature_dim(self):
        return self.backbone.num_features

    def update_classifier(self, num_classes, with_norm=False, with_bias=False, freeze_old=True, norm_layer=None):
        if self.classifier == None:
            self.classifier = ContinualLinear(
                self.feature_dim, num_classes, with_norm=with_norm, with_bias=with_bias, norm_layer=norm_layer
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

    def get_features(self, x, return_layer_features=False):
        self.layer_features = []

        if return_layer_features:
            hooks = []
            # timm ViT: blocks; HuggingFace ViT: encoder.layer
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                blocks = getattr(getattr(self.backbone, "encoder", None), "layer", None)

            if blocks is not None:
                def _make_hook():
                    def _hook(module, input, output):
                        feat = output[0] if isinstance(output, tuple) else output
                        self.layer_features.append(feat)
                    return _hook

                for block in blocks:
                    hooks.append(block.register_forward_hook(_make_hook()))

        z = self.backbone(x)

        if return_layer_features:
            for hook in hooks:
                hook.remove()

        if self.norm is not None:
            z = self.norm(z)
        return z

    def forward_from_block(self, feats, start_block):
        """Run backbone from `start_block` onwards.

        Args:
            feats:       (B, seq_len, D) full token sequence, or (B, D) CLS-only
            start_block: int, first block index to execute (0-indexed)
        Returns:
            (B, D) final CLS features after norm
        """
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise RuntimeError("backbone.blocks not found; only timm ViT is supported")

        if feats.dim() == 2:
            # CLS-only path: pad with one dummy patch token to keep attention valid
            B, D = feats.shape
            dummy_patch = torch.zeros(B, 1, D, device=feats.device, dtype=feats.dtype)
            z = torch.cat([feats.unsqueeze(1), dummy_patch], dim=1)  # (B, 2, D)
        else:
            z = feats  # (B, seq_len, D) — full token sequence passed as-is

        for i in range(start_block, len(blocks)):
            z = grad_ckpt(blocks[i], z, use_reentrant=False)

        z = self.backbone.norm(z)[:, 0]  # CLS token after backbone norm
        if self.norm is not None:
            z = self.norm(z)
        return z

    def forward(self, x):
        z = self.get_features(x)
        y = self.classifier(z)
        return y

    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params:,}, total_params={total_params:,}, percentage={trainable_params * 100 / total_params:.2f})"

    def get_backbone_info(self):
        trainable_params = count_parameters(self.backbone, trainable=True)
        total_params = count_parameters(self.backbone)
        return f"Backbone(trainable_params={trainable_params:,}, total_params={total_params:,}, percentage={trainable_params * 100 / total_params:.2f})"


# =============================================================================


# Feature Sampler =============================================================

def _make_generator(class_idx: int, layer_key, n_samples: int, call_count: int = 0) -> torch.Generator:
    """Deterministic per-call generator seeded from (class_idx, layer_key, n_samples, call_count).

    call_count is incremented by the sampler on each sample() call, giving different
    draws across training epochs while remaining fully reproducible for a given call index.
    """
    layer_key_int = layer_key if isinstance(layer_key, int) else sum(ord(c) for c in str(layer_key))
    seed = (class_idx * 2654435761 ^ layer_key_int * 40503 ^ n_samples * 12345 ^ call_count * 6700417) % (2 ** 32)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


class IntermediateFeatureSampler:
    """Store per-class, per-layer Gaussian statistics and draw synthetic samples.

    Feature shapes:
      - Intermediate layers : (N, T, D)  — N samples, T tokens (e.g. 197), D dims
      - Final layer         : (N, D)     — stored as (N, 1, D) internally

    Stored per layer_key (int or "final"):
      _means[layer_key]  : (num_classes, T, D)
      _sigmas[layer_key] : (num_classes, T, D)
    """

    def __init__(
        self,
        total_classes: int,
        token_length: int = 197,
        feature_dim: int = 768,
    ) -> None:
        self.total_classes = total_classes
        self.token_length = token_length
        self.feature_dim = feature_dim

        self._means  = {}  # layer_key -> (total_classes, T, D)
        self._sigmas = {}  # layer_key -> (total_classes, T, D)

        self._sample_counts: Dict[tuple, int] = {}  # (class_idx, layer_key) -> call count

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def update(self, features: torch.Tensor, class_idx: int, layer_key) -> None:
        """Compute and store mean/sigma for class_idx at layer_key.

        Args:
            features:  (N, T, D) for intermediate layers, (N, D) for the final layer.
            class_idx: Global class index.
            layer_key: Layer identifier (int index or "final").
        """
        features = features.detach().cpu().float()
        if features.ndim == 2:
            features = features.unsqueeze(1)   # (N, 1, D)

        _, T, D = features.shape
        self._ensure_capacity(layer_key, T, D)

        sigma, mean = torch.std_mean(features, dim=0, correction=1)
        self._means[layer_key][class_idx] = mean
        self._sigmas[layer_key][class_idx] = sigma.nan_to_num(0.0)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, class_idx: int, layer_key, n_samples: int) -> torch.Tensor:
        """Draw n_samples synthetic features from per-element Gaussian.

        Each call increments an internal counter so repeated calls produce
        different (but reproducible) samples.

        Returns:
            (n_samples, T, D) on CPU.
        """
        key = (class_idx, layer_key)
        call_count = self._sample_counts.get(key, 0)
        self._sample_counts[key] = call_count + 1

        mean  = self._means[layer_key][class_idx]
        sigma = self._sigmas[layer_key][class_idx]
        gen = _make_generator(class_idx, layer_key, n_samples, call_count)
        return mean.unsqueeze(0) + torch.randn(n_samples, *mean.shape, generator=gen) * sigma.unsqueeze(0)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_mean(self, class_idx: int, layer_key) -> torch.Tensor:
        """(T, D)"""
        return self._means[layer_key][class_idx]

    def get_cls_mean(self, class_idx: int, layer_key) -> torch.Tensor:
        """CLS token mean — (D,)."""
        return self._means[layer_key][class_idx][0]

    def get_sigma(self, class_idx: int, layer_key) -> torch.Tensor:
        """CLS token sigma — (D,)."""
        return self._sigmas[layer_key][class_idx][0]

    def expand_to(self, new_total_classes: int) -> None:
        """Grow storage to accommodate more classes (preserves existing data)."""
        for key in list(self._means.keys()):
            old_m = self._means[key]
            if new_total_classes <= old_m.shape[0]:
                continue
            T, D = old_m.shape[1], old_m.shape[2]
            new_m = torch.zeros(new_total_classes, T, D)
            new_m[:old_m.shape[0]] = old_m
            self._means[key] = new_m

            # _sigmas is always allocated for all strategies.
            old_s = self._sigmas[key]
            new_s = torch.zeros(new_total_classes, T, D)
            new_s[:old_s.shape[0]] = old_s
            self._sigmas[key] = new_s

        self.total_classes = new_total_classes

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ensure_capacity(self, layer_key, T: int, D: int) -> None:
        if layer_key not in self._means:
            self._means[layer_key] = torch.zeros(self.total_classes, T, D)
            # _sigmas always allocated — needed for patch tokens in all strategies.
            self._sigmas[layer_key] = torch.zeros(self.total_classes, T, D)

# =============================================================================
