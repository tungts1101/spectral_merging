"""Unified loading shim for the robustness suite.

Every exported final model loads into the same interface:

    m = load_model(path)
    m.logits(x)    -> [B, C]  (x: [0,1] 224x224 tensor, see m.preprocess)
    m.features(x)  -> [B, d]
    m.class_stats  -> (means [C,d], covs [C,d,d]) or None
    m.classes      -> C
    m.map_labels(y) original dataset labels -> the model's shuffled class indices

Exports understood:
  main        : {"kind": "main", config, backbone (LoRA trainables, plus the
                 model-level norm.* if model_use_norm), classifier, known_classes,
                 class_means / class_covs}. Written by main.run_single_experiment
                 when config["final_model_dir"] is set (run_robustness_models.py).
  main_pickle : {"kind": "main_pickle", config, model (whole module),
                 known_classes}. E2-LoRA / InfLoRA / KeepLoRA exports.
  exp21 / exp21_pickle : the same payloads under the pre-publication module name.
  pilot       : {"kind": "pilot", network, model_name, seed}. LAMDA-PILOT
                 baselines (SLCA / EASE / MOS / TUNA) exported with a
                 `save_final_model` hook added to the LAMDA-PILOT trainer. Needs a
                 LAMDA-PILOT checkout at <repo>/LAMDA-PILOT; that patched copy is
                 not part of this repository.
"""
import os
import re
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _label_map(seed, n):
    """utils.data_manager.DataManager permutes the class order with
    np.random.seed(seed); np.random.permutation(n) and remaps labels to
    positions in that order. label_map[original] = model idx."""
    np.random.seed(seed)
    order = np.random.permutation(n)
    inv = np.zeros(n, dtype=np.int64)
    for pos, cls in enumerate(order):
        inv[cls] = pos
    return inv


def _preprocess_direct224(x01):
    """Our pipeline: Resize((224,224), bicubic)."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    return TF.resize(x01, [224, 224],
                     interpolation=InterpolationMode.BICUBIC,
                     antialias=True).clamp(0, 1)


def _preprocess_256crop224(x01):
    """LAMDA-PILOT pipeline: Resize(256, bicubic) + CenterCrop(224)."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    x = TF.resize(x01, 256, interpolation=InterpolationMode.BICUBIC,
                  antialias=True)
    return TF.center_crop(x, [224, 224]).clamp(0, 1)


class WrappedModel:
    def __init__(self, logits_fn, features_fn, classes, class_stats=None,
                 module=None, label_map=None, preprocess=None):
        self._logits, self._features = logits_fn, features_fn
        self.classes, self.class_stats, self.module = classes, class_stats, module
        self.label_map = label_map
        # raw [0,1] NCHW (any size) -> the 224x224 tensor this model was
        # trained to see (no channel normalization in either pipeline).
        self.preprocess = preprocess or _preprocess_direct224

    def map_labels(self, y):
        """Original dataset labels -> the model's shuffled class indices."""
        if self.label_map is None:
            return y
        if isinstance(y, torch.Tensor):
            return torch.as_tensor(self.label_map, device=y.device)[y]
        return self.label_map[y]

    def logits(self, x):
        return self._logits(x)

    def features(self, x):
        return self._features(x)


def _load_main(payload):
    import helper
    from main import Model21

    config = payload["config"]
    backbone = helper.get_backbone(config)
    model = Model21(config, backbone)

    # Rebuild the ContinualLinear head-by-head to match the saved structure
    # (one head per task), mirroring Learner.train_backbone's growth call.
    head_sizes = {}
    for k, v in payload["classifier"].items():
        m = re.match(r"heads\.(\d+)\..*\.?weight$", k)
        if m and v.dim() == 2 and v.shape[1] == model.feature_dim:
            head_sizes[int(m.group(1))] = v.shape[0]
    sizes = [head_sizes[i] for i in sorted(head_sizes)]
    use_norm = config.get("model_classifier_use_norm",
                          not config.get("model_use_norm", False))
    for s in sizes:
        model.update_classifier(
            s, with_norm=use_norm,
            with_bias=config.get("model_classifier_bias", False),
            freeze_old=True,
            norm_layer=config.get("model_classifier_norm_layer", "ln"))
    model.classifier.load_state_dict(payload["classifier"])

    bb_params = payload["backbone"]
    # "norm.*" keys belong to the MODEL-level feature LayerNorm, not the
    # backbone (routing them into the backbone would clobber the ViT's own
    # final norm via suffix matching).
    norm_sd = {k[len("norm."):]: v for k, v in bb_params.items()
               if k.startswith("norm.")}
    if norm_sd:
        assert model.norm is not None, "export has norm.* but model_use_norm off"
        model.norm.load_state_dict(norm_sd)
    rest = {k: v for k, v in bb_params.items() if not k.startswith("norm.")}
    sd = model.backbone.state_dict()
    missing = [k for k in rest if k not in sd]
    assert not missing, f"unmatched backbone keys: {missing[:5]}"
    sd.update(rest)
    model.backbone.load_state_dict(sd)
    model.cuda().eval()

    stats = None
    if "class_means" in payload and payload.get("class_covs") is not None:
        stats = (payload["class_means"].cuda(), payload["class_covs"].cuda())

    def feats(x):
        return model.get_features(x)

    def logits(x):
        out = model(x)
        return out["logits"] if isinstance(out, dict) else out

    n = payload["known_classes"]
    return WrappedModel(logits, feats, n, stats, model,
                        label_map=_label_map(config["seed"], n),
                        preprocess=_preprocess_direct224)


def _load_pilot(payload):
    """Per-method inference wrappers replicating each learner's REAL eval
    path (EASE test-mode forward; MOS iterative adapter routing; TUNA
    entropy-selected specific + general-adapter fusion; SLCA plain).
    MOS/TUNA routing is non-differentiable -> `attack_logits` exposes each
    method's differentiable surrogate for gradient-based attacks."""
    sys.path.insert(0, os.path.join(REPO, "LAMDA-PILOT"))
    net = payload["network"].cuda().eval()
    # Some methods keep grown modules in plain python lists (e.g. EASE's
    # backbone.adapters) which nn.Module.cuda() does not traverse.
    import torch.nn as nn

    def _to_cuda(obj):
        if isinstance(obj, nn.Module):
            obj.cuda().eval()
        elif isinstance(obj, torch.Tensor):
            return obj.cuda()
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                r = _to_cuda(v)
                if r is not None:
                    obj[i] = r
        return None

    for holder in (net, getattr(net, "backbone", None)):
        if holder is None:
            continue
        for attr in vars(holder):
            val = getattr(holder, attr)
            if isinstance(val, list):
                _to_cuda(val)
    name = payload.get("model_name", "")
    INC = 10  # CIFAR-100 10x10 protocol

    def plain_logits(x):
        out = net(x)
        return out["logits"] if isinstance(out, dict) else out

    attack_logits = None

    if name == "ease":
        # Export happens AFTER the final after_task: add_adapter_to_list has
        # (a) appended the trained task-T adapter to adapter_list and
        # (b) RE-INITIALIZED cur_adapter to a fresh random one. Restore the
        # eval-time state: cur_adapter <- trained adapter_list[-1], list gets
        # tasks 0..T-2.
        bb = net.backbone
        expected = net.fc.in_features // 768 - (1 if net.use_init_ptm else 0)
        if len(bb.adapter_list) + 1 > expected:
            bb.cur_adapter = bb.adapter_list[-1]
            bb.adapter_list = list(bb.adapter_list)[:expected - 1]

        # vit_ease.forward_test deepcopies activation TENSORS, which breaks
        # under grad (adversarial attacks). Patch its module-level `copy` to
        # clone tensors instead (value-identical, autograd-safe).
        import copy as _copy
        import types as _types
        _ve = sys.modules.get("backbone.vit_ease")
        if _ve is not None and not getattr(_ve, "_copy_patched", False):
            shim = _types.SimpleNamespace(
                deepcopy=lambda o: o.clone() if torch.is_tensor(o)
                else _copy.deepcopy(o))
            _ve.copy = shim
            _ve._copy_patched = True

        def logits(x):
            return net(x, test=True)["logits"]
    elif name == "mos":
        with torch.no_grad():
            C = net.forward_orig(
                torch.zeros(1, 3, 224, 224).cuda())["logits"].shape[-1]
        T = C // INC

        def logits(x):
            with torch.no_grad():
                orig = net.forward_orig(x)["logits"][:, :C]
                idx = orig.argmax(1) // INC
                feats_all = torch.stack(
                    [net.backbone(x, adapter_id=t, train=False)["features"]
                     for t in range(T)], dim=1)
                out = torch.empty_like(orig)
                for i in range(x.shape[0]):
                    prev = int(idx[i])
                    for _ in range(4):
                        cl = net.backbone(feats_all[i, prev].unsqueeze(0),
                                          fc_only=True)["logits"][:, :C]
                        cur = int(cl.argmax(1)[0]) // INC
                        if cur == prev:
                            break
                        prev = cur
                    out[i] = cl[0]
                return out

        def attack_logits(x):
            return net.forward_orig(x)["logits"][:, :C]
    elif name in ("tuna", "tuna_matched"):
        scale = 20.0
        with torch.no_grad():
            f0 = net.backbone(torch.zeros(1, 3, 224, 224).cuda(),
                              adapter_id=0, train=False)["features"]
            C = net.fc(f0)["logits"].shape[-1]
        T = C // INC

        def logits(x):
            with torch.no_grad():
                lgs, ents = [], []
                for i in range(T):
                    f = net.backbone(x, adapter_id=i, train=False)["features"]
                    lg = net.fc(f)["logits"][:, :C] * scale
                    p = torch.softmax(lg, 1)
                    ents.append(-(p * (p + 1e-10).log()).sum(1))
                    lgs.append(lg)
                lgs, ents = torch.stack(lgs), torch.stack(ents)
                spec = lgs[ents.argmin(0), torch.arange(x.shape[0])]
                fgen = net.backbone(x, adapter_id=T, train=False)["features"]
                gen = net.fc(fgen)["logits"][:, :C] * scale
                # Combined score = softmax(general) + softmax(specific);
                # sums to 2 -> halve so downstream softmax/MSP treat it sanely.
                return (torch.softmax(gen, 1) + torch.softmax(spec, 1)).log()

        def attack_logits(x):
            f = net.backbone(x, adapter_id=T, train=False)["features"]
            return net.fc(f)["logits"][:, :C] * scale
    else:  # slca and anything with a standard forward
        logits = plain_logits

    def feats(x):
        if hasattr(net, "extract_vector"):
            return net.extract_vector(x)
        if hasattr(net, "backbone"):
            try:
                f = net.backbone(x, adapter_id=0, train=False)
            except TypeError:
                f = net.backbone(x)
            return f["features"] if isinstance(f, dict) else f
        raise AttributeError("no feature extractor on pilot network")

    with torch.no_grad():
        c = logits(torch.zeros(1, 3, 224, 224).cuda()).shape[-1]
    wm = WrappedModel(logits, feats, c, None, net,
                      label_map=_label_map(payload["seed"], c),
                      preprocess=_preprocess_256crop224)
    if attack_logits is not None:
        wm.attack_logits = attack_logits
    return wm


def _use_pilot_namespace():
    """Make LAMDA-PILOT's packages (utils/models/backbone) take precedence so
    pickled PILOT modules resolve. Purges same-named packages already imported
    from this repo. NOTE: within one process, evaluate models of ONE kind, or
    our exports FIRST; after this switch, `utils` is PILOT's."""
    pilot = os.path.join(REPO, "LAMDA-PILOT")
    if pilot not in sys.path:
        sys.path.insert(0, pilot)
    for mod in list(sys.modules):
        root = mod.split(".")[0]
        if root in ("utils", "models", "backbone"):
            m = sys.modules[mod]
            if getattr(m, "__file__", "") and pilot not in (m.__file__ or ""):
                del sys.modules[mod]
    # timm >=1.0 moved timm.models.layers.* to timm.layers.*; pickles carry
    # the old paths. Alias every timm.layers submodule under the old prefix.
    import timm.layers
    import importlib
    import pkgutil
    sys.modules.setdefault("timm.models.layers", timm.layers)
    for info in pkgutil.iter_modules(timm.layers.__path__):
        try:
            sub = importlib.import_module(f"timm.layers.{info.name}")
            sys.modules.setdefault(f"timm.models.layers.{info.name}", sub)
        except ImportError:
            pass


def load_model(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError:
        _use_pilot_namespace()
        payload = torch.load(path, map_location="cpu", weights_only=False)
    kind = payload.get("kind")
    if kind in ("main", "exp21"):
        return _load_main(payload)
    if kind in ("main_pickle", "exp21_pickle"):
        model = payload["model"].cuda().eval()
        config = payload["config"]

        def logits(x):
            out = model(x)
            return out["logits"] if isinstance(out, dict) else out

        n = payload["known_classes"]
        return WrappedModel(logits, model.get_features, n, None, model,
                            label_map=_label_map(config["seed"], n),
                            preprocess=_preprocess_direct224)
    if kind == "pilot":
        return _load_pilot(payload)
    raise ValueError(f"unknown export kind {kind!r} in {path}")
