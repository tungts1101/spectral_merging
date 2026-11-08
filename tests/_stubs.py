"""Stub modules so that helper.py / main.py import on a machine that has torch
but not the vision stack (timm, peft, tqdm, gdown, kagglehub).

install() registers a minimal stand-in in sys.modules for every package in
the list that is NOT importable, and returns the names it stubbed. Where the
real packages exist (the server) it is a no-op, so the tests never shadow a
real installation. The stubs only satisfy module-level imports; any call into
them (timm.create_model, peft.get_peft_model, ...) raises RuntimeError.
torchvision and PIL are not stubbed: utils/data.py builds transforms at
import time, so `import main` needs the real packages.

Usage (before the first `import helper` / `import main`):
    import _stubs; _stubs.install()
"""
import sys
import types


def _unavailable(name):
    def _raise(*args, **kwargs):
        raise RuntimeError(f"stub {name}: not available in this environment")
    return _raise


def _missing(name):
    try:
        __import__(name)
        return False
    except ImportError:
        return True


def install():
    stubbed = []

    if _missing("timm"):
        timm = types.ModuleType("timm")
        timm.__version__ = "0.0-stub"
        timm.create_model = _unavailable("timm.create_model")
        models = types.ModuleType("timm.models")
        layers = types.ModuleType("timm.models.layers")

        def trunc_normal_(tensor, *args, **kwargs):
            return tensor

        layers.trunc_normal_ = trunc_normal_
        models.layers = layers
        timm.models = models
        sys.modules["timm"] = timm
        sys.modules["timm.models"] = models
        sys.modules["timm.models.layers"] = layers
        stubbed.append("timm")

    if _missing("peft"):
        peft = types.ModuleType("peft")

        class LoraConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        peft.LoraConfig = LoraConfig
        peft.get_peft_model = _unavailable("peft.get_peft_model")
        sys.modules["peft"] = peft
        stubbed.append("peft")

    if _missing("tqdm"):
        tqdm_mod = types.ModuleType("tqdm")

        def tqdm(iterable=None, *args, **kwargs):
            return iterable

        tqdm_mod.tqdm = tqdm
        sys.modules["tqdm"] = tqdm_mod
        stubbed.append("tqdm")

    for name in ("gdown", "kagglehub"):
        if _missing(name):
            mod = types.ModuleType(name)
            mod.download = _unavailable(f"{name}.download")
            mod.dataset_download = _unavailable(f"{name}.dataset_download")
            sys.modules[name] = mod
            stubbed.append(name)

    return stubbed


if __name__ == "__main__":
    print("stubbed:", install() or "nothing (all packages present)")
