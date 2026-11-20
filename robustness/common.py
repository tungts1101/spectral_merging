"""Shared pieces for the robustness suite: input convention, the clean
CIFAR-100 test set, and the accuracy helper.

Data root: the `DATA_ROOT` environment variable (the same convention as
utils/data.py). The suite reads
    $DATA_ROOT/cifar-100-python/            torchvision CIFAR-100 (clean test set)
    $DATA_ROOT/CIFAR-100-C/*.npy            Hendrycks & Dietterich, Zenodo 3555552
    $DATA_ROOT/CIFAR-100-P/*.npy            Zenodo 3555553 (the tar extracts to
                                            CIFAR-100-P/CIFAR-100-P/ on some
                                            versions; both layouts are accepted)
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_ROOT = os.environ.get("DATA_ROOT", "/home/lis/data")


def normalize(x01):
    """IDENTITY. The training pipeline (utils.data with use_input_norm=False)
    feeds raw ToTensor [0,1] images with NO ImageNet normalization, so the
    evaluated models expect unnormalized inputs. Kept as a named hook so the
    eval scripts stay explicit about the convention."""
    return x01


def upscale224(x01):
    """Match the training pipeline's test resize: bicubic with antialias
    (the torchvision PIL-equivalent), not plain bilinear interpolate."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    return TF.resize(x01, [224, 224],
                     interpolation=InterpolationMode.BICUBIC,
                     antialias=True).clamp(0, 1)


def cifar_test_raw():
    """CIFAR-100 test set as uint8 [N,32,32,3] + ORIGINAL labels."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(DATA_ROOT, train=False, download=True)
    return ds.data, np.array(ds.targets)


def batched_uint8_loader(images_u8, labels, batch_size=128):
    x = torch.from_numpy(np.ascontiguousarray(images_u8))
    y = torch.from_numpy(np.ascontiguousarray(labels)).long()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False,
                      num_workers=2)


@torch.no_grad()
def accuracy_u8(model, images_u8, labels, batch_size=128):
    """Accuracy on uint8 HWC images (any resolution; upscaled to 224).
    Labels are ORIGINAL dataset labels; mapped to the model's shuffled class
    indices via model.map_labels."""
    labels = model.map_labels(np.asarray(labels))
    correct = total = 0
    for x, y in batched_uint8_loader(images_u8, labels, batch_size):
        x = x.cuda().permute(0, 3, 1, 2).float() / 255.0
        x = normalize(model.preprocess(x))
        pred = model.logits(x).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total
