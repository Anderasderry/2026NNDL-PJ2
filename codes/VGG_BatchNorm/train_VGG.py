"""Shared training utilities for VGG Task 2 (used by starter and full experiments)."""

import os
import random
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

from common.paths import VGG_FIGURES_DIR, VGG_MODELS_DIR, VGG_OUTPUT_DIR

OUTPUT_DIR = VGG_OUTPUT_DIR
FIGURES_DIR = VGG_FIGURES_DIR
MODELS_DIR = VGG_MODELS_DIR

device = None
torch = None
nn = None
tqdm = None
get_cifar_loader = None
VGG_A = None
VGG_A_BatchNorm = None
_set_seed = None


def init_training():
    """Lazy-import PyTorch (skipped when only replotting from JSON)."""
    global device, torch, nn, tqdm, get_cifar_loader, VGG_A, VGG_A_BatchNorm, _set_seed
    if torch is not None:
        return
    from tqdm import tqdm as _tqdm
    from torch import nn as _nn

    from common.data.loaders import get_cifar_loader as _get_cifar_loader
    from common.utils.device import resolve_device, set_seed
    from models.vgg import VGG_A as _VGG_A, VGG_A_BatchNorm as _VGG_A_BatchNorm

    tqdm = _tqdm
    nn = _nn
    torch = __import__('torch')
    get_cifar_loader = _get_cifar_loader
    VGG_A = _VGG_A
    VGG_A_BatchNorm = _VGG_A_BatchNorm
    _set_seed = set_seed
    device = resolve_device()


def set_random_seeds(seed_value=2020):
    init_training()
    _set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


def get_accuracy(model, loader):
    init_training()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    epochs_n=20,
    best_model_path=None,
    curve_path=None,
):
    """Train and record per-step loss / last-FC gradient norm."""
    init_training()
    model.to(device)
    learning_curve = [0.0] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0.0

    batches_n = len(train_loader)
    losses_list = []
    grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch', leave=False):
        model.train()
        loss_list = []
        grad_list = []
        epoch_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)

            loss_list.append(loss.item())
            epoch_loss += loss.item()

            loss.backward()
            weight_grad = model.classifier[4].weight.grad
            if weight_grad is not None:
                grad_list.append(weight_grad.norm().item())
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad_list)
        learning_curve[epoch] = epoch_loss / batches_n

        val_acc = get_accuracy(model, val_loader)
        val_accuracy_curve[epoch] = val_acc
        if best_model_path and val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)

        if curve_path:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
            axes[0].plot(range(1, epoch + 2), learning_curve[: epoch + 1])
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Train loss')
            axes[1].plot(range(1, epoch + 2), val_accuracy_curve[: epoch + 1])
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Val accuracy')
            fig.tight_layout()
            fig.savefig(curve_path)
            plt.close(fig)

    return losses_list, grads, learning_curve, val_accuracy_curve


def build_model(use_bn=False):
    init_training()
    return VGG_A_BatchNorm() if use_bn else VGG_A()
