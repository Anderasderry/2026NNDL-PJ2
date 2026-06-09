"""Visualize training curves, conv filters, interpretation plots, and loss landscape."""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from common.utils.device import get_default_device, resolve_device, torch
import torch.nn as nn
from tqdm import tqdm

from common.data.loaders import CIFAR10_MEAN, CIFAR10_STD, get_cifar_loader
from common.paths import CIFAR10_OUTPUT_DIR
from models.cnn import CIFARNet

OUTPUT_DIR = CIFAR10_OUTPUT_DIR

CIFAR10_CLASSES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
)


def _load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    model = CIFARNet(
        width=config.get('width', 64),
        dropout=config.get('dropout', 0.5),
        activation=config.get('activation', 'relu'),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint, config


def _save_params(model):
    return [param.data.clone() for param in model.parameters()]


def _restore_params(model, saved):
    for param, value in zip(model.parameters(), saved):
        param.data.copy_(value)


def _apply_perturbation(model, saved, direction, step):
    offset = 0
    for param, base in zip(model.parameters(), saved):
        size = param.numel()
        delta = direction[offset:offset + size].view_as(param)
        param.data.copy_(base + step * delta)
        offset += size


def _flatten_grad(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    return torch.cat(grads)


@torch.no_grad()
def _eval_loss(model, images, labels, criterion):
    model.eval()
    return criterion(model(images), labels).item()


def plot_history(history_path, save_path):
    with open(history_path, encoding='utf-8') as f:
        history = json.load(f)

    epochs = [row['epoch'] for row in history]
    train_loss = [row['train_loss'] for row in history]
    test_loss = [row['test_loss'] for row in history]
    train_acc = [row['train_acc'] for row in history]
    test_acc = [row['test_acc'] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label='train')
    axes[0].plot(epochs, test_loss, label='test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label='train')
    axes[1].plot(epochs, test_acc, label='test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_filters(checkpoint_path, save_path, max_filters=64, device='cpu'):
    model, _, _ = _load_model_from_checkpoint(checkpoint_path, resolve_device(device))
    weights = model.first_conv_weights().numpy()
    n_filters = min(weights.shape[0], max_filters)
    grid = int(math.ceil(math.sqrt(n_filters)))

    fig, axes = plt.subplots(grid, grid, figsize=(grid * 1.2, grid * 1.2))
    axes = np.array(axes).reshape(grid, grid)

    for idx in range(grid * grid):
        row, col = divmod(idx, grid)
        ax = axes[row, col]
        ax.axis('off')
        if idx >= n_filters:
            continue
        filt = weights[idx]
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        filt = np.transpose(filt, (1, 2, 0))
        ax.imshow(filt)

    fig.suptitle('First Conv Layer Filters', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _denormalize_image(image_chw):
    img = np.transpose(image_chw, (1, 2, 0))
    img = img * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
    return np.clip(img, 0.0, 1.0)


@torch.no_grad()
def _collect_test_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_images = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_images.append(images.cpu())

    labels = torch.cat(all_labels)
    preds = torch.cat(all_preds)
    images = torch.cat(all_images)
    return labels.numpy(), preds.numpy(), images.numpy()


def plot_confusion_matrix(checkpoint_path, save_path, device='cpu', batch_size=256, num_workers=0):
    device = resolve_device(device)
    model, _, _ = _load_model_from_checkpoint(checkpoint_path, device)
    loader = get_cifar_loader(
        train=False,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        augment=False,
    )
    labels, preds, _ = _collect_test_predictions(model, loader, device)

    matrix = np.zeros((len(CIFAR10_CLASSES), len(CIFAR10_CLASSES)), dtype=int)
    for true_label, pred_label in zip(labels, preds):
        matrix[true_label, pred_label] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(normalized, cmap='Blues', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticks(range(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix (row-normalized test accuracy)')

    for i in range(len(CIFAR10_CLASSES)):
        for j in range(len(CIFAR10_CLASSES)):
            count = matrix[i, j]
            if count == 0:
                continue
            color = 'white' if normalized[i, j] > 0.5 else 'black'
            ax.text(j, i, str(count), ha='center', va='center', color=color, fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Recall fraction')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    summary_path = os.path.splitext(save_path)[0] + '.json'
    per_class_acc = {
        CIFAR10_CLASSES[i]: float(matrix[i, i] / max(row_sums[i, 0], 1))
        for i in range(len(CIFAR10_CLASSES))
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'per_class_accuracy': per_class_acc, 'confusion_matrix': matrix.tolist()}, f, indent=2)
    return per_class_acc


def plot_misclassified_examples(
    checkpoint_path,
    save_path,
    device='cpu',
    batch_size=256,
    num_workers=0,
    max_examples=12,
):
    device = resolve_device(device)
    model, _, _ = _load_model_from_checkpoint(checkpoint_path, device)
    loader = get_cifar_loader(
        train=False,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        augment=False,
    )
    labels, preds, images = _collect_test_predictions(model, loader, device)
    wrong_idx = np.where(labels != preds)[0]
    if wrong_idx.size == 0:
        return []

    chosen = wrong_idx[:max_examples]
    cols = min(4, max_examples)
    rows = int(math.ceil(len(chosen) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.atleast_2d(axes)

    for plot_idx, sample_idx in enumerate(chosen):
        row, col = divmod(plot_idx, cols)
        ax = axes[row, col]
        true_label = int(labels[sample_idx])
        pred_label = int(preds[sample_idx])
        ax.imshow(_denormalize_image(images[sample_idx]))
        ax.set_title(
            f'true: {CIFAR10_CLASSES[true_label]}\npred: {CIFAR10_CLASSES[pred_label]}',
            fontsize=9,
        )
        ax.axis('off')

    for plot_idx in range(len(chosen), rows * cols):
        row, col = divmod(plot_idx, cols)
        axes[row, col].axis('off')

    fig.suptitle('Misclassified Test Examples', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return chosen.tolist()


def plot_loss_landscape(
    checkpoint_path,
    save_path,
    device='cpu',
    step_min=-0.2,
    step_max=0.2,
    num_steps=41,
    batch_size=128,
    n_batches=4,
    num_workers=0,
):
    """
    Plot 1D loss landscape along the normalized loss gradient at a checkpoint.

    For each step size t, parameters are perturbed as:
        theta(t) = theta(0) + t * direction
    where direction is the normalized gradient on a fixed mini-batch.
    """
    device = resolve_device(device)
    model, checkpoint, _ = _load_model_from_checkpoint(checkpoint_path, device)
    criterion = nn.CrossEntropyLoss()

    loader = get_cifar_loader(
        train=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        augment=False,
    )
    images_list, labels_list = [], []
    for batch_idx, (images, labels) in enumerate(loader):
        images_list.append(images)
        labels_list.append(labels)
        if batch_idx + 1 >= n_batches:
            break
    images = torch.cat(images_list, dim=0).to(device)
    labels = torch.cat(labels_list, dim=0).to(device)

    saved = _save_params(model)
    model.train()
    model.zero_grad(set_to_none=True)
    loss = criterion(model(images), labels)
    loss.backward()
    direction = _flatten_grad(model)
    direction = direction / (direction.norm() + 1e-8)
    base_loss = loss.item()
    _restore_params(model, saved)

    steps = np.linspace(step_min, step_max, num_steps)
    losses = []
    for step in tqdm(steps, desc='loss landscape', leave=False):
        _apply_perturbation(model, saved, direction, float(step))
        losses.append(_eval_loss(model, images, labels, criterion))
    _restore_params(model, saved)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, marker='o', markersize=3, linewidth=1.5)
    ax.axvline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.scatter([0.0], [base_loss], color='red', zorder=5, label='checkpoint')
    ax.set_xlabel('Step size along normalized gradient direction')
    ax.set_ylabel('Loss')
    epoch = checkpoint.get('epoch')
    title = 'Loss Landscape (1D gradient direction)'
    if epoch is not None:
        title += f' @ epoch {epoch}'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    landscape_data = {
        'checkpoint': checkpoint_path,
        'base_loss': base_loss,
        'epoch': checkpoint.get('epoch'),
        'step_min': step_min,
        'step_max': step_max,
        'num_steps': num_steps,
        'steps': steps.tolist(),
        'losses': losses,
    }
    json_path = os.path.splitext(save_path)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(landscape_data, f, indent=2)
    return landscape_data


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CIFARNet training results')
    parser.add_argument('--run-name', type=str, default='cifarnet')
    parser.add_argument('--history', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default=get_default_device())
    parser.add_argument('--skip-landscape', action='store_true', help='Skip loss landscape plot')
    parser.add_argument(
        '--skip-interpretation',
        action='store_true',
        help='Skip confusion matrix and misclassified example plots',
    )
    parser.add_argument(
        '--num-misclassified',
        type=int,
        default=12,
        help='Number of misclassified examples to plot',
    )
    parser.add_argument('--landscape-step-min', type=float, default=-0.2)
    parser.add_argument('--landscape-step-max', type=float, default=0.2)
    parser.add_argument('--landscape-steps', type=int, default=41)
    parser.add_argument('--landscape-batches', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.join(OUTPUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    history_path = args.history or os.path.join(run_dir, 'history.json')
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.normpath(os.path.join(os.path.dirname(__file__), checkpoint_path))
    else:
        checkpoint_path = os.path.join(run_dir, 'best_model.pt')

    curves_path = os.path.join(run_dir, 'training_curves.png')
    filters_path = os.path.join(run_dir, 'first_conv_filters.png')
    confusion_path = os.path.join(run_dir, 'confusion_matrix.png')
    misclassified_path = os.path.join(run_dir, 'misclassified_examples.png')
    landscape_path = os.path.join(run_dir, 'loss_landscape.png')

    if os.path.exists(history_path):
        plot_history(history_path, curves_path)
        print(f'Saved: {curves_path}')
    else:
        print(f'History not found: {history_path}')

    if os.path.exists(checkpoint_path):
        plot_filters(checkpoint_path, filters_path, device=args.device)
        print(f'Saved: {filters_path}')

        if not args.skip_interpretation:
            per_class_acc = plot_confusion_matrix(
                checkpoint_path,
                confusion_path,
                device=args.device,
                num_workers=args.num_workers,
            )
            print(f'Saved: {confusion_path}')
            print(f'Saved: {os.path.splitext(confusion_path)[0] + ".json"}')
            plot_misclassified_examples(
                checkpoint_path,
                misclassified_path,
                device=args.device,
                num_workers=args.num_workers,
                max_examples=args.num_misclassified,
            )
            print(f'Saved: {misclassified_path}')
            weakest = sorted(per_class_acc.items(), key=lambda item: item[1])[:3]
            print('Lowest per-class accuracy:', ', '.join(f'{name}={acc:.3f}' for name, acc in weakest))

        if not args.skip_landscape:
            plot_loss_landscape(
                checkpoint_path,
                landscape_path,
                device=args.device,
                step_min=args.landscape_step_min,
                step_max=args.landscape_step_max,
                num_steps=args.landscape_steps,
                n_batches=args.landscape_batches,
                num_workers=args.num_workers,
            )
            print(f'Saved: {landscape_path}')
            print(f'Saved: {os.path.splitext(landscape_path)[0] + ".json"}')
    else:
        print(f'Checkpoint not found: {checkpoint_path}')


if __name__ == '__main__':
    main()
