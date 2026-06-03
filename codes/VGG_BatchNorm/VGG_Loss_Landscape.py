"""Train VGG-A / VGG-A+BN on CIFAR-10 and visualize loss landscape (Task 2)."""

import argparse
import json
import os
import random
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils.device import resolve_device, set_seed, torch
from common.data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm
from torch import nn

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
DEFAULT_LEARNING_RATES = [1e-3, 2e-3, 1e-4, 5e-4]

device = resolve_device()


def set_random_seeds(seed_value=2020):
    set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


def get_accuracy(model, loader):
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
    """Train model and record per-step loss / gradient for loss landscape."""
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


def flatten_steps(nested_list):
    return [value for epoch_values in nested_list for value in epoch_values]


def compute_min_max_curves(all_series):
    """Build max/min curves across runs keyed by learning rate."""
    flats = [flatten_steps(series) for series in all_series.values()]
    num_steps = min(len(flat) for flat in flats)
    steps = list(range(num_steps))
    max_curve, min_curve = [], []

    for step in range(num_steps):
        values_at_step = [flat[step] for flat in flats]
        max_curve.append(max(values_at_step))
        min_curve.append(min(values_at_step))

    return steps, min_curve, max_curve


def compute_grad_predictiveness(flat_grads):
    """Step-to-step change of gradient norm (gradient predictiveness)."""
    if len(flat_grads) < 2:
        return [], []
    deltas = [abs(flat_grads[i] - flat_grads[i - 1]) for i in range(1, len(flat_grads))]
    return list(range(1, len(flat_grads))), deltas


def compute_max_diff_curve(min_curve, max_curve):
    return [max_val - min_val for max_val, min_val in zip(max_curve, min_curve)]


def flatten_losses(losses_list):
    return flatten_steps(losses_list)


def plot_loss_landscape(steps, min_curve, max_curve, save_path, title='Loss landscape'):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, max_curve, label='max', color='C0')
    plt.plot(steps, min_curve, label='min', color='C1')
    plt.fill_between(steps, min_curve, max_curve, alpha=0.25, color='C0')
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_grad_landscape(steps, min_curve, max_curve, save_path, title='Gradient landscape'):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, max_curve, label='max', color='C2')
    plt.plot(steps, min_curve, label='min', color='C3')
    plt.fill_between(steps, min_curve, max_curve, alpha=0.25, color='C2')
    plt.xlabel('Training step')
    plt.ylabel('Gradient norm')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_max_diff_curve(steps, diff_curve, save_path, title='Max gradient difference'):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, diff_curve, color='C4', linewidth=1.2)
    plt.xlabel('Training step')
    plt.ylabel('max(grad) - min(grad)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_grad_norm_comparison(flat_a, flat_b, save_path, label_a='VGG-A', label_b='VGG-A+BN'):
    steps = list(range(min(len(flat_a), len(flat_b))))
    plt.figure(figsize=(9, 5))
    plt.plot(steps, flat_a[: len(steps)], label=label_a, alpha=0.85)
    plt.plot(steps, flat_b[: len(steps)], label=label_b, alpha=0.85)
    plt.xlabel('Training step')
    plt.ylabel('Gradient norm')
    plt.title('Gradient norm during training')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_grad_predictiveness(steps, delta_a, delta_b, save_path, label_a='VGG-A', label_b='VGG-A+BN'):
    plt.figure(figsize=(9, 5))
    plt.plot(steps, delta_a, label=label_a, alpha=0.85)
    plt.plot(steps, delta_b, label=label_b, alpha=0.85)
    plt.xlabel('Training step')
    plt.ylabel('|grad_t - grad_{t-1}|')
    plt.title('Gradient predictiveness')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_grad_landscape_comparison(
    steps_a,
    min_a,
    max_a,
    steps_b,
    min_b,
    max_b,
    save_path,
):
    plt.figure(figsize=(9, 5))
    plt.plot(steps_a, max_a, color='C0', linewidth=1.2, label='VGG-A max')
    plt.plot(steps_a, min_a, color='C0', linewidth=1.2, linestyle='--', label='VGG-A min')
    plt.fill_between(steps_a, min_a, max_a, alpha=0.20, color='C0')

    plt.plot(steps_b, max_b, color='C1', linewidth=1.2, label='VGG-A+BN max')
    plt.plot(steps_b, min_b, color='C1', linewidth=1.2, linestyle='--', label='VGG-A+BN min')
    plt.fill_between(steps_b, min_b, max_b, alpha=0.20, color='C1')

    plt.xlabel('Training step')
    plt.ylabel('Gradient norm')
    plt.title('Gradient landscape: VGG-A vs VGG-A+BN')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_landscape_comparison(
    steps_a,
    min_a,
    max_a,
    steps_b,
    min_b,
    max_b,
    save_path,
    ylim=None,
):
    plt.figure(figsize=(9, 5))
    plt.plot(steps_a, max_a, color='C0', linewidth=1.2, label='VGG-A max')
    plt.plot(steps_a, min_a, color='C0', linewidth=1.2, linestyle='--', label='VGG-A min')
    plt.fill_between(steps_a, min_a, max_a, alpha=0.20, color='C0')

    plt.plot(steps_b, max_b, color='C1', linewidth=1.2, label='VGG-A+BN max')
    plt.plot(steps_b, min_b, color='C1', linewidth=1.2, linestyle='--', label='VGG-A+BN min')
    plt.fill_between(steps_b, min_b, max_b, alpha=0.20, color='C1')

    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.title('Loss landscape: VGG-A vs VGG-A+BN')
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def load_landscape_curves(tag):
    path = os.path.join(OUTPUT_DIR, f'{tag}_landscape.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data['steps'], data['min_curve'], data['max_curve']


def replot_loss_landscape_comparison(ylim=None):
    steps_a, min_a, max_a = load_landscape_curves('vgg_a')
    steps_b, min_b, max_b = load_landscape_curves('vgg_a_bn')
    save_path = os.path.join(FIGURES_DIR, 'loss_landscape_comparison.png')
    plot_landscape_comparison(
        steps_a, min_a, max_a, steps_b, min_b, max_b, save_path, ylim=ylim,
    )
    print(f'Saved comparison plot: {save_path}' + (f' (ylim={ylim})' if ylim else ''))


def build_model(use_bn=False):
    return VGG_A_BatchNorm() if use_bn else VGG_A()


def save_losses(losses_list, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(losses_list, f)


def run_model_comparison(train_loader, val_loader, epochs_n, lr):
    criterion = nn.CrossEntropyLoss()
    results = {}

    for use_bn, tag in ((False, 'vgg_a'), (True, 'vgg_a_bn')):
        set_random_seeds()
        model = build_model(use_bn=use_bn)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        run_dir = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(run_dir, exist_ok=True)

        losses, grads, train_curve, val_curve = train(
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            epochs_n=epochs_n,
            best_model_path=os.path.join(MODELS_DIR, f'{tag}.pt'),
            curve_path=os.path.join(FIGURES_DIR, f'{tag}_training_curve.png'),
        )

        save_losses(losses, os.path.join(run_dir, 'losses.json'))
        with open(os.path.join(run_dir, 'grads.json'), 'w', encoding='utf-8') as f:
            json.dump(grads, f)

        summary = {
            'model': tag,
            'epochs': epochs_n,
            'lr': lr,
            'best_val_acc': float(max(val_curve)),
            'final_val_acc': float(val_curve[-1]),
        }
        with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        results[tag] = {
            'losses': losses,
            'grads': grads,
            'val_curve': val_curve,
            'summary': summary,
        }
        print(f'[{tag}] best val acc: {summary["best_val_acc"]:.4f}')

    flat_a = flatten_steps(results['vgg_a']['grads'])
    flat_b = flatten_steps(results['vgg_a_bn']['grads'])
    plot_grad_norm_comparison(
        flat_a,
        flat_b,
        os.path.join(FIGURES_DIR, 'grad_norm_comparison.png'),
    )
    _, delta_a = compute_grad_predictiveness(flat_a)
    _, delta_b = compute_grad_predictiveness(flat_b)
    n = min(len(delta_a), len(delta_b))
    plot_grad_predictiveness(
        list(range(1, n + 1)),
        delta_a[:n],
        delta_b[:n],
        os.path.join(FIGURES_DIR, 'grad_predictiveness_comparison.png'),
    )

    return results


def run_loss_landscape(train_loader, val_loader, epochs_n, learning_rates, use_bn=False):
    tag = 'vgg_a_bn' if use_bn else 'vgg_a'
    criterion = nn.CrossEntropyLoss()
    all_losses = {}
    all_grads = {}

    for lr in learning_rates:
        set_random_seeds()
        model = build_model(use_bn=use_bn)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses, grads, _, _ = train(
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            epochs_n=epochs_n,
        )
        all_losses[lr] = losses
        all_grads[lr] = grads
        print(f'[{tag}] finished lr={lr:g}')

    steps, min_curve, max_curve = compute_min_max_curves(all_losses)
    landscape_path = os.path.join(FIGURES_DIR, f'{tag}_loss_landscape.png')
    plot_loss_landscape(
        steps,
        min_curve,
        max_curve,
        landscape_path,
        title=f'Loss landscape ({tag})',
    )

    grad_steps, grad_min, grad_max = compute_min_max_curves(all_grads)
    plot_grad_landscape(
        grad_steps,
        grad_min,
        grad_max,
        os.path.join(FIGURES_DIR, f'{tag}_grad_landscape.png'),
        title=f'Gradient landscape ({tag})',
    )
    grad_diff = compute_max_diff_curve(grad_min, grad_max)
    plot_max_diff_curve(
        grad_steps,
        grad_diff,
        os.path.join(FIGURES_DIR, f'{tag}_grad_max_diff.png'),
        title=f'Max gradient difference ({tag})',
    )

    landscape_data = {
        'model': tag,
        'learning_rates': learning_rates,
        'steps': steps,
        'min_curve': min_curve,
        'max_curve': max_curve,
        'grad_steps': grad_steps,
        'grad_min_curve': grad_min,
        'grad_max_curve': grad_max,
        'grad_max_diff': grad_diff,
    }
    with open(os.path.join(OUTPUT_DIR, f'{tag}_landscape.json'), 'w', encoding='utf-8') as f:
        json.dump(landscape_data, f, indent=2)

    return steps, min_curve, max_curve, grad_steps, grad_min, grad_max


def parse_args():
    parser = argparse.ArgumentParser(description='VGG-A / VGG-A+BN training and loss landscape')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--n-items', type=int, default=-1, help='Use subset for quick debug')
    parser.add_argument('--lr', type=float, default=1e-3, help='LR for VGG-A vs BN comparison')
    parser.add_argument('--skip-comparison', action='store_true')
    parser.add_argument('--skip-landscape', action='store_true')
    parser.add_argument(
        '--learning-rates',
        type=float,
        nargs='+',
        default=DEFAULT_LEARNING_RATES,
        help='LRs for loss landscape experiment',
    )
    parser.add_argument(
        '--replot-comparison',
        action='store_true',
        help='Replot loss_landscape_comparison.png from saved JSON (no training)',
    )
    parser.add_argument(
        '--loss-ylim',
        type=float,
        nargs=2,
        metavar=('YMIN', 'YMAX'),
        default=None,
        help='Y-axis limits for loss landscape comparison, e.g. --loss-ylim 0 4',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.replot_comparison:
        ylim = tuple(args.loss_ylim) if args.loss_ylim is not None else None
        replot_loss_landscape_comparison(ylim=ylim)
        return

    print(f'Device: {device}')
    train_loader = get_cifar_loader(
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_items=args.n_items,
    )
    val_loader = get_cifar_loader(
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        n_items=args.n_items,
    )

    for images, labels in train_loader:
        print(f'batch shape: {images.shape}, labels shape: {labels.shape}')
        print(f'label example: {labels[:5].tolist()}')
        break

    if not args.skip_comparison:
        print('\n=== VGG-A vs VGG-A+BN comparison ===')
        run_model_comparison(train_loader, val_loader, args.epochs, args.lr)

    if not args.skip_landscape:
        print('\n=== Loss landscape (multiple learning rates) ===')
        steps_a, min_a, max_a, grad_steps_a, grad_min_a, grad_max_a = run_loss_landscape(
            train_loader,
            val_loader,
            args.epochs,
            args.learning_rates,
            use_bn=False,
        )
        steps_b, min_b, max_b, grad_steps_b, grad_min_b, grad_max_b = run_loss_landscape(
            train_loader,
            val_loader,
            args.epochs,
            args.learning_rates,
            use_bn=True,
        )
        ylim = tuple(args.loss_ylim) if args.loss_ylim is not None else None
        compare_path = os.path.join(FIGURES_DIR, 'loss_landscape_comparison.png')
        plot_landscape_comparison(steps_a, min_a, max_a, steps_b, min_b, max_b, compare_path, ylim=ylim)
        print(f'Saved comparison plot: {compare_path}')

        grad_compare_path = os.path.join(FIGURES_DIR, 'grad_landscape_comparison.png')
        plot_grad_landscape_comparison(
            grad_steps_a,
            grad_min_a,
            grad_max_a,
            grad_steps_b,
            grad_min_b,
            grad_max_b,
            grad_compare_path,
        )
        print(f'Saved gradient comparison plot: {grad_compare_path}')

    print(f'\nOutputs saved under: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
