"""Visualize training curves and first-layer conv filters."""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.cnn import CIFARNet

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


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


def plot_filters(checkpoint_path, save_path, max_filters=64):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    model = CIFARNet(
        width=config.get('width', 64),
        dropout=config.get('dropout', 0.5),
        activation=config.get('activation', 'relu'),
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    weights = model.first_conv_weights().numpy()
    n_filters = min(weights.shape[0], max_filters)
    grid = int(math.ceil(math.sqrt(n_filters)))

    fig, axes = plt.subplots(grid, grid, figsize=(grid * 1.2, grid * 1.2))
    axes = np.array(axes).reshape(grid, grid)

    for idx in range(grid * grid):
        r, c = divmod(idx, grid)
        ax = axes[r, c]
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


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CIFARNet training results')
    parser.add_argument('--run-name', type=str, default='cifarnet')
    parser.add_argument('--history', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.join(OUTPUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    history_path = args.history or os.path.join(run_dir, 'history.json')
    checkpoint_path = args.checkpoint or os.path.join(run_dir, 'best_model.pt')

    curves_path = os.path.join(run_dir, 'training_curves.png')
    filters_path = os.path.join(run_dir, 'first_conv_filters.png')

    if os.path.exists(history_path):
        plot_history(history_path, curves_path)
        print(f'Saved: {curves_path}')
    else:
        print(f'History not found: {history_path}')

    if os.path.exists(checkpoint_path):
        plot_filters(checkpoint_path, filters_path)
        print(f'Saved: {filters_path}')
    else:
        print(f'Checkpoint not found: {checkpoint_path}')


if __name__ == '__main__':
    main()
