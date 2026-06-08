"""Matplotlib helpers for VGG loss / gradient landscape figures.

Functions ``plot_loss_landscape``, ``plot_grad_landscape``, ``plot_grad_landscape_comparison``,
and ``plot_grad_norm_comparison`` are kept for reference but are not called by ``experiments.py``.
"""

import matplotlib.pyplot as plt

from core import subsample_aligned

COLOR_VGG_A = 'tab:green'
COLOR_VGG_BN = 'tab:red'


def _plot_landscape_band(steps, min_curve, max_curve, color, label=None, fill_alpha=0.20):
    plt.plot(steps, max_curve, color=color, linewidth=1.2, label=label)
    plt.plot(steps, min_curve, color=color, linewidth=1.2)
    plt.fill_between(steps, min_curve, max_curve, alpha=fill_alpha, color=color)


def plot_loss_landscape(steps, min_curve, max_curve, save_path, title='Loss landscape', fill_alpha=0.25):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, max_curve, label='max', color='C0')
    plt.plot(steps, min_curve, label='min', color='C1')
    plt.fill_between(steps, min_curve, max_curve, alpha=fill_alpha, color='C0')
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_grad_landscape(steps, min_curve, max_curve, save_path, title='Gradient landscape', fill_alpha=0.25):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, max_curve, label='max', color='C2')
    plt.plot(steps, min_curve, label='min', color='C3')
    plt.fill_between(steps, min_curve, max_curve, alpha=fill_alpha, color='C2')
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


def plot_grad_predictiveness(
    steps,
    delta_a,
    delta_b,
    save_path,
    label_a='VGG-A',
    label_b='VGG-A+BN',
    ylim=None,
    plot_stride=50,
):
    steps, delta_a, delta_b = subsample_aligned(plot_stride, steps, delta_a, delta_b)
    plt.figure(figsize=(9, 5))
    plt.plot(steps, delta_a, label=label_a, alpha=0.85)
    plt.plot(steps, delta_b, label=label_b, alpha=0.85)
    plt.xlabel('Training step')
    plt.ylabel('|grad_t - grad_{t-1}|')
    plt.title('Gradient predictiveness')
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_landscape_comparison(
    steps_a, min_a, max_a, steps_b, min_b, max_b, save_path,
    ylim=None, fill_alpha=0.20, plot_stride=50,
):
    steps_a, min_a, max_a = subsample_aligned(plot_stride, steps_a, min_a, max_a)
    steps_b, min_b, max_b = subsample_aligned(plot_stride, steps_b, min_b, max_b)
    plt.figure(figsize=(9, 5))
    _plot_landscape_band(steps_a, min_a, max_a, COLOR_VGG_A, label='VGG-A', fill_alpha=fill_alpha)
    _plot_landscape_band(steps_b, min_b, max_b, COLOR_VGG_BN, label='VGG-A+BN', fill_alpha=fill_alpha)
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.title('Loss landscape: VGG-A vs VGG-A+BN')
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_grad_landscape_comparison(
    steps_a, min_a, max_a, steps_b, min_b, max_b, save_path,
    ylim=None, fill_alpha=0.20, plot_stride=50,
):
    steps_a, min_a, max_a = subsample_aligned(plot_stride, steps_a, min_a, max_a)
    steps_b, min_b, max_b = subsample_aligned(plot_stride, steps_b, min_b, max_b)
    plt.figure(figsize=(9, 5))
    _plot_landscape_band(steps_a, min_a, max_a, COLOR_VGG_A, label='VGG-A', fill_alpha=fill_alpha)
    _plot_landscape_band(steps_b, min_b, max_b, COLOR_VGG_BN, label='VGG-A+BN', fill_alpha=fill_alpha)
    plt.xlabel('Training step')
    plt.ylabel('Gradient norm')
    plt.title('Gradient landscape: VGG-A vs VGG-A+BN')
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
