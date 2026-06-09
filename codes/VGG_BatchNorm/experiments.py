"""Full Task 2 experiment pipelines (Part A comparison + Part B landscape + CLI)."""

import argparse
import json
import os

from core import (
    DEFAULT_LEARNING_RATES,
    compute_grad_predictiveness,
    compute_max_diff_curve,
    compute_min_max_curves,
    flatten_steps,
)
from plots import (
    plot_grad_predictiveness,
    plot_landscape_comparison,
    plot_max_diff_curve,
)
import train_VGG
from train_VGG import (
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    build_model,
    init_training,
    set_random_seeds,
    train,
)


def load_landscape_curves(tag):
    with open(os.path.join(OUTPUT_DIR, f'{tag}_landscape.json'), encoding='utf-8') as f:
        data = json.load(f)
    return data['steps'], data['min_curve'], data['max_curve']


def load_comparison_grads(tag):
    with open(os.path.join(OUTPUT_DIR, tag, 'grads.json'), encoding='utf-8') as f:
        return flatten_steps(json.load(f))


def replot_loss_landscape_comparison(loss_ylim=None, fill_alpha=0.20, plot_stride=50):
    """Replot loss_landscape_comparison.png from vgg_*_landscape.json."""
    steps_a, min_a, max_a = load_landscape_curves('vgg_a')
    steps_b, min_b, max_b = load_landscape_curves('vgg_a_bn')
    loss_path = os.path.join(FIGURES_DIR, 'loss_landscape_comparison.png')
    plot_landscape_comparison(
        steps_a, min_a, max_a, steps_b, min_b, max_b,
        loss_path, ylim=loss_ylim, fill_alpha=fill_alpha, plot_stride=plot_stride,
    )
    print(f'Saved: {loss_path} (stride={plot_stride})')


def replot_grad_predictiveness_comparison(predictiveness_ylim=None, plot_stride=50):
    """Replot grad_predictiveness_comparison.png from vgg_*/grads.json."""
    flat_a = load_comparison_grads('vgg_a')
    flat_b = load_comparison_grads('vgg_a_bn')
    _, delta_a = compute_grad_predictiveness(flat_a)
    _, delta_b = compute_grad_predictiveness(flat_b)
    n = min(len(delta_a), len(delta_b))
    pred_path = os.path.join(FIGURES_DIR, 'grad_predictiveness_comparison.png')
    plot_grad_predictiveness(
        list(range(1, n + 1)), delta_a[:n], delta_b[:n], pred_path,
        ylim=predictiveness_ylim, plot_stride=plot_stride,
    )
    print(f'Saved: {pred_path} (stride={plot_stride})')


def replot_comparison_figures(
    loss_ylim=None,
    predictiveness_ylim=None,
    fill_alpha=0.20,
    plot_stride=50,
):
    """Replot both comparison figures (loss landscape + gradient predictiveness)."""
    replot_loss_landscape_comparison(
        loss_ylim=loss_ylim, fill_alpha=fill_alpha, plot_stride=plot_stride,
    )
    replot_grad_predictiveness_comparison(
        predictiveness_ylim=predictiveness_ylim, plot_stride=plot_stride,
    )


def run_model_comparison(train_loader, val_loader, epochs_n, lr, plot_stride=50):
    init_training()
    from torch import nn
    import torch

    criterion = nn.CrossEntropyLoss()
    results = {}

    for use_bn, tag in ((False, 'vgg_a'), (True, 'vgg_a_bn')):
        set_random_seeds()
        model = build_model(use_bn=use_bn)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        run_dir = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(run_dir, exist_ok=True)

        losses, grads, _, val_curve = train(
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            epochs_n=epochs_n,
            best_model_path=os.path.join(MODELS_DIR, f'{tag}.pt'),
            curve_path=os.path.join(FIGURES_DIR, f'{tag}_training_curve.png'),
        )

        with open(os.path.join(run_dir, 'losses.json'), 'w', encoding='utf-8') as f:
            json.dump(losses, f)
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

        results[tag] = {'losses': losses, 'grads': grads, 'summary': summary}
        print(f'[{tag}] best val acc: {summary["best_val_acc"]:.4f}')

    flat_a = flatten_steps(results['vgg_a']['grads'])
    flat_b = flatten_steps(results['vgg_a_bn']['grads'])
    _, delta_a = compute_grad_predictiveness(flat_a)
    _, delta_b = compute_grad_predictiveness(flat_b)
    n = min(len(delta_a), len(delta_b))
    plot_grad_predictiveness(
        list(range(1, n + 1)), delta_a[:n], delta_b[:n],
        os.path.join(FIGURES_DIR, 'grad_predictiveness_comparison.png'),
        plot_stride=plot_stride,
    )
    return results


def run_loss_landscape(train_loader, val_loader, epochs_n, learning_rates, use_bn=False):
    init_training()
    from torch import nn
    import torch

    tag = 'vgg_a_bn' if use_bn else 'vgg_a'
    criterion = nn.CrossEntropyLoss()
    all_losses = {}
    all_grads = {}

    for lr in learning_rates:
        set_random_seeds()
        model = build_model(use_bn=use_bn)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses, grads, _, _ = train(
            model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs_n,
        )
        all_losses[lr] = losses
        all_grads[lr] = grads
        print(f'[{tag}] finished lr={lr:g}')

    steps, min_curve, max_curve = compute_min_max_curves(all_losses)

    grad_steps, grad_min, grad_max = compute_min_max_curves(all_grads)
    grad_diff = compute_max_diff_curve(grad_min, grad_max)
    plot_max_diff_curve(
        grad_steps, grad_diff,
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

    return steps, min_curve, max_curve


def parse_args():
    parser = argparse.ArgumentParser(description='VGG-A / VGG-A+BN training and loss landscape')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--n-items', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--skip-comparison', action='store_true')
    parser.add_argument('--skip-landscape', action='store_true')
    parser.add_argument('--learning-rates', type=float, nargs='+', default=DEFAULT_LEARNING_RATES)
    parser.add_argument('--replot-comparison', action='store_true',
                        help='Replot loss landscape and gradient predictiveness comparison figures')
    parser.add_argument('--replot-loss-landscape', action='store_true',
                        help='Replot loss_landscape_comparison.png only')
    parser.add_argument('--replot-predictiveness', action='store_true',
                        help='Replot grad_predictiveness_comparison.png only')
    parser.add_argument('--loss-ylim', type=float, nargs=2, default=None)
    parser.add_argument('--predictiveness-ylim', type=float, nargs=2, default=None)
    parser.add_argument('--fill-alpha', type=float, default=0.15)
    parser.add_argument('--plot-stride', type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.replot_comparison or args.replot_loss_landscape or args.replot_predictiveness:
        loss_ylim = tuple(args.loss_ylim) if args.loss_ylim else None
        predictiveness_ylim = tuple(args.predictiveness_ylim) if args.predictiveness_ylim else None
        if args.replot_comparison or args.replot_loss_landscape:
            replot_loss_landscape_comparison(
                loss_ylim=loss_ylim,
                fill_alpha=args.fill_alpha,
                plot_stride=args.plot_stride,
            )
        if args.replot_comparison or args.replot_predictiveness:
            replot_grad_predictiveness_comparison(
                predictiveness_ylim=predictiveness_ylim,
                plot_stride=args.plot_stride,
            )
        return

    init_training()
    print(f'Device: {train_VGG.device}')
    train_loader = train_VGG.get_cifar_loader(
        train=True, batch_size=args.batch_size,
        num_workers=args.num_workers, n_items=args.n_items,
    )
    val_loader = train_VGG.get_cifar_loader(
        train=False, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, n_items=args.n_items,
    )

    if not args.skip_comparison:
        print('\n=== VGG-A vs VGG-A+BN comparison ===')
        run_model_comparison(train_loader, val_loader, args.epochs, args.lr, args.plot_stride)

    if not args.skip_landscape:
        print('\n=== Loss landscape (multiple learning rates) ===')
        steps_a, min_a, max_a = run_loss_landscape(
            train_loader, val_loader, args.epochs, args.learning_rates, use_bn=False,
        )
        steps_b, min_b, max_b = run_loss_landscape(
            train_loader, val_loader, args.epochs, args.learning_rates, use_bn=True,
        )
        plot_landscape_comparison(
            steps_a, min_a, max_a, steps_b, min_b, max_b,
            os.path.join(FIGURES_DIR, 'loss_landscape_comparison.png'),
            ylim=tuple(args.loss_ylim) if args.loss_ylim else None,
            fill_alpha=args.fill_alpha, plot_stride=args.plot_stride,
        )
        print('Saved loss landscape comparison.')

    print(f'\nOutputs saved under: {OUTPUT_DIR}')
