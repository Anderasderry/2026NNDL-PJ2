"""
Run all formal Task 1 (CIFAR-10) experiments sequentially.

Usage (from project root):
    python run_all_experiments.py
    python run_all_experiments.py --dry-run
    python run_all_experiments.py --group width --skip-existing
    python run_all_experiments.py --device cuda --num-workers 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'codes', 'CIFAR10', 'train.py')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'codes'))
from common.paths import CIFAR10_OUTPUT_DIR

OUTPUT_ROOT = CIFAR10_OUTPUT_DIR

# Shared baseline for control-variable ablations.
BASELINE: Dict[str, object] = {
    'epochs': 50,
    'batch_size': 128,
    'lr': 0.1,
    'weight_decay': 5e-4,
    'width': 64,
    'dropout': 0.5,
    'activation': 'relu',
    'optimizer': 'sgd',
    'loss': 'ce',
    'scheduler': 'cosine',
    'seed': 42,
}


@dataclass
class Experiment:
    run_name: str
    group: str
    description: str
    overrides: Dict[str, object] = field(default_factory=dict)

    def resolve_args(self, device: str, num_workers: int, epochs: Optional[int]) -> Dict[str, object]:
        args = dict(BASELINE)
        args.update(self.overrides)
        if epochs is not None:
            args['epochs'] = epochs
        args['device'] = device
        args['num_workers'] = num_workers
        return args


EXPERIMENTS: List[Experiment] = [
    Experiment(
        run_name='cifarnet',
        group='baseline',
        description='Main model: CIFARNet baseline (width=64, ReLU, CE, SGD)',
    ),
    # (a) Different number of filters
    Experiment(
        run_name='width32',
        group='width',
        description='Ablation (a): base width=32',
        overrides={'width': 32},
    ),
    Experiment(
        run_name='width96',
        group='width',
        description='Ablation (a): base width=96',
        overrides={'width': 96},
    ),
    # (b) Different loss functions / regularization
    Experiment(
        run_name='loss_label_smooth',
        group='loss',
        description='Ablation (b): label smoothing (0.1)',
        overrides={'loss': 'label_smooth'},
    ),
    Experiment(
        run_name='wd_1e4',
        group='loss',
        description='Ablation (b): lighter L2 regularization (weight_decay=1e-4)',
        overrides={'weight_decay': 1e-4},
    ),
    Experiment(
        run_name='wd_1e3',
        group='loss',
        description='Ablation (b): stronger L2 regularization (weight_decay=1e-3)',
        overrides={'weight_decay': 1e-3},
    ),
    # (c) Different activations
    Experiment(
        run_name='act_gelu',
        group='activation',
        description='Ablation (c): GELU activation',
        overrides={'activation': 'gelu'},
    ),
    Experiment(
        run_name='act_leaky_relu',
        group='activation',
        description='Ablation (c): LeakyReLU activation',
        overrides={'activation': 'leaky_relu'},
    ),
    # Optimizer comparison (Task 1 optimizer strategy)
    Experiment(
        run_name='optim_adamw',
        group='optimizer',
        description='Optimizer ablation: AdamW (lr=3e-4)',
        overrides={'optimizer': 'adamw', 'lr': 3e-4},
    ),
]


def build_command(resolved: Dict[str, object]) -> List[str]:
    cmd = [sys.executable, TRAIN_SCRIPT]
    flag_map = {
        'epochs': '--epochs',
        'batch_size': '--batch-size',
        'lr': '--lr',
        'weight_decay': '--weight-decay',
        'width': '--width',
        'dropout': '--dropout',
        'activation': '--activation',
        'optimizer': '--optimizer',
        'loss': '--loss',
        'scheduler': '--scheduler',
        'num_workers': '--num-workers',
        'seed': '--seed',
        'device': '--device',
        'run_name': '--run-name',
    }
    for key, flag in flag_map.items():
        if key in resolved:
            cmd.extend([flag, str(resolved[key])])
    return cmd


def summary_path(run_name: str) -> str:
    return os.path.join(OUTPUT_ROOT, run_name, 'summary.json')


def load_summary(run_name: str) -> Optional[dict]:
    path = summary_path(run_name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def collect_results(experiments: List[Experiment]) -> List[dict]:
    rows = []
    for exp in experiments:
        summary = load_summary(exp.run_name)
        row = {
            'run_name': exp.run_name,
            'group': exp.group,
            'description': exp.description,
            'status': 'done' if summary else 'missing',
        }
        if summary:
            row.update({
                'best_test_acc': summary.get('best_test_acc'),
                'best_test_error': summary.get('best_test_error'),
                'parameters': summary.get('parameters'),
                'train_time_sec': summary.get('train_time_sec'),
            })
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description='Run all Task 1 CIFAR-10 experiments')
    parser.add_argument(
        '--group',
        choices=['all', 'baseline', 'width', 'loss', 'activation', 'optimizer'],
        default='all',
        help='Run a subset of experiment groups',
    )
    parser.add_argument('--device', default=_default_device())
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs for all runs')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running')
    parser.add_argument('--skip-existing', action='store_true', help='Skip runs with summary.json')
    return parser.parse_args()


def _default_device() -> str:
    codes_dir = os.path.join(PROJECT_ROOT, 'codes')
    if codes_dir not in sys.path:
        sys.path.insert(0, codes_dir)
    from common.utils.device import get_default_device

    return get_default_device()


def select_experiments(group: str) -> List[Experiment]:
    if group == 'all':
        return EXPERIMENTS
    return [exp for exp in EXPERIMENTS if exp.group == group]


def main():
    args = parse_args()
    experiments = select_experiments(args.group)

    if not os.path.isfile(TRAIN_SCRIPT):
        raise FileNotFoundError(f'Training script not found: {TRAIN_SCRIPT}')

    print(f'Project root : {PROJECT_ROOT}')
    print(f'Train script : {TRAIN_SCRIPT}')
    print(f'Experiments  : {len(experiments)} ({args.group})')
    print(f'Device       : {args.device}')
    print('-' * 72)

    failed = []
    skipped = []

    for index, exp in enumerate(experiments, start=1):
        resolved = exp.resolve_args(args.device, args.num_workers, args.epochs)
        resolved['run_name'] = exp.run_name
        cmd = build_command(resolved)

        print(f'\n[{index}/{len(experiments)}] {exp.run_name} ({exp.group})')
        print(f'  {exp.description}')

        if args.skip_existing and os.path.isfile(summary_path(exp.run_name)):
            print('  -> skipped (summary.json exists)')
            skipped.append(exp.run_name)
            continue

        print('  ->', ' '.join(cmd))

        if args.dry_run:
            continue

        start = time.time()
        try:
            subprocess.run(cmd, check=True, cwd=os.path.dirname(TRAIN_SCRIPT))
            elapsed = time.time() - start
            print(f'  -> finished in {elapsed / 60:.1f} min')
        except subprocess.CalledProcessError:
            print('  -> FAILED')
            failed.append(exp.run_name)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    report = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'group': args.group,
        'device': args.device,
        'experiments': collect_results(experiments),
        'failed': failed,
        'skipped': skipped,
    }
    report_path = os.path.join(OUTPUT_ROOT, 'experiments_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('\n' + '=' * 72)
    print(f'Report saved: {report_path}')
    if skipped:
        print(f'Skipped: {", ".join(skipped)}')
    if failed:
        print(f'Failed : {", ".join(failed)}')
        sys.exit(1)
    if args.dry_run:
        print('Dry run complete (no training executed).')
    else:
        print('All selected experiments completed.')


if __name__ == '__main__':
    main()
