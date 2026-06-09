"""Project directory layout and output paths."""

import os

CODES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.normpath(os.path.join(CODES_DIR, '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
LOGS_ROOT = os.path.join(PROJECT_ROOT, 'logs')

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'outputs')
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
CIFAR10_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'CIFAR10')
VGG_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'VGG_BatchNorm')
VGG_FIGURES_DIR = os.path.join(VGG_OUTPUT_DIR, 'figures')
VGG_MODELS_DIR = os.path.join(VGG_OUTPUT_DIR, 'models')


def cifar10_run_dir(run_name: str) -> str:
    return os.path.join(CIFAR10_OUTPUT_DIR, run_name)
