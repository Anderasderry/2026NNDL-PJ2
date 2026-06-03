"""Shared utilities for NNDL Project 2."""

from .paths import (
    CIFAR10_OUTPUT_DIR,
    CODES_DIR,
    DATA_ROOT,
    LOGS_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    VGG_FIGURES_DIR,
    VGG_MODELS_DIR,
    VGG_OUTPUT_DIR,
    cifar10_run_dir,
)


def setup_import_paths():
    """Allow `from common.xxx import ...` when running scripts inside task folders."""
    import sys

    if CODES_DIR not in sys.path:
        sys.path.insert(0, CODES_DIR)
