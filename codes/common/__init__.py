"""Shared utilities for NNDL Project 2."""

import os

CODES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.normpath(os.path.join(CODES_DIR, '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')


def setup_import_paths():
    """Allow `from common.xxx import ...` when running scripts inside task folders."""
    import sys

    if CODES_DIR not in sys.path:
        sys.path.insert(0, CODES_DIR)
