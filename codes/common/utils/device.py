"""Device setup for Ascend NPU / CUDA / CPU.

Import this module before using ``torch`` so NPU backends are initialized
correctly on Ascend toolkit environments.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Union

import numpy as np

# Prevent broken torch_npu autoload (triton mismatch) before torch import.
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')


def _patch_triton_attrs_descriptor() -> None:
    """Patch missing AttrsDescriptor in older triton builds used with torch_npu."""

    class AttrsDescriptor:
        property_values: dict

        @classmethod
        def from_dict(cls, payload):
            obj = cls()
            arg_properties = payload.get('arg_properties', {})
            obj.property_values = {
                'tt.divisibility': arg_properties.get('tt.divisibility'),
                'tt.equal_to': arg_properties.get('tt.equal_to'),
            }
            return obj

    for module_name in ('triton.compiler.compiler', 'triton.backends.compiler'):
        try:
            module = __import__(module_name, fromlist=['AttrsDescriptor'])
        except ImportError:
            continue
        if not hasattr(module, 'AttrsDescriptor'):
            module.AttrsDescriptor = AttrsDescriptor


_patch_triton_attrs_descriptor()

import torch

_NPU_AVAILABLE = False
try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except Exception:
    _NPU_AVAILABLE = False


def npu_available() -> bool:
    return _NPU_AVAILABLE


def cuda_available() -> bool:
    return torch.cuda.is_available()


def get_default_device() -> str:
    if _NPU_AVAILABLE:
        return 'npu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        device = get_default_device()
    dev = torch.device(device)
    if dev.type == 'npu' and not _NPU_AVAILABLE:
        raise RuntimeError('NPU requested but Ascend NPU is not available.')
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but CUDA is not available.')
    if dev.type == 'npu':
        torch.npu.set_device(dev)
    return dev


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if _NPU_AVAILABLE:
        torch.npu.manual_seed_all(seed)
