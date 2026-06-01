from .device import (
    cuda_available,
    get_default_device,
    npu_available,
    resolve_device,
    set_seed,
    torch,
)
from .nn import init_weights_, init_weights_kaiming_, init_weights_xavier_
