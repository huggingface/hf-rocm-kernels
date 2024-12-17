import torch
from torch import Tensor

import hf_rocm_kernels._C_HFRK # noqa: F401


def _sparse_k(a: Tensor, b: Tensor, d: Tensor, W: int) -> None:
    torch.ops._C_HFRK.sparse_k(a, b, d, W)
