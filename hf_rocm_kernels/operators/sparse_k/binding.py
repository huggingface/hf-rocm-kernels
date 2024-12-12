import torch
from torch import Tensor

import hf_rocm_kernels._C # noqa: F401


def _sparse_k(a: Tensor, b: Tensor, d: Tensor) -> None:
    torch.ops._C.sparse_k(a, b, d)
