import torch

import hf_rocm_kernels._C


def _increment(x: torch.Tensor) -> None:
    torch.ops._C.increment(x)
