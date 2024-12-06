import torch

import hf_rocm_kernels._C  # noqa: F401


def _increment(x: torch.Tensor) -> None:
    torch.ops._C.increment(x)
