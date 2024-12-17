import torch

import hf_rocm_kernels._C_HFRK  # noqa: F401


def _increment(x: torch.Tensor) -> None:
    torch.ops._C_HFRK.increment(x)
