import torch

import hf_rocm_kernels._HFRK_C  # noqa: F401


def _increment(x: torch.Tensor) -> None:
    torch.ops._HFRK_C.increment(x)
