import torch

#import hf_rocm_kernels._HFRK_C # noqa: F401


def _increment(x: torch.Tensor) -> None:
    torch.hfrk.increment(x)
