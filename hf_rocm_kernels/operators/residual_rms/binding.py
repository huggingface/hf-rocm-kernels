import torch
from torch import Tensor

import hf_rocm_kernels._HFRK_C # noqa: F401


def _residual_rms(
    input: Tensor, 
    residual: Tensor, 
    weight: Tensor, 
    scale_tensor: Tensor, 
    output: Tensor, 
    epsilon: float, 
    mode: int,
    num_threads: int,
) -> None:
    torch.ops._HFRK_C.residual_rms(input, residual, weight, scale_tensor, output, epsilon, mode, num_threads)
