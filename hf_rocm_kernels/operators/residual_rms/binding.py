import torch
from torch import Tensor

import hf_rocm_kernels._HFRK_C # noqa: F401


def _residual_rms(
    input: Tensor, 
    residual: Tensor, 
    weight: Tensor, 
    scale_tensor: Tensor, 
    epsilon: float, 
    output: Tensor, 
    next_buffer: Tensor, 
    mode: int,
    num_threads: int,
) -> None:
    torch.ops._HFRK_C.residual_rms(
        input, residual, weight, scale_tensor, epsilon, output, next_buffer, mode, num_threads
    )
