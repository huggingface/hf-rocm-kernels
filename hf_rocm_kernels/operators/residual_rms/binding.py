import torch
from torch import Tensor

import hf_rocm_kernels._C


def _residual_rms(
    input: Tensor, 
    residual: Tensor, 
    weight: Tensor, 
    output: Tensor, 
    epsilon: float, 
    scale: float,
    mode: int,
    num_threads: int,
) -> None:
    torch.ops._C.residual_rms(input, residual, weight, output, epsilon, scale, mode, num_threads)
