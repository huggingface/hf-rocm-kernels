import torch
from torch import Tensor

import hf_rocm_kernels._HFRK_C # noqa: F401


def _skinny_gemm(
    A: Tensor,
    B: Tensor,
    scale_tensor: Tensor,
    D: Tensor,
    b_lanes: int,
    split_k: int,
) -> None:
    torch.ops._HFRK_C.skinny_gemm(A, B, D, scale_tensor, b_lanes, split_k)
