import torch
from torch import Tensor

import hf_rocm_kernels._HFRK_C # noqa: F401


def _skinny_gemm(
    A: Tensor,
    B: Tensor,
    scale_tensor: Tensor,
    D: Tensor,
    split_k: int,
    A_producers: int,
    B_producers: int,
    consumers: int,
    a_lanes: int,
    b_lanes: int,
    qsize: int,
    op_m: int,
    ops: int,
) -> int:
    return torch.ops._HFRK_C.skinny_gemm_tb(
        A, B, D, scale_tensor,
        split_k,
        A_producers, B_producers, consumers,
        a_lanes, b_lanes, qsize,
        op_m, ops
    )
