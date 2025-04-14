import torch
from torch import Tensor

import hf_rocm_kernels._HFRK_C  # noqa: F401


def _all_reduce_init(
    rank: int,
    world_size: int,
    port: int,
    comms_A: Tensor,
    comms_B: Tensor
) -> int:
    return torch.ops._HFRK_C.all_reduce_init(rank, world_size, port, comms_A, comms_B)


def _all_reduce(
    allreduce_engine_ptr: int,
    A: Tensor,
    B: Tensor,
    D: Tensor,
    scale_tensor: Tensor,
    b_lanes: int,
    split_k: int,
    is_capturing: bool
) -> Tensor:
    torch.ops._HFRK_C.all_reduce(allreduce_engine_ptr, A, B, D, scale_tensor, b_lanes, split_k, is_capturing)

    return D
