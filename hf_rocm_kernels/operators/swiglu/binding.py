import torch
from torch import Tensor

import hf_rocm_kernels._HFRK_C  # noqa: F401


def _swiglu(
    gate_up_proj: Tensor,
    scale_tensor: Tensor,
    swiglu_out: Tensor,
    next_buffer: Tensor,
    num_threads: int,
    force_scalar: bool,
) -> None:
    torch.ops._HFRK_C.swiglu(
        gate_up_proj, scale_tensor, swiglu_out, next_buffer, num_threads, force_scalar
    )
