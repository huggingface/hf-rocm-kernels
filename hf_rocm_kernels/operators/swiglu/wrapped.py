from typing import Tuple, Optional
import torch
from math import ceil
from torch import Tensor

from .binding import _swiglu


_HIGHEST_VECTORIZED_SWIGLU_MODE = 2


def swiglu_checks(
    gate_up_proj: Tensor, 
    scale_tensor: Tensor, 
    next_buffer: Tensor,
) -> None:
    # Check shapes
    assert gate_up_proj.dim() == 2, f"Expected gate_up_proj to have 2 dims but got {gate_up_proj.dim() = } instead."
    assert gate_up_proj.size(1) % 2 == 0, f"Expected gate_up_proj an even number of columns: {gate_up_proj.shape}."
    assert next_buffer.size(0) == gate_up_proj.size(0), \
        f"Expected same number of rows: {next_buffer.shape = } but {gate_up_proj.shape = }."
    assert scale_tensor.numel() == 1, f"Expected {scale_tensor.numel() = } to be 1."
    # Check devices
    device = gate_up_proj.device
    assert device.type == "cuda", f"Expected gate_up_proj.device to be of type cuda, but got {device.type = } instead."
    assert scale_tensor.device == device, f"Expected {scale_tensor.device = } to be the same as {device = }"
    assert next_buffer.device == device, f"Expected {next_buffer.device = } to be the same as {device = }"
    # Check layouts
    assert gate_up_proj.is_contiguous(), f"Expected gate_up_proj to be contiguous but got {gate_up_proj.stride() = }"
    assert next_buffer.is_contiguous(), f"Expected residual to be contiguous but got {next_buffer.stride() = }"

def swiglu_choose_mode(gate_up_proj: Tensor, next_buffer: Tensor, mode: int) -> int:
    cols_is_multiple_of_8 = (gate_up_proj.size(1) % 8 == 0) and (next_buffer.size(1) % 8 == 0)
    tensors_are_16b_aligned = all([x.data_ptr() % 16 == 0 for x in [gate_up_proj, next_buffer]])
    if mode == -1:
        mode = _HIGHEST_VECTORIZED_SWIGLU_MODE if (tensors_are_16b_aligned and cols_is_multiple_of_8) else 0
    elif mode > 0:
        assert tensors_are_16b_aligned, (
            f"Requested {mode = } > 0 requires tensors to be 16 bits aligned but {gate_up_proj.data_ptr() % 16 = } "
            f"and {next_buffer.data_ptr() % 16 = }"
        )
        assert cols_is_multiple_of_8, (
            f"Requested {mode = } > 0 requires {gate_up_proj.size(1) = } and {next_buffer.size(1) = } be multiple of 8"
        )
    return mode

def infer_num_threads(rows: int, hidden_dim: int, mode: int, num_threads: int) -> int:
    # If a number of threads is given, use it
    if num_threads != -1:
        return num_threads
    
    # For non-vectorized mode, just use as many threads as possible
    max_threads_per_sm = 1024
    if mode == 0:
        return max_threads_per_sm

    # For vectorized modes (mode > 0) use a pre-computed interpolation table
    work = rows * hidden_dim
    if work <= 32 * 16384:
        return 256
    if work <= 64 * 16384:
        return 448
    if work <= 128 * 16384:
        return 64
    if work <= 256 * 16384:
        return 192
    if work <= 1024 * 16384:
        return 512
    if work <= 2048 * 16384:
        return 128
    return 1024

def swiglu(
    gate_up_proj: Tensor,
    scale_tensor: Tensor,
    next_buffer: Optional[Tensor] = None,
    mode: int = -1,
    nb_threads: int = -1,
) -> Tensor:
    """Kernel that fuses a swiglu activation and a conversion to fp8. Can also initialize a buffer with as many rows as 
    the input to zero.
    Args:
        - gate_up_proj: a fp16 tensor of shape (rows, cols) in row-major format
        - scale_tensor: a fp32 one-item tensor to divide the output of the RMS norm before their conversion to fp8
        - next_buffer: an optional tensor of shape (rows, .) to initialize to zero
        - mode: the dispatch mode used for the C++ operation. Default value is -1, which sets the mode automatically
            depending on tensor alignment. If a specific mode is chosen and needs tensor alignment, an error is raised
    Outputs:
        - an fp8 tensor of shape (rows, cols) in row-major format
    """
    # If there is no buffer to initialize, create a dummy one
    if next_buffer is None:
        next_buffer = torch.empty(size=(gate_up_proj.size(0), 0), device=gate_up_proj.device, dtype=torch.float16)
    swiglu_checks(gate_up_proj, scale_tensor, next_buffer)
    mode = swiglu_choose_mode(gate_up_proj, next_buffer, mode)
    num_threads = infer_num_threads(gate_up_proj.size(0), gate_up_proj.size(1), mode, nb_threads)
    swiglu_out = torch.empty(
        size=(gate_up_proj.size(0), gate_up_proj.size(1) // 2), 
        dtype=torch.float8_e4m3fnuz, 
        device=gate_up_proj.device,
    )
    _swiglu(
        gate_up_proj=gate_up_proj,
        scale_tensor=scale_tensor,
        swiglu_out=swiglu_out,
        next_buffer=next_buffer,
        mode=mode,
        nb_threads=num_threads,
    )
    return swiglu_out
