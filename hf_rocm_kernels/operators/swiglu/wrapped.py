from typing import Optional
import torch
from torch import Tensor

from .binding import _swiglu


MAX_THREADS_PER_SM = 1024


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

def infer_num_threads(rows: int, force_scalar: bool, num_threads: int) -> int:
    # If a valid number of threads is given, use it
    if num_threads > 0 and num_threads <= MAX_THREADS_PER_SM:
        return num_threads
    # For non-vectorized mode, just use as many threads as possible
    if force_scalar:
        return MAX_THREADS_PER_SM
    # For vectorized mode, use a somewhat pre-computed interpolation table
    if rows <= 8: return 64
    if rows <= 32: return 256
    if rows <= 2048: return 192
    if rows <= 4096: return 512
    return 256

def swiglu(
    gate_up_proj: Tensor,
    scale_tensor: Tensor,
    next_buffer: Optional[Tensor] = None,
    force_scalar: bool = False,
    num_threads: int = -1,
) -> Tensor:
    """Kernel that fuses a swiglu activation and a conversion to fp8. Can also initialize a buffer with as many rows as 
    the input to zero.
    Args:
        - gate_up_proj: a fp16 tensor of shape (rows, cols) in row-major format
        - scale_tensor: a fp32 one-item tensor to divide the output of the RMS norm before their conversion to fp8
        - next_buffer: an optional tensor of shape (rows, .) to initialize to zero
        - force_scalar: if True, the operation will be scalarized even if the input tensors are 16-bit aligned
    Outputs:
        - an fp8 tensor of shape (rows, cols) in row-major format
    """
    # If there is no buffer to initialize, create a dummy one
    if next_buffer is None:
        next_buffer = torch.empty(size=(gate_up_proj.size(0), 0), device=gate_up_proj.device, dtype=torch.float16)
    swiglu_checks(gate_up_proj, scale_tensor, next_buffer)
    num_threads = infer_num_threads(gate_up_proj.size(0), force_scalar, num_threads)
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
        force_scalar=force_scalar,
        num_threads=num_threads,
    )
    return swiglu_out
