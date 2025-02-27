from typing import Tuple, Optional
import torch
from torch import Tensor

from .binding import _residual_rms


_HIGHEST_RESIDUAL_RMS_MODE = 3


def residual_rms_checks(
    input: Tensor, 
    residual: Tensor, 
    weight: Tensor, 
    scale_tensor: Tensor,
    epsilon: float,
    next_buffer: Tensor,
) -> None:
    # Check shapes
    assert input.dim() == 2, f"Expected input to have 2 dimensions but got {input.dim() = } instead."
    assert residual.shape == input.shape, \
        f"Expected input and residual to have same shape but got {input.shape = } and {residual.shape = }"
    assert weight.shape == (input.size(1), ), \
        f"Expected weight to have shape {(input.size(1), ) = } but got {weight.shape = }"
    # Check devices
    device = input.device
    assert device.type == "cuda", f"Expected input.device to be of type cuda, but got {device.type = } instead."
    assert residual.device == device, f"Expected {residual.device = } to be the same as {input.device = }"
    if scale_tensor is not None:
        assert scale_tensor.device == device, f"Expected {scale_tensor.device = } to be the same as {input.device = }"
    assert next_buffer.device == device, f"Expected {next_buffer.device = } to be the same as {input.device = }"
    # Check layouts
    assert input.is_contiguous(), f"Expected input to be contiguous but got {input.stride() = }"
    assert residual.is_contiguous(), f"Expected residual to be contiguous but got {residual.stride() = }"
    # Check scalars
    assert epsilon > 0, f"Expected RMS epsilon to be > 0 to avoid division by zero, but got {epsilon = }"


def residual_rms_choose_mode(
    input: Tensor, 
    residual: Tensor, 
    weight: Tensor, 
    next_buffer: Tensor, 
    mode: int,
) -> int:
    cols_is_multiple_of_8 = (input.size(1) % 8 == 0) and (next_buffer.size(1) % 8 == 0)
    tensors_are_16b_aligned = all([x.data_ptr() % 16 == 0 for x in [input, residual, weight]])
    if mode == -1:
        mode = _HIGHEST_RESIDUAL_RMS_MODE if (tensors_are_16b_aligned and cols_is_multiple_of_8) else 0
    elif mode > 0:
        assert tensors_are_16b_aligned, (
            f"Requested a {mode = } > 0 requires tensors to be 16 bits aligned but got {input.data_ptr() % 16 = }, "
            f"{residual.data_ptr() % 16 = }, {weight.data_ptr() % 16 = }"
        )
        assert cols_is_multiple_of_8, f"Requested {mode = } requires {input.size(1) = } to be a multiple of 8."
    return mode


def infer_num_threads(rows: int, num_threads: int) -> int:
    # Error case
    if num_threads < 0 or num_threads > 1024:
        raise ValueError(f"{num_threads = } is not between 0 and 1024")
    # Case: num_threads was specified
    elif num_threads != 0:
        return num_threads
    # Otherwise, we branch upon the number of rows
    if rows <= 32:
        return 1024
    elif rows <= 128:
        return 768
    return 384


def residual_rms(
    input: Tensor, 
    residual: Tensor, 
    weight: Tensor,
    epsilon: float, 
    scale_tensor: Optional[Tensor] = None,
    next_buffer: Optional[Tensor] = None,
    num_threads: int = 0,
    force_scalar: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Kernel that fuses a residual connection, an RMS normalization and a conversion to fp8. The resdiual argument is
    modified inplace (residual <- input + residual).
    Args:
        - input: a fp16 tensor of shape (rows, cols) in row-major format
        - residual: a fp16 tensor of shape (rows, cols) in row-major format
        - weight: a fp16 tensor of shape (cols, ) in row-major format which contains the weight of the RMS norm
        - epsilon: the small epsilon used inside the RMS norm to avoid division by zero
        - scale_tensor: a fp32 one-item tensor to divide the output of the RMS norm before their conversion to fp8. If
            set to None, then the output dtype is fp16
        - next_buffer: an optional tensor of shape (rows, .) to initialize to zero if the output dtype in fp8
        - num_threads: the number of threads per block in the kernel. Default value is 0, which then defaults to 1024
    Outputs:
        - an fp8 tensor of shape (rows, cols) in row-major format
        - the residual modified in place
    """
    if next_buffer is None:
        next_buffer = torch.empty(size=(input.size(0), 0), device=input.device, dtype=torch.float16)

    residual_rms_checks(input, residual, weight, scale_tensor, epsilon, next_buffer)
    num_threads = infer_num_threads(input.size(0), num_threads)

    if scale_tensor is not None:
        output = torch.empty(size=input.shape, dtype=torch.float8_e4m3fnuz, device=input.device)
    else:
        # TODO: here, we could use input as the output tensor
        output = torch.empty(size=input.shape, dtype=torch.float16, device=input.device)
    _residual_rms(
        input=input,
        residual=residual,
        weight=weight,
        scale_tensor=scale_tensor,
        epsilon=epsilon,
        output=output,
        next_buffer=next_buffer,
        num_threads=num_threads,
        force_scalar=force_scalar,
    )
    return output, residual
