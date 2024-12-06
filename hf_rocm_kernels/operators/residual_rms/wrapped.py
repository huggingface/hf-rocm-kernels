import torch
from torch import Tensor

from .binding import _residual_rms


def residual_rms_checks(
    input: Tensor, 
    residual: Tensor, 
    weight: Tensor, 
    epsilon: float,
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
    # Check layouts
    assert input.is_contiguous(), f"Expected input to be contiguous but got {input.stride() = }"
    assert residual.is_contiguous(), f"Expected residual to be contiguous but got {residual.stride() = }"
    # Check scalars
    assert epsilon > 0, f"Expected RMS epsilon to be > 0 to avoid division by zero, but got {epsilon = }"


def infer_num_threads(num_threads: int) -> int:
    if num_threads < 0 or num_threads > 1024:
        raise ValueError(f"{num_threads = } is not between 0 and 1024")
    elif num_threads == 0:
        return 1024 # TODO: refine this thinking
    else:
        return num_threads

def residual_rms(
    input: Tensor, 
    residual: Tensor, 
    weight: Tensor,
    epsilon: float, 
    scale: float,
    mode: int = 0,
    num_threads: int = 0,
) -> Tensor:
    """Kernel that fuses a residual connection, an RMS normalization and a conversion to fp8. The resdiual argument is
    modified inplace (residual <- input + residual).
    Args:
        - input: a fp16 tensor of shape (rows, cols) in row-major format
        - residual: a fp16 tensor of shape (rows, cols) in row-major format
        - weight: a fp16 tensor of shape (cols, ) in row-major format which contains the weight of the RMS norm
        - epsilon: the small epsilon used inside the RMS norm to avoid division by zero
        - scale: a float to scale the output of the RMS norm before their conversion to fp8
        - mode: the dispatch mode used for the C++ operation. Default value is 0
        - num_threads: the number of threads per block in the kernel. Default value is 0, which then defaults to 1024
    Outputs:
        an fp8 tensor of shape (rows, cols) in row-major format.
    """
    residual_rms_checks(input, residual, weight, epsilon)
    num_threads = infer_num_threads(num_threads)
    output = torch.empty(size=input.shape, dtype=torch.float8_e4m3fnuz, device=input.device)
    _residual_rms(
        input=input,
        residual=residual,
        weight=weight,
        output=output,
        epsilon=epsilon,
        scale=scale,
        mode=mode,
        num_threads=infer_num_threads(num_threads),
    )
    return output
