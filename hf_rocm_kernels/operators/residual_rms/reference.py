from typing import Tuple, Optional
from torch import Tensor
import torch

from hf_rocm_kernels.utils.fp8 import fp8_quantize

def reference_residual_rms(
    input: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    scale_tensor: Optional[Tensor],
    next_buffer: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, float]:
    """Reference for the residual_rms operation. Check its docstring for more details, the only difference here is that
    the scale needs to be passed a tensor and not a float."""
    # Type checking
    assert input.dtype == torch.float16, f"Expected torch.float16 but got {input.dtype = }"
    assert residual.dtype == torch.float16, f"Expected torch.float16 but got {residual.dtype = }"
    # FastRMSNorm
    input += residual
    residual = input
    input = reference_rms(input, epsilon)
    # Convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)
    input = weight * input
    if scale_tensor is not None:
        # Convert to fp8
        qinput, scale_tensor = fp8_quantize(input, scale_tensor)
        # Zero-init the next buffer
        if next_buffer is not None:
            next_buffer.fill_(0)
    else:
        qinput = input
    return qinput, residual, scale_tensor


def precise_residual_rms(
    input: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    scale_tensor: Optional[Tensor],
    next_buffer: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Reference for the residual_rms operation. Check its docstring for more details, the only difference here is that
    the scale needs to be passed a tensor and not a float."""
    # Conversion to fp64
    input = input.to(torch.float64)
    residual = residual.to(torch.float64)
    weight = weight.to(torch.float64)
    # FastRMSNorm
    input += residual
    residual = input
    input = reference_rms(input, epsilon)
    input = weight * input
    if scale_tensor is not None:
        # Convert to fp8
        qinput, scale_tensor = fp8_quantize(input, scale_tensor)
        # Zero-init the next buffer
        if next_buffer is not None:
            next_buffer.fill_(0)
    else:
        qinput = input
    return qinput, residual


def reference_rms(x: Tensor, eps: float) -> Tensor:
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def generate_residual_rms_data(
    rows: int, cols: int, buffer_cols: int = 0, dtype: torch.dtype = torch.float16, seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, float, Optional[Tensor]]:
    """Generates random inputs for the residual_rms operation. The generated input's shape is determined by (rows) and
    (cols), and one can pass a (seed) to ensure repeatability."""
    if seed is not None:
        torch.manual_seed(seed)
    input = torch.normal(0, 1, size=(rows, cols), device="cuda", dtype=torch.float16)
    residual = torch.normal(0, 1, size=(rows, cols), device="cuda", dtype=torch.float16)
    weights = torch.normal(0, 1, size=(cols, ), device="cuda", dtype=torch.float16)
    epsilon = torch.rand(size=(1,)).add(1).mul(1e-5).item()
    if dtype == torch.float8_e4m3fnuz:
        scale_tensor = torch.rand(size=(1,), device="cuda", dtype=torch.float32).mul(2).add(1)
    else:
        scale_tensor = None
    next_buffer = None if buffer_cols == 0 else torch.empty((rows, buffer_cols), device="cuda", dtype=torch.float16)
    return input, residual, weights, epsilon, scale_tensor, next_buffer
