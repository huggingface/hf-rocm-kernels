from typing import Tuple
from torch import Tensor
import torch

from hf_rocm_kernels.utils.fp8 import fp8_quantize

def reference_residual_rms(
    input: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    tensor_scale: Tensor,
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
    # Convert to fp8
    qinput, tensor_scale = fp8_quantize(input, tensor_scale)
    return qinput, residual, tensor_scale


def reference_rms(x: Tensor, eps: float) -> Tensor:
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)
