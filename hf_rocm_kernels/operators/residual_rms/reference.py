from typing import Tuple, Optional
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


def generate_residual_rms_data(
    rows: int, cols: int, seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, float, Tensor]:
    """Generates random inputs for the residual_rms operation. The generated input's shape is determined by (rows) and 
    (cols), and one can pass a (seed) to ensure repeatability."""
    if seed is not None:
        torch.manual_seed(seed)
    input = torch.normal(0, 1, size=(rows, cols), device="cuda", dtype=torch.float16)
    residual = torch.normal(0, 1, size=(rows, cols), device="cuda", dtype=torch.float16)
    weights = torch.normal(0, 1, size=(cols, ), device="cuda", dtype=torch.float16)
    epsilon = torch.rand(size=(1,)).add(1).mul(1e-5).item()
    scale = torch.rand(size=(1,), device="cuda", dtype=torch.float32).mul(2).add(1)
    return input, residual, weights, epsilon, scale
