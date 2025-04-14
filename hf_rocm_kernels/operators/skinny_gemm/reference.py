from typing import Tuple, Optional
import torch
from torch import Tensor

from hf_rocm_kernels.utils.fp8 import fp8_quantize


def reference_skinny_gemm(
    skinny_a: Tensor,
    b: Tensor,
    scale_a: Tensor,
    scale_b: Tensor,
    output: Optional[Tensor] = None,
) -> Tensor:
    """Reference function for the skinny_gemm operator. For details, check the operator's docstring."""
    if output is None:
        output = torch.zeros(size=(skinny_a.size(0), b.size(1)), dtype=torch.float16, device=skinny_a.device)
    torch._scaled_mm(
        input=skinny_a,
        mat2=b,
        scale_a=scale_a,
        scale_b=scale_b,
        out=output,
        out_dtype=torch.float16,
    )
    return output


def generate_skinny_gemm_data(
    m: int, n: int, k: int, seed: Optional[int] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generates random inputs for the skinny_gemm operation. The generated input's shape is determined by (m), (n) and
    (k), and one can pass a (seed) to ensure repeatability."""
    if seed is not None:
        torch.manual_seed(seed)
    scale_tensor = torch.rand(size=(1,), device="cuda", dtype=torch.float32).mul(2).add(1)
    skinny_a = fp8_quantize(
        torch.ones(size=(m, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0]
    b = fp8_quantize(
        torch.ones(size=(n, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0].t()
    output = torch.zeros(size=(m, n), dtype=torch.float16, device="cuda")
    return skinny_a, b, scale_tensor, output
