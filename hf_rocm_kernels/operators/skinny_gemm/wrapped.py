from typing import Tuple, Optional
import torch
from torch import Tensor

from .binding import _skinny_gemm


def skinny_gemm_checks(
    skinny_a: Tensor,
    b: Tensor,
    scale_tensor: Tensor,
    output: Tensor,
) -> None:
    # Temporary restrictions (TODO)
    assert skinny_a.size(0) <= 16, f"Right now, {skinny_a.size(0) = } must be below 16."
    # Check shapes
    assert skinny_a.dim() == 2, f"Expected skinny_a to have 2 dimensions but got {skinny_a.dim() = } instead."
    assert skinny_a.size(1) == b.size(0), f"Expected {skinny_a.size(1) = } and {b.size(0) = } to be the same."
    assert scale_tensor.numel() == 1, f"Expected {scale_tensor.numel() = } to be 1."
    assert output.shape == (skinny_a.size(0), b.size(1)), (
        f"{output.shape} does not match with {skinny_a.shape = } and {b.shape =}" )
    # Check layouts
    assert skinny_a.is_contiguous(), f"Expected skinny_a to be contiguous but got {skinny_a.stride() = }"
    assert b.stride(0) == 1, f"Expected b to be column-major but got {b.stride() = }"
    assert output.is_contiguous(), f"Expected output to be contiguous but got {output.stride() = }"
    # Check dtypes
    assert skinny_a.dtype == torch.float8_e4m3fnuz, f"Expected {skinny_a.dtype = } to be torch.float8_e4m3fnuz"
    assert b.dtype == torch.float8_e4m3fnuz, f"Expected {b.dtype = } to be torch.float8_e4m3fnuz"
    assert scale_tensor.dtype == torch.float32, f"Expected {scale_tensor.dtype = } to be torch.float32"
    assert output.dtype == torch.float16, f"Expected {output.dtype = } to be torch.float16"
    # Check devices
    device = skinny_a.device
    assert device.type == "cuda", f"Expected input.device to be of type cuda, but got {device.type = } instead."
    assert b.device == device, f"Expected {b.device = } to be the same as {device = }"
    assert scale_tensor.device == device, f"Expected {scale_tensor.device = } to be the same as {device = }"
    assert output.device == device, f"Expected {output.device = } to be the same as {device = }"

def skinny_gemm(
    skinny_a: Tensor, 
    b: Tensor,
    scale_tensor: Tensor,
    output: Optional[Tensor] = None,
    split_k: Optional[int] = None,
    b_lanes: Optional[int] = None,
) -> Tensor:
    """Skinny GEMM kernel that leverages artifical sparsity.
    Args:
        - skinny_a: a fp8 tensor of shape (m, k) in row-major format
        - b: a fp8 tensor of shape (k, n) in col-major format
        - scale_tensor: a fp32 one-item tensor to scale the output of the RMS norm before their conversion to fp8
        - output: an optional tensor to store the output, which MUST be zero initialized (or the bias)
        - split_k: an optional int to determine how many splits there are along the K axis
    Outputs:
        an fp16 tensor of shape (m, n) in row-major format
    """
    if output is None:
        output = torch.zeros(size=(skinny_a.size(0), b.size(1)), dtype=torch.float16, device=skinny_a.device)
    skinny_gemm_checks(skinny_a, b, scale_tensor, output)
    split_k = 1 if split_k is None else split_k
    b_lanes = 3 if b_lanes is None else b_lanes
    assert b_lanes in [2, 3, 4, 5]
    _skinny_gemm(skinny_a, b, scale_tensor, output, b_lanes, split_k)
    return output
