from typing import Tuple
import torch
from torch import Tensor

from .binding import _sparse_k


_HIGHEST_RESIDUAL_RMS_MODE = 1


def sparse_k_checks(
    skinny_a: Tensor,
    b: Tensor,
) -> None:
    # Check shapes
    assert skinny_a.dim() == 2, f"Expected skinny_a to have 2 dimensions but got {skinny_a.dim() = } instead."
    assert skinny_a.size(1) == b.size(0), f"Expected {skinny_a.size(1) = } and {b.size(0) = } to be the same."
    # Temporary restrictions (TODO)
    assert skinny_a.size(0) == 8, f"Right now, {skinny_a.size(0) = } must be 8."
    assert skinny_a.size(1) % 64 == 0, f"Right now, {skinny_a.size(1) = } must be a multiple of 64."
    assert b.size(1) % 16 == 0, f"Right now, {b.size(0) = } must be a multiple of 16."
    # Check layouts
    assert skinny_a.is_contiguous(), f"Expected skinny_a to be contiguous but got {skinny_a.stride() = }"
    assert b.stride(0) == 1, f"Expected b to be column-major but got {b.stride() = }"
    # Check dtypes
    assert skinny_a.dtype == torch.float8_e4m3fnuz, f"Expected {skinny_a.dtype = } to be torch.float8_e4m3fnuz"
    assert b.dtype == torch.float8_e4m3fnuz, f"Expected {b.dtype = } to be torch.float8_e4m3fnuz"
    # Check devices
    device = skinny_a.device
    assert device.type == "cuda", f"Expected input.device to be of type cuda, but got {device.type = } instead."
    assert b.device == device, f"Expected {b.device = } to be the same as {b.device = }"


def infer_warps_per_block(n: int) -> int:
    for warps_per_block in [8, 4, 2, 1]:
        if n % (16 * warps_per_block) == 0:
            return warps_per_block
    raise ValueError(f"{n = } is not divisible by 16")


def sparse_k(
    skinny_a: Tensor, 
    b: Tensor,
) -> Tensor:
    """Skinny GEMM kernel that leverages artifical sparsity.
    Args:
        - skinny_a: a fp8 tensor of shape (m, k) in row-major format
        - b: a fp8 tensor of shape (k, n) in col-major format
    Outputs:
        an fp32 tensor of shape (m, n) in row-major format
    """
    sparse_k_checks(skinny_a, b)
    warps_per_block = infer_warps_per_block(b.size(1))
    output = torch.zeros(size=(16, b.size(1)), dtype=torch.float32, device=skinny_a.device)
    _sparse_k(skinny_a, b, output, warps_per_block)
    return output
