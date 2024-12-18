from typing import Tuple, Optional
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
    assert skinny_a.size(0) % 8 == 0, f"Right now, {skinny_a.size(0) = } must be a multiple of 8."
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


def infer_warps_per_block(m: int, warps_per_block: Optional[int]) -> int:
    if warps_per_block is not None:
        # assert warps_per_block in [1, 2, 3, 4, 5], f"Incorrect value for {warps_per_block = }"
        return warps_per_block
    return {8: 6, 16: 6, 24: 3}.get(m, 1)


def sparse_k(
    skinny_a: Tensor, 
    b: Tensor,
    warps_per_block: Optional[int] = None,
) -> Tensor:
    """Skinny GEMM kernel that leverages artifical sparsity.
    Args:
        - skinny_a: a fp8 tensor of shape (m, k) in row-major format
        - b: a fp8 tensor of shape (k, n) in col-major format
    Outputs:
        an fp32 tensor of shape (m, n) in row-major format
    """
    sparse_k_checks(skinny_a, b)
    warps_per_block = infer_warps_per_block(skinny_a.size(0), warps_per_block)
    output = torch.empty(size=(skinny_a.size(0), b.size(1)), dtype=torch.float16, device=skinny_a.device)
    _sparse_k(skinny_a, b, output, warps_per_block)
    return output
