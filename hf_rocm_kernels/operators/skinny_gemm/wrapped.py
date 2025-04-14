from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from math import ceil

from .binding import _skinny_gemm


def skinny_gemm_checks(
    skinny_a: Tensor,
    b: Tensor,
    scale_tensor: Tensor,
    output: Tensor,
) -> None:
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

def infer_skinny_gemm_m_params(m: int) -> Tuple[int, int]:
    a_lanes = 2 if m > 16 else 1
    op_m = 16 if m > 8 else 8
    return a_lanes, op_m

def infer_skinny_gemm_params(skinny_a: Tensor, b: Tensor) -> Dict[str, int]:
    m, k = skinny_a.shape
    n = b.shape[1]
    # Output projection
    if (n, k) == (16384, 2048):
        if m <= 8:
            return {"A_producers": 3, "B_producers": 5, "consumers": 2, "split_k": 1,
                    "a_lanes": 1, "b_lanes": 4, "qsize": 3, "op_m": 8, "ops": 4} # 11.114 ± 0.275 -> 104.06%
        if m <= 16:
            return {"A_producers": 2, "B_producers": 3, "consumers": 2, "split_k": 1,
                    "a_lanes": 1, "b_lanes": 4, "qsize": 2, "op_m": 16, "ops": 8}
        else:
            return {"A_producers": 2, "B_producers": 4, "consumers": 2, "split_k": 1,
                    "a_lanes": 2, "b_lanes": 4, "qsize": 3, "op_m": 16, "ops": 4}
    # QKV projection
    if (n, k) == (2304, 16384):
        if m <= 8:
            return {"A_producers": 3, "B_producers": 6, "consumers": 2, "split_k": 16,
                    "a_lanes": 1, "b_lanes": 4, "qsize": 3, "op_m": 8, "ops": 4} # 12.485 ± 0.241
        if m <= 16:
            return {"A_producers": 3, "B_producers": 4, "consumers": 2, "split_k": 8,
                    "a_lanes": 1, "b_lanes": 4, "qsize": 3, "op_m": 16, "ops": 8} # 14.301 ± 0.108
        else:
            return {"A_producers": 2, "B_producers": 4, "consumers": 2, "split_k": 8,
                    "a_lanes": 2, "b_lanes": 4, "qsize": 2, "op_m": 16, "ops": 8} # 17.220 ± 0.183
    # Gate / up projection
    if (n, k) == (13312, 16384):
        if m <= 8:
            return {"A_producers": 3, "B_producers": 6, "consumers": 2, "split_k": 3,
                    "a_lanes": 1, "b_lanes": 3, "qsize": 4, "op_m": 8, "ops": 4} # 131.06%
        if m <= 16:
            return {"A_producers": 2, "B_producers": 5, "consumers": 3, "split_k": 3,
                    "a_lanes": 1, "b_lanes": 3, "qsize": 3, "op_m": 16, "ops": 8} # 62.136 ± 1.008 -> 118.38%
        else:
            return {"A_producers": 3, "B_producers": 6, "consumers": 3, "split_k": 1,
                    "a_lanes": 2, "b_lanes": 3, "qsize": 3, "op_m": 16, "ops": 8} # 69.251 ± 1.794 -> 116.40%
    # Down projection
    if (n, k) == (16384, 6656):
        if m <= 8:
            return {"A_producers": 2, "B_producers": 7, "consumers": 2, "split_k": 1,
                    "a_lanes": 1, "b_lanes": 4, "qsize": 2, "op_m": 8, "ops": 4} # 28.668 ± 0.275 -> 112.45 %
        if m <= 16:
            return {"A_producers": 2, "B_producers": 5, "consumers": 3, "split_k": 1,
                    "a_lanes": 1, "b_lanes": 4, "qsize": 3, "op_m": 16, "ops": 8}
        else:
            return {"A_producers": 2, "B_producers": 4, "consumers": 2, "split_k": 1,
                    "a_lanes": 2, "b_lanes": 4, "qsize": 2, "ops": 8, "op_m": 16}
    # Default
    op_m = 8 if m <= 8 else 16
    ops = 4 if op_m == 8 else 8
    a_lanes = 2 if m > 16 else 1
    b_lanes = n // (304 * 16)
    b_lanes = min(4, max(1, b_lanes))
    qsize = 3
    consumers = 3 # = qsize
    A_producers = 3 # = qsize
    B_producers = min(6, qsize * b_lanes)
    split_k = max(1, 304 // ceil(n / (16 * b_lanes)))
    return {"A_producers": A_producers, "B_producers": B_producers, "consumers": consumers,
            "a_lanes": a_lanes, "b_lanes": b_lanes,
            "qsize": qsize, "op_m": op_m, "ops": ops, "split_k": split_k}

def skinny_gemm(
    skinny_a: Tensor,
    b: Tensor,
    scale_tensor: Tensor,
    output: Optional[Tensor] = None,
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
    kwargs = infer_skinny_gemm_params(skinny_a, b)
    _skinny_gemm(
        A=skinny_a, B=b, scale_tensor=scale_tensor, D=output,
        **kwargs
    )
    return output


# def infer_skinny_gemm_other_params(skinny_a: Tensor, b: Tensor) -> Tuple[int, int, int, int, int, int]:
#     # A, B, C,   Bl, Qs, Ops   Sk
#     m, k = skinny_a.shape
#     n = b.shape[1]
#     # Gate / up projection
#     if (n, k) == (13312, 16384):
#         if m == 32:
#             return 5, 7, 4,   3, 4,   1
#         else:
#             return 2, 6, 3,   3, 3,   3
#     # Down projection
#     if (n, k) == (16384, 6656):
#         if m == 32:
#             return 3, 5, 2,   4, 2,   1
#         else:
#             return 2, 6, 3,   3, 3,   3
#     return 2, 6, 3,   3, 3,   3
