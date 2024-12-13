import torch.random
import torch.random
from hf_rocm_kernels.operators.sparse_k.binding import _sparse_k
from hf_rocm_kernels.operators.sparse_k.wrapped import sparse_k
import torch
from torch import Tensor


def reference_sparse_8r2x8x64_matmul(a: Tensor, b: Tensor) -> Tensor:
    scale = torch.tensor(1, dtype=torch.float32, device="cuda")
    a = torch.vstack([a, a])
    for i in range(0, 64, 4):
        a[:8, i + 2] = 0
        a[:8, i + 3] = 0
        a[8:, i + 0] = 0
        a[8:, i + 1] = 0
    ref = torch._scaled_mm(
        a.to(torch.float8_e4m3fnuz), 
        b.to(torch.float8_e4m3fnuz), 
        scale_a=scale, 
        scale_b=scale, 
        out_dtype=torch.float32,
    )
    return ref

def reference_skinny_fp8_gemm_WS4_Awpl2(a: Tensor, b: Tensor) -> Tensor:
    k = a.size(1)
    d = torch.zeros((16, b.size(1)), device="cuda")
    for i in range(0, k, 64):
        r = reference_sparse_8r2x8x64_matmul(a[:, i:i+64], b[i:i+64, :])
        d += r
    return d

if __name__ == "__main__":

    Awpl = 1
    n = 16 * 16

    a = torch.rand(size=(8, 64 * Awpl), device="cuda")
    b = torch.rand(size=(n, 64 * Awpl), device="cuda").T
    
    d = sparse_k(a.to(torch.float8_e4m3fnuz), b.to(torch.float8_e4m3fnuz))
    # d = torch.ones(size=(16, 16 * WS), dtype=torch.float32, device="cuda") * 0.01
    # _sparse_k(
    #     a.to(torch.float8_e4m3fnuz), 
    #     b.to(torch.float8_e4m3fnuz), 
    #     d,
    #     WS,
    # )
    # print(c)
    # print(d)
    ref = reference_skinny_fp8_gemm_WS4_Awpl2(a, b)

    # print("REF:\n", ref)
    # print()
    # print("OUR:\n", d.tolist())
    print((ref - d).abs().sum())
