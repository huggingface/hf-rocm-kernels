import torch.random
import torch.random
from hf_rocm_kernels.operators.sparse_k.binding import _sparse_k
import torch
from torch import Tensor


def reference_sparse_8r2x8x64_matmul(a: Tensor, b: Tensor) -> Tensor:
    scale = torch.tensor(1, dtype=torch.float32, device="cuda")
    a = torch.vstack([a, a])
    for i in range(0, 64, 4):
        a[8:, i + 0] = 0
        a[8:, i + 1] = 0
        a[:8, i + 2] = 0
        a[:8, i + 3] = 0
    ref = torch._scaled_mm(
        a.to(torch.float8_e4m3fnuz), 
        b.to(torch.float8_e4m3fnuz), 
        scale_a=scale, 
        scale_b=scale, 
        out_dtype=torch.float32,
    ).float()
    return ref


if __name__ == "__main__":

    a = torch.rand(size=(8, 64), device="cuda")
    b = torch.rand(size=(16, 64), device="cuda").T
    
    d = torch.ones(size=(16, 16), dtype=torch.float32, device="cuda") * 0.01
    _sparse_k(
        a.to(torch.float8_e4m3fnuz), 
        b.to(torch.float8_e4m3fnuz), 
        d,
    )
    d = d.float()
    print(d[0, :8].tolist())

    ref = reference_sparse_8r2x8x64_matmul(a, b)

    print("REF:\n", ref)
    # print()
    print("OUR:\n", d)
    print((ref - d).abs().sum())
