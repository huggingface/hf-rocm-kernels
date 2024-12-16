import torch.random
import torch.random
from hf_rocm_kernels.operators.sparse_k.binding import _sparse_k
from hf_rocm_kernels.operators.sparse_k.wrapped import sparse_k
import torch
from torch import Tensor
import matplotlib.pyplot as plt



def reference_gemm(a: Tensor, b: Tensor) -> Tensor:
    scale = torch.tensor(1, dtype=torch.float32, device="cuda")
    return torch._scaled_mm(
        a.to(torch.float8_e4m3fnuz), 
        b.to(torch.float8_e4m3fnuz), 
        scale_a=scale, 
        scale_b=scale, 
        out_dtype=torch.float32,
    )

if __name__ == "__main__":

    m = 16
    n = 6656
    k = 16384

    a = torch.rand(size=(m, k), device="cuda")
    b = torch.rand(size=(n, k), device="cuda").T
    
    d = sparse_k(a.to(torch.float8_e4m3fnuz), b.to(torch.float8_e4m3fnuz), 2)
    ref = reference_gemm(a, b)

    print("REF:\n", ref)
    # print()
    print("OUR:\n", d)
    print("Max error:", (ref - d).abs().max().item())

    fig, axs = plt.subplots(1, 2)
    axs[0].matshow(ref.numpy(force=True))
    axs[0].set_title("Reference")
    axs[1].matshow(d.numpy(force=True))
    axs[1].set_title("Ours")
    fig.savefig("test.png")
