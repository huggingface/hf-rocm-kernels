from tqdm import tqdm
import torch
from torch import Tensor

from hf_rocm_kernels.operators.sparse_k import sparse_k # TODO:, generate_sparse_k_data, reference_sparse_k
from hf_rocm_kernels.utils.benchmarking import Bench


def reference_gemm(a: Tensor, b: Tensor, scale: Tensor, d) -> Tensor:
    return torch._scaled_mm(    
        a, 
        b, 
        scale_a=scale, 
        scale_b=scale, 
        out_dtype=d,
    )


if __name__ == "__main__":
    bench = Bench()

    m = 8
    n = 6656 # to imitate Llama3.1 405B in TP8
    k = 16384

    a0 = torch.rand(size=(m, k), device="cuda").to(torch.float8_e4m3fnuz)
    b0 = torch.rand(size=(n, k), device="cuda").to(torch.float8_e4m3fnuz).T
    scale0 = torch.tensor(1, dtype=torch.float32, device="cuda")
    d0 = reference_gemm(a0, b0, scale0, torch.float16)
    d1 = reference_gemm(a0, b0, scale0, torch.float32)
    print(d0.sub(d1).abs().max().item())

    a0 = torch.rand(size=(m, k), device="cuda").to(torch.float8_e4m3fnuz)
    b0 = torch.rand(size=(n, k), device="cuda").to(torch.float8_e4m3fnuz).T
    scale0 = torch.tensor(1, dtype=torch.float32, device="cuda")
    a1= torch.rand(size=(m, k), device="cuda").to(torch.float8_e4m3fnuz)
    b1= torch.rand(size=(n, k), device="cuda").to(torch.float8_e4m3fnuz).T
    scale1 = torch.tensor(1, dtype=torch.float32, device="cuda")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        reference_gemm(a0, b0, scale0, torch.float16)
        sparse_k(a1, b1, 6)
        reference_gemm(a0, b0, scale0, torch.float16)
        sparse_k(a1, b1, 6)
        reference_gemm(a0, b0, scale0, torch.float16)
        sparse_k(a1, b1, 6)
    p.export_chrome_trace("profi.json")

