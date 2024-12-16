from tqdm import tqdm
import torch
from torch import Tensor

from hf_rocm_kernels.operators.sparse_k import sparse_k # TODO:, generate_sparse_k_data, reference_sparse_k
from hf_rocm_kernels.utils.benchmarking import Bench


def reference_gemm(a: Tensor, b: Tensor, scale: Tensor) -> Tensor:
    return torch._scaled_mm(    
        a, 
        b, 
        scale_a=scale, 
        scale_b=scale, 
        out_dtype=torch.float32,
    )


if __name__ == "__main__":
    bench = Bench()

    ms = [8, 16, 24]
    n = 6656 # to imitate Llama3.1 405B in TP8
    k = 16384


    for m in ms:
        print(f"# {m = } --------------------------------------------------")

        a = torch.rand(size=(m, k), device="cuda").to(torch.float8_e4m3fnuz)
        b = torch.rand(size=(n, k), device="cuda").to(torch.float8_e4m3fnuz).T
        scale = torch.tensor(1, dtype=torch.float32, device="cuda")
        
        t = bench.benchmark_fn(fn=lambda: reference_gemm(a, b, scale))
        print(f"Reference time: {t}")

        for warps_per_block in range(16):
            warps_per_block += 1
            a = torch.rand(size=(m, k), device="cuda").to(torch.float8_e4m3fnuz)
            b = torch.rand(size=(n, k), device="cuda").to(torch.float8_e4m3fnuz).T

            t = bench.benchmark_fn(fn=lambda: sparse_k(a, b, warps_per_block))
            print(f"{warps_per_block = }: {t}")
