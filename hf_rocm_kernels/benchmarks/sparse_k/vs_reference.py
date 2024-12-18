from tqdm import tqdm
import torch
from torch import Tensor
import vllm._C

from hf_rocm_kernels.operators.sparse_k import sparse_k # TODO:, generate_sparse_k_data, reference_sparse_k
from hf_rocm_kernels.utils.benchmarking import Bench


def reference_gemm(a: Tensor, b: Tensor, scale: Tensor) -> Tensor:
    return torch._scaled_mm(    
        a, 
        b, 
        scale_a=scale, 
        scale_b=scale, 
        out_dtype=torch.float16,
        use_fast_accum=False,
    )


if __name__ == "__main__":
    bench = Bench()

    ms = [8]
    n = 13312 # = 2 * (53248 // 8) # to imitate Llama3.1 405B in TP8
    k = 16384


    for m in ms:
        print(f"# {m = } --------------------------------------------------")

        a = torch.rand(size=(m, k), device="cuda").to(torch.float8_e4m3fnuz)
        b = torch.rand(size=(n, k), device="cuda").to(torch.float8_e4m3fnuz).T
        scale = torch.tensor(1, dtype=torch.float32, device="cuda")
        
        t = bench.benchmark_fn(fn=lambda: reference_gemm(a, b, scale))
        print(f"Reference time: {t}")

        for warps_per_block in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            a = torch.rand(size=(m, k), device="cuda").to(torch.float8_e4m3fnuz)
            b = torch.rand(size=(n, k), device="cuda").to(torch.float8_e4m3fnuz).T

            t = bench.benchmark_fn(fn=lambda: sparse_k(a, b, warps_per_block))
            print(f"{warps_per_block = }: {t}")
