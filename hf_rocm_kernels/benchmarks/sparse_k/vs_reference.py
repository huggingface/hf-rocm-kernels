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

    ms = [8]
    n = 6656 # to imitate Llama3.1 405B in TP8
    k = 16384


    for m in tqdm(ms):

        a = torch.rand(size=(m, k), device="cuda").to(torch.float8_e4m3fnuz)
        b = torch.rand(size=(n, k), device="cuda").to(torch.float8_e4m3fnuz).T
        scale = torch.tensor(1, dtype=torch.float32, device="cuda")
        
        bench.add_measure(
            header="Ref (μs)", 
            label=m, 
            fn=lambda: reference_gemm(a, b, scale),
        )
        bench.add_measure(
            header=f"Sparse K (μs)", 
            label=m, 
            fn=lambda: sparse_k(a, b),
        )

    bench.display_table(row_header="M")
