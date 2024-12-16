from tqdm import tqdm
import torch
from torch import Tensor
import matplotlib.pyplot as plt

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

    m = 8
    max_n = 6656 * 2
    k = 16384

    ns = []
    ref_ts = []
    ts = []

    for n in tqdm(range(32, max_n, 128)):

        a = torch.ones(size=(m, k), device="cuda").to(torch.float8_e4m3fnuz)
        b = torch.ones(size=(n, k), device="cuda").to(torch.float8_e4m3fnuz).T
        scale = torch.tensor(1, dtype=torch.float32, device="cuda")
        
        ns.append(n)
        ref_ts.append(bench.benchmark_fn(fn=lambda: reference_gemm(a, b, scale)))
        ts.append(bench.benchmark_fn(fn=lambda: sparse_k(a, b, 2)))

    plt.plot(ns, ref_ts)
    plt.plot(ns, ts)
    plt.savefig("bench.png")
