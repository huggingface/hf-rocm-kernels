import torch

from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":
    bench = Bench() # to enter benchmarking mode
    device = "cuda"

    M = 16
    N = 13312
    K = 16384

    SPLIT_K = 3
    B_LANES = 3

    G_SIZE = 8
    WARMUPS = 128
    ITERS = 128

    reference_t = benchmark_skinny_gemm(M, N, K, 0, 0, G_SIZE, WARMUPS, ITERS)
    print(f"Reference: {reference_t:.3f} ns")

    our_t = benchmark_skinny_gemm(M, N, K, SPLIT_K, B_LANES, G_SIZE, WARMUPS, ITERS)
    print(f"Ours:      {our_t:.3f} ns")

    print(f"Speedup:   {(100 * reference_t / our_t):.2f} %")
