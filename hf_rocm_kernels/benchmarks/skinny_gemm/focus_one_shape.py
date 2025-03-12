from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":
    bench = Bench() # to enter benchmarking mode
    device = "cuda"


    M = 8
    N = 2304
    K = 16384

    G_SIZE = 8
    WARMUPS = 200
    ITERS = 1000

    reference_t = benchmark_skinny_gemm(M, N, K, 0, 0, G_SIZE, WARMUPS, ITERS)
    best = [1e8, 0, 0]

    for b_lanes in [2, 3, 4, 5]:
        for split_k in range(16):
            split_k += 1

            t = benchmark_skinny_gemm(M, N, K, split_k, b_lanes, G_SIZE, WARMUPS, ITERS)
            speedup = reference_t / t
            print(f"{b_lanes=} {split_k=} | t = {t:.2f} -> speedup = {speedup:.2f}")

            if t < best[0]:
                best = [t, b_lanes, split_k]

    print(f"Best is t={best} for b_lanes={best[1]} and split_k={best[2]}")
