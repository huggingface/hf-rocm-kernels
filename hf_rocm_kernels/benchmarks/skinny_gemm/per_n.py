from tqdm import tqdm
import matplotlib.pyplot as plt

from hf_rocm_kernels.utils.benchmarking import Bench
from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm


if __name__ == "__main__":
    bench = Bench() # to enter benchmarking mode

    M = 8
    N_resolution = 256
    N_max = 17000
    K = 16384

    # Ours
    for split_k in [0, 1, 2, 3]:
        for b_lanes in [3, 4, 5]:

            if split_k == 0 and b_lanes != 3:
                continue

            ts = []
            ns = list(range(N_resolution, N_max, N_resolution))
            for n in tqdm(ns):

                t = benchmark_skinny_gemm(M, n, K, split_k, b_lanes)
                ts.append(t)

            label = "Torch" if split_k == 0 else f"Split-k / B_lanes = {(split_k, b_lanes)}"
            plt.plot(ns, ts, label=label)

    plt.legend()
    plt.xlabel("n")
    plt.ylabel("Latency (ns)")
    plt.title(f"Kernel latency for ({M}, {K}) * ({K}, n) matmul (init assumed)")
    plt.savefig("bench.png")

