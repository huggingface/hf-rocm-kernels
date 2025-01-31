from tqdm import tqdm
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from hf_rocm_kernels.operators.skinny_gemm import skinny_gemm, generate_skinny_gemm_data, reference_skinny_gemm
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":
    bench = Bench()

    M = 8
    NS = list(range(1024, 17000, 1024))
    K = 16384
    B_LANES = 3

    # Ours
    for split_k in [0, 3]:

        ts = []
        for n in tqdm(NS):

            skinny_a, b, scale_a, output = generate_skinny_gemm_data(M, n, K)
            scale_b = torch.ones_like(scale_a)

            # TODO: cg=False ?
            if split_k:
                ts.append(bench.benchmark_fn(
                    fn=lambda: skinny_gemm(skinny_a, b, scale_a, output, split_k, B_LANES), cg=False)
                )
                label = f"{split_k = }"
            else:
                reference_skinny_gemm(skinny_a, b, scale_a, scale_b, output)
                ts.append(bench.benchmark_fn(
                    fn=lambda: reference_skinny_gemm(skinny_a, b, scale_a, scale_b, output), cg=False)
                )
                label = "Torch"

        plt.plot(NS, ts, label=label)

    plt.legend()
    plt.xlabel("n")
    plt.ylabel("Latency (ns)")
    plt.title(f"Kernel latency for ({M}, {K}) * ({K}, n) matmul (init assumed)")
    plt.savefig("bench.png")

