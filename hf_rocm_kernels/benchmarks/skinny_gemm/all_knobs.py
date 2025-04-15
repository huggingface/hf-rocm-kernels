"""
This script benchmarks the performance of the Skinny GEMM operator when manually setting each hyperparameter.
"""

import argparse
from typing import Tuple

from hf_rocm_kernels.utils.benchmarking import Bench
from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm


# Parameters
DEVICE_NB = 0

G_SIZE = 12
WARMUPS = 500
ITERS = 2000


# Configuration
A_PRODUCERS = 2
B_PRODUCERS = 7
CONSUMERS = 2

A_LANES = 1
B_LANES = 4
QSIZE = 2

OP_M = 8
OPS = 4


# Functions
def get_reference_time(
    m: int, n: int, k: int,
    graph_size: int, warmups: int, iterations: int,
    device: str = "cuda"
) -> Tuple[float, float, bool]:
    # Get reference time without TunableOps
    Bench.enter_benchmark_mode(enable_tunable=False)
    reference_t1, reference_std1 = benchmark_skinny_gemm(
        m=m, n=n, k=k,
        skg_kwargs=None,
        graph_size=graph_size, warmups=warmups, iterations=iterations, device=device
    )
    # Get time with TunableOps
    Bench.enter_benchmark_mode(enable_tunable=True)
    reference_t2, reference_std2 = benchmark_skinny_gemm(
        m=m, n=n, k=k,
        skg_kwargs=None,
        graph_size=graph_size, warmups=warmups, iterations=iterations, device=device
    )
    # Choose the smallest time
    if reference_t1 < reference_t2:
        return reference_t1, reference_std1, False
    return reference_t2, reference_std2, True

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", "-m", type=int, default=1)
    parser.add_argument("--n", "-n", type=int, default=16384)
    parser.add_argument("--k", "-k", type=int, default=6656)
    parser.add_argument("--split-k", "-sk", nargs="+", type=int, default=[1])
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k
    split_ks = args.split_k
    print(f"{m=}, {n=}, {k=}")

    reference_t, reference_std, with_tunable = get_reference_time(
        m=m, n=n, k=k,
        graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS, device=f"cuda:{DEVICE_NB}"
    )
    with_or_without = "with" if with_tunable else "without"
    print(f"Reference: {reference_t:.3f} ns ± {reference_std:.3f} ({with_or_without} TunableOps)")

    # Get time with Skinny GEMM
    our_t, our_std, our_split_k = 1e20, 0, 0

    for split_k in split_ks:
        skg_kwargs = {
            "split_k": split_k,
            "A_producers": A_PRODUCERS, "B_producers": B_PRODUCERS, "consumers": CONSUMERS,
            "a_lanes": A_LANES, "b_lanes": B_LANES, "qsize": QSIZE,
            "op_m": OP_M, "ops": OPS,
        }
        t, std = benchmark_skinny_gemm(
            m=m, n=n, k=k, skg_kwargs=skg_kwargs,
            graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS,
            device=f"cuda:{DEVICE_NB}"
        )
        if t < our_t:
            our_t, our_std, our_split_k = t, std, split_k

    # Display results
    print(f"Ours:      {our_t:.3f} ns ± {our_std:.3f} (split_k={our_split_k})")
    print(f"Speedup:   {(100 * reference_t / our_t):.2f} %")
    skg_kwargs["split_k"] = our_split_k
    print("\nConfig:    " + " ".join(map(str, skg_kwargs.items())))
