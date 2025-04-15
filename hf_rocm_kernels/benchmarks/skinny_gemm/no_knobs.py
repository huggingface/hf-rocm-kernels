"""
This script benchmarks the performance of the Skinny GEMM operator when called through its wrapper, so without setting
any hyperparameter.
"""

import argparse
import itertools
import tabulate

from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm
from hf_rocm_kernels.benchmarks.skinny_gemm.all_knobs import get_reference_time


# Benchmarking parameters
G_SIZE = 10
WARMUPS = 500
ITERS = 2000
DEVICE_NB = 2

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", "-m", type=int, nargs="+", default=[1])
    parser.add_argument("--n", "-n", type=int, nargs="+", default=[16384])
    parser.add_argument("--k", "-k", type=int, nargs="+", default=[6656])
    args = parser.parse_args()

    for m, n, k in itertools.product(args.m, args.n, args.k):

        # Get reference time
        t_reference, std_reference, with_tunable = get_reference_time(
            m=m, n=n, k=k,
            graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS,
            device=f"cuda:{DEVICE_NB}"
        )

        # Get time with SkG
        our_t, std = benchmark_skinny_gemm(
            m=m, n=n, k=k,
            skg_kwargs={}, # this makes the benchmark use the wrapped version of the operator
            graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS,
            device=f"cuda:{DEVICE_NB}",
            mode="median"
        )

        print(f" {m = }, {n = }, {k = }")
        print(tabulate.tabulate([
            ["Reference", "Skinny GEMM", "Speedup"],
            [f"{t_reference:.3f} ± {std_reference:.3f}", f"{our_t:.3f} ± {std:.3f}", f"{100 * t_reference / our_t:.2f} %"]
        ], tablefmt="rounded_grid"))
        print()
