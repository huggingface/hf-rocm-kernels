"""
This script benchmarks the performance of the Skinny GEMM operator when exploring the space of hyperparameters.
"""

import json
from math import ceil
import argparse
import itertools

from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm
from hf_rocm_kernels.benchmarks.skinny_gemm.all_knobs import get_reference_time


# A subset of possible configs for SkG (A_lanes, B_lanes, qsize, op_m, ops)
CONFIGS = [
    (1, 1, 1, 8, 4),
    (1, 1, 1, 16, 4),
    (1, 1, 1, 16, 8),
    (1, 1, 1, 32, 4),
    (1, 1, 1, 32, 8),
    (1, 1, 2, 8, 4),
    (1, 1, 2, 16, 4),
    (1, 1, 2, 16, 8),
    (1, 1, 2, 32, 4),
    (1, 1, 2, 32, 8),
    (1, 1, 3, 8, 4),
    (1, 1, 3, 16, 4),
    (1, 1, 3, 16, 8),
    (1, 1, 4, 8, 4),
    (1, 1, 4, 16, 4),
    (1, 1, 4, 16, 8),
    (1, 1, 5, 8, 4),
    (1, 1, 5, 16, 4),
    (1, 1, 5, 16, 8),
    (1, 1, 6, 8, 4),
    (1, 1, 6, 16, 4),
    (1, 1, 6, 16, 8),
    (1, 2, 1, 8, 4),
    (1, 2, 1, 16, 4),
    (1, 2, 1, 16, 8),
    (1, 2, 2, 8, 4),
    (1, 2, 2, 16, 4),
    (1, 2, 2, 16, 8),
    (1, 2, 3, 8, 4),
    (1, 2, 3, 16, 4),
    (1, 2, 3, 16, 8),
    (1, 2, 4, 8, 4),
    (1, 2, 4, 16, 4),
    (1, 2, 4, 16, 8),
    (1, 2, 5, 8, 4),
    (1, 2, 5, 16, 4),
    (1, 2, 5, 16, 8),
    (1, 2, 6, 8, 4),
    (1, 2, 6, 16, 4),
    (1, 3, 1, 8, 4),
    (1, 3, 1, 16, 4),
    (1, 3, 1, 16, 8),
    (1, 3, 2, 8, 4),
    (1, 3, 2, 16, 4),
    (1, 3, 2, 16, 8),
    (1, 3, 3, 8, 4),
    (1, 3, 3, 16, 4),
    (1, 3, 3, 16, 8),
    (1, 3, 4, 8, 4),
    (1, 3, 4, 16, 4),
    (1, 3, 5, 16, 4),
    (1, 3, 6, 16, 4),
    (1, 4, 1, 8, 4),
    (1, 4, 1, 16, 4),
    (1, 4, 1, 16, 8),
    (1, 4, 2, 8, 4),
    (1, 4, 2, 16, 4),
    (1, 4, 2, 16, 8),
    (1, 4, 3, 8, 4),
    (1, 4, 3, 16, 4),
    (1, 4, 3, 16, 8),
    (1, 4, 4, 16, 4),
    (1, 4, 5, 16, 4),
    (1, 4, 6, 16, 4),
    (1, 5, 1, 8, 4),
    (1, 5, 1, 16, 4),
    (1, 5, 1, 16, 8),
    (1, 5, 2, 8, 4),
    (1, 5, 2, 16, 4),
    (1, 5, 2, 16, 8),
    (1, 5, 3, 16, 4),
    (1, 5, 4, 16, 4),
    (1, 5, 5, 16, 4),
    (1, 6, 1, 8, 4),
    (1, 6, 1, 16, 4),
    (1, 6, 1, 16, 8),
    (1, 6, 2, 8, 4),
    (1, 6, 2, 16, 4),
    (1, 6, 2, 16, 8),
    (1, 6, 3, 16, 4),
    (1, 6, 4, 16, 4),
    (2, 1, 1, 16, 4),
    (2, 1, 1, 16, 8),
    (2, 1, 2, 16, 4),
    (2, 1, 2, 16, 8),
    (2, 1, 3, 16, 4),
    (2, 1, 3, 16, 8),
    (2, 1, 4, 16, 4),
    (2, 1, 4, 16, 8),
    (2, 1, 5, 16, 4),
    (2, 1, 5, 16, 8),
    (2, 1, 6, 16, 4),
    (2, 2, 1, 16, 4),
    (2, 2, 1, 16, 8),
    (2, 2, 2, 16, 4),
    (2, 2, 2, 16, 8),
    (2, 2, 3, 16, 4),
    (2, 2, 3, 16, 8),
    (2, 2, 4, 16, 4),
    (2, 2, 5, 16, 4),
    (2, 2, 6, 16, 4),
    (2, 3, 1, 16, 4),
    (2, 3, 1, 16, 8),
    (2, 3, 2, 16, 4),
    (2, 3, 2, 16, 8),
    (2, 3, 3, 16, 4),
    (2, 3, 3, 16, 8),
    (2, 3, 4, 16, 4),
    (2, 3, 5, 16, 4),
    (2, 3, 6, 16, 4),
    (2, 4, 1, 16, 4),
    (2, 4, 1, 16, 8),
    (2, 4, 2, 16, 4),
    (2, 4, 2, 16, 8),
    (2, 4, 3, 16, 4),
    (2, 4, 4, 16, 4),
    (2, 4, 5, 16, 4),
    (2, 5, 1, 16, 4),
    (2, 5, 1, 16, 8),
    (2, 5, 2, 16, 4),
    (2, 5, 2, 16, 8),
    (2, 5, 3, 16, 4),
    (2, 5, 4, 16, 4),
    (2, 6, 1, 16, 4),
    (2, 6, 1, 16, 8),
    (2, 6, 2, 16, 4),
    (2, 6, 3, 16, 4),
    (3, 1, 1, 16, 4),
    (3, 1, 1, 16, 8),
    (3, 1, 2, 16, 4),
    (3, 1, 2, 16, 8),
    (3, 1, 3, 16, 4),
    (3, 1, 3, 16, 8),
    (3, 1, 4, 16, 4),
    (3, 1, 5, 16, 4),
    (3, 1, 6, 16, 4),
    (3, 2, 1, 16, 4),
    (3, 2, 1, 16, 8),
    (3, 2, 2, 16, 4),
    (3, 2, 2, 16, 8),
    (3, 2, 3, 16, 4),
    (3, 2, 3, 16, 8),
    (3, 2, 4, 16, 4),
    (3, 2, 5, 16, 4),
    (3, 2, 6, 16, 4),
    (3, 3, 1, 16, 4),
    (3, 3, 1, 16, 8),
    (3, 3, 2, 16, 4),
    (3, 3, 2, 16, 8),
    (3, 3, 3, 16, 4),
    (3, 3, 4, 16, 4),
    (3, 3, 5, 16, 4),
    (3, 4, 1, 16, 4),
    (3, 4, 1, 16, 8),
    (3, 4, 2, 16, 4),
    (3, 4, 2, 16, 8),
    (3, 4, 3, 16, 4),
    (3, 4, 4, 16, 4),
    (3, 5, 1, 16, 4),
    (3, 5, 1, 16, 8),
    (3, 5, 2, 16, 4),
    (3, 5, 3, 16, 4),
    (3, 6, 1, 16, 4),
    (3, 6, 1, 16, 8),
    (3, 6, 2, 16, 4),
    (3, 6, 3, 16, 4),
    (4, 1, 1, 16, 4),
    (4, 1, 1, 16, 8),
    (4, 1, 2, 16, 4),
    (4, 1, 2, 16, 8),
    (4, 1, 3, 16, 4),
    (4, 1, 3, 16, 8),
    (4, 1, 4, 16, 4),
    (4, 1, 5, 16, 4),
    (4, 1, 6, 16, 4),
    (4, 2, 1, 16, 4),
    (4, 2, 1, 16, 8),
    (4, 2, 2, 16, 4),
    (4, 2, 2, 16, 8),
    (4, 2, 3, 16, 4),
    (4, 2, 4, 16, 4),
    (4, 2, 5, 16, 4),
    (4, 3, 1, 16, 4),
    (4, 3, 1, 16, 8),
    (4, 3, 2, 16, 4),
    (4, 3, 2, 16, 8),
    (4, 3, 3, 16, 4),
    (4, 3, 4, 16, 4),
    (4, 4, 1, 16, 4),
    (4, 4, 1, 16, 8),
    (4, 4, 2, 16, 4),
    (4, 4, 3, 16, 4),
    (4, 5, 1, 16, 4),
    (4, 5, 1, 16, 8),
    (4, 5, 2, 16, 4),
    (4, 5, 3, 16, 4),
    (4, 6, 1, 16, 4),
    (4, 6, 1, 16, 8),
    (4, 6, 2, 16, 4),
    (4, 6, 3, 16, 4),
]

# Benchmarking parameters
G_SIZE = 8
WARMUPS = 300
ITERS = 700
DEVICE_NB = 0


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", "-m", type=int, default=1)
    parser.add_argument("--n", "-n", type=int, default=16384)
    parser.add_argument("--k", "-k", type=int, default=6656)
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k
    print(f"{m=}, {n=}, {k=}")


    # Get reference time
    t_reference, _, _ = get_reference_time(
        m=m, n=n, k=k,
        graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS, device=f"cuda:{DEVICE_NB}"
    )


    # Grid search
    results = {}

    for config in CONFIGS:

        # Unpack config and maybe skip some of them
        a_lanes, b_lanes, qsize, op_m, ops = config
        if not all([
            (a_lanes > 1), # (eg. for less than 8 rows)
            (op_m > 8),    # (eg. for less than 8 rows)
            (qsize >=2),   # (eg. to avoid many configs with qsize=1)
        ]):
            continue

        # Prepare list of some possible values for warp specialization
        list_a_producers = [i for i in (2, 3, 4) if i <= qsize * a_lanes]
        list_b_producers = [i for i in (2, 3, 4, 5, 6, 7, 8) if i <= qsize * b_lanes]
        list_consumers =   [i for i in (2, 3, 4) if i <= qsize]

        # Skip triplets with too many warps
        abcs = list(itertools.product(list_a_producers, list_b_producers, list_consumers))
        abcs = list(filter(lambda abc: sum(abc) <= 16, abcs))
        for a_producers, b_producers, consumers in abcs:

            # Prepare list of some possible values for split_k
            work = ceil(n / (16 * b_lanes))
            split_ks = [1] + [304 // work] # here, we can add the split_k values that we want to always try
            split_ks = list(filter(lambda x: x >= 1, set(split_ks)))

            for split_k in split_ks:

                skg_kwargs = {
                    "split_k": split_k,
                    "A_producers": a_producers, "B_producers": b_producers, "consumers": consumers,
                    "a_lanes": a_lanes, "b_lanes": b_lanes, "qsize": qsize,
                    "op_m": op_m, "ops": ops,
                }
                key = f"{a_lanes},{b_lanes},{qsize},{op_m},{ops} {a_producers},{b_producers},{consumers} {split_k}"
                print(f"{key}", end="\r")

                our_t, std = benchmark_skinny_gemm(
                    m=m, n=n, k=k,
                    skg_kwargs=skg_kwargs,
                    graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS,
                    device=f"cuda:{DEVICE_NB}"
                )
                results[key] = our_t

                speedup = 100 * t_reference / our_t
                print(f"{key}: {our_t:.3f} ± {std:.3f} -> {speedup:.2f}%")

            # Save results as we go
            json.dump(results, open(f"SkG_grid_search_results_{m}x{n}x{k}.json", "w"), indent=4)
