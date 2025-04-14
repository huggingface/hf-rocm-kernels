import torch
import matplotlib.pyplot as plt
import json
from math import ceil
import itertools
import random
from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm
from hf_rocm_kernels.utils.benchmarking import Bench


CONFIGS = [
    (1, 1, 1, 8, 4),
    # (1, 1, 1, 8, 8),
    (1, 1, 1, 16, 4),
    (1, 1, 1, 16, 8),
    (1, 1, 1, 32, 4),
    (1, 1, 1, 32, 8),
    (1, 1, 2, 8, 4),
    # (1, 1, 2, 8, 8),
    (1, 1, 2, 16, 4),
    (1, 1, 2, 16, 8),
    (1, 1, 2, 32, 4),
    (1, 1, 2, 32, 8),
    (1, 1, 3, 8, 4),
    # (1, 1, 3, 8, 8),
    (1, 1, 3, 16, 4),
    (1, 1, 3, 16, 8),
    (1, 1, 4, 8, 4),
    # (1, 1, 4, 8, 8),
    (1, 1, 4, 16, 4),
    (1, 1, 4, 16, 8),
    (1, 1, 5, 8, 4),
    # (1, 1, 5, 8, 8),
    (1, 1, 5, 16, 4),
    (1, 1, 5, 16, 8),
    (1, 1, 6, 8, 4),
    (1, 1, 6, 16, 4),
    (1, 1, 6, 16, 8),
    (1, 2, 1, 8, 4),
    # (1, 2, 1, 8, 8),
    (1, 2, 1, 16, 4),
    (1, 2, 1, 16, 8),
    (1, 2, 2, 8, 4),
    # (1, 2, 2, 8, 8),
    (1, 2, 2, 16, 4),
    (1, 2, 2, 16, 8),
    (1, 2, 3, 8, 4),
    # (1, 2, 3, 8, 8),
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
    # (1, 3, 1, 8, 8),
    (1, 3, 1, 16, 4),
    (1, 3, 1, 16, 8),
    (1, 3, 2, 8, 4),
    # (1, 3, 2, 8, 8),
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
    # (1, 4, 1, 8, 8),
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
    # (1, 5, 1, 8, 8),
    (1, 5, 1, 16, 4),
    (1, 5, 1, 16, 8),
    (1, 5, 2, 8, 4),
    (1, 5, 2, 16, 4),
    (1, 5, 2, 16, 8),
    (1, 5, 3, 16, 4),
    (1, 5, 4, 16, 4),
    (1, 5, 5, 16, 4),
    (1, 6, 1, 8, 4),
    # (1, 6, 1, 8, 8),
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
    (2, 5, 2, 16, 8),# start here
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


def benchmark_t_reference(m: int, n: int, k: int, graph_size: int, warmups: int, iterations: int) -> float:
    tnop, _ = benchmark_skinny_gemm(
        m=m, n=n, k=k,
        skg_kwargs=None,
        graph_size=graph_size, warmups=warmups, iterations=iterations,
        device="cuda:2"
    )
    print(f"Reference without TunableOps: {tnop:.3f} ns")
    Bench() # to enter benchmarking mode
    top, _ = benchmark_skinny_gemm(
        m=m, n=n, k=k,
        skg_kwargs=None,
        graph_size=graph_size, warmups=warmups, iterations=iterations,
        device="cuda:2"
    )
    print(f"Reference with TunableOps: {top:.3f} ns")
    return min(tnop, top)


if __name__ == "__main__":

    M = 128
    N = 2304
    K = 16384

    G_SIZE = 8
    WARMUPS = 300
    ITERS = 700

    t_reference = benchmark_t_reference(m=M, n=N, k=K, graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS)

    results = {}

    for config in CONFIGS:
        a_lanes, b_lanes, qsize, op_m, ops = config
        if not all([
            (a_lanes > 1),
            (op_m > 8),
            # (qsize >=2),
        ]):
            continue

        # max_a_producers = min(qsize * a_lanes, 4)
        # list_a_producers = list(range(2, max_a_producers + 1))
        # max_consumers = min(qsize, 4)
        # list_consumers = list(range(2, max_consumers + 1))
        # max_b_producers = min(qsize * b_lanes, 8)
        # list_b_producers = list(range(2, max_b_producers + 1))
        list_a_producers = [i for i in (2, 3, 4) if i <= qsize * a_lanes]
        list_b_producers = [i for i in (2, 3, 4, 5, 6, 7, 8) if i <= qsize * b_lanes]
        list_consumers =   [i for i in (2, 3, 4) if i <= qsize]

        abcs = list(itertools.product(list_a_producers, list_b_producers, list_consumers))
        # random.shuffle(abcs)
        # abcs = abcs[:4]

        for a_producers, b_producers, consumers in abcs:
            if sum([a_producers, b_producers, consumers]) > 16:
                continue

            work = ceil(N / (16 * b_lanes))
            mul_cus = [1, 2]
            split_ks = [1, 3] + [(304 * mul_cus) // work for mul_cus in mul_cus]
            for split_k in set(split_ks):
                if split_k < 1:
                    continue

                skg_kwargs = {
                    "split_k": split_k,
                    "A_producers": a_producers, "B_producers": b_producers, "consumers": consumers,
                    "a_lanes": a_lanes, "b_lanes": b_lanes, "qsize": qsize,
                    "op_m": op_m, "ops": ops,
                }
                key = f"{a_producers},{b_producers},{consumers} {split_k} {a_lanes},{b_lanes},{qsize},{op_m},{ops}"
                print(f"{key}", end="\r")

                our_t, std = benchmark_skinny_gemm(
                    m=M, n=N, k=K,
                    skg_kwargs=skg_kwargs,
                    graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS,
                    device="cuda:2"
                )
                results[key] = our_t
                print(f"{key}: {our_t:.3f} ± {std:.3f} -> {100 * t_reference / our_t:.2f}%")

                json.dump(results, open("results__.json", "w"), indent=4)
