
import os
import matplotlib.pyplot as plt
import argparse
import torch
from typing import List
from tqdm import tqdm
from hf_rocm_kernels.operators.swiglu import swiglu, generate_swiglu_data
from hf_rocm_kernels.utils.benchmarking import Bench

try:
    from hf_rocm_kernels.benchmarks.swiglu.vs_vllm import get_vllm_time
except ImportError:
    get_vllm_time = lambda *args, **kwargs: 0


# Parameters
INTERMEDIATE_SIZE = 6656  # to imitate Llama3.1 405B in TP8
BUFFER_COLS = 16384
DTYPE = torch.float8_e4m3fnuz


def draw_nb_threads_plot(list_rows: List[int]) -> None:
    bench = Bench()
    for rows in list_rows:
        # Gather measures
        ns, ts = [], []
        for nthreads in tqdm([64 * i for i in range(1, 17)]):
            gate_up_proj, scale_tensor, next_buffer = generate_swiglu_data(rows, INTERMEDIATE_SIZE, BUFFER_COLS)
            t = bench.benchmark_fn(fn=lambda: swiglu(gate_up_proj, scale_tensor, next_buffer, num_threads=nthreads))
            ns.append(nthreads)
            ts.append(t)
        # Find the minimum time
        min_t = min(ts)
        min_n = [n for i, n in enumerate(ns) if ts[i] == min_t][0]
        # Plot the curve
        plt.plot(ns, ts, label=f"{rows=} w/ min=({min_n}, {min_t:.2f})")
        # Add VLLM's time if available
        vllm_time = get_vllm_time(bench, rows, INTERMEDIATE_SIZE)
        if vllm_time > 0:
            plt.axhline(vllm_time, color="red", linestyle="--", label=f"VLLM w/ {rows}")
        plt.legend()
        plt.savefig("__bench__.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark swiglu operation with different thread counts")
    parser.add_argument( "--rows", "-r", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048])
    args = parser.parse_args()
    
    draw_nb_threads_plot(args.rows)

