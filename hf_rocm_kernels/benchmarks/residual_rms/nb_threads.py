from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from typing import List
from hf_rocm_kernels.operators.residual_rms import residual_rms, generate_residual_rms_data
from hf_rocm_kernels.utils.benchmarking import Bench

try:
    from hf_rocm_kernels.benchmarks.residual_rms.vs_vllm import get_vllm_time
except ImportError as e:
    print("WARNING: when trying to import get_vllm_time, got an ImportError. VLLM time will be set to 0.\n", e)
    def get_vllm_time(*args, **kwargs): return 0


# Parameters
DTYPE = torch.float8_e4m3fnuz


def draw_nb_threads_plot(list_rows: List[int], cols: int, buffer_cols: int, figure_name: str) -> None:
    bench = Bench()
    cmap = cm.get_cmap("tab10")
    # Prepare the plot
    plt.xlabel("Number of threads")
    plt.ylabel("Latency (μs)")
    plt.title(f"RMS latency for {cols=} and {buffer_cols=} (dotted is VLLM)")

    for i, rows in enumerate(list_rows):
        # Gather measures
        ns, ts = [], []
        for nthreads in tqdm([64 * i for i in range(1, 17)]):
            args = generate_residual_rms_data(rows, cols, buffer_cols, DTYPE)
            t = bench.benchmark_fn(fn=lambda: residual_rms(*args, num_threads=nthreads))
            # t = benchmark_cuda_graph_no_cache(residual_rms, args, {"num_threads": nthreads})
            ns.append(nthreads)
            ts.append(t)
        # Find the minimum time
        min_t = min(ts)
        min_n = [n for i, n in enumerate(ns) if ts[i] == min_t][0]
        # Plot the curve
        plt.plot(ns, ts, label=f"{rows=} w/ min=({min_n}, {min_t:.2f})", color=cmap(i % 10))
        # Add VLLM's time if available
        vllm_time = get_vllm_time(bench, rows, cols, buffer_cols, DTYPE)
        if vllm_time > 0:
            plt.axhline(vllm_time, linestyle="--", color=cmap(i % 10))
        plt.legend()
        plt.savefig(figure_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", "-r", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64]) # , 128, 256, 1024])
    parser.add_argument("--cols", "-c", type=int, default=16384)
    parser.add_argument("--buffer-cols", "-b", type=int, default=0)
    parser.add_argument("--rows-multiplier", "-m", type=int, default=1)
    parser.add_argument("--figure-name", type=str, default="__bench__.png")
    args = parser.parse_args()

    rows = [r * args.rows_multiplier for r in args.rows]
    cols = args.cols
    buffer_cols = args.buffer_cols
    figure_name = args.figure_name

    draw_nb_threads_plot(rows, cols, buffer_cols, figure_name)
