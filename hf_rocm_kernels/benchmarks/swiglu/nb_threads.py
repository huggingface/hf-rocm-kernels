
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from typing import List
from tqdm import tqdm
from hf_rocm_kernels.operators.swiglu import swiglu, generate_swiglu_data
from hf_rocm_kernels.utils.benchmarking import Bench

try:
    from hf_rocm_kernels.benchmarks.swiglu.vs_vllm import get_vllm_time
except ImportError as e:
    print("WARNING: when trying to import get_vllm_time, got an ImportError. VLLM time will be set to 0.\n", e)
    def get_vllm_time(*args, **kwargs): return 0


def draw_nb_threads_plot(list_rows: List[int], intermediate_size: int, buffer_cols: int, figure_name: str) -> None:
    bench = Bench()
    cmap = cm.get_cmap("tab10")
    # Prepare the plot
    plt.xlabel("Number of threads")
    plt.ylabel("Latency (μs)")
    plt.title(f"SwiGLU latency for i_size={intermediate_size} and {buffer_cols=} (dotted is VLLM)")

    for i, rows in enumerate(list_rows):
        # Gather measures
        ns, ts = [], []
        for nthreads in tqdm([64 * i for i in range(1, 17)]):
            args = generate_swiglu_data(rows, intermediate_size, buffer_cols)
            t = bench.benchmark_fn(lambda: swiglu(*args, num_threads=nthreads))
            ns.append(nthreads)
            ts.append(t)
        # Find the minimum time
        min_t = min(ts)
        min_n = [n for i, n in enumerate(ns) if ts[i] == min_t][0]
        # Plot the curve
        plt.plot(ns, ts, label=f"{rows=} w/ min=({min_n}, {min_t:.2f})", color=cmap(i % 10))
        # Add VLLM's time if available
        vllm_time = get_vllm_time(bench, rows, intermediate_size)
        if vllm_time > 0:
            plt.axhline(vllm_time, color=cmap(i % 10), linestyle="--")
        plt.legend()
        plt.savefig(figure_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", "-r", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256]) #, 1024, 2048])
    parser.add_argument("--intermediate-size", "-i", type=int, default=6656) # to imitate Llama3.1 405B in TP8
    parser.add_argument("--buffer-cols", "-b", type=int, default=0)
    parser.add_argument("--multiplier", "-m", type=int, default=1)
    parser.add_argument("--figure-name", type=str, default="__bench__.png")
    args = parser.parse_args()

    rows = [r * args.multiplier for r in args.rows]
    intermediate_size = args.intermediate_size
    buffer_cols = args.buffer_cols
    figure_name = args.figure_name

    draw_nb_threads_plot(rows, intermediate_size, buffer_cols, figure_name)

