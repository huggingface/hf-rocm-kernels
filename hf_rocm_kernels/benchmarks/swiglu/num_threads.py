from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from hf_rocm_kernels.operators.swiglu import swiglu, generate_swiglu_data
from hf_rocm_kernels.operators.swiglu.wrapped import infer_num_threads
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":
    bench = Bench()

    # Parameters
    list_rows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048]
    hidden_dim = 16384
    buffer_cols = 0
    force_scalar = False

    # Delete old figure 
    if os.path.exists("__bench__.png"):
        os.remove("__bench__.png")

    # Loop over the number of rows
    for rows in list_rows:
        ns, ts = [], []
        for num_threads in tqdm([64 + i for i in range(0, 1024, 64)]):
            args = generate_swiglu_data(rows, hidden_dim, buffer_cols)
            t = bench.benchmark_fn(fn=lambda: swiglu(*args, num_threads=num_threads, force_scalar=force_scalar))
            ns.append(num_threads)
            ts.append(t)
        min_t = min(ts)
        min_n = [n for i, n in enumerate(ns) if ts[i] == min_t][0]
        
        plt.plot(ns, ts, label=f"{rows=} w/ min=({min_n}, {min_t:.2f})")
        plt.legend()
        plt.savefig("__bench__.png")


    # Loop over the number of rows
    for rows in list_rows:
        args = generate_swiglu_data(rows, hidden_dim, buffer_cols)
        t = bench.benchmark_fn(fn=lambda: swiglu(*args, num_threads=-1, force_scalar=force_scalar))
        print(rows, t, infer_num_threads(rows, hidden_dim, force_scalar, -1))
