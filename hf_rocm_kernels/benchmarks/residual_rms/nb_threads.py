from tqdm import tqdm
import matplotlib.pyplot as plt

from hf_rocm_kernels.operators.residual_rms import residual_rms, generate_residual_rms_data
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":
    bench = Bench()

    # Parameters
    list_rows = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    cols = 16384
    mode = 1

    # Loop over the number of rows
    for rows in list_rows:

        ns, ts = [], []
        for nthreads in tqdm([64 + i for i in range(0, 1024, 64)]):
            args = generate_residual_rms_data(rows, cols)
            t = bench.benchmark_fn(fn=lambda: residual_rms(*args[:-1], args[-1].item(), mode, nthreads))
            ns.append(nthreads)
            ts.append(t)
        
        plt.plot(ns, ts, label=f"{rows = }")
        plt.legend()
        plt.savefig("__bench__.png")
