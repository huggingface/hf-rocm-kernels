from tqdm import tqdm
import torch
import argparse

from hf_rocm_kernels.operators.residual_rms import residual_rms, generate_residual_rms_data, reference_residual_rms
from hf_rocm_kernels.utils.benchmarking import Bench


DTYPE = torch.float8_e4m3fnuz


if __name__ == "__main__":

    # Retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", "-r", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 2048])
    parser.add_argument("--cols", "-c", type=int, default=16384)
    parser.add_argument("--buffer-cols", "-b", type=int, default=0)
    args = parser.parse_args()

    rows = [r for r in args.rows]
    cols = args.cols
    buffer_cols = args.buffer_cols


    # Bench RMS
    bench = Bench()

    for rows in tqdm(rows, "Gathering measures"):
        args = generate_residual_rms_data(rows, cols, buffer_cols, DTYPE)
        bench.add_measure(
            header="Ref (μs)",
            label=rows,
            fn=lambda: reference_residual_rms(*args),
        )
        bench.add_measure(
            header="Pointwise (μs)",
            label=rows,
            fn=lambda: residual_rms(*args, force_scalar=True),
        )
        bench.add_measure(
            header="Vectorized (μs)",
            label=rows,
            fn=lambda: residual_rms(*args, force_scalar=False),
        )

    bench.display_table(row_header="Nb. rows")


#   Nb. rows    Ref (μs)    Pointwise (μs)    Vectorized (μs)
# ----------  ----------  ----------------  -----------------
#          1     42.4464           11.2799            4.66131
#          2     46.4053           11.3722            4.62339
#          4     47.881            11.4576            4.6868
#          8     47.928            11.5175            4.74966
#         16     49.0487           11.626             4.81408
#         32     57.6287           13.9424            5.03623
#         64     78.5644           13.0868            5.67018
#        128    101.934            14.1814            6.55099
#        256    122.353            15.2455            9.46788
#       2048    667.768            80.5618           65.8073
