from tqdm import tqdm
import argparse

from hf_rocm_kernels.operators.swiglu import swiglu, generate_swiglu_data, reference_swiglu
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":

    bench = Bench()

    # Retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", "-r", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 2048])
    parser.add_argument("--intermediate-size", "-i", type=int, default=6656)
    parser.add_argument("--buffer-cols", "-b", type=int, default=0)
    args = parser.parse_args()

    rows = [r for r in args.rows]
    intermediate_size = args.intermediate_size
    buffer_cols = args.buffer_cols

    for row in tqdm(rows, "Gathering measures"):
        gate_up_proj, scale_tensor, next_buffer = generate_swiglu_data(row, intermediate_size, buffer_cols, seed=0)
        bench.add_measure(
            header="Torch (μs)", label=row, fn=lambda: reference_swiglu(gate_up_proj, scale_tensor, next_buffer)
        )
        bench.add_measure(
            header="Scalarized (μs)",
            label=row,
            fn=lambda: swiglu(gate_up_proj, scale_tensor, next_buffer, force_scalar=True)
        )
        bench.add_measure(
            header="Vectorized (μs)",
            label=row,
            fn=lambda: swiglu(gate_up_proj, scale_tensor, next_buffer, force_scalar=False)
        )

    print("-" * 14, f"i_size = {intermediate_size} AND buffer_cols = {buffer_cols}", "-" * 13)
    bench.display_table(row_header="Nb. rows")


# ------------- i_size = 6656 AND buffer_cols = 0 --------------
#   Nb. rows    Torch (μs)    Scalarized (μs)    Vectorized (μs)
# ----------  ------------  -----------------  -----------------
#          1       42.7103            2.37437            2.09134
#          2       32.606             2.45989            2.07332
#          4       37.5817            2.46205            2.12491
#          8       23.889             2.55379            2.15295
#         16       23.9401            2.63905            2.04731
#         32       26.4887            2.76049            2.1212
#         64       33.3452            3.6288             2.27648
#        128       43.3875            4.88383            2.79315
#        256       54.6133            7.98832            3.80765
#       2048      244.129            54.6449            19.9479
