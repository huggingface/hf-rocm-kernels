from tqdm import tqdm
import torch

from hf_rocm_kernels.operators.residual_rms import residual_rms, generate_residual_rms_data, reference_residual_rms
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":
    bench = Bench()

    list_rows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 2048]
    cols = 16384
    buffer_cols = 0
    dtype = torch.float8_e4m3fnuz

    for rows in tqdm(list_rows, "Gathering measures"):
        args = generate_residual_rms_data(rows, cols, buffer_cols, dtype)
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



#   Nb. rows    Ref (μs)    Mode 0 (μs)    Mode 1 (μs)    Mode 2 (μs)    Mode 3 (μs)    Mode 4 (μs)
# ----------  ----------  -------------  -------------  -------------  -------------  -------------
#          1     39.6799        10.4178        5.69333        5.56819        4.79682        4.73507
#          2     42.8523        10.5678        5.60803        5.6303         4.95249        4.82551
#          4     43.4262        10.5658        5.74438        5.72863        4.96111        4.87793
#          8     43.1202        10.6513        5.77915        5.75665        5.00844        4.89718
#         16     46.4581        10.7354        5.85507        5.8458         5.08137        4.97483
#         32     55.4413        10.9484        6.11379        6.09975        5.33902        5.2539
#         64     78.0771        11.646         6.51273        6.54157        5.78304        5.74506
#        128     97.0771        12.6915        7.14067        7.21665        6.52303        6.54999
#        256    122.57          13.6316       11.9258        11.7987        11.0201        11.2677
