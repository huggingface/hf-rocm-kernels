from tqdm import tqdm

from hf_rocm_kernels.operators.swiglu import swiglu, generate_swiglu_data, reference_swiglu
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":
    bench = Bench()

    list_rows = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    hidden_dim = 6656
    buffer_cols = 0

    for rows in tqdm(list_rows, "Gathering measures"):
        args = generate_swiglu_data(rows, hidden_dim, buffer_cols)
        bench.add_measure(
            header="Ref (μs)", 
            label=rows, 
            fn=lambda: reference_swiglu(*args),
        )
        for mode in [0, 1]:
            bench.add_measure(
                header=f"Mode {mode} (μs)", 
                label=rows, 
                fn=lambda: swiglu(*args, mode=mode),
            )

    bench.display_table(row_header="Nb. rows")


#  Nb. rows    Ref (μs)    Mode 0 (μs)    Mode 1 (μs)
# ----------  ----------  -------------  -------------
#          1     28.3177        3.96984        3.59183
#          2     21.1967        4.0992         3.65824
#          4     21.4435        4.09017        3.71209
#          8     22.106         4.14422        3.69264
#         16     22.4885        4.21684        3.71242
#         32     25.3838        4.27407        3.73407
#         64     31.1103        4.6278         3.81875
#        128     40.8495        4.79683        3.91772
#        256     53.0349        4.73076        4.19682

