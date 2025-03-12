from tqdm import tqdm

from hf_rocm_kernels.operators.swiglu import swiglu, generate_swiglu_data, reference_swiglu
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":

    bench = Bench()
    rows=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048]
    hidden_dim=6656  # to imitate Llama3.1 405B in TP8
    buffer_cols=0

    for row in tqdm(rows, "Gathering measures"):
        gate_up_proj, scale_tensor, next_buffer = generate_swiglu_data(row, hidden_dim, buffer_cols, seed=0)
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
    
    print("-" * 40, f"hidden_dim = {hidden_dim} AND buffer_cols = {buffer_cols}", "-" * 40)
    bench.display_table(row_header="Nb. rows")


# --------- hidden_dim = 6656 AND buffer_cols = 16384 ---------
#   Nb. rows    Torch (μs)    VLLM (μs)    Ours (μs)    Speedup
# ----------  ------------  -----------  -----------  ---------
#          1       41.3421      3.73786      1.78521   2.0938
#          2       32.1678      3.83803      1.87497   2.04699
#          4       36.3197      3.92113      1.89572   2.06841
#          8       25.0688      3.92996      1.93089   2.0353
#         16       24.4134      3.98036      2.01746   1.97296
#         32       26.8584      4.03421      2.22225   1.81537
#         64       33.6491      4.11144      2.92414   1.40603
#        128       44.9737      4.22152      4.42907   0.953139
#        256       58.1091      4.75704      7.42158   0.640974
#       1024      137.45       13.8092      25.6809    0.537724
#       2048      272.996      25.9755      46.2979    0.561051
