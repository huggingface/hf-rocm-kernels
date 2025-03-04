from tqdm import tqdm

from hf_rocm_kernels.operators.swiglu.benchmarking_fn import benchmark_swiglu
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":
    bench = Bench()

    list_rows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048]
    hidden_dim = 6656
    buffer_cols = 0

    for rows in tqdm(list_rows, "Gathering measures"):

        kwargs = {
            "batch_size": rows,
            "intermediate_size": hidden_dim,
            "graph_size": 16,
            "warmups": 100,
            "iterations": 500,
        }
        
        bench.add_raw_measure(
            header="Ref (μs)", 
            label=rows, 
            measure=benchmark_swiglu(**kwargs, num_threads=0),
        )
        for force_scalar in [True, False]:
            bench.add_raw_measure(
                header="Scalar (μs)" if force_scalar else "Vectorized (μs)", 
                label=rows, 
                measure=benchmark_swiglu(**kwargs, force_scalar=force_scalar, num_threads=-1),
            )

    bench.display_table(row_header="Nb. rows")


#   Nb. rows    Ref (μs)    Mode 0 (μs)    Mode 1 (μs)    Mode 2 (μs) 
# ----------  ----------  -------------  -------------  -------------
#          1     39.0035        4.30232        5.09437        2.26692
#          2     29.7115        4.42433        5.16787        2.33263
#          4     34.785         4.48482        5.27177        2.34975
#          8     22.9076        4.54016        5.29588        2.36881
#         16     23.2147        4.53861        5.2784         2.32557
#         32     25.7494        4.60842        5.32699        2.34867
#         64     34.0951        4.96762        5.20797        3.09333
#        128     40.2988        5.35101       18.7006         3.57764
#        256     55.5152        5.59839        8.08354        4.46844
#       1024    136.599        18.6276        16.7173        13.6792
#       2048    243.659        32.5537        33.1646        24.5257
