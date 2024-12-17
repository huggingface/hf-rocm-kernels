from tqdm import tqdm

from hf_rocm_kernels.operators.residual_rms import residual_rms, generate_residual_rms_data, reference_residual_rms
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":
    bench = Bench()

    list_rows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]
    cols = 6656 # to imitate Llama3.1 405B in TP8

    for rows in tqdm(list_rows, "Gathering reference times"): #TODO: change tag
        args = generate_residual_rms_data(rows, cols)
        bench.add_measure(
            header="Ref (μs)", 
            label=rows, 
            fn=lambda: reference_residual_rms(*args),
        )
        for mode in [0, 1, 2, 3, 4]:
            scale = args[-1].item()
            bench.add_measure(
                header=f"Mode {mode} (μs)", 
                label=rows, 
                fn=lambda: residual_rms(*args[:-1], scale, mode=mode),
            )

    bench.display_table(row_header="Nb. rows")
