from tqdm import tqdm
import torch
from torch import Tensor

from hf_rocm_kernels.operators.residual_rms import residual_rms, generate_residual_rms_data, reference_residual_rms
from hf_rocm_kernels.utils.benchmarking import Bench

try:
    import vllm._custom_ops as ops
    _ = repr(ops.scaled_fused_add_rms_norm)
except BaseException as e:
    if isinstance(e, ImportError):
        raise ModuleNotFoundError("It seems you don't have VLLM installed. Get it from: https://github.com/rocm/vllm")
    else:
        raise NotImplementedError(
            "It seems you  don't have the right version of VLLM installed. Get it from https://github.com/rocm/vllm"
        )


def vllm_resdual_rms(input: Tensor, residual: Tensor, weights: Tensor, epsilon: float, scale: Tensor) -> Tensor:
    out = torch.empty_like(input, dtype=torch.float8_e4m3fnuz)
    ops.scaled_fused_add_rms_norm(out, input, residual, weights, scale, epsilon)
    return out


if __name__ == "__main__":
    bench = Bench()

    list_rows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048]
    cols = 16384 # to imitate Llama3.1 405B in TP8

    for rows in tqdm(list_rows, "Gathering measures"):
        args = generate_residual_rms_data(rows, cols)
        bench.add_measure(
            header="Torch (μs)", 
            label=rows, 
            fn=lambda: reference_residual_rms(*args),
        )
        bench.add_measure(
            header="VLLM (μs)", 
            label=rows, 
            fn=lambda: vllm_resdual_rms(*args),
        )
        bench.add_measure(
            header="Ours (μs)", 
            label=rows, 
            fn=lambda: residual_rms(*args),
        )

    bench.display_table(row_header="Nb. rows")
