import argparse
from tqdm import tqdm
import torch
from typing import Optional, List
from torch import Tensor

from hf_rocm_kernels.operators.residual_rms import residual_rms, generate_residual_rms_data, reference_residual_rms
from hf_rocm_kernels.utils.benchmarking import Bench

try:
    import vllm._custom_ops as ops
    _ = repr(ops.fused_add_rms_norm)
    _ = repr(ops.scaled_fused_add_rms_norm)
except BaseException as e:
    if isinstance(e, ImportError):
        raise ModuleNotFoundError("It seems you don't have VLLM installed. Get it from: https://github.com/rocm/vllm")
    else: 
        raise NotImplementedError(
            "It seems you  don't have the right version of VLLM installed. Get it from https://github.com/rocm/vllm"
        )


def vllm_residual_rms(
    input: Tensor, 
    residual: Tensor, 
    weights: Tensor, 
    epsilon: float, 
    scale_tensor: Optional[Tensor],
) -> Tensor:
    # Case: fp16
    if scale_tensor is None:
        ops.fused_add_rms_norm(input, residual, weights, epsilon)
        out = input
    # Case: fp8
    else:
        out = torch.empty_like(input, dtype=torch.float8_e4m3fnuz)
        ops.scaled_fused_add_rms_norm(out, input, residual, weights, scale_tensor, epsilon)
    return out


def get_vllm_time(bench: Bench, rows: int, cols: int, buffer_cols: int, dtype: torch.dtype) -> float:
    input, residual, weights, epsilon, scale_tensor, next_buffer = generate_residual_rms_data(rows, cols, buffer_cols, dtype)
    return bench.benchmark_fn(fn=lambda: vllm_residual_rms(input, residual, weights, epsilon, scale_tensor))


def run_benchmark(rows: List[int], cols: int, buffer_cols: int, dtype: torch.dtype) -> None:
    bench = Bench()
    for rows in tqdm(rows, "Gathering measures"):
        input, residual, weights, epsilon, scale_tensor, next_buffer = generate_residual_rms_data(rows, cols, buffer_cols, dtype)
        bench.add_measure(
            header="Torch (μs)", 
            label=rows,     
            fn=lambda: reference_residual_rms(input, residual, weights, epsilon, scale_tensor, None),
        )
        bench.add_raw_measure(header="VLLM (μs)", label=rows, measure=get_vllm_time(bench, rows, cols, buffer_cols, dtype))
        bench.add_measure(
            header="Ours (μs)", 
            label=rows,     
            fn=lambda: residual_rms(input, residual, weights, epsilon, scale_tensor, next_buffer),
        )
    bench.add_speedup_column(ref_header="VLLM (μs)", our_header="Ours (μs)")
    print("-" * 40, f"{dtype = } AND {buffer_cols = }", "-" * 40)
    bench.display_table(row_header="Nb. rows")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", "-r", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048])
    parser.add_argument("--buffer-cols", "-b", type=int, default=13312)
    args = parser.parse_args()

    run_benchmark(
        rows=args.rows,
        cols=16384, # to imitate Llama3.1 405B in TP8,
        buffer_cols=args.buffer_cols,
        dtype=torch.float8_e4m3fnuz
    )

    run_benchmark(
        rows=args.rows,
        cols=16384, # to imitate Llama3.1 405B in TP8,
        buffer_cols=0, # no buffer in fp16 yet
        dtype=torch.float16
    )
