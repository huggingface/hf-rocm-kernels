import argparse
from tqdm import tqdm
from typing import List
from torch import Tensor

from hf_rocm_kernels.operators.swiglu import swiglu, generate_swiglu_data, reference_swiglu
from hf_rocm_kernels.utils.benchmarking import Bench

try:
    from vllm.model_executor.layers.activation import SiluAndMul
    SILU_AND_MUL_MODULE = SiluAndMul()
except BaseException as e:
    if isinstance(e, ImportError):
        raise ModuleNotFoundError("It seems you don't have VLLM installed. Get it from: https://github.com/rocm/vllm")
    else: 
        raise NotImplementedError(
            "It seems you don't have the right version of VLLM installed. Get it from https://github.com/rocm/vllm"
        )


def vllm_silu_and_mul(gate_up_proj: Tensor, scale_tensor: Tensor) -> Tensor:
    out = SILU_AND_MUL_MODULE(gate_up_proj, scale_tensor)
    return out


def get_vllm_time(bench: Bench, rows: int, hidden_dim: int) -> float:
    gate_up_proj, scale_tensor, _ = generate_swiglu_data(rows, hidden_dim, buffer_cols=0, seed=0)
    return bench.benchmark_fn(fn=lambda: vllm_silu_and_mul(gate_up_proj, scale_tensor))


def run_benchmark(rows: List[int], hidden_dim: int, buffer_cols: int) -> None:
    bench = Bench()
    for row in tqdm(rows, "Gathering measures"):
        gate_up_proj, scale_tensor, next_buffer = generate_swiglu_data(row, hidden_dim, buffer_cols, seed=0)
        bench.add_measure(
            header="Torch (μs)", label=row, fn=lambda: reference_swiglu(gate_up_proj, scale_tensor, next_buffer)
        )
        bench.add_raw_measure(header="VLLM (μs)", label=row, measure=get_vllm_time(bench, row, hidden_dim))
        bench.add_measure(header="Ours (μs)", label=row, fn=lambda: swiglu(gate_up_proj, scale_tensor, next_buffer))
    print("-" * 40, f"hidden_dim = {hidden_dim} AND buffer_cols = {buffer_cols}", "-" * 40)
    bench.add_speedup_column(ref_header="VLLM (μs)", our_header="Ours (μs)")
    bench.display_table(row_header="Nb. rows")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", "-r", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048])
    parser.add_argument("--buffer-cols", "-b", type=int, default=0)
    parser.add_argument("--multiplier", "-m", type=int, default=1)
    args = parser.parse_args()

    rows = [r * args.multiplier for r in args.rows]

    # Only benchmark fp8 version
    run_benchmark(
        rows=rows,
        hidden_dim=6656,  # to imitate Llama3.1 405B in TP8
        buffer_cols=args.buffer_cols,
    )
