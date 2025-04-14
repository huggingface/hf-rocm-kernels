import torch
from typing import Optional, Dict, Any, Tuple
import math
from hf_rocm_kernels.utils.fp8 import fp8_quantize
from hf_rocm_kernels.operators.skinny_gemm.wrapped import skinny_gemm
from hf_rocm_kernels.operators.skinny_gemm.binding import _skinny_gemm

def benchmark_skinny_gemm(
    m: int,
    n: int,
    k: int,
    skg_kwargs: Optional[Dict[str, Any]] = None,
    graph_size: int = 8,
    warmups: int = 32,
    iterations: int = 128,
    device: str = "cuda",
    mode: str = "mean"
) -> Tuple[float, float]:
    # Create input
    input = torch.normal(0, 1, size=(m, k), device=device, dtype=torch.float32)
    input_scale = torch.mean(input)
    input = fp8_quantize(input, input_scale)[0]
    # Create weights
    weights = [torch.normal(0, 1, size=(n, k), device=device, dtype=torch.float32) for _ in range(graph_size)]
    scales = [torch.mean(weight) for weight in weights]
    q_weights = [fp8_quantize(weight, scale)[0].t() for weight, scale in zip(weights, scales)]
    # Prepare output
    output = torch.zeros(size=(m, n), device=device, dtype=torch.float16)

    # If skg_kwargs is None, then we are timing the torch function
    if skg_kwargs is None:
        def fn(i: int) -> None:
            torch._scaled_mm(
                input=input,
                mat2=q_weights[i],
                scale_a=input_scale,
                scale_b=scales[i],
                out_dtype=torch.float16,
                out=output
            )
        # To initialize tunable ops
        fn(0)
    # Otherwise if skg_kwargs is empty, we call the wrapped function
    elif not skg_kwargs:
        def fn(i: int) -> None:
            skinny_gemm(
                skinny_a=input,
                b=q_weights[i],
                scale_tensor=scales[i],
                output=output
            )
    else:
        def fn(i: int) -> None:
            _skinny_gemm(
                A=input,
                B=q_weights[i],
                scale_tensor=scales[i],
                D=output,
                **skg_kwargs
            )

    # Create a side-stream to benchmark in
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):

        # Create graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(graph_size):
                fn(i)
        torch.cuda.synchronize()

        # Benchmark its replays
        ts = []
        for i in range(warmups + iterations):

            # Prepare events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Run
            start_event.record()
            graph.replay()
            end_event.record()

            # Accumulate
            torch.cuda.synchronize()
            if i > warmups:
                ts.append(start_event.elapsed_time(end_event))

    # Post-process time
    ts = [1000 * t / graph_size for t in ts]
    mean = sum(ts) / len(ts)
    median = torch.tensor(ts).median().item()
    std = torch.tensor(ts).std().item()
    if mode == "mean":
        return mean, std
    elif mode == "median":
        return median, std
    else:
        raise ValueError(f"Invalid mode: {mode}")

def benchmark_skinny_gemm_general(
    m: int,
    n: int,
    k: int,
    split_k: int,
    A_producers: int = 0,
    B_producers: int = 0,
    consumers: int = 0,
    a_lanes: int = 0,
    b_lanes: int = 0,
    qsize: int = 0,
    op_m: int = 0,
    ops: int = 0,
    graph_size: int = 8,
    warmups: int = 32,
    iterations: int = 128,
    device: str = "cuda"
) -> Tuple[float, float]:
    # Create input
    input = torch.normal(0, 1, size=(m, k), device=device, dtype=torch.float32)
    input_scale = torch.mean(input)
    input = fp8_quantize(input, input_scale)[0]
    # Create weights
    weights = [torch.normal(0, 1, size=(n, k), device=device, dtype=torch.float32) for _ in range(graph_size)]
    scales = [torch.mean(weight) for weight in weights]
    q_weights = [fp8_quantize(weight, scale)[0].t() for weight, scale in zip(weights, scales)]
    # Prepare output
    output = torch.zeros(size=(m, n), device=device, dtype=torch.float16)

    # If split_k == 0, then we are timing the torch function
    if split_k == 0:
        def fn(i: int) -> None:
            torch._scaled_mm(
                input=input,
                mat2=q_weights[i],
                scale_a=input_scale,
                scale_b=scales[i],
                out_dtype=torch.float16,
                out=output
            )
        # To initialize tunable ops
        fn(0)
    # Otherwise, we are timing skinny_gemm
    else:
        def fn(i: int) -> None:
            _skinny_gemm(
                A=input,
                B=q_weights[i],
                scale_tensor=scales[i],
                D=output,
                split_k=split_k,
                A_producers=A_producers,
                B_producers=B_producers,
                consumers=consumers,
                a_lanes=a_lanes,
                b_lanes=b_lanes,
                qsize=qsize,
                op_m=op_m,
                ops=ops
            )

    # Create a side-stream to benchmark in
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):

        # Create graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(graph_size):
                fn(i)
        torch.cuda.synchronize()

        # Benchmark its replays
        ts = []
        for i in range(warmups + iterations):

            # Prepare events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Run
            start_event.record()
            graph.replay()
            end_event.record()

            # Accumulate
            torch.cuda.synchronize()
            if i > warmups:
                ts.append(start_event.elapsed_time(end_event))

    # Post-process time
    ts = [1000 * t / (iterations * graph_size) for t in ts]
    mean = sum(ts) / len(ts)
    std = math.sqrt(sum((t - mean) ** 2 for t in ts) / len(ts))
    return mean, std
