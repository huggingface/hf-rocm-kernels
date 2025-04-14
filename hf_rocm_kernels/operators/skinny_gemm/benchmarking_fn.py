import torch
from hf_rocm_kernels.utils.fp8 import fp8_quantize
from hf_rocm_kernels.operators.skinny_gemm import skinny_gemm

def benchmark_skinny_gemm(
    m: int,
    n: int,
    k: int,
    split_k: int,
    b_lanes: int,
    graph_size: int = 8,
    warmups: int = 32,
    iterations: int = 128,
    device: str = "cuda"
) -> float:
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
            skinny_gemm(
                skinny_a=input,
                b=q_weights[i],
                scale_tensor=scales[i],
                split_k=split_k,
                b_lanes=b_lanes,
                output=output
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
        t = 0
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
                t += start_event.elapsed_time(end_event)

    # Post-process time
    t *= 1000 / (iterations * graph_size)
    return t
