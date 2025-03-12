import torch
from hf_rocm_kernels.operators.swiglu import swiglu, reference_swiglu

def benchmark_swiglu(
    batch_size: int,
    intermediate_size: int,
    num_threads: int, # 0 for reference, -1 for automatic, >0 for specific number of threads
    force_scalar: bool = False,
    graph_size: int = 16,
    warmups: int = 50,
    iterations: int = 250,
    device: str = "cuda",
) -> float:
    # Create input
    inputs = [
        torch.normal(0, 1, size=(batch_size, 2 * intermediate_size), device=device, dtype=torch.float16)
        for _ in range(graph_size)
    ]
    scale_tensors = [
        torch.rand(size=(1,), device=device, dtype=torch.float32).mul(2).sub(1)
        for _ in range(graph_size)
    ]

    # Define function to benchmark
    if num_threads == 0:
        def fn(i: int) -> None:
            reference_swiglu(inputs[i], scale_tensors[i], next_buffer=None)
    elif num_threads == -1:
        def fn(i: int) -> None:
            swiglu(inputs[i], scale_tensors[i], next_buffer=None, force_scalar=force_scalar, num_threads=num_threads)
    
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
