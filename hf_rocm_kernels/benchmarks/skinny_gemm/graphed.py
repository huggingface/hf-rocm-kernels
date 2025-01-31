import torch

from hf_rocm_kernels.operators.skinny_gemm import skinny_gemm
from hf_rocm_kernels.utils.benchmarking import Bench
from hf_rocm_kernels.utils.fp8 import fp8_quantize


if __name__ == "__main__":
    bench = Bench()
    device = "cuda"

    M = 8
    N = 13312
    K = 16384

    SPLIT_K = 3
    B_LANES = 3

    G_SIZE = 16
    ITERS = 64

    input = torch.normal(0, 1, size=(M, K), device=device, dtype=torch.float32)
    input_scale = torch.mean(input)
    input = fp8_quantize(input, input_scale)[0]

    weights = [torch.normal(0, 1, size=(N, K), device=device, dtype=torch.float32) for _ in range(G_SIZE)]
    scales = [torch.mean(weight) for weight in weights]
    q_weights = [fp8_quantize(weight, scale)[0].t() for weight, scale in zip(weights, scales)]

    output = torch.zeros(size=(M, N), device=device, dtype=torch.float16)

    torch._scaled_mm(
        input=input, 
        mat2=q_weights[0], 
        scale_a=input_scale, 
        scale_b=scales[0],
        out_dtype=torch.float16,
        out=output
    )

    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):

        # Reference graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for weight, scale in zip(q_weights, scales):
                torch._scaled_mm(
                    input=input, 
                    mat2=weight, 
                    scale_a=input_scale, 
                    scale_b=scale,
                    out_dtype=torch.float16,
                    out=output
                )
        torch.cuda.synchronize()

        # Reference timing
        reference_t = 0
        for _ in range(ITERS):

            # Prepare events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Run
            start_event.record()
            graph.replay()
            end_event.record()

            # Accumulate
            torch.cuda.synchronize()
            reference_t += start_event.elapsed_time(end_event)
        reference_t *= 1000 / (ITERS * G_SIZE)
        print(f"Reference: {reference_t:.3f} ns")

        # Our graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for weight, scale in zip(q_weights, scales):
                skinny_gemm(
                    skinny_a=input, 
                    b=weight, 
                    scale_tensor=scale,
                    split_k=SPLIT_K,
                    b_lanes=B_LANES,
                    output=output,
                )
        torch.cuda.synchronize()

        # Our timing
        our_t = 0
        for _ in range(ITERS):

            # Prepare events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Run
            start_event.record()
            graph.replay()
            end_event.record()

            # Accumulate
            torch.cuda.synchronize()
            our_t += start_event.elapsed_time(end_event)
        our_t *= 1000 / (ITERS * G_SIZE)
        print(f"Ours:      {our_t:.3f} ns")

        print(f"Speedup:   {(100 * reference_t / our_t):.2f} %")
