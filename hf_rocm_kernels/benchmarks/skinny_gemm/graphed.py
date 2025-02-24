from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm
from hf_rocm_kernels.utils.benchmarking import Bench


if __name__ == "__main__":
    bench = Bench() # to enter benchmarking mode
    device = "cuda"


    M = 8
    N = 16384
    K = 6656

    SPLIT_K = 1
    B_LANES = 4

    G_SIZE = 12
    WARMUPS = 200
    ITERS = 1000

    reference_t = benchmark_skinny_gemm(M, N, K, 0, 0, G_SIZE, WARMUPS, ITERS)
    print(f"Reference: {reference_t:.3f} ns")

    our_t = benchmark_skinny_gemm(M, N, K, SPLIT_K, B_LANES, G_SIZE, WARMUPS, ITERS)
    print(f"Ours:      {our_t:.3f} ns")

    print(f"Speedup:   {(100 * reference_t / our_t):.2f} %")

    # from hf_rocm_kernels.operators.skinny_gemm import generate_skinny_gemm_data, skinny_gemm
    # prof_ = torch.profiler
    # args = [generate_skinny_gemm_data(M, N, K) for _ in range(2)]
    # with prof_.profile(activities=[prof_.ProfilerActivity.CUDA]) as prof:
    #     torch._scaled_mm(
    #         input=args[0][0], mat2=args[0][1], scale_a=args[0][2], scale_b=args[0][2], out_dtype=torch.float16, out=args[0][3]
    #     )
    #     skinny_gemm(
    #         skinny_a=args[1][0], b=args[1][1], scale_tensor=args[1][2], split_k=SPLIT_K,  b_lanes=B_LANES, output=args[1][3]
    #     )
    # prof.export_chrome_trace("trace_skinny_gemm_graphed.json")
