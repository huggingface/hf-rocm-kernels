from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm
from hf_rocm_kernels.utils.benchmarking import Bench
from hf_rocm_kernels.operators.skinny_gemm.reference import generate_skinny_gemm_data
from hf_rocm_kernels.operators.skinny_gemm.wrapped import skinny_gemm
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image

if __name__ == "__main__":
    bench = Bench() # to enter benchmarking mode
    device = "cuda"


    M = 32
    N = 13312
    K = 16384

    G_SIZE = 12
    WARMUPS = 200
    ITERS = 1000

    reference_t = benchmark_skinny_gemm(M, N, K, None, G_SIZE, WARMUPS, ITERS)
    print(f"Reference: {reference_t:.3f} ns")

    our_t = benchmark_skinny_gemm(M, N, K, {}, G_SIZE, WARMUPS, ITERS)
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

    a, b, scale_tensor, our = generate_skinny_gemm_data(M, N, K)

    ref = torch._scaled_mm(
        input=a, mat2=b, scale_a=scale_tensor, scale_b=scale_tensor.clone().fill_(1), out_dtype=torch.float16)
    our = skinny_gemm(a, b, scale_tensor)
    precise = torch.mm(a.float(), b.float()) * scale_tensor.float().item()

    delta_ref_our = (ref - our).abs()
    delta_precise_our = (precise - our).abs()
    delta_precise_ref = (precise - ref).abs()

    # Max
    print(
        "", "Delta max",
        f"\tref - our: {delta_ref_our.max():.3f}",
        f"\tprecise - our: {delta_precise_our.max():.3f}",
        f"\tprecise - ref: {delta_precise_ref.max():.3f}",
        sep="\n"
    )
    # Mean
    print(
        "", "Delta mean",
        f"\tref - our: {delta_ref_our.mean():.3f}",
        f"\tprecise - our: {delta_precise_our.mean():.3f}",
        f"\tprecise - ref: {delta_precise_ref.mean():.3f}",
        sep="\n"
    )
    # Median
    print(
        "", "Delta median",
        f"\tref - our: {delta_ref_our.median():.3f}",
        f"\tprecise - our: {delta_precise_our.median():.3f}",
        f"\tprecise - ref: {delta_precise_ref.median():.3f}",
        sep="\n"
    )

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].matshow(delta_ref_our.numpy(force=True))
    axs[1].matshow(delta_precise_ref.numpy(force=True))
    axs[2].matshow(delta_precise_our.numpy(force=True))
    plt.savefig("skinny_gemm_delta.png")

    matplotlib.image.imsave("skinny_gemm_delta_ref_our.png", delta_ref_our.numpy(force=True))
