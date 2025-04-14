import torch
import matplotlib.pyplot as plt
import matplotlib

from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm
from hf_rocm_kernels.operators.skinny_gemm import generate_skinny_gemm_data
from hf_rocm_kernels.operators.skinny_gemm.binding import _skinny_gemm

device = "cuda:3"

if __name__ == "__main__":

    M = 1
    N = 16384
    K = 6656

    SPLIT_K = 1

    ABC = (2, 7, 2)

    A_LANES = 1
    B_LANES = 4
    QSIZE = 2

    OP_M = 8
    OPS = 4

    G_SIZE = 12
    WARMUPS = 500
    ITERS = 2000

    # bench = Bench() # to enter benchmarking mode
    reference_t, reference_std = benchmark_skinny_gemm(m=M, n=N, k=K, skg_kwargs=None, graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS, device=device)
    print(f"Reference: {reference_t:.3f} ns ± {reference_std:.3f}")

    skg_kwargs = {
        "split_k": SPLIT_K,
        "A_producers": ABC[0], "B_producers": ABC[1], "consumers": ABC[2],
        "a_lanes": A_LANES, "b_lanes": B_LANES, "qsize": QSIZE,
        "op_m": OP_M, "ops": OPS,
    }
    our_t, our_std = benchmark_skinny_gemm(
        m=M, n=N, k=K, skg_kwargs=skg_kwargs,
        graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS,
        device=device
    )
    print(f"Ours:      {our_t:.3f} ns ± {our_std:.3f}")
    print(f"Speedup:   {(100 * reference_t / our_t):.2f} %")
    print("\n" + " ".join(map(str, skg_kwargs.items())))

    a, b, scale_tensor, our = generate_skinny_gemm_data(M, N, K)
    ref = torch._scaled_mm(
        input=a, mat2=b, scale_a=scale_tensor, scale_b=scale_tensor.clone().fill_(1), out_dtype=torch.float16
    )
    _skinny_gemm(
        A=a, B=b, D=our, scale_tensor=scale_tensor,
        split_k=SPLIT_K,
        A_producers=ABC[0], B_producers=ABC[1], consumers=ABC[2],
        a_lanes=A_LANES, b_lanes=B_LANES, qsize=QSIZE,
        op_m=OP_M, ops=OPS
    )
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
