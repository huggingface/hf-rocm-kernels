from hf_rocm_kernels.operators.skinny_gemm.benchmarking_fn import benchmark_skinny_gemm
from hf_rocm_kernels.utils.benchmarking import Bench


def benchmark_t_reference(m: int, n: int, k: int, graph_size: int, warmups: int, iterations: int) -> float:
    tnop, stdnop = benchmark_skinny_gemm(
        m=m, n=n, k=k,
        skg_kwargs=None,
        graph_size=graph_size, warmups=warmups, iterations=iterations,
        device="cuda:2",
        mode="median"
    )
    print(f"Reference without TunableOps: {tnop:.3f} ns")
    Bench() # to enter benchmarking mode
    top, stdtop = benchmark_skinny_gemm(
        m=m, n=n, k=k,
        skg_kwargs=None,
        graph_size=graph_size, warmups=warmups, iterations=iterations,
        device="cuda:2",
        mode="median"
    )
    print(f"Reference with TunableOps: {top:.3f} ns")
    if tnop < top:
        return tnop, stdnop
    else:
        return top, stdtop


if __name__ == "__main__":

    M = 32
    N = 16384
    K = 6656

    G_SIZE = 10
    WARMUPS = 500
    ITERS = 2000

    t_reference, std_reference = benchmark_t_reference(m=M, n=N, k=K, graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS)

    our_t, std = benchmark_skinny_gemm(
        m=M, n=N, k=K,
        skg_kwargs={},
        graph_size=G_SIZE, warmups=WARMUPS, iterations=ITERS,
        device="cuda:2",
        mode="median"
    )

    print(f"{t_reference:.3f} ± {std_reference:.3f} , {our_t:.3f} ± {std:.3f} , {100 * t_reference / our_t:.2f} %")
