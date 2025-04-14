import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hf_rocm_kernels.utils.benchmarking import Bench
from hf_rocm_kernels.operators.fused_gemm_ar import fused_gemm_ar_init
from hf_rocm_kernels.operators.fused_gemm_ar.benchmarking_fn import benchmark_fused_gemm_ar

G_SIZE = 8
WARMUPS = 5
ITERS = 5

def init_process(rank, world_size, master_addr):
    """Initialize process group and set environment variables"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Set device
    torch.cuda.set_device(rank)

    return rank, world_size

def _bench(rank: int, world_size: int, m: int, n: int, k: int, split_k: int, b_lanes: int):
    try:
        rank, world_size = init_process(rank, world_size, "localhost")

        comms_a = torch.zeros(size=(m, n), dtype=torch.float16, device="cuda")
        comms_b = torch.zeros(size=(m, n), dtype=torch.float16, device="cuda")

        port = 50004
        allreduce_engine = fused_gemm_ar_init(
            rank=rank,
            world_size=world_size,
            port=port,
            comms_a=comms_a,
            comms_b=comms_b
        )

        reference_t = benchmark_fused_gemm_ar(m, n, k, 0, 0, G_SIZE, WARMUPS, ITERS)

        t = benchmark_fused_gemm_ar(allreduce_engine, m, n, k, split_k, b_lanes)
        speedup = reference_t / t
        print(f"{b_lanes=} {split_k=} | t = {t:.2f} -> speedup = {speedup:.2f}")

    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise

    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    bench = Bench() # to enter benchmarking mode
    device = "cuda"


    M = 8
    N = 2304
    K = 16384

    SPLIT_K = 5
    B_LANES = 4

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")

    mp.spawn(
        _bench,
        args=(world_size, M, N, K, SPLIT_K, B_LANES),
        nprocs=world_size,
        join=True,
        start_method='spawn'
    )
