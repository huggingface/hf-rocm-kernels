//include "consumer.cu"
//#include "producer.cu"
#include "./consumers/consumers.cuh"
#include "./producers/producers.cuh"
//#include "producers/producers.cuh"
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/packet_device.hpp>


template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::MemoryChannel> constRingChannelsA[7];
__constant__ DeviceHandle<mscclpp::MemoryChannel> constRingChannelsB[7];

__device__ mscclpp::DeviceSyncer deviceSyncer;

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;

#define __device_inline__   __device__ __forceinline__
__device_inline__ static void buffer_store_dwordx4(int32x4_t data,
                        int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device__ void device_ring_barrier(int rank, int world_size) {
    // Compute peer ranks in the ring
    int next_rank = (rank + 1) % world_size;
    int prev_rank = (rank - 1 + world_size) % world_size;

    // Compute peer indices for A and B channels (adjusted for local indexing)
    int send_id_a = (next_rank < rank) ? next_rank : next_rank - 1;
    int recv_id_a = (prev_rank < rank) ? prev_rank : prev_rank - 1;

    int send_id_b = (prev_rank < rank) ? prev_rank : prev_rank - 1;
    int recv_id_b = (next_rank < rank) ? next_rank : next_rank - 1;

    // Forward token passing ring (A)
    if (rank != 0) {
        constRingChannelsA[recv_id_a].wait();  // Wait for token from previous rank
    }

    constRingChannelsA[send_id_a].signal();   // Send token to next rank

    if (rank == 0) {
        constRingChannelsA[recv_id_a].wait();  // Wait for token to return
    }

    // Backward release ring (B)
    if (rank != 0) {
        constRingChannelsB[recv_id_b].wait();  // Wait for release from next rank
    }

    constRingChannelsB[send_id_b].signal();   // Send release to previous rank

    if (rank == 0) {
        constRingChannelsB[recv_id_b].wait();  // Wait for release to return
    }
}

__device__ __forceinline__ int ring_peer_id(int me, int peer) {
    return (peer < me) ? peer : peer - 1;
}

__device__ void device_global_barrier(int rank, int world_size) {
    // Phase 1: All-to-one gather (A)
    if (rank != 0) {
        int peer_id = ring_peer_id(rank, 0);
        constRingChannelsA[peer_id].signal();  // Notify rank 0
        constRingChannelsB[peer_id].wait();    // Wait for release from rank 0
    } else {
        // Rank 0 waits for all other GPUs
        for (int peer = 1; peer < world_size; ++peer) {
            int peer_id = ring_peer_id(0, peer);
            constRingChannelsA[peer_id].wait();  // Wait for arrival from peer
        }

        // Then send release to all peers
        for (int peer = 1; peer < world_size; ++peer) {
            int peer_id = ring_peer_id(0, peer);
            constRingChannelsB[peer_id].signal();  // Release peer
        }
    }

    // Optional per-block sync for all threads
    //__syncthreads();
}


__global__ void vectorized_reduce_inplace(__half* __restrict__ D,
                                          __half* __restrict__ buff_a,
                                          __half* __restrict__ buff_b,
                                          int size, int rank, int world_size,
                                          int compute_warps, bool is_capturing) {
    using half2_t = __half2;

    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    const int warp_size = 64;
    const int warps_per_block = blockDim.x / warp_size;
    const int warp_id = thread_id / warp_size;
    const int lane_id = thread_id % warp_size;

    const int comm_warps = warps_per_block - compute_warps;

    const int total_blocks = gridDim.x;
    const int compute_threads_per_block = compute_warps * warp_size;
    const int comm_threads_per_block = comm_warps * warp_size;
    const int total_compute_threads = total_blocks * compute_threads_per_block;
    const int total_comm_threads = total_blocks * comm_threads_per_block;

    const int total_half2s = size / 2;
    const int total_half2s_with_tail = (size + 1) / 2;

    const bool is_compute = warp_id < compute_warps;
    const bool is_comm = warp_id >= compute_warps;

    int peerSendRank = (rank + 1) % world_size;
    int peerRecvRank = (rank + world_size - 1) % world_size;
    int peerSendId = peerSendRank < rank ? peerSendRank : peerSendRank - 1;
    int peerRecvId = peerRecvRank < rank ? peerRecvRank : peerRecvRank - 1;
    //printf("rank: %d, peerSendId: %d, peerRecvId: %d\n", rank, peerSendId, peerRecvId);
    // rank: 0, peerSendId: 0, peerRecvId: 6
    // rank: 1, peerSendId: 1, peerRecvId: 0
    // rank: 2, peerSendId: 2, peerRecvId: 1
    // rank: 3, peerSendId: 3, peerRecvId: 2
    // rank: 4, peerSendId: 4, peerRecvId: 3
    // rank: 5, peerSendId: 5, peerRecvId: 4
    // rank: 6, peerSendId: 6, peerRecvId: 5
    // rank: 7, peerSendId: 0, peerRecvId: 6

    DeviceHandle<mscclpp::MemoryChannel>& left_a = constRingChannelsA[peerRecvId];
    DeviceHandle<mscclpp::MemoryChannel>& right_a = constRingChannelsA[peerSendId];
    DeviceHandle<mscclpp::MemoryChannel>& left_b = constRingChannelsB[peerRecvId];
    DeviceHandle<mscclpp::MemoryChannel>& right_b = constRingChannelsB[peerSendId];

    half2_t* D_ = reinterpret_cast<half2_t*>(D);
    half2_t* buff_a_ = reinterpret_cast<half2_t*>(buff_a);
    half2_t* buff_b_ = reinterpret_cast<half2_t*>(buff_b);
    for (int step = 0; step < world_size - 1; ++step) {
        deviceSyncer.sync(gridDim.x);

        half2_t* buff = (step % 2 == 0) ? buff_b_ : buff_a_;

        __syncthreads();
        if (is_compute) {
            int compute_warp_id = warp_id;
            int compute_lane_id = lane_id;
            int logical_compute_tid = compute_warp_id * warp_size + compute_lane_id;
            int global_compute_tid = block_id * compute_threads_per_block + logical_compute_tid;

            for (int i = global_compute_tid; i < total_half2s; i += total_compute_threads) {
                D_[i] += buff[i];
            }

            if (global_compute_tid == 0 && (size % 2) != 0) {
                __half b = (step % 2 == 0 ? buff_b : buff_a)[size - 1];
                D[size - 1] += b;
            }
        }

        if (is_comm && !is_capturing) {
            int comm_warp_id = warp_id - compute_warps;
            int comm_lane_id = lane_id;
            int local_comm_tid = comm_warp_id * warp_size + comm_lane_id;
            int comm_threads_per_block = comm_warps * warp_size;
            int global_comm_tid = block_id * comm_threads_per_block + local_comm_tid;

            int elems_per_block = (total_half2s_with_tail + total_blocks - 1) / total_blocks;
            int block_start_idx = block_id * elems_per_block;
            int block_end_idx = min(block_start_idx + elems_per_block, total_half2s_with_tail);

            if (block_start_idx < block_end_idx) {
                uint64_t offset_bytes = block_start_idx * sizeof(half2_t);
                uint64_t chunk_bytes = (block_end_idx - block_start_idx) * sizeof(half2_t);

                if (step % 2 == 0) {
                    if (global_comm_tid == 0) {
                        left_a.signal();
                        right_a.wait();
                    }
                    deviceSyncer.sync(gridDim.x);
                    //right_b.put(offset_bytes, chunk_bytes, local_comm_tid, comm_threads_per_block);
                    int global_comm_tid = block_id * comm_threads_per_block + local_comm_tid;
                    right_b.put(offset_bytes, chunk_bytes, local_comm_tid, comm_threads_per_block);
                    if (global_comm_tid == 0) {
                        //right_b.put(0, size*2, 0, 1);
                        right_b.signal();
                        left_b.wait();
                        left_b.signal();
                        right_b.wait();
                    }
                } else {
                    if (global_comm_tid == 0) {
                        left_b.signal();
                        right_b.wait();
                    }
                    deviceSyncer.sync(gridDim.x);
                    //right_a.put(offset_bytes, chunk_bytes, local_comm_tid, comm_threads_per_block);
                    int global_comm_tid = block_id * comm_threads_per_block + local_comm_tid;
                    right_a.put(offset_bytes, chunk_bytes, local_comm_tid, comm_threads_per_block);
                    if (global_comm_tid == 0) {
                        //right_a.put(0, size*2, 0, 1);
                        right_a.signal();
                        left_a.wait();
                        left_b.signal();
                        right_b.wait();
                    }
                }
                __threadfence_system();
            }
        }
    }
    // Reset buffer A to ensure we do not accumulate between allreduce runs
    //for (int i = idx * 2; i < size; i += stride * 2) {
    //    reinterpret_cast<int32_t*>(buff_a)[i / 2] = 0;
    //}
}



#define launch_tsr(BL, AP, BP, C, COMM, QS)                                                                                  \
    block.x = WARPSIZE * (AP + BP + C) + COMM;                                                                                \
    _tsr_kernel<BL, AP, BP, C, COMM, QS><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, b_stride, split_k, rank, world_size, buff_a_, is_capturing); \
    break;

template <int A_LANES, int B_LANES, int QSIZE, int OP_M, int OPS>
void __global__ _skinny_gemm_kernel(const fp8* __restrict__ A, const fp8* __restrict__ B, half* __restrict__ D,
                            const float* scale_tensor, const int m, const int n, const int k, const int b_stride,
                            const int split_k, const int A_producers, const int B_producers, const int consumers,
                            const int rank, const int world_size, half* scratch, bool is_capturing) {
    // Compile-time constants
    static constexpr int WARPTILE_M = OP_M * A_LANES;  // is either 8, 16 or 32
    static constexpr int WARPTILE_N = (OP_M == 32 ? 32 : 16) * B_LANES;
    static constexpr int WARPTILE_K = (512 / OP_M) * OPS;

    // Initialize shared queue
    __shared__ int queue[(A_LANES + B_LANES) * QSIZE];
    if (threadIdx.x < (A_LANES + B_LANES) * QSIZE) {
        queue[threadIdx.x] = 0;
    }
    // Declare shared buffer
    __shared__ fp8 A_buffer[WARPTILE_M * WARPTILE_K * QSIZE];
    __shared__ fp8 B_buffer[WARPTILE_N * WARPTILE_K * QSIZE];
    __syncthreads();

    // Infer warp-specialization-related variables
    const int warp_id = get_warp_id();
    int role_id = (warp_id < A_producers) ? warp_id : (warp_id - A_producers);
    role_id = (warp_id < A_producers + B_producers) ? role_id : (role_id - B_producers);
    int index = role_id;
    int p_state = (warp_id >= A_producers + B_producers);

    // Figure out peer channels for comms
    int peerSendRank = (rank + 1) % world_size;
    int peerRecvRank = (rank - 1 + world_size) % world_size;
    int peerSendId = peerSendRank < rank ? peerSendRank : peerSendRank - 1;
    int peerRecvId = peerRecvRank < rank ? peerRecvRank : peerRecvRank - 1;
    DeviceHandle<mscclpp::MemoryChannel>& left = constRingChannelsA[peerRecvId];
    DeviceHandle<mscclpp::MemoryChannel>& right = constRingChannelsA[peerSendId];

    // Tiles loop
    const int rows_of_tiles = CDIV(m, WARPTILE_M);
    const int tiles_per_row = CDIV(n, WARPTILE_N);
    const int total_tiles = rows_of_tiles * tiles_per_row * split_k;
    const int tiles_per_block = CDIV(total_tiles, CU);

    const int stop_tile = min(total_tiles, tiles_per_block * (blockIdx.x + 1));

    for (int tile = tiles_per_block * blockIdx.x; tile < stop_tile; tile++) {
        // Compute tile position
        int idx_m = tile % rows_of_tiles;
        int curr_m = idx_m * WARPTILE_M;

        int idx_nk = tile / rows_of_tiles;
        int curr_n = (idx_nk % tiles_per_row) * WARPTILE_N;
        int curr_k = (idx_nk / tiles_per_row) * WARPTILE_K * NUM_WARPTILE_K(k, split_k);

        // Compute tile's K blocks (number of blocks along the K axis)
        bool last_tile = (idx_nk / tiles_per_row) == (split_k - 1);
        int full_tiles = (k / WARPTILE_K);
        int k_blocks = last_tile ? full_tiles - (split_k - 1) * (full_tiles / split_k) : (full_tiles / split_k);

        // Account for column overflow
        int dropped_rows = max(0, curr_m + WARPTILE_M - m);
        int dropped_cols = max(0, curr_n + WARPTILE_N - n);
        curr_n -= dropped_cols;  // make sure we are in the bounds of B

        // A producer warp
        if (warp_id < A_producers) {
            produce_A_tiles<A_LANES, QSIZE, OP_M, OPS>(A + curr_m * k + curr_k, &A_buffer[0], A_producers, &queue[0],
                                                       index, p_state, role_id, k, k_blocks, dropped_rows);
        }
        // B producer warp
        else if (warp_id < A_producers + B_producers) {
            // TODO: investigate the reuse parameter forB (true = faster but goes wrong because no sc1)
            produce_B_tiles<B_LANES, QSIZE, OP_M, OPS>(B + curr_n * b_stride + curr_k, &B_buffer[0], B_producers,
                                                       &queue[A_LANES * QSIZE], index, p_state, role_id, b_stride,
                                                       k_blocks);
        }
        // Consumers warp
        else if (warp_id < A_producers + B_producers + consumers) {
            consume_tiles<A_LANES, B_LANES, QSIZE, OP_M, OPS>(&A_buffer[0], &B_buffer[0], D + curr_m * n + curr_n,
                                                              scale_tensor[0], consumers, &queue[0], index, p_state,
                                                              role_id, dropped_rows, dropped_cols, n, k, k_blocks);
        }
        //asm volatile("s_waitcnt vmcnt(0)");
        __syncthreads();
        size_t tid = threadIdx.x - (A_producers + B_producers + consumers) * WARPSIZE;
        if (tid >= 0 && tid < COMMS) {
            if (!is_capturing) {
                size_t comms_threads = COMMS;
                size_t chunk_bytes = (16*B_LANES) * sizeof(half);
                size_t chunk_offset = 2*(curr_n + tid * n);
                right.put(chunk_offset, chunk_bytes, 0, 1);
            }
        }
        __syncthreads();
    }
    //deviceSyncer.sync(gridDim.x);
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     //printf("Sending data");
    //     right.put(0, m*n*2, 0, 1);
    //     right.signal();
    //     left.wait();
    // }
    // deviceSyncer.sync(gridDim.x);
    __threadfence_system();
}

void skinny_gemm(
    const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& D, const torch::Tensor& scale_tensor,
    const int64_t split_k, const int64_t A_producers, const int64_t B_producers, const int64_t consumers,
    const int64_t a_lanes, const int64_t b_lanes, const int64_t qsize, const int64_t op_m, const int64_t ops,
    const int rank, const int world_size, uint8_t* buff_a, uint8_t* buff_b, cudaEvent_t lock,
    const int nBlocks, const int compute_warps, const bool is_capturing
) {
    // Retrieve pointers
    const fp8* __restrict__ A_ = (const fp8* __restrict__)A.data_ptr();
    const fp8* __restrict__ B_ = (const fp8* __restrict__)B.data_ptr();
    half* __restrict__ D_ = (half* __restrict__)D.data_ptr();
    float* __restrict__ scale_tensor_ = (float* __restrict__)scale_tensor.data_ptr();

    // Retrieve shapes
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);
    const int b_stride = B.stride(1);

    half* __restrict__ buff_a_ = (half* __restrict__)buff_a;
    half* __restrict__ buff_b_ = (half* __restrict__)buff_b;

    // Check shape
    if (m > WARPTILE_M) {
        std::cerr << "m = " << k << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
        exit(1);
    }
    if (k % WARPTILE_K != 0) {
        std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
        exit(1);
    }

    // Retrieve stream
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

    skinny_gemm_caller(A_, B_, D_, scale_tensor_, m, n, k, b_stride, split_k, A_producers, B_producers, consumers,
                    a_lanes, b_lanes, qsize, op_m, ops, stream);

    int threads = 1024;
    int blocks = (D.numel() / 2 + threads - 1) / threads;
    vectorized_reduce_inplace<<<blocks, threads, 0, stream>>>(D_, buff_a_, buff_b_, D.numel(), rank, world_size, compute_warps, is_capturing);
    cudaDeviceSynchronize();
}

#define COND_LAUCNH_ONE_SKINNY_GEMM(__al, __bl, __qs, __om, __ops)                                   \
    else if (a_lanes == __al && b_lanes == __bl && qsize == __qs && op_m == __om && ops == __ops) {  \
        _skinny_gemm_kernel<__al, __bl, __qs, __om, __ops><<<grid, block, 0, stream>>>(              \
            A, B, D, scale_tensor, m, n, k, b_stride, split_k, A_producers, B_producers, consumers); \
    }

enum SkinnyGemmReturnCode {
    SUCCESS = 0,
    M_ABOVE_WARPTILE_M = 1,
    N_NOT_EVEN = 2,
    K_NOT_DIVISIBLE_BY_WARPTILE_K = 3,
    TOO_MANY_WARPS = 4,
    QSIZE_TOO_SMALL = 5,
    INVALID_CONFIG = 6
};

int skinny_gemm_caller(
    // Tensors
    const fp8* __restrict__ A, const fp8* __restrict__ B, half* __restrict__ D, const float* scale_tensor,
    // Shapes
    const int m, const int n, const int k, const int b_stride, const int split_k,
    // Async non-templated
    const int A_producers, const int B_producers, const int consumers,
    // Async templated
    const int a_lanes, const int b_lanes, const int qsize, const int op_m, const int ops,
    // Cuda-related
    hipStream_t stream) {
    // Deduce other constants
    const int OP_K = 512 / op_m;
    const int WARPTILE_M = op_m;
    const int WARPTILE_K = OP_K * ops;

    // Check shapes
    // if (m > WARPTILE_M) {
    //     std::cerr << "m = " << m << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
    //     return SkinnyGemmReturnCode::M_ABOVE_WARPTILE_M;
    // }
    if (n % 2 != 0) {
        std::cerr << "n = " << n << " is not even" << std::endl;
        return SkinnyGemmReturnCode::N_NOT_EVEN;
    }
    if (k % WARPTILE_K != 0) {
        std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
        return SkinnyGemmReturnCode::K_NOT_DIVISIBLE_BY_WARPTILE_K;
    }

    // Check async
    if (A_producers + B_producers + consumers > 16) {
        std::cerr << "A_producers = " << A_producers << ", B_producers = " << B_producers << ", consumers = ";
        std::cerr << consumers << " is greater than 16" << std::endl;
        return SkinnyGemmReturnCode::TOO_MANY_WARPS;
    }
    if (qsize < (A_producers / a_lanes) || qsize < (B_producers / b_lanes) || qsize < consumers) {
        std::cerr << "qsize = " << qsize << " is less than A_producers / a_lanes = " << A_producers / a_lanes;
        std::cerr << ", B_producers / b_lanes = " << B_producers / b_lanes;
        std::cerr << ", or consumers = " << consumers << std::endl;
        return SkinnyGemmReturnCode::QSIZE_TOO_SMALL;
    }

    // Prepare kernel launch
    dim3 grid(CU);
    dim3 block((A_producers + B_producers + consumers) * WARPSIZE);

    // Dispatch to the correct kernel
    if (b_lanes == 0) {
        // This is a dummy if because the macro begins with an else if
        return SkinnyGemmReturnCode::INVALID_CONFIG;
    }  // TODO: remove some possibilities
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 2, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 3, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 3, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 4, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 4, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 4, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 4, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 5, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 5, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 5, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 5, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 6, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 6, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 6, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 2, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 3, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 3, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 4, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 4, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 4, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 5, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 5, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 5, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 6, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 2, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 3, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 4, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 3, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 4, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 4, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 5, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 5, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 3, 32, 4)
    else {
        return SkinnyGemmReturnCode::INVALID_CONFIG;
    }

    return SkinnyGemmReturnCode::SUCCESS;
}
