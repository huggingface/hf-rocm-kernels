#include "consumer.cu"
#include "producer.cu"
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
    int peerRecvRank = (rank - 1 + world_size) % world_size;
    int peerSendId = peerSendRank < rank ? peerSendRank : peerSendRank - 1;
    int peerRecvId = peerRecvRank < rank ? peerRecvRank : peerRecvRank - 1;

    DeviceHandle<mscclpp::MemoryChannel>& left_a = constRingChannelsA[peerRecvId];
    DeviceHandle<mscclpp::MemoryChannel>& right_a = constRingChannelsA[peerSendId];
    DeviceHandle<mscclpp::MemoryChannel>& left_b = constRingChannelsB[peerRecvId];
    DeviceHandle<mscclpp::MemoryChannel>& right_b = constRingChannelsB[peerSendId];

    for (int step = 0; step < world_size - 1; ++step) {
        deviceSyncer.sync(gridDim.x);

        __syncthreads();
        if (is_compute) {
            int compute_warp_id = warp_id;
            int compute_lane_id = lane_id;
            int logical_compute_tid = compute_warp_id * warp_size + compute_lane_id;
            int global_compute_tid = block_id * compute_threads_per_block + logical_compute_tid;

            for (int i = global_compute_tid; i < total_half2s; i += total_compute_threads) {
                half2_t a = reinterpret_cast<half2_t*>(D)[i];
                half2_t b = reinterpret_cast<half2_t*>((step % 2 == 0) ? buff_b : buff_a)[i];

                reinterpret_cast<half2_t*>(D)[i] = __hadd2(a, b);
            }

            if (global_compute_tid == 0 && (size % 2) != 0) {
                __half b = (step % 2 == 0 ? buff_b : buff_a)[size - 1];
                D[size - 1] = __hadd(D[size - 1], b);
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
}



#define launch_tsr(BL, AP, BP, C, COMM, QS)                                                                                  \
    block.x = WARPSIZE * (AP + BP + C) + COMM;                                                                                \
    _tsr_kernel<BL, AP, BP, C, COMM, QS><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, b_stride, split_k, rank, world_size, buff_a_, is_capturing); \
    break;

template <int B_LANES, int A_PRODUCERS, int B_PRODUCERS, int CONSUMERS, int COMMS, int QSIZE>
void __global__ _tsr_kernel(const fp8* __restrict__ A, const fp8* __restrict__ B, half* __restrict__ D,
                            const float* scale_tensor, const int m, const int n, const int k, const int b_stride,
                            const int split_k, const int rank, const int world_size, half* scratch, bool is_capturing) {
    // Initialize shared queue
    __shared__ int queue[2 * B_LANES * QSIZE];
    if (threadIdx.x < 2 * B_LANES * QSIZE) {
        queue[threadIdx.x] = 0;
    }
    // Declare shared buffer
    __shared__ fp8 A_buffer[WARPTILE_M * WARPTILE_K * QSIZE];
    __shared__ fp8 B_buffer[(OP_N * B_LANES) * WARPTILE_K * QSIZE];
    __syncthreads();

    // Infer index and p-state
    int role_id;
    int index;
    int p_state;

    // A producer warp
    if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
        role_id = threadIdx.x / WARPSIZE;
        index = (OPS == 1 ? 2 : 1) * role_id;
        p_state = 0;
    }
    // B producer warp
    else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
        role_id = (threadIdx.x / WARPSIZE) - A_PRODUCERS;
        index = role_id;
        p_state = 0;
    }
    // Consumers warp
    else {
        role_id = (threadIdx.x / WARPSIZE) - (A_PRODUCERS + B_PRODUCERS);
        index = role_id;
        p_state = 1;
    }

    // Figure out peer channels for comms
    int peerSendRank = (rank + 1) % world_size;
    int peerRecvRank = (rank - 1 + world_size) % world_size;
    int peerSendId = peerSendRank < rank ? peerSendRank : peerSendRank - 1;
    int peerRecvId = peerRecvRank < rank ? peerRecvRank : peerRecvRank - 1;
    DeviceHandle<mscclpp::MemoryChannel>& left = constRingChannelsA[peerRecvId];
    DeviceHandle<mscclpp::MemoryChannel>& right = constRingChannelsA[peerSendId];

    // if(threadIdx.x == 0) {
    //     device_ring_barrier(rank, world_size);
    // }
    __syncthreads();

    // Tiles loop
    int curr_n, curr_k, k_blocks, dropped_rows, dropped_cols;
    const int warptile_per_row = CDIV(n, (OP_N * B_LANES));
    const int tiles = warptile_per_row * split_k;
    const int tpw = max(CDIV(tiles, CU), 1);

    for (int warptile = (tpw * blockIdx.x); warptile < min(tiles, tpw * (blockIdx.x + 1)); warptile++) {
        // Compute tile position
        curr_n = (warptile % warptile_per_row) * (OP_N * B_LANES);
        curr_k = (warptile / warptile_per_row) * WARPTILE_K * K_BLOCKS(k, split_k);
        k_blocks = ((warptile / warptile_per_row) == (split_k - 1))
                       ? (k / WARPTILE_K) - (split_k - 1) * K_BLOCKS(k, split_k)
                       : K_BLOCKS(k, split_k);

        // Account for column overflow
        dropped_rows = max(0, 0 + WARPTILE_M - m);
        dropped_cols = max(0, curr_n + (OP_N * B_LANES) - n);
        curr_n -= dropped_cols;

        // A producer warp
        if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
            _tsr_A_producer<A_PRODUCERS, B_LANES, QSIZE>(A + curr_k, &A_buffer[0], &queue[0], index, p_state, role_id,
                                                         dropped_rows, k, k_blocks);
        }
        // B producer warp
        else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
            _tsr_B_producer<B_PRODUCERS, B_LANES, QSIZE>(B + curr_n * b_stride + curr_k, &B_buffer[0], &queue[1], index,
                                                         p_state, role_id, b_stride, k_blocks);
        }
        // Consumers warp
        else if (threadIdx.x < (A_PRODUCERS + B_PRODUCERS + CONSUMERS) * WARPSIZE) {
            _tsr_consumer<CONSUMERS, B_LANES, QSIZE>(&A_buffer[0], &B_buffer[0], D + curr_n, scale_tensor[0], &queue[0],
                                                     index, p_state, role_id, n, dropped_rows, dropped_cols, k,
                                                     k_blocks, scratch + curr_n);
        }
        //asm volatile("s_waitcnt vmcnt(0)");
        __syncthreads();
        size_t tid = threadIdx.x - (A_PRODUCERS + B_PRODUCERS + CONSUMERS) * WARPSIZE;
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

void skinny_gemm(torch::Tensor& A, torch::Tensor& B, torch::Tensor& D, torch::Tensor& scale_tensor, int64_t b_lanes,
                 int64_t split_k, const int rank, const int world_size, uint8_t* buff_a, uint8_t* buff_b, cudaEvent_t lock, int nBlocks, int compute_warps, bool is_capturing) {
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);
    const int b_stride = B.stride(1);

    const fp8* __restrict__ A_ = (const fp8* __restrict__)A.data_ptr();
    const fp8* __restrict__ B_ = (const fp8* __restrict__)B.data_ptr();
    half* __restrict__ D_ = (half* __restrict__)D.data_ptr();
    float* __restrict__ scale_tensor_ = (float* __restrict__)scale_tensor.data_ptr();

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

    // Prepare kernel launch
    dim3 grid(CU, 1, 1);
    dim3 block(1, 1, 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel (branched on B_LANES)
    switch (b_lanes) {
        case 2:
            launch_tsr(2, 3, 8, 4, 8, 5);
        case 3:
            launch_tsr(3, 3, 5, 2, 8, 4);  // Perforamnce on MI300: 8_13312_16384:57.54
        case 4:
            launch_tsr(4, 2, 6, 3, 8, 3);  // Perforamnce on MI300: 8_16384_6656:29.5
        case 5:
            launch_tsr(5, 2, 6, 2, 8, 2);
        default:
            break;
    }
    // switch (b_lanes) {
    //     case 2:
    //         launch_tsr(2, 3, 8, 1, 8, 5);
    //     case 3:
    //         launch_tsr(3, 3, 5, 1, 8, 4);  // Perforamnce on MI300: 8_13312_16384:57.54
    //     case 4:
    //         launch_tsr(4, 2, 6, 1, 8, 3);  // Perforamnce on MI300: 8_16384_6656:29.5
    //     case 5:
    //         launch_tsr(5, 2, 6, 1, 8, 2);
    //     default:
    //         break;
    // }

    int threads = 1024;
    //int blocks = (D.numel() / 2 + threads - 1) / threads;
    vectorized_reduce_inplace<<<nBlocks, threads, 0, stream>>>(D_, buff_a_, buff_b_, D.numel(), rank, world_size, compute_warps, is_capturing);
    cudaDeviceSynchronize();
}
