#include "consumer.cu"
#include "producer.cu"
#include <mscclpp/concurrency_device.hpp>


template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::PortChannel> constRingChannelsA[7];
__constant__ DeviceHandle<mscclpp::PortChannel> constRingChannelsB[7];

__device__ mscclpp::DeviceSyncer deviceSyncer;

__global__ void vectorized_reduce_inplace(__half* __restrict__ D, __half* __restrict__ buff_a, __half* __restrict__ buff_b, int size, int rank, int world_size, bool is_capturing) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    using half2_t = __half2;


    // Figure out peer channels for comms
    int peerSendRank = (rank + 1) % world_size;
    int peerRecvRank = (rank - 1 + world_size) % world_size;
    int peerSendId = peerSendRank < rank ? peerSendRank : peerSendRank - 1;
    int peerRecvId = peerRecvRank < rank ? peerRecvRank : peerRecvRank - 1;

    DeviceHandle<mscclpp::PortChannel>& left_a = constRingChannelsA[peerRecvId];
    DeviceHandle<mscclpp::PortChannel>& right_a = constRingChannelsA[peerSendId];

    DeviceHandle<mscclpp::PortChannel>& left_b = constRingChannelsB[peerRecvId];
    DeviceHandle<mscclpp::PortChannel>& right_b = constRingChannelsB[peerSendId];

    // Ring all-reduce
    for (int step = 0; step < world_size - 1; ++step) {
        if (idx == 0) {
            if (step % 2 == 0) {
                // Let's wait for the parallel transfer to complete (B->A)
                //printf("Allreduce Rank %d: Sending data B->A to %d\n", rank, peerSendRank);
                if(!is_capturing) {
                    right_b.put(0, size*2);
                    right_b.signal();
                }

            } else {
                // Let's wait for the parallel transfer to complete (A->B)
                //printf("Allreduce Rank %d: Sending data A->B to %d\n", rank, peerSendRank);
                if(!is_capturing) {
                    right_a.put(0, size*2);
                    right_a.signal();
                }
            }
        }

        // The first comms iteration in the ring is performed during
        // the GEMM operation, so we can start with a reduce here
        for (int i = idx * 2; i < size; i += stride * 2) {
            half2_t a = reinterpret_cast<half2_t*>(D)[i / 2];
            //const __half2* buff = step % 2 == 0 ? buff_a : buff_b;
            half2_t b = reinterpret_cast<half2_t*>(step % 2 == 0 ? buff_b : buff_a)[i / 2];
            //half2_t b = reinterpret_cast<const half2_t*>(buff_a)[i / 2];
            reinterpret_cast<half2_t*>(D)[i / 2] = __hadd2(a, b);  // In-place addition
        }

        // Handle odd-length case (if n is odd)
        if (idx == 0 && (size % 2) != 0) {
            __half b = (step % 2 == 0 ? buff_b : buff_a)[size - 1];
            D[size - 1] = __hadd(D[size - 1], b);
        }

        deviceSyncer.sync(gridDim.x, -1);
        if (idx == 0) {
            if (step % 2 == 0) {
                // Let's wait for the parallel transfer to complete (B->A)
                if(!is_capturing) {
                    right_b.flush();
                    left_b.wait();
                }
            } else {
                // Let's wait for the parallel transfer to complete (A->B)
                if(!is_capturing) {
                    right_a.flush();
                    left_a.wait();
                }
            }
        }
        deviceSyncer.sync(gridDim.x, -1);
     }

     // Reset buffer A to ensure we do not accumulate between allreduce runs
     for (int i = idx * 2; i < size; i += stride * 2) {
         reinterpret_cast<int32_t*>(buff_a)[i / 2] = 0;
     }
}

#define launch_tsr(BL, AP, BP, C, QS)                                                                        \
    block.x = WARPSIZE * (AP + BP + C); \
    _tsr_kernel<BL, AP, BP, C, QS><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, split_k, rank, world_size, buff_a_, is_capturing); \
    break;

template <int B_LANES, int A_PRODUCERS, int B_PRODUCERS, int CONSUMERS, int QSIZE>
void __global__ _tsr_kernel(const fp8* __restrict__ A, const fp8* __restrict__ B, half* __restrict__ D,
                            const float* scale_tensor, const int m, const int n, const int k, const int split_k, const int rank, const int world_size, half* scratch, bool is_capturing) {
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
    DeviceHandle<mscclpp::PortChannel>& left = constRingChannelsA[peerRecvId];
    DeviceHandle<mscclpp::PortChannel>& right = constRingChannelsA[peerSendId];

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
            __syncthreads();
        }
        // B producer warp
        else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
            _tsr_B_producer<B_PRODUCERS, B_LANES, QSIZE>(B + curr_n * k + curr_k, &B_buffer[0], &queue[1], index,
                                                         p_state, role_id, k, k_blocks);
            __syncthreads();
        }
        // Consumers warp
        else if (threadIdx.x < (A_PRODUCERS + B_PRODUCERS + CONSUMERS) * WARPSIZE) {
            _tsr_consumer<CONSUMERS, B_LANES, QSIZE>(&A_buffer[0], &B_buffer[0], D + curr_n, scale_tensor[0], &queue[0],
                                                     index, p_state, role_id, n, dropped_rows, dropped_cols, k,
                                                     k_blocks, scratch + curr_n, curr_n);
            __syncthreads();
            if (threadIdx.x == (A_PRODUCERS + B_PRODUCERS) * WARPSIZE) {
                // Send the result around the ring, only one thread needs to do this
                //printf("Rank %d: Sending data to %d\n", rank, peerSendRank);
                //for (int row = 0; row < 8 /* 8 rows tile */; row++) {
                //    right.put(2*(curr_n + row * n), (16*B_LANES) * 2);
                //}
                //printf("Rank %d: Received data from %d\n", rank, peerRecvRank);
            }
        }
    }
    deviceSyncer.sync(gridDim.x, -1);
    if (threadIdx.x == (A_PRODUCERS + B_PRODUCERS) * WARPSIZE && blockIdx.x == 0) {
        // Send the result around the ring, only one thread needs to do this.

        if(!is_capturing) {
            right.put(0, m*n*2);
            right.signal();
            right.flush();
            left.wait();
        }
        //printf("Rank %d: Received data from %d\n", rank, peerRecvRank);
    }
    deviceSyncer.sync(gridDim.x, -1);
}

void skinny_gemm(torch::Tensor& A, torch::Tensor& B, torch::Tensor& D, torch::Tensor& scale_tensor, int64_t b_lanes,
                 int64_t split_k, const int rank, const int world_size, uint8_t* buff_a, uint8_t* buff_b, cudaEvent_t lock, bool is_capturing) {
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);

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
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel (branched on B_LANES)
    //cudaEventSynchronize(lock);

    cudaStreamCaptureStatus status;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) {
        printf("Error from cudaStreamIsCapturing: %s\n", cudaGetErrorString(err));
    }
    //bool is_capturing = (status != hipStreamCaptureStatusNone);

    //printf("capturing state: %d\n", status);
    //printf("current stream: %p\n", stream);
    //printf("capturing: %d\n", is_capturing);

    switch (b_lanes) {
        case 2:
            launch_tsr(2, 3, 8, 4, 5);
        case 3:
            launch_tsr(3, 3, 5, 2, 4);  // Perforamnce on MI300: 8_13312_16384:57.54
        case 4:
            launch_tsr(4, 2, 6, 3, 3);  // Perforamnce on MI300: 8_16384_6656:29.5
        case 5:
            launch_tsr(5, 2, 6, 2, 2);
        default:
            break;
    }

    int threads = 256;
    int blocks = (D.numel() / 2 + threads - 1) / threads;
    //printf("Allreduce: sizes: %d %d %d %d %d %d\n", D.numel(), m, n, k, m*n, m*n*2);
    vectorized_reduce_inplace<<<blocks, threads, 0, stream>>>(D_, buff_a_, buff_b_, D.numel(), rank, world_size, is_capturing);
    //cudaEventRecord(lock, stream);
    //cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();
}
