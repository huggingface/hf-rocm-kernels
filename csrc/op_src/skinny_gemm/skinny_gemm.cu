#include "./consumers/consumers.cuh"
#include "./producers/4_half_tiles.cu"
#include "./producers/n_full_tiles.cu"

// TODO: try uint16 for the p state

template<int B_LANES, int A_PRODUCERS, int B_PRODUCERS, int CONSUMERS, int QSIZE>
void __global__ _skinny_gemm_kernel(
    const fp8* __restrict__ A,
    const fp8* __restrict__ B,
    half* __restrict__ D,
    const float* scale_tensor,
    const int m,
    const int n,
    const int k,
    const int B_stride,
    const int split_k
) {
    // Compile-time constants
    static constexpr int OP_M = 16;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 32;
    static constexpr int WARPTILE_M = OP_M;
    static constexpr int WARPTILE_N = OP_N * B_LANES;
    static constexpr int WARPTILE_K = OP_K * OPS;

    // Initialize shared queue
    __shared__ int queue[2 * B_LANES * QSIZE];
    if (threadIdx.x < 2 * B_LANES * QSIZE) {
        queue[threadIdx.x] = 0;
    }
    // Declare shared buffer
    __shared__ fp8 A_buffer[WARPTILE_M * WARPTILE_K * QSIZE];
    __shared__ fp8 B_buffer[WARPTILE_N * WARPTILE_K * QSIZE];
    __syncthreads();

    // Infer warp-specialization-related variables
    const int warp_id = get_warp_id();
    int role_id;
    if (warp_id < A_PRODUCERS) {
        role_id = warp_id;
    }
    else if (warp_id < A_PRODUCERS + B_PRODUCERS) {
        role_id = warp_id - A_PRODUCERS;
    }
    else {
        role_id = warp_id - A_PRODUCERS - B_PRODUCERS;
    }
    int index = role_id;
    int p_state = (warp_id >= A_PRODUCERS + B_PRODUCERS);

    // Tiles loop
    int curr_n, curr_k, k_blocks, dropped_rows, dropped_cols;
    const int tiles_per_row = CDIV(n, WARPTILE_N);
    const int total_tiles = tiles_per_row * split_k;
    const int tiles_per_block = CDIV(total_tiles, CU);

    const int stop_tile = min(total_tiles, tiles_per_block * (blockIdx.x + 1));
    for (
        int tile = tiles_per_block * blockIdx.x;
        tile < stop_tile;
        tile++
    ) {

        // Compute tile position
        curr_n = (tile % tiles_per_row) * WARPTILE_N;
        curr_k = (tile / tiles_per_row) * WARPTILE_K * NUM_WARPTILE_K(k, split_k);

        // Compute tile's K blocks (number of blocks along the K axis)
        k_blocks = (
            (tile / tiles_per_row) == (split_k - 1)
            ? (k / WARPTILE_K) - (split_k - 1) * NUM_WARPTILE_K(k, split_k)
            : NUM_WARPTILE_K(k, split_k)
        );

        // Account for column overflow
        dropped_rows = max(0, 0      + WARPTILE_M - m);
        dropped_cols = max(0, curr_n + WARPTILE_N - n);
        curr_n -= dropped_cols; // make sure we are in the bounds of B

        // A producer warp
        if (warp_id < A_PRODUCERS) {
            produce_n_full_tiles<A_PRODUCERS, 1, QSIZE, OP_K, OP_M, true>(
                A + curr_k,
                &A_buffer[0],
                &queue[0], B_LANES,
                index, p_state, role_id,
                dropped_rows,
                k, // A is supposed to always be contiguous
                k_blocks
            );
        }
        // B producer warp
        else if (warp_id < A_PRODUCERS + B_PRODUCERS) {
            // TODO: investigate the reuse parameter forB (true = faster but goes wrong because no sc1)
            produce_n_full_tiles<B_PRODUCERS, B_LANES, QSIZE, OP_K, OP_N, false>(
                B + curr_n * B_stride + curr_k,
                &B_buffer[0],
                &queue[1], 1,
                index, p_state, role_id,
                0, // for B, we simply offset the tiles to not load OOB
                B_stride,
                k_blocks
            );
        }
        // Consumers warp
        else if (warp_id < A_PRODUCERS + B_PRODUCERS + CONSUMERS) {
            consume_tiles_dense_16x16x32<CONSUMERS, B_LANES, QSIZE>(
                &A_buffer[0],
                &B_buffer[0],
                D + curr_n,
                scale_tensor[0],
                &queue[0],
                index, p_state, role_id,
                dropped_rows, dropped_cols,
                n, k, k_blocks
            );
        }
    }
}

void skinny_gemm_notorch(
    const fp8* __restrict__ A,
    const fp8* __restrict__ B,
    half* __restrict__ D,
    const float* scale_tensor,
    const int m,
    const int n,
    const int k
) {
    // Compile-time constants
    static constexpr int OP_M = 16;
    static constexpr int OP_K = 32;
    static constexpr int WARPTILE_M = OP_M;
    static constexpr int WARPTILE_K = OP_K * OPS;

    // Check shape
    if (m > WARPTILE_M) {
        std::cerr << "m = " << m << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
        exit(1);
    }
    if (k % WARPTILE_K != 0) {
        std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
        exit(1);
    }

    // Prepare kernel launch
    dim3 grid(CU, 1, 1);

    int warps = 0;
    warps += A_PRODUCERS_;
    warps += B_PRODUCERS_;
    warps += CONSUMERS_;
    dim3 block(warps * WARPSIZE, 1, 1);

    // Launch kernel
    _skinny_gemm_kernel<B_LANES_, A_PRODUCERS_, B_PRODUCERS_, CONSUMERS_, QSIZE_><<<grid, block, 0, 0>>>(A, B, D, scale_tensor, m, n, k, k, SK);
}




void skinny_gemm(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D,
    torch::Tensor& scale_tensor,
    int64_t b_lanes,
    int64_t split_k
) {
    // Compile-time constants
    static constexpr int OP_M = 16;
    static constexpr int OP_K = 32;
    static constexpr int WARPTILE_M = OP_M;
    static constexpr int WARPTILE_K = OP_K * OPS;

    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);

    const int B_stride = B.stride(1);

    const fp8* __restrict__ A_ = (const fp8* __restrict__) A.data_ptr();
    const fp8* __restrict__ B_ = (const fp8* __restrict__) B.data_ptr();
    half* __restrict__ D_ = (half* __restrict__) D.data_ptr();
    float* __restrict__ scale_tensor_ = (float* __restrict__) scale_tensor.data_ptr();

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
        case 1:
            block.x = WARPSIZE * (1 + 1 + 3);
            _skinny_gemm_kernel<1, 1, 1, 3, 3><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, B_stride, split_k);
            break;
        case 2:
            block.x = WARPSIZE * (1 + 1 + 3);
            _skinny_gemm_kernel<2, 1, 1, 3, 3><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, B_stride, split_k);
            break;
        case 3:
            block.x = WARPSIZE * (2 + 6 + 3);
            _skinny_gemm_kernel<3, 2, 6, 3, 3><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, B_stride, split_k);
            break;
        case 4:
            block.x = WARPSIZE * (2 + 6 + 3);
            _skinny_gemm_kernel<4, 2, 6, 3, 3><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, B_stride, split_k);
            break;
        case 5:
            block.x = WARPSIZE * (2 + 6 + 2);
            _skinny_gemm_kernel<5, 2, 6, 2, 2><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, B_stride, split_k);
            break;
        default:
            break;
    }
}
