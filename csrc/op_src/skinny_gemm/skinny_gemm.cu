#include "./consumer.cu"
#include "./producer.cu"

template<int B_LANES, int A_PRODUCERS, int B_PRODUCERS, int CONSUMERS, int QSIZE>
void __global__ _tsr_kernel(
    const fp8* __restrict__ A, 
    const fp8* __restrict__ B,
    half* __restrict__ D,
    const float* scale_tensor,
    const int m,
    const int n,
    const int k,
    const int split_k
) {
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

    // Tiles loop
    int curr_n, curr_k, k_blocks, dropped_rows, dropped_cols;
    const int warptile_per_row = CDIV(n, (OP_N * B_LANES));
    const int tiles = warptile_per_row * split_k;
    const int tpw = max(CDIV(tiles, CU), 1);

    for (int warptile = (tpw * blockIdx.x); warptile < min(tiles, tpw * (blockIdx.x + 1)); warptile++) {

        // Compute tile position
        curr_n = (warptile % warptile_per_row) * (OP_N * B_LANES);
        curr_k = (warptile / warptile_per_row) * WARPTILE_K * K_BLOCKS(k, split_k);
        k_blocks = ((warptile / warptile_per_row) == (split_k - 1)) ? (k / WARPTILE_K) - (split_k - 1) * K_BLOCKS(k, split_k) : K_BLOCKS(k, split_k);

        // Account for column overflow
        dropped_rows = max(0, 0      + WARPTILE_M - m);
        dropped_cols = max(0, curr_n + (OP_N * B_LANES) - n);
        curr_n -= dropped_cols;

        // A producer warp
        if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
            _tsr_A_producer<A_PRODUCERS, B_LANES, QSIZE>(
                A + curr_k, 
                &A_buffer[0], 
                &queue[0],
                index, p_state, role_id,
                dropped_rows,
                k, k_blocks
            ); 
        } 
        // B producer warp
        else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
            _tsr_B_producer<B_PRODUCERS, B_LANES, QSIZE>(
                B + curr_n * k + curr_k,
                &B_buffer[0],
                &queue[1],
                index, p_state, role_id,
                k, k_blocks
            ); 
        }
        // Consumers warp
        else if (threadIdx.x < (A_PRODUCERS + B_PRODUCERS + CONSUMERS) * WARPSIZE) {
            _tsr_consumer<CONSUMERS, B_LANES, QSIZE>(
                &A_buffer[0],
                &B_buffer[0],
                D + curr_n,
                scale_tensor[0],
                &queue[0],
                index, p_state, role_id,
                n, 
                dropped_rows, dropped_cols,
                k, k_blocks
            );
        }
    }    
}

void async_gemm(
    const fp8* __restrict__ A, 
    const fp8* __restrict__ B,
    half* __restrict__ D, 
    const float* scale_tensor,
    const int m, 
    const int n, 
    const int k
) {
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

    int warps = 0;
    warps += A_PRODUCERS_;
    warps += B_PRODUCERS_;
    warps += CONSUMERS_;
    dim3 block(warps * WARPSIZE, 1, 1);

    // Launch kernel
    _tsr_kernel<B_LANES_, A_PRODUCERS_, B_PRODUCERS_, CONSUMERS_, QSIZE_><<<grid, block, 0, 0>>>(A, B, D, scale_tensor, m, n, k, SK);
}




void skinny_gemm(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D,
    torch::Tensor& scale_tensor,
    int64_t b_lanes,
    int64_t split_k
) {
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);
    
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
        case 2:
            block.x = WARPSIZE * (3 + 8 + 4);
            _tsr_kernel<2, 3, 8, 4, 5><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, split_k);
            break;
        case 3:
            block.x = WARPSIZE * (3 + 5 + 2); 
            _tsr_kernel<3, 3, 5, 2, 4><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, split_k);
            break; // 8_13312_16384:57.54
        case 4:
            block.x = WARPSIZE * (2 + 6 + 3);
            _tsr_kernel<4, 2, 6, 3, 3><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, split_k);
            break; // 8_16384_6656:29.5
        case 5:
            block.x = WARPSIZE * (2 + 6 + 2);
            _tsr_kernel<5, 2, 6, 2, 2><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, split_k);
            break;
        default:
            break;
    }
}
