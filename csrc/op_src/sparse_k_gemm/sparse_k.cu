#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "utils/macros.h"

#define WARPSIZE 64
#define WARPTILE_M 8
#define WARPTILE_N 16
#define WARPTILE_K 64

using fp8 = __hip_fp8_storage_t;
using fp8x2 = short;
using fp8x4 = short;

using fp8x2_4 = __attribute__( (__vector_size__(4 * sizeof(short)) )) short;
using fp8x2_8 = __attribute__( (__vector_size__(8 * sizeof(short)) )) short;

using fp8x4_2 = __attribute__( (__vector_size__(2 * sizeof(int)) )) int;
using fp8x4_4 = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;

using f32x4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;

__device__ inline void sparse_8x16x64_load(
    const fp8x2* warptile_A,                     // Shape: [8, 64],  Layout: row-major
    fp8x2_4 &thread_buffer_A,
    const int stride_A,
    const fp8x4_4* warptile_B,                     // Shape: [64, 16], Layout: col-major
    fp8x4_4 &thread_buffer_B,
    const int stride_B,
    const int thread_id
) {
    // Compute the thread position in A according to the matrix layout AND accounting for the fact that A has 8 rows
    int row_A = thread_id % 8;
    int col_A = 16 * (thread_id / 16);
    // This threads covers: row_A, [col_A, col_A + 16[

    // Compute the thread sparsity indices
    const int thread_group = (thread_id % 16) / 8;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        thread_buffer_A[i] = warptile_A[(row_A * stride_A + col_A) / 2 + 2 * i + thread_group];
    }

    fp8x4_4 tB = warptile_B[(stride_B / 16) * (thread_id / 4) + (thread_id % 4)];

    int src_lane = (thread_id % 16) * 4 + (thread_id / 16);
    thread_buffer_B[0] = __shfl(tB[0], src_lane, WARPSIZE);
    thread_buffer_B[1] = __shfl(tB[1], src_lane, WARPSIZE);
    thread_buffer_B[2] = __shfl(tB[2], src_lane, WARPSIZE);
    thread_buffer_B[3] = __shfl(tB[3], src_lane, WARPSIZE);

}

__device__ void sparse_8x16x64_wgemm(
    const fp8x2* warptile_A,                     // Shape: [8, 64],  Layout: row-major
    const int stride_A,
    const fp8x4_4* warptile_B,                     // Shape: [64, 16], Layout: col-major
    const int stride_B,
    f32x4 &thread_buffer_D,
    const int thread_id
) {
    fp8x2_4 thread_buffer_A;
    fp8x4_4 thread_buffer_B;
    sparse_8x16x64_load(warptile_A, thread_buffer_A, stride_A, warptile_B, thread_buffer_B, stride_B, thread_id);

    // Compute the thread sparsity indices
    const int thread_group = (thread_id % 16) / 8;
    const int sparsity_indices = (thread_group) ? 0x0000EEEE : 0x00004444;

    // Compute
    thread_buffer_D = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
        reinterpret_cast<fp8x4_2>(thread_buffer_A),
        reinterpret_cast<fp8x4_4>(thread_buffer_B), 
        thread_buffer_D, 
        sparsity_indices, // src2
        7, // cbsz
        1  // abid
    );
    
}


__global__ void skinny_fp8_gemm(
    const fp8* A,
    const fp8* B,
    float* D,
    const int m,
    const int n,
    const int k,
    const int grid_m,
    const int grid_n
) {
    // Infer thread role
    const int thread_id = threadIdx.x % WARPSIZE;
    const int warp_id = threadIdx.x / WARPSIZE;
    const int warps_per_block = blockDim.x / WARPSIZE;
    // Infer block role
    const int block_x = blockIdx.x % grid_m;
    const int block_y = blockIdx.x / grid_m;

    // Prepare shared memory result buffer
    __shared__ float block_buffer_D[16 * 16]; // = 64 * 4
    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            block_buffer_D[4 * thread_id + i] = 0.0f;
        }
    }
    __syncthreads();

    // Account for the number of threads per blocktile
    int block_iterations = (k / WARPTILE_K);
    int iterations_per_warp = block_iterations / warps_per_block;
    // int warp_iter_offs, warp_start, warp_end;
    // warp_start = warp_id * iterations_per_warp - ( (warp_id % 2 == 0) ? 0 : 2 * (1 + (warp_id / 2)) );
    // warp_end = (warp_id + 1) * iterations_per_warp - ( (warp_id % 2 == 0) ? 2 * (1 + (warp_id / 2)) : 0 );
    // warp_end = (warp_id == (warps_per_block - 1)) ? block_iterations : warp_end;
    int warp_start = warp_id * iterations_per_warp;
    int warp_end = (warp_id == warps_per_block - 1) ? block_iterations : ((warp_id + 1) * iterations_per_warp);

    // Relocate pointers according to the thread's position
    A += (block_x * WARPTILE_M * k) + (warp_start * WARPTILE_K);
    B += (block_y * WARPTILE_N * k) + (warp_start * WARPTILE_K);
    D += (block_x * WARPTILE_M * n) + (block_y * WARPTILE_N);

    // Prepare MFMA arguments
    f32x4 thread_buffer_D = {0, 0, 0, 0};

    // K-wise loop
    for (int iter = warp_start; iter < warp_end; iter++) {

        // Matrix fuse-mul-add
        sparse_8x16x64_wgemm(
            reinterpret_cast<const fp8x2*>(A),
            k,
            reinterpret_cast<const fp8x4_4*>(B),
            k,
            thread_buffer_D,
            thread_id
        );
        A += WARPTILE_K;
        B += WARPTILE_K;

    }

    // Atomic addtion if the result buffer
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        atomicAdd(block_buffer_D + (4 * thread_id + i), thread_buffer_D[i]);
    }
    __syncthreads();

    // Leading half-warp takes care of the final reduce and store
    float tmp = 0.0f;
    if (threadIdx.x < 32) {
        const int row_D = 4 * (thread_id / 16);
        const int col_D = thread_id % 16;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            tmp = block_buffer_D[4 * thread_id + i];
            tmp += block_buffer_D[4 * (thread_id + 32) + i];
            D[(row_D + i) * n + col_D] = tmp;
        }        
    }
}


#define LAUNCH_skinny_fp8_gemm                       \
    skinny_fp8_gemm<<<grid, block, 0, stream>>>(     \
        (fp8*)A.data_ptr(),                                         \
        (fp8*)B.data_ptr(),                                         \
        (float*)D.data_ptr(),                                       \
        m, n, k, grid_m, grid_n)                                                    \

void sparse_k(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D,
    int64_t W
) {
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);
    
    int grid_m = m / 8;
    int grid_n = n / 16;
    dim3 grid(grid_m * grid_n, 1, 1);

    dim3 block(WARPSIZE * W, 1, 1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    LAUNCH_skinny_fp8_gemm;
}
