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

#define M 16
#define N 16
#define K 64
#define WARPSIZE 64
#define WARPTILE_M 8
#define WARPTILE_N 16
#define WARPTILE_K 64
#define WpB 4
#define VSIZE 16

using fp8 = __hip_fp8_storage_t;

using fp8x2_4 = __attribute__( (__vector_size__(4 * sizeof(short)) )) short;
using fp8x2_8 = __attribute__( (__vector_size__(8 * sizeof(short)) )) short;

using fp8x4_2 = __attribute__( (__vector_size__(2 * sizeof(int)) )) int;
using fp8x4_4 = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;

using f32x4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;

__device__ void sparse_8x16x64_wgemm(
    const fp8x2_8* warptile_A,                     // Shape: [8, 64],  Layout: row-major
    fp8x2_4 thread_buffer_A,
    const int stride_A,
    const fp8x2_8* warptile_B,                     // Shape: [64, 16], Layout: col-major
    fp8x2_8 thread_buffer_B,
    const int stride_B,
    f32x4 &thread_buffer_D,
    const int thread_id
) {
    // Compute the thread position in A according to the matrix layout AND accounting for the fact that A has 8 rows
    int row_A = thread_id % 8;
    int col_A = 16 * (thread_id / 16);
    // This threads covers: row_A, [col_A, col_A + 16[
    
    // Compute the thread position in B according to the matrix layout
    int row_B = 16 * (thread_id / 16);
    int col_B = thread_id % 16;
    // This threads covers: [row_B, row_B + 16[, col_B

    // Compute the thread sparsity indices
    const int thread_group = (thread_id % 16) / 8;
    const int sparsity_indices = (thread_group) ? 0x0000EEEE : 0x00004444;

    // Load 16 coeffs of A in the buffer of B, because it has the right size
    thread_buffer_B = warptile_A[(row_A * stride_A + col_A) / 16];
    // Only keep the right elements for that thread
    thread_buffer_A[0] = thread_buffer_B[0 + thread_group];
    thread_buffer_A[1] = thread_buffer_B[2 + thread_group];
    thread_buffer_A[2] = thread_buffer_B[4 + thread_group];
    thread_buffer_A[3] = thread_buffer_B[6 + thread_group];

    // Now load the 16 coeffs of B in the buffer of B
    thread_buffer_B = warptile_B[(row_B + col_B * stride_B) / 16];

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


template<unsigned int W>
__global__ void skinny_fp8_gemm_Awpl1_pipe1(
    const fp8* A,
    const fp8* B,
    float* D,
    const int m,
    const int n,
    const int k
) {
    // This buffer contains all warptiles for A
    static constexpr int Awpl = 1;
    // typename static constexpr int W = 1;
    __shared__ fp8 warptiles_A[Awpl * 512]; // = Awpl * WARPTILE_M * WARPTILE_K = 2*8*64
    // BUG: if we set W = 16, this falls to 0 if Awpl is 1. As a temporary fix, we never set W to 16
    static constexpr int A_loads_per_thread = (Awpl * 8) / W; // = (Awpl * WARPTILE_M * WARPTILE_K) / (W * WARPSIZE)
    // This buffer contains all warptiles for B
    __shared__ fp8 warptiles_B[1024 * W]; // = WARPTILE_K * WARPTILE_N * W 
    static constexpr int B_loads_per_thread = 16; // = (WARPTILE_K * WARPTILE_N) / WARPSIZE

    // Relocate pointers according to the thread's position
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;
    const int block_offs = blockIdx.x * (blockDim.x / WARPSIZE) * WARPTILE_N;
    // for A
    const int row_A = (A_loads_per_thread * threadIdx.x) / WARPTILE_K;
    const int col_A = (A_loads_per_thread * threadIdx.x) % WARPTILE_K;
    A += row_A * k + col_A;
    fp8* warptiles_A_loadptr = warptiles_A + row_A * WARPTILE_K + col_A;
    // for B
    const int row_B = (B_loads_per_thread * threadIdx.x) % WARPTILE_K;
    const int col_B = (B_loads_per_thread * threadIdx.x) / WARPTILE_K;
    B += (col_B + blockIdx.x * (blockDim.x / WARPSIZE) * WARPTILE_N) * k + row_B;
    fp8* warptiles_B_loadptr = warptiles_B + col_B * WARPTILE_K + row_B;
    // for D
    D += (warp_id + blockIdx.x * (blockDim.x / WARPSIZE)) * WARPTILE_N;

    // Prepare MFMA arguments
    fp8x2_8* warptile_A_compute_ptr = reinterpret_cast<fp8x2_8*>(warptiles_A);
    fp8x2_8* warptile_B_compute_ptr = reinterpret_cast<fp8x2_8*>(warptiles_B + WARPTILE_K * WARPTILE_N * warp_id);
    fp8x2_4 thread_buffer_A;
    fp8x2_8 thread_buffer_B;
    f32x4 thread_buffer_D = {0, 0, 0, 0};

    // K-wise loop [ 1 | 2 | ... | iterations *  WARPTILE_K) ]
    const int iterations = k / WARPTILE_K;
    for (int iter = 0; iter < iterations; iter++) {
        
        // If we have no more A warptiles to consume, load the A warptiles
        if (iter % Awpl == 0) {
            #pragma unroll
            for (int i = 0; i < A_loads_per_thread; i++) {
                warptiles_A_loadptr[i] = A[i];
            }
            A += Awpl * WARPTILE_K;
            warptile_A_compute_ptr = reinterpret_cast<fp8x2_8*>(warptiles_A);
        // Otherwise advance to the next A warptile to consume
        } else {
            warptile_A_compute_ptr += WARPTILE_K / 16;
        }

        // Load the B warptiles
        #pragma unroll
        for (int i = 0; i < B_loads_per_thread; i++) {
            warptiles_B_loadptr[i] = B[i];
        }
        B += WARPTILE_K;

        __syncthreads();
        // Matrix fuse-mul-add
        sparse_8x16x64_wgemm(
            warptile_A_compute_ptr,
            thread_buffer_A,
            WARPTILE_K,
            warptile_B_compute_ptr,
            thread_buffer_B,
            WARPTILE_K,
            thread_buffer_D,
            thread_id
        );
        __syncthreads();
    }

    // Final store
    const int row_D = 4 * (thread_id / 16); // 4
    const int col_D = thread_id % 16;
    for (int i = 0; i < 4; ++i) {
        D[(row_D + i) * n + col_D] = thread_buffer_D[i]; // D[64 + col_D]
    }

}

#define LAUNCH_skinny_fp8_gemm_Awpl1_pipe1(W)                       \
    skinny_fp8_gemm_Awpl1_pipe1<W><<<grid, block, 0, stream>>>(     \
        (fp8*)A.data_ptr(),                                         \
        (fp8*)B.data_ptr(),                                         \
        (float*)D.data_ptr(),                                       \
        m, n, k)                                                    \

void sparse_k(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D,
    int64_t W
) {
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);

    dim3 grid(n / (WARPTILE_N * W), 1, 1);
    dim3 block(WARPSIZE * W, 1, 1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (W) {
        // case 1 is default
        case 2:
            LAUNCH_skinny_fp8_gemm_Awpl1_pipe1(2);
            break;
        case 4:
            LAUNCH_skinny_fp8_gemm_Awpl1_pipe1(4);
            break;
        case 8:
            LAUNCH_skinny_fp8_gemm_Awpl1_pipe1(8);
            break;
        // case 16:
        //     LAUNCH_skinny_fp8_gemm_Awpl1_pipe1(16);
        //     break;
        default:
            LAUNCH_skinny_fp8_gemm_Awpl1_pipe1(1);
            break;
    }
}
