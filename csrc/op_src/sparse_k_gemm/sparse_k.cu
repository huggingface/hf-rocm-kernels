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

using fp8x2_4 = __attribute__( (__vector_size__(4 * sizeof(short)) )) short;
using fp8x2_8 = __attribute__( (__vector_size__(8 * sizeof(short)) )) short;

using fp8x4_2 = __attribute__( (__vector_size__(2 * sizeof(int)) )) int;
using fp8x4_4 = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;

using f32x4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;

__global__ void sparse_8r2x8x64_matmul(
    const fp8x2_8* A,                     // Shape: [8, 64],  Layout: row-major
    const fp8x2_8* B,                     // Shape: [64, 16], Layout: col-major
    float* D                              // Shape: [16, 16], Layout: row-major
) {
    fp8x2_4 thread_buffer_A;
    fp8x2_8 thread_buffer_B;

    // Compute the thread position in A according to the matrix layout
    int row_A = threadIdx.x % 16;
    int col_A = 16 * (threadIdx.x / 16);
    // Account for the fact that A is a [8, 64] dense matrix
    row_A = row_A % 8;
    // This threads covers: row_A, [col_A, col_A + 16[
    
    // Compute the thread position in B according to the matrix layout
    int row_B = 16 * (threadIdx.x / 16);
    int col_B = threadIdx.x % 16;
    // This threads covers: [row_B, row_B + 16[, col_B

    // Compute the thread sparsity indices
    const int thread_group = (threadIdx.x % 16) / 8;
    const int sparsity_indices = (thread_group) ? 0x0000EEEE : 0x00004444;

    // Load 16 coeffs of A in the buffer of B, because it has the right size
    thread_buffer_B = A[(row_A * K + col_A) / 16];
    // Only keep the right elements for that thread
    thread_buffer_A[0] = thread_buffer_B[0 + thread_group];
    thread_buffer_A[1] = thread_buffer_B[2 + thread_group];
    thread_buffer_A[2] = thread_buffer_B[4 + thread_group];
    thread_buffer_A[3] = thread_buffer_B[6 + thread_group];

    // Now load the 16 coeffs of B in the buffer of B
    thread_buffer_B = B[(row_B + col_B * K) / 16];

    // Result registers
    f32x4 dmn = {0, 0, 0, 0};

    // Compute
    dmn = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
        reinterpret_cast<fp8x4_2>(thread_buffer_A), 
        reinterpret_cast<fp8x4_4>(thread_buffer_B), 
        dmn, 
        sparsity_indices, // src2
        7, // cbsz
        1  // abid
    );
    
    // Store
    const int row_D = 4 * (threadIdx.x / 16);
    const int col_D = threadIdx.x % 16;
    for (int i = 0; i < 4; ++i) {
        D[(row_D + i) * 16 + col_D] = dmn[i];
    }
}

void sparse_k(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D
) {
    dim3 grid(1, 1, 1);
    dim3 block(64, 1, 1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sparse_8r2x8x64_matmul<<<grid, block, 0, stream>>>(
        (fp8x2_8*)A.data_ptr(), 
        (fp8x2_8*)B.data_ptr(), 
        (float*)D.data_ptr()
    );
}
