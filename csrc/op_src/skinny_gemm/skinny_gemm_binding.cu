// Torch-related
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__
#define SKINNY_GEMM_FULL_COMPILE

// Main function
#include "./skinny_gemm_caller.cu"


// Torch bind of the main function
int skinny_gemm_tb(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D,
    torch::Tensor& scale_tensor,
    int64_t split_k,
    int64_t A_producers,
    int64_t B_producers,
    int64_t consumers,
    int64_t a_lanes,
    int64_t b_lanes,
    int64_t qsize,
    int64_t op_m,
    int64_t ops
) {
    // Retrieve pointers
    const fp8* __restrict__ A_ = (const fp8* __restrict__) A.data_ptr();
    const fp8* __restrict__ B_ = (const fp8* __restrict__) B.data_ptr();
    half* __restrict__ D_ = (half* __restrict__) D.data_ptr();
    float* __restrict__ scale_tensor_ = (float* __restrict__) scale_tensor.data_ptr();

    // Retrieve shapes
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);
    const int b_stride = B.stride(1);

    // TODO: Depending on the number of rows in A, we have different OP_M and A_LANES

    // Retrieve stream
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

    // Launch kernel (branched on B_LANES)
    return skinny_gemm(
        A_, B_, D_, scale_tensor_,
        m, n, k, b_stride, split_k,
        A_producers, B_producers, consumers,
        a_lanes, b_lanes, qsize, op_m, ops,
        stream
    );
}
