#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include "op_src/residual_rms/residual_rms_v0.cu"
#include "op_src/residual_rms/residual_rms_v1.cu"

void residual_rms(torch::Tensor& input,     // Shape: [m, n] / Layout: row-major / Dtype: fp16
                  torch::Tensor& residual,  // Shape: [m, n] / Layout: row-major / Dtype: fp16
                  torch::Tensor& weight,    // Shape: [m,  ] / Layout: row-major / Dtype: fp16
                  torch::Tensor& output,    // Shape: [m, n] / Layout: row-major / Dtype: fp8
                  double epsilon, double scale, int64_t mode,
                  int64_t num_threads) {  // TODO: add fp16 output mode

    // Retrieve shapes
    const int rows = input.size(0);
    const int cols = input.size(1);
    // Activate device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    // Prepare kernel launch arguments
    dim3 grid(rows);
    dim3 block(num_threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    switch (mode)
    {
        case 1:
            LAUNCH_RESIDUAL_RMS_V1;
            break;
        default:
            LAUNCH_RESIDUAL_RMS_V0;
            break;
    }
}

/*
    Versions:
        0. non-vectorized version
        1. vectorizes loads and stores
*/
