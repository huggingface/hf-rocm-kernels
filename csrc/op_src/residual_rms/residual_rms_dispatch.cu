#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include "op_src/residual_rms/residual_rms_vectorized.cu"
#include "op_src/residual_rms/residual_rms_scalar.cu"

void residual_rms(torch::Tensor& input,         // Shape: [m, n] / Layout: row-major / Dtype: fp16
                  torch::Tensor& residual,      // Shape: [m, n] / Layout: row-major / Dtype: fp16
                  torch::Tensor& weight,        // Shape: [m,  ] / Layout: row-major / Dtype: fp16
                  torch::Tensor& scale_tensor,  // Shape: [1,  ] / Layout: row-major / Dtype: fp32
                  double epsilon,
                  torch::Tensor& output,       // Shape: [m, n] / Layout: row-major / Dtype: fp8 or fp16
                  torch::Tensor& next_buffer,  // Shape: [m, o] / Layout: dont-care / Dtype: fp16
                  int64_t num_threads, bool force_scalar) {
    // Retrieve shapes
    const int rows = input.size(0);
    const int cols = input.size(1);
    // Activate device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    // Prepare kernel launch arguments
    dim3 grid(rows);
    dim3 block(num_threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Check tensors alignment
    bool vectorized_available = IS_16B_ALIGNED(input) && IS_16B_ALIGNED(residual) && IS_16B_ALIGNED(weight);
    vectorized_available = vectorized_available && (!force_scalar) && (cols <= 32000);

    // Case: output is fp16
    if (output.dtype() == torch::kFloat16) {
        vectorized_available = vectorized_available && IS_16B_ALIGNED(output);

        if (vectorized_available) {
            _residual_rms_vectorized<half2, false><<<grid, block, 0, stream>>>(
                (half*)input.data_ptr(), (half*)residual.data_ptr(), (half*)weight.data_ptr(), (float*)NULL,
                (half2*)output.data_ptr(), (half*)NULL, epsilon, cols, 0);
        } else {
            _residual_rms_scalar<half, false><<<grid, block, 0, stream>>>(
                (half*)input.data_ptr(), (half*)residual.data_ptr(), (half*)weight.data_ptr(), (float*)NULL,
                (half*)output.data_ptr(), (half*)NULL, epsilon, cols, 0);
        }
    }

    // Case: output is fp8e3m4fnuz
    else {
        vectorized_available = vectorized_available && IS_8B_ALIGNED(output) && (next_buffer.size(1) % 8 == 0);

        // Launch kernel
        if (vectorized_available) {
            _residual_rms_vectorized<__hip_fp8x2_storage_t, true><<<grid, block, 0, stream>>>(
                (half*)input.data_ptr(), (half*)residual.data_ptr(), (half*)weight.data_ptr(),
                (float*)scale_tensor.data_ptr(), (__hip_fp8x2_storage_t*)output.data_ptr(),
                (half*)next_buffer.data_ptr(), epsilon, cols, next_buffer.size(1));
        } else {
            _residual_rms_scalar<__hip_fp8_storage_t, true><<<grid, block, 0, stream>>>(
                (half*)input.data_ptr(), (half*)residual.data_ptr(), (half*)weight.data_ptr(),
                (float*)scale_tensor.data_ptr(), (__hip_fp8_storage_t*)output.data_ptr(), (half*)next_buffer.data_ptr(),
                epsilon, cols, next_buffer.size(1));
        }
    }
}
