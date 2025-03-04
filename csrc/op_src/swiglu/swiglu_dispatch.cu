#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include "op_src/swiglu/swiglu_scalar.cu"
#include "op_src/swiglu/swiglu_vectorized.cu"

void swiglu(torch::Tensor& gate_up,  // Shape: [m, 2*n] / Layout: row-major / Dtype: fp16
            torch::Tensor& scale_tensor,  // Shape: [1,    ] / Layout: row-major / Dtype: fp32
            torch::Tensor& output,    // Shape: [m,   n] / Layout: row-major / Dtype: fp8
            torch::Tensor& next_buffer,   // Shape: [m,   o] / Layout: dont-care / Dtype: fp16
            int64_t num_threads,
            bool force_scalar) {
    // Retrieve shapes
    const int rows = gate_up.size(0);
    const int output_cols = gate_up.size(1) / 2;
    const int buffer_cols = next_buffer.size(1);
    // Activate device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(gate_up));

    // Prepare kernel launch arguments
    dim3 grid(1);
    dim3 block(num_threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Check tensors alignment
    bool vectorized_available = IS_16B_ALIGNED(gate_up) && IS_8B_ALIGNED(output) && (!force_scalar);

    // Right now, output is always fp8e3m4fnuz
    vectorized_available = vectorized_available && (buffer_cols % 8 == 0);

    // Launch kernel
    if (vectorized_available) {
        grid.x = CDIV((output_cols * rows) / 8, num_threads);
        _swiglu_vectorized<<<grid, block, 0, stream>>>((half*)gate_up.data_ptr(), (float*)scale_tensor.data_ptr(),
                                                (__hip_fp8_storage_t*)output.data_ptr(),
                                                (half*)next_buffer.data_ptr(), rows, output_cols, buffer_cols);
    } else {
        grid.x = CDIV(output_cols * rows, num_threads);
        _swiglu_scalar<<<grid, block, 0, stream>>>((half*)gate_up.data_ptr(), (float*)scale_tensor.data_ptr(),
                                                (__hip_fp8_storage_t*)output.data_ptr(),
                                                (half*)next_buffer.data_ptr(), rows, output_cols, buffer_cols);
    }

    // Case: output is fp16 (TODO)
    // if (output.dtype() == torch::kFloat16) {
        // 
        // vectorized_available = vectorized_available && IS_16B_ALIGNED(output);

        // if (vectorized_available) {
        //     _residual_rms_vectorized<half2, false><<<grid, block, 0, stream>>>(
        //         (half*)gate_up.data_ptr(), (half*)residual.data_ptr(), (half*)weight.data_ptr(), (float*)NULL,
        //         (half2*)output.data_ptr(), (half*)NULL, epsilon, cols, 0);
        // } else {
        //     _residual_rms_scalar<half, false><<<grid, block, 0, stream>>>(
        //         (half*)gate_up.data_ptr(), (half*)residual.data_ptr(), (half*)weight.data_ptr(), (float*)NULL,
        //         (half*)output.data_ptr(), (half*)NULL, epsilon, cols, 0);
        // }
}

