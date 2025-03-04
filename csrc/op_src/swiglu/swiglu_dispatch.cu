#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include "op_src/swiglu/swiglu_v0.cu"
#include "op_src/swiglu/swiglu_v1.cu"
#include "op_src/swiglu/swiglu_v2.cu"

void swiglu(torch::Tensor& gate_up_proj,  // Shape: [m, 2*n] / Layout: row-major / Dtype: fp16
            torch::Tensor& scale_tensor,  // Shape: [1,    ] / Layout: row-major / Dtype: fp32
            torch::Tensor& swiglu_out,    // Shape: [m,   n] / Layout: row-major / Dtype: fp8
            torch::Tensor& next_buffer,   // Shape: [m,   o] / Layout: dont-care / Dtype: fp16
            int64_t mode,
            int64_t thread_per_block) {
    // Retrieve shapes
    const int rows = gate_up_proj.size(0);
    const int hidden_dim = gate_up_proj.size(1) / 2;
    const int buffer_cols = next_buffer.size(1);

    // Activate device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(gate_up_proj));

    // Prepare kernel launch arguments
    dim3 grid(rows);
    dim3 block(thread_per_block);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    switch (mode) {
        // TODO : propagate nb threads
        case 2:
            block.x = thread_per_block;
            grid.x = CDIV((hidden_dim * rows) / 8, block.x);
            _swiglu_v2<<<grid, block, 0, stream>>>((half*)gate_up_proj.data_ptr(), (float*)scale_tensor.data_ptr(),
                                                   (__hip_fp8_storage_t*)swiglu_out.data_ptr(), rows, hidden_dim);
                                                   // TODO - WARNING : add buffer cols back
            break;
        case 1:
            _swiglu_v1<<<grid, block, 0, stream>>>((half*)gate_up_proj.data_ptr(), (float*)scale_tensor.data_ptr(),
                                                   (__hip_fp8_storage_t*)swiglu_out.data_ptr(),
                                                   (half*)next_buffer.data_ptr(), hidden_dim, buffer_cols);
            break;
        default:
            _swiglu_v0<<<grid, block, 0, stream>>>((half*)gate_up_proj.data_ptr(), (float*)scale_tensor.data_ptr(),
                                                   (__hip_fp8_storage_t*)swiglu_out.data_ptr(),
                                                   (half*)next_buffer.data_ptr(), hidden_dim, buffer_cols);
            break;
    }
}

// TODO: reformat the dispatch as was done with RMS, with a scalar and a vectorized version
