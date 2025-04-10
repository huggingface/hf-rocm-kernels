#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "../../utils/macros.h"

__device__ void initialize_buffer_scalar(half* __restrict__ next_buffer, int rows, int buffer_cols) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int buffer_elems_per_thread = CDIV(rows * buffer_cols, blockDim.x * gridDim.x);

    int offs = thread_id * buffer_elems_per_thread;
    offs = min(offs, rows * buffer_cols - buffer_elems_per_thread);
    next_buffer += offs;

    for (int i = 0; i < buffer_elems_per_thread; i++) {
        next_buffer[i] = (half)0;
    }
}

__global__ void _swiglu_scalar(const half* __restrict__ gate_up, const float* __restrict__ scale_tensor,
                               __hip_fp8_storage_t* __restrict__ output, half* __restrict__ next_buffer, int rows,
                               int output_cols, int buffer_cols) {
    // Advance pointers according to the position of the thread in the grid
    const int elems_per_threads = CDIV(rows * output_cols, blockDim.x * gridDim.x);
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int offs = thread_id * elems_per_threads;

    // Prepare swiglu loop
    float inv_scale = 1 / scale_tensor[0];

    // Swiglu loop
    for (int i = 0; i < elems_per_threads; i++) {
        int row_id = (offs + i) / output_cols;
        int col_id = (offs + i) % output_cols;

        if (row_id >= rows) {
            initialize_buffer_scalar(next_buffer, rows, buffer_cols);
            return;
        }

        const half* __restrict__ gate_ptr = gate_up + (row_id * 2 * output_cols) + col_id;
        const half* __restrict__ up_ptr = gate_ptr + output_cols;
        __hip_fp8_storage_t* __restrict__ output_ptr = output + (row_id * output_cols) + col_id;

        half gate = gate_ptr[0];
        half up = up_ptr[0];

        float gate_f32 = __half2float(gate);
        float up_f32 = __half2float(up);

        gate_f32 = gate_f32 / (1 + exp2(-gate_f32 * 1.44269504089f));
        gate_f32 *= up_f32;
        gate_f32 *= inv_scale;

        gate_f32 = __builtin_amdgcn_fmed3f(gate_f32, 448.0, -448.0);

        output_ptr[0] = __hip_cvt_float_to_fp8(gate_f32, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    }

    // Initialize next buffer
    initialize_buffer_scalar(next_buffer, rows, buffer_cols);
}
