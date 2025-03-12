#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "utils/macros.h"

__device__ void initialize_buffer_vectorized(half* __restrict__ next_buffer, int rows, int buffer_cols) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int buffer_elems_per_thread = CDIV(rows * buffer_cols, blockDim.x * gridDim.x);
    const int chunks_of_8 = CDIV(buffer_elems_per_thread, 8);

    for (int i = 0; i < chunks_of_8; i++) {
        int offs = 8 * (thread_id * chunks_of_8 + i);
        half* buffer_ptr = next_buffer + (offs % (rows * buffer_cols));

#pragma unroll
        for (int j = 0; j < 8; j++) {
            buffer_ptr[j] = (half)0;
        }
    }
}

__global__ void _swiglu_vectorized(const half* __restrict__ gate_up, const float* __restrict__ scale_tensor,
                                   __hip_fp8_storage_t* __restrict__ output, half* __restrict__ next_buffer, int rows,
                                   int output_cols, int buffer_cols) {
    static constexpr int elems_per_threads = 8;

    // Advance pointers according to the position of the thread in the grid
    const int threads_per_row = output_cols / elems_per_threads;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    const int row_id = thread_id / threads_per_row;
    const int col_id = thread_id % threads_per_row;

    if (row_id >= rows) {
        initialize_buffer_vectorized(next_buffer, rows, buffer_cols);
        return;
    }

    const half* __restrict__ gate_ptr = gate_up + (row_id * 2 * output_cols) + col_id * elems_per_threads;
    const half* __restrict__ up_ptr = gate_ptr + output_cols;
    __hip_fp8x2_storage_t* __restrict__ output_ptr =
        reinterpret_cast<__hip_fp8x2_storage_t*>(output + (row_id * output_cols) + col_id * elems_per_threads);

    // Prepare swiglu loop
    half2 gate_regs[elems_per_threads / 2];
    half2 up_regs[elems_per_threads / 2];
    __hip_fp8x2_storage_t output_regs[elems_per_threads / 2];

    // Add protection against division by zero
    float scale_value = scale_tensor[0];
    float inv_scale = (scale_value != 0.0f) ? (1.0f / scale_value) : 1.0f;

// Load gate and up elements using half2 (packed)
#pragma unroll
    for (int j = 0; j < elems_per_threads / 2; j++) {
        gate_regs[j] = reinterpret_cast<const half2*>(gate_ptr)[j];
        up_regs[j] = reinterpret_cast<const half2*>(up_ptr)[j];
    }

// Compute SwiGLU and fp8 pre-conversion using float2 (packed)
#pragma unroll
    for (int j = 0; j < elems_per_threads / 2; j++) {
        // Convert half2 to float2
        float2 gate_f32x2 = __half22float2(gate_regs[j]);

        // Apply sigmoid to gate values using exp2 instead of expf
        // log2(e) ≈ 1.44269504089f, so we scale the gate_up by this factor
        gate_f32x2.x = gate_f32x2.x / (1 + exp2(-gate_f32x2.x * 1.44269504089f));
        gate_f32x2.y = gate_f32x2.y / (1 + exp2(-gate_f32x2.y * 1.44269504089f));

        // Multiply by up projection and scale
        float2 up_f32x2 = __half22float2(up_regs[j]);
        gate_f32x2 *= up_f32x2;
        gate_f32x2 *= inv_scale;

        // Clamp values
        gate_f32x2.x = __builtin_amdgcn_fmed3f(gate_f32x2.x, FP8_MAX, -FP8_MAX);
        gate_f32x2.y = __builtin_amdgcn_fmed3f(gate_f32x2.y, FP8_MAX, -FP8_MAX);

        // Store fp8x2
        output_regs[j] = __hip_cvt_float2_to_fp8x2(gate_f32x2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    }

// Store fp8x2
#pragma unroll
    for (int j = 0; j < elems_per_threads / 2; j++) {
        output_ptr[j] = output_regs[j];
    }

    // Initialize next buffer
    initialize_buffer_vectorized(next_buffer, rows, buffer_cols);
}
