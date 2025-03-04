#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "utils/macros.h"

__global__ void _swiglu_v2(const half* __restrict__ gate_up_proj, const float* __restrict__ scale_tensor,
                           __hip_fp8_storage_t* __restrict__ swiglu_out, int rows, int hidden_dim) {
    static constexpr int elems_per_threads = 8;

    // Advance pointers according to the position of the thread in the grid
    const int threads_per_row = hidden_dim / elems_per_threads;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    const int row_id = thread_id / threads_per_row;
    const int col_id = thread_id % threads_per_row;

    if (row_id >= rows) {
        return;
    }

    const half* __restrict__ gate_ptr = gate_up_proj + (row_id * 2 * hidden_dim) + col_id * elems_per_threads;
    const half* __restrict__ up_ptr = gate_ptr + hidden_dim;
    __hip_fp8x2_storage_t* __restrict__ swiglu_out_ptr = reinterpret_cast<__hip_fp8x2_storage_t*>(
        swiglu_out + (row_id * hidden_dim) + col_id * elems_per_threads);

    // Prepare swiglu loop
    half2 gate_regs[elems_per_threads/2];
    half2 up_regs[elems_per_threads/2];
    __hip_fp8x2_storage_t swiglu_out_regs[elems_per_threads/2];

    float inv_scale = 1 / scale_tensor[0];

    // Load gate and up elements using half2 (packed)
    #pragma unroll
    for (int j = 0; j < elems_per_threads/2; j++) {
        gate_regs[j] = reinterpret_cast<const half2*>(gate_ptr)[j];
        up_regs[j] = reinterpret_cast<const half2*>(up_ptr)[j];
    }

    // Compute SwiGLU and fp8 pre-conversion using float2 (packed)
    #pragma unroll
    for (int j = 0; j < elems_per_threads/2; j++) {
        // Convert half2 to float2
        float2 gate_f32x2 = __half22float2(gate_regs[j]);
        
        // Apply sigmoid to gate values using exp2 instead of expf
        // log2(e) ≈ 1.44269504089f, so we scale the input by this factor
        gate_f32x2.x = gate_f32x2.x / (1 + exp2(-gate_f32x2.x * 1.44269504089f));
        gate_f32x2.y = gate_f32x2.y / (1 + exp2(-gate_f32x2.y * 1.44269504089f));
        
        // Multiply by up projection and scale
        float2 up_f32x2 = __half22float2(up_regs[j]);
        gate_f32x2 *= up_f32x2;
        gate_f32x2 *= inv_scale;
        
        // Clamp values
        gate_f32x2.x = __builtin_amdgcn_fmed3f(gate_f32x2.x, 448.0, -448.0);
        gate_f32x2.y = __builtin_amdgcn_fmed3f(gate_f32x2.y, 448.0, -448.0);

        // Store fp8x2
        swiglu_out_regs[j] = __hip_cvt_float2_to_fp8x2(gate_f32x2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    }

    // Store fp8x2
    #pragma unroll
    for (int j = 0; j < elems_per_threads/2; j++) {
        swiglu_out_ptr[j] = swiglu_out_regs[j];
    }
//     // Initialize next buffer
//     for (int i = elems_per_threads * threadIdx.x; i < buffer_cols; i += elems_per_threads * blockDim.x) {
// #pragma unroll
//         for (int j = 0; j < elems_per_threads; j++) {
//             next_buffer[j] = 0;
//         }
//         next_buffer += elems_per_threads * blockDim.x;
//     }
}
