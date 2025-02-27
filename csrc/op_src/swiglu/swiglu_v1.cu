#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "utils/macros.h"

__global__ void _swiglu_v1(const half* __restrict__ gate_up_proj, const float* __restrict__ scale_tensor,
                           __hip_fp8_storage_t* __restrict__ swiglu_out, half* __restrict__ next_buffer, int hidden_dim,
                           int buffer_cols) {
    static constexpr int elems_per_threads = 4;

    // Advance pointers according to the position of the thread in the grid
    gate_up_proj += (blockIdx.x * 2 * hidden_dim) + elems_per_threads * threadIdx.x;
    swiglu_out += (blockIdx.x * hidden_dim) + elems_per_threads * threadIdx.x;
    next_buffer += (blockIdx.x * buffer_cols) + elems_per_threads * threadIdx.x;

    // Prepare swiglu loop
    half gate_regs[elems_per_threads];
    half up_regs[elems_per_threads];
    float f32_acc[elems_per_threads];

    const half* __restrict__ gate_ptr = gate_up_proj;
    const half* __restrict__ up_ptr = gate_up_proj + hidden_dim;

    float inv_scale = 1 / scale_tensor[0];

    // Swiglu loop
    for (int i = elems_per_threads * threadIdx.x; i < hidden_dim; i += elems_per_threads * blockDim.x) {
// Load gate elements
#pragma unroll
        for (int j = 0; j < elems_per_threads; j++) {
            gate_regs[j] = gate_ptr[j];
        }
// Load up elements
#pragma unroll
        for (int j = 0; j < elems_per_threads; j++) {
            up_regs[j] = up_ptr[j];
        }

// Compute SwiGLU and fp8 pre-conversion
#pragma unroll
        for (int j = 0; j < elems_per_threads; j++) {
            f32_acc[j] = (float)gate_regs[j];
            f32_acc[j] = f32_acc[j] / (1 + expf(-f32_acc[j]));
            f32_acc[j] = f32_acc[j] * (float)up_regs[j];
            f32_acc[j] = f32_acc[j] * inv_scale;
            f32_acc[j] = std::clamp(f32_acc[j], -448.0f, 448.0f);
        }

// Convert and store
#pragma unroll
        for (int j = 0; j < elems_per_threads; j++) {
            swiglu_out[j] = __hip_cvt_float_to_fp8(f32_acc[j], __HIP_SATFINITE, __HIP_E4M3_FNUZ);
        }
        // Advance
        gate_ptr += elems_per_threads * blockDim.x;
        up_ptr += elems_per_threads * blockDim.x;
        swiglu_out += elems_per_threads * blockDim.x;
    }

    // Initialize next buffer
    for (int i = elems_per_threads * threadIdx.x; i < buffer_cols; i += elems_per_threads * blockDim.x) {
#pragma unroll
        for (int j = 0; j < elems_per_threads; j++) {
            next_buffer[j] = 0;
        }
        next_buffer += elems_per_threads * blockDim.x;
    }
}
