#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "utils/macros.h"

__global__ void _swiglu_v0(
    const half* __restrict__ gate_up_proj, 
    const float* __restrict__ scale_tensor,
    __hip_fp8_storage_t* __restrict__ swiglu_out, 
    half* __restrict__ next_buffer, 
    int hidden_dim, 
    int buffer_cols
) {
    // Advance pointers according to the position of the thread in the grid
    gate_up_proj += blockIdx.x * 2 * hidden_dim;
    swiglu_out += blockIdx.x * hidden_dim;
    next_buffer += blockIdx.x * buffer_cols;

    // Swiglu loop
    float x;
    float inv_scale = 1 / scale_tensor[0];

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        x = (float) gate_up_proj[i];
        x = x / (1 + expf(-x));
        x = x * (float) gate_up_proj[i + hidden_dim];
        x = x * inv_scale;
        FP8_CLAMP(x, float);
        swiglu_out[i] = __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    }

    // Initialize next buffer
    for (int i = threadIdx.x; i < buffer_cols; i += blockDim.x) {
        next_buffer[i] = 0;
    }
}
