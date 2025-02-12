#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "utils/macros.h"

__global__ void _residual_rms_v0(const half* __restrict__ input, half* __restrict__ residual,
                                 const half* __restrict__ weight, const float* __restrict__ scale_tensor,
                                 __hip_fp8_storage_t* __restrict__ output, half* __restrict__ next_buffer, 
                                 const float epsilon, const int cols, const int buffer_cols) {
    // Advance pointers according to the position of the thread in the grid
    input += blockIdx.x * cols;
    residual += blockIdx.x * cols;
    output += blockIdx.x * cols;

    // Residual connection: inplace add of input to residual, accumulate norm along the way
    float variance = 0.0f;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        half z = input[i];
        z += residual[i];
        float x = (float)z;
        variance += (x * x);
        residual[i] = z;
    }
    variance /= cols;

    // Block reduce to compute the total norm
    __shared__ float shared_normalizer;
    using BlockReduce = hipcub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;

    variance = BlockReduce(reduceStore).Reduce(variance, hipcub::Sum{}, blockDim.x);
    if (threadIdx.x == 0) {
        shared_normalizer = rsqrtf(variance + epsilon);
    }
    __syncthreads();

    // Normalize and convert
    float inv_scale = 1 / scale_tensor[0];
    for (int idx = threadIdx.x; idx < cols; idx += blockDim.x) {
        float x = (float)residual[idx];
        half y = (half)(x * shared_normalizer);
        y = (y * weight[idx]);
        x = (float)y;
        x *= inv_scale;
        FP8_CLAMP(x, float);
        output[idx] = __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    }

    // Initialize next buffer
    next_buffer += blockIdx.x * buffer_cols;
    for (int i = threadIdx.x; i < buffer_cols; i+=blockDim.x) {
        next_buffer[i] = 0;
    }
}

#define LAUNCH_RESIDUAL_RMS_V0                                                                                       \
    (_residual_rms_v0<<<grid, block, 0, stream>>>((half*)input.data_ptr(), (half*)residual.data_ptr(),               \
                                                  (half*)weight.data_ptr(), (float*)scale_tensor.data_ptr(),        \
                                                   (__hip_fp8_storage_t*)output.data_ptr(), \
                                                   (__half*) next_buffer.data_ptr(), epsilon, cols, buffer_cols))
