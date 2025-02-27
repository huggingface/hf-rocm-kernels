#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "utils/macros.h"

template <typename T, bool clean_next_buffer>
__global__ void _residual_rms_scalar(const half* __restrict__ input, half* __restrict__ residual,
                                     const half* __restrict__ weight, const float* __restrict__ scale_tensor,
                                     T* __restrict__ output, half* __restrict__ next_buffer, const float epsilon,
                                     const int cols, const int buffer_cols) {
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

    // Get inverse scale (only for fp8)
    float inv_scale = 1.0f;
    if constexpr (std::is_same_v<T, __hip_fp8_storage_t>) {
        inv_scale = 1 / scale_tensor[0];
    }

    // Normalize and store
    for (int idx = threadIdx.x; idx < cols; idx += blockDim.x) {
        float x = (float)residual[idx];
        half y = (half)(x * shared_normalizer);
        y = (y * weight[idx]);

        if constexpr (std::is_same_v<T, __hip_fp8_storage_t>) {
            x = (float)y;
            x *= inv_scale;
            FP8_CLAMP(x, float);
            output[idx] = __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
        }
        if constexpr (std::is_same_v<T, half>) {
            output[idx] = y;
        }
    }

    // Initialize next buffer
    if constexpr (clean_next_buffer) {
        next_buffer += blockIdx.x * buffer_cols;
        for (int i = threadIdx.x; i < buffer_cols; i += blockDim.x) {
            next_buffer[i] = 0;
        }
    }
}
