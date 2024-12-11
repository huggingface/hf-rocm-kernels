#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "utils/macros.h"

#define WPT 8                           // WorkPerThreads
#define CDIV(a, b) ((a + b - 1) / (b))  // Ceiling division

__global__ void _residual_rms_v2(const half* __restrict__ input, half* __restrict__ residual,
                                 const half* __restrict__ weight, __hip_fp8_storage_t* __restrict__ output,
                                 const float epsilon, const float scale, const int cols) {
    // Advance pointers according to the position of the thread in the grid
    input += blockIdx.x * cols + WPT * threadIdx.x;
    residual += blockIdx.x * cols + WPT * threadIdx.x;
    weight += WPT * threadIdx.x;
    output += blockIdx.x * cols + WPT * threadIdx.x;
    half* residual_start = residual;

    // Residual connection: inplace add of input to residual, accumulate norm along the way
    float variance = 0.0f;
    float fp32_residual;
    half input_buffer[WPT];
    half residual_buffer[WPT];

    const int loop_stride = WPT * blockDim.x;
    const int iterations = CDIV(cols - WPT * threadIdx.x, loop_stride);
    for (int i = 0; i < iterations; i++) {
// Load data using 128-bits loads
#pragma unroll
        for (int j = 0; j < WPT; j++) {
            input_buffer[j] = input[j];
        }
#pragma unroll
        for (int j = 0; j < WPT; j++) {
            residual_buffer[j] = residual[j];
        }

// Add everything in the residual buffer and accumulate variance
#pragma unroll
        for (int j = 0; j < WPT; j++) {
            residual_buffer[j] += input_buffer[j];
            fp32_residual = (float)residual_buffer[j];
            variance += fp32_residual * fp32_residual;
        }

// 128-bits store
#pragma unroll
        for (int j = 0; j < WPT; j++) {
            residual[j] = residual_buffer[j];
        }

        // Advance pointers
        input += loop_stride;
        residual += loop_stride;
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
    float tmp_float;
    half residual_buffer_[WPT];
    half weight_buffer[WPT];
    __hip_fp8_storage_t fp8_buffer[WPT];

    residual = residual_start;
    for (int i = 0; i < iterations; i++) {
// 128-bits loads
#pragma unroll
        for (int j = 0; j < WPT; j++) {
            residual_buffer_[j] = residual[j];
        }
#pragma unroll
        for (int j = 0; j < WPT; j++) {
            weight_buffer[j] = weight[j];
        }

// Compute and fill buffer
#pragma unroll
        for (int j = 0; j < WPT; j++) {
            tmp_float = (float)residual_buffer_[j] * shared_normalizer;
            tmp_float = (float)((half)(tmp_float)*weight_buffer[j]);
            tmp_float *= scale;
            FP8_CLAMP(tmp_float, float);
            fp8_buffer[j] = __hip_cvt_float_to_fp8(tmp_float, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
        }

// 64b store
#pragma unroll
        for (int j = 0; j < WPT; j++) {
            output[j] = fp8_buffer[j];
        }

        // Advance pointers
        residual += loop_stride;
        weight += loop_stride;
        output += loop_stride;
    }
}

#define LAUNCH_RESIDUAL_RMS_V2                                                                                       \
    (_residual_rms_v2<<<grid, block, 0, stream>>>((half*)input.data_ptr(), (half*)residual.data_ptr(),               \
                                                  (half*)weight.data_ptr(), (__hip_fp8_storage_t*)output.data_ptr(), \
                                                  epsilon, scale, cols))
