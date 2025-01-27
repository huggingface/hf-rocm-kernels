#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "utils/macros.h"

#define WPT 8                           // WorkPerThreads
#define CDIV(a, b) ((a + b - 1) / (b))  // Ceiling division

__global__ void _residual_rms_v4(const __half2* __restrict__ input, __half2* __restrict__ residual,
                                 const __half2* __restrict__ weight, const float* __restrict__ scale_tensor,
                                 __hip_fp8x2_storage_t* __restrict__ output, half* __restrict__ next_buffer,
                                 const float epsilon, const int cols, const int buffer_cols) {
    // Advance pointers according to the position of the thread in the grid
    input += (blockIdx.x * cols + WPT * threadIdx.x) / 2;
    residual += (blockIdx.x * cols + WPT * threadIdx.x) / 2;
    weight += (WPT * threadIdx.x) / 2;
    output += (blockIdx.x * cols + WPT * threadIdx.x) / 2;

    // Residual connection: inplace add of input to residual, accumulate norm along the way
    float variance = 0.0f;
    float fp32_residual;
    __half2 input_buffer[WPT / 2];
    __half2 residual_buffer[WPT / 2];

    const int loop_stride = blockDim.x * (WPT / 2);
    const int iterations = CDIV(cols - WPT * threadIdx.x, 2 * loop_stride);
    for (int i = 0; i < iterations; i++) {
// Load data using 128-bits loads
#pragma unroll
        for (int j = 0; j < WPT / 2; j++) {
            input_buffer[j] = input[j];
        }
#pragma unroll
        for (int j = 0; j < WPT / 2; j++) {
            residual_buffer[j] = residual[j];
        }

// Residual connection and variance accumulation
#pragma unroll
        for (int j = 0; j < WPT / 2; j++) {
            asm volatile(
                "V_PK_ADD_F16 %0, %2, %3\n\t"
                "V_DOT2C_F32_F16 %1, %2, %2"
                : "=v"(residual_buffer[j]), "=v"(variance)
                : "0"(residual_buffer[j]), "v"(input_buffer[j]));
        }

// 128-bits store
#pragma unroll
        for (int j = 0; j < WPT / 2; j++) {
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
    float2 tmp_float2;
    __half2 residual_buffer_[WPT / 2];
    __half2 weight_buffer[WPT / 2];
    __hip_fp8x2_storage_t fp8x2_buffer[WPT / 2];
    float scale = scale_tensor[0];

    residual -= iterations * loop_stride;
    for (int i = 0; i < iterations; i++) {
// 128-bits loads
#pragma unroll
        for (int j = 0; j < WPT / 2; j++) {
            residual_buffer_[j] = residual[j];
        }
#pragma unroll
        for (int j = 0; j < WPT / 2; j++) {
            weight_buffer[j] = weight[j];
        }

// Compute and fill buffer
#pragma unroll
        for (int j = 0; j < WPT / 2; j++) {
            // .x
            tmp_float2.x = (float)residual_buffer_[j].x * shared_normalizer;
            tmp_float2.x = (float)((half)(tmp_float2.x) * weight_buffer[j].x);
            tmp_float2.x *= scale;
            FP8_CLAMP(tmp_float2.x, float);
            // .y
            tmp_float2.y = (float)residual_buffer_[j].y * shared_normalizer;
            tmp_float2.y = (float)((half)(tmp_float2.y) * weight_buffer[j].y);
            tmp_float2.y *= scale;
            FP8_CLAMP(tmp_float2.y, float);
            // convert
            fp8x2_buffer[j] = __hip_cvt_float2_to_fp8x2(tmp_float2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
        }

// 64b store
#pragma unroll
        for (int j = 0; j < WPT / 2; j++) {
            output[j] = fp8x2_buffer[j];
        }

        // Advance pointers
        residual += loop_stride;
        weight += loop_stride;
        output += loop_stride;
    }

    // Initialize next buffer
    next_buffer += blockIdx.x * buffer_cols;
    for (int i = 8 * threadIdx.x; i < buffer_cols; i += 8 * blockDim.x) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            next_buffer[i + j] = 0;
        }
    }
}

#define LAUNCH_RESIDUAL_RMS_V4                                                                               \
    (_residual_rms_v4<<<grid, block, 0, stream>>>((__half2*)input.data_ptr(), (__half2*)residual.data_ptr(), \
                                                  (__half2*)weight.data_ptr(), (float*)scale_tensor.data_ptr(),                               \
                                                  (__hip_fp8x2_storage_t*)output.data_ptr(), \
                                                  (__half*) next_buffer.data_ptr(), epsilon, cols, buffer_cols))
