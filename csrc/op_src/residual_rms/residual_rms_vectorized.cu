#include <torch/all.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp8.h>

#include "utils/macros.h"

template <typename T, bool clean_next_buffer>
__global__ void _residual_rms_vectorized(const half* __restrict__ input, half* __restrict__ residual,
                                         const half* __restrict__ weight, const float* __restrict__ scale_tensor,
                                         T* __restrict__ output,  // half2 or __hip_fp8x2_storage_t
                                         half* __restrict__ next_buffer, const float epsilon, const int cols,
                                         const int buffer_cols) {
    static constexpr int elems_per_load = 8;

    // Advance pointers according to the position of the thread in the grid
    input += blockIdx.x * cols + elems_per_load * threadIdx.x;
    residual += blockIdx.x * cols + elems_per_load * threadIdx.x;
    weight += elems_per_load * threadIdx.x;
    output += (blockIdx.x * cols + elems_per_load * threadIdx.x) / 2;
    half* residual_start = residual;

    // Residual connection: inplace add of input to residual, accumulate norm along the way
    float variance = 0.0f;
    float fp32_residual;
    half input_buffer[elems_per_load];
    half residual_buffer[elems_per_load];

    const int loop_stride = elems_per_load * blockDim.x;
    const int iterations = CDIV(cols - elems_per_load * threadIdx.x, loop_stride);
    for (int i = 0; i < iterations; i++) {
        // Load data using 128-bits loads
#pragma unroll
        for (int j = 0; j < elems_per_load; j++) {
            input_buffer[j] = input[j];
        }
#pragma unroll
        for (int j = 0; j < elems_per_load; j++) {
            residual_buffer[j] = residual[j];
        }

        // Add everything in the residual buffer and accumulate variance
#pragma unroll
        for (int j = 0; j < elems_per_load; j++) {
            residual_buffer[j] += input_buffer[j];
            float float_res = (float)residual_buffer[j];
            variance += float_res * float_res;
            // asm volatile(
            //     "v_pk_add_f16 %0, %2, %3\n\t"
            //     "v_dot2c_f32_f16 %1, %2, %2"
            //     : "=v"(residual_buffer[j]), "=v"(variance)
            //     : "0"(residual_buffer[j]), "v"(input_buffer[j])
            // );
        }

        // 128-bits store
#pragma unroll
        for (int j = 0; j < elems_per_load; j++) {
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
    half residual_buffer_[elems_per_load];
    half weight_buffer[elems_per_load];
    T output_buffer[elems_per_load / 2];

    // Get inverse scale (only for fp8)
    float inv_scale = 1.0f;
    if constexpr (std::is_same_v<T, __hip_fp8x2_storage_t>) {
        inv_scale = 1 / scale_tensor[0];
    }

    residual = residual_start;
    for (int i = 0; i < iterations; i++) {
// 128-bits loads
#pragma unroll
        for (int j = 0; j < elems_per_load; j++) {
            residual_buffer_[j] = residual[j];
        }
#pragma unroll
        for (int j = 0; j < elems_per_load; j++) {
            weight_buffer[j] = weight[j];
        }

// Compute and fill buffer
#pragma unroll
        for (int j = 0; j < elems_per_load / 2; j++) {
            // Output is fp8
            if constexpr (std::is_same_v<T, __hip_fp8x2_storage_t>) {
                tmp_float2.x = (float)residual_buffer_[2 * j] * shared_normalizer;
                tmp_float2.x = (float)((half)(tmp_float2.x) * weight_buffer[2 * j]);
                tmp_float2.x *= inv_scale;
                FP8_CLAMP(tmp_float2.x, float);

                tmp_float2.y = (float)residual_buffer_[2 * j + 1] * shared_normalizer;
                tmp_float2.y = (float)((half)(tmp_float2.y) * weight_buffer[2 * j + 1]);
                tmp_float2.y *= inv_scale;
                FP8_CLAMP(tmp_float2.y, float);

                output_buffer[j] = __hip_cvt_float2_to_fp8x2(tmp_float2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
            }

            // Output is fp16
            if constexpr (std::is_same_v<T, half2>) {
                tmp_float2.x = (float)residual_buffer_[2 * j];
                tmp_float2.y = (float)residual_buffer_[2 * j + 1];
                tmp_float2 *= shared_normalizer;
                half2 tmp = {(half)tmp_float2.x, (half)tmp_float2.y};
                tmp *= reinterpret_cast<const half2*>(weight_buffer)[j];
                output_buffer[j] = tmp;
            }
        }

// 64b store
#pragma unroll
        for (int j = 0; j < elems_per_load / 2; j++) {
            output[j] = output_buffer[j];
        }

        // Advance pointers
        residual += loop_stride;
        weight += loop_stride;
        output += loop_stride / 2;
    }

    // Initialize next buffer TODO: add this as a template (eventualy w/ vector granularity)
    if constexpr (clean_next_buffer) {
        next_buffer += blockIdx.x * buffer_cols;
        for (int i = elems_per_load * threadIdx.x; i < buffer_cols; i += elems_per_load * blockDim.x) {
#pragma unroll
            for (int j = 0; j < elems_per_load; j++) {
                next_buffer[i + j] = 0;
            }
        }
    }
}
