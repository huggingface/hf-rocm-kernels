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
    __shared__ half _smem[32000];

    // Advance pointers according to the position of the thread in the grid
    input += blockIdx.x * cols + elems_per_load * threadIdx.x;
    residual += blockIdx.x * cols + elems_per_load * threadIdx.x;
    weight += elems_per_load * threadIdx.x;
    output += (blockIdx.x * cols + elems_per_load * threadIdx.x) / 2;

    half* residual_start = residual;
    half* residual_smem_buffer = &_smem[0] + elems_per_load * threadIdx.x;

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
        }

        // 128-bits smem store
        #pragma unroll
        for (int j = 0; j < elems_per_load; j++) {
            residual_smem_buffer[j] = residual_buffer[j];
        }

        // Advance pointers
        input += loop_stride;
        residual += loop_stride;
        residual_smem_buffer += loop_stride;
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
    half weight_buffer[elems_per_load];
    T output_buffer[elems_per_load / 2];

    // Get inverse scale (only for fp8)
    float inv_scale = 1.0f;
    if constexpr (std::is_same_v<T, __hip_fp8x2_storage_t>) {
        inv_scale = 1 / scale_tensor[0];
    }

    residual = residual_start;
    residual_smem_buffer = &_smem[0] + elems_per_load * threadIdx.x;

    for (int i = 0; i < iterations; i++) {
        // 128-bits loads
        #pragma unroll
        for (int j = 0; j < elems_per_load; j++) {
            residual_buffer[j] = residual_smem_buffer[j];
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
                tmp_float2.x = (float)residual_buffer[2 * j] * shared_normalizer;
                tmp_float2.x = (float)((half)(tmp_float2.x) * weight_buffer[2 * j]);
                tmp_float2.x *= inv_scale;
                FP8_CLAMP(tmp_float2.x, float);

                tmp_float2.y = (float)residual_buffer[2 * j + 1] * shared_normalizer;
                tmp_float2.y = (float)((half)(tmp_float2.y) * weight_buffer[2 * j + 1]);
                tmp_float2.y *= inv_scale;
                FP8_CLAMP(tmp_float2.y, float);

                output_buffer[j] = __hip_cvt_float2_to_fp8x2(tmp_float2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
            }

            // Output is fp16
            if constexpr (std::is_same_v<T, half2>) {
                tmp_float2.x = (float)residual_buffer[2 * j];
                tmp_float2.y = (float)residual_buffer[2 * j + 1];
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
        // 128b store
        #pragma unroll
        for (int j = 0; j < elems_per_load; j++) {
            residual[j] = residual_buffer[j];
        }


        // Advance pointers
        residual += loop_stride;
        residual_smem_buffer += loop_stride;
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

//   Nb. rows    Ref (μs)    Pointwise (μs)    Vectorized (μs)
// ----------  ----------  ----------------  -----------------
//          1     40.6864           10.4857            4.8905
//          2     42.8676           10.5499            5.04421
//          4     43.7978           10.5729            5.05962
//          8     44.0237           10.6909            5.10061
//         16     47.1026           10.7823            5.19516
//         32     56.3393           11.0192            5.45101
//         64     74.0383           14.0895            5.86153
//        128     98.3725           15.2012            6.59527
//        256    119.426            27.7393           11.5191
