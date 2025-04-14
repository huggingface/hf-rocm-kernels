#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>



using fp8 = __hip_fp8_storage_t;
using fp8_4 = int;
using fp8x8 = __attribute__( (__vector_size__(8 * sizeof(fp8)) )) fp8;
using fp8x16 = __attribute__( (__vector_size__(16 * sizeof(fp8)) )) fp8;
using fp8_4x2 = __attribute__( (__vector_size__(2 * sizeof(int)) )) int;
using fp8_4x4 = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;
using f32x4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
using f32x16 = __attribute__( (__vector_size__(16 * sizeof(float)) )) float;
using uint8 = unsigned char;
using uint16 = unsigned short;
using uint32 = unsigned int;
using uint64 = unsigned long long;

// Absolute constants
#define WARPSIZE 64
#define NB_BANKS 32
#define CU 304

// Parameters
#define A_LANES_ 2
#define B_LANES_ 3
#define QSIZE_ 3
#define OP_M_ 16
#define OPS_ 8

#define A_PRODUCERS_ 2
#define B_PRODUCERS_ 6
#define CONSUMERS_ 3

#define SK_ 1

// Macros
#define NUM_WARPTILE_K(k, split_k) (((k / WARPTILE_K) / split_k))
#define CDIV(a, b) ((a + b - 1) / (b))
#define DB8TO32(x) (float)__hip_cvt_fp8_to_halfraw(x, __HIP_E4M3_FNUZ).data

// Ids
__device__ __forceinline__ int get_warp_id() { return threadIdx.x >> 6; }
__device__ __forceinline__ int get_lane_id() { return threadIdx.x & 0x3f; }
__device__ __forceinline__ int get_thread_id() { return threadIdx.x % WARPSIZE; }
