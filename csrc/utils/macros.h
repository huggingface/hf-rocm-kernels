#pragma once

#define WARPSIZE 64

#define FP8_CLAMP(x, type)                   \
    x = (x > (type)448.0) ? (type)448.0 : x; \
    x = (x < (type) - 448.0) ? (type) - 448.0 : x;
// TODO: reformat clamping

#define IS_8B_ALIGNED(tensor) (reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 4 == 0)
#define IS_16B_ALIGNED(tensor) (reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0)

#define CDIV(a, b) ((a + b - 1) / (b))

#define FP8_MAX 224.0f  // TODO: check if this or 448.0f
