#pragma once

#define FP8_CLAMP(x, type) x = (x > (type) 448.0) ? (type) 448.0 : x; x = (x < (type) -448.0) ? (type) -448.0 : x;

#define IS_16B_ALIGNED(tensor) ( reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0)
// TODO: optimize clamping if possible
