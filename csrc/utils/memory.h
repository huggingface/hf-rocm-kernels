#define IS_16B_ALIGNED(tensor) ( reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0)
