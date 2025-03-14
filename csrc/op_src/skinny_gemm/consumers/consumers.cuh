#include "./../utils.cuh"
#include "./smfma_16x16x64_consumer.cu"
#include "./mfma_16x16x32_consumer.cu"

template<int CONSUMERS, int B_LANES, int QSIZE>
void __device__ consume_tiles_sparse_16x16x64(
    fp8* A_buffer,
    fp8* B_buffer,
    half* D,
    float scale,
    int* queue,
    int &index,
    int &p_state,
    int &role_id,
    const int dropped_rows,
    const int dropped_cols,
    const int n,
    const int k,
    const int k_blocks
);

template<int CONSUMERS, int B_LANES, int QSIZE>
void __device__ consume_tiles_dense_16x16x32(
    fp8* A_buffer,
    fp8* B_buffer,
    half* D,
    float scale,
    int* queue,
    int &index,
    int &p_state,
    int &role_id,
    const int dropped_rows,
    const int dropped_cols,
    const int n,
    const int k,
    const int k_blocks
);
