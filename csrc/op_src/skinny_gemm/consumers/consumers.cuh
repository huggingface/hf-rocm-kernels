#include "./../utils.cuh"
#include "./smfma_16x16x64_consumer.cu"
#include "./mfma_16x16x32_consumer.cu"
#include "./mfma_32x32x16_consumer.cu"

template<int A_LANES, int B_LANES, int QSIZE, int OPS>
void __device__ consume_tiles_sparse_16x16x64(
    fp8* A_buffer,
    fp8* B_buffer,
    half* D,
    float scale,
    const int consumers,
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

template<int A_LANES, int B_LANES, int QSIZE, int OPS>
void __device__ consume_tiles_dense_16x16x32(
    fp8* A_buffer,
    fp8* B_buffer,
    half* D,
    float scale,
    const int consumers,
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

template<int A_LANES, int B_LANES, int QSIZE, int OPS>
void __device__ consume_tiles_dense_32x32x16(
    fp8* A_buffer,
    fp8* B_buffer,
    half* D,
    float scale,
    const int consumers,
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

template<int A_LANES, int B_LANES, int QSIZE, int OP_M, int OPS>
void __device__ consume_tiles(
    fp8* A_buffer,
    fp8* B_buffer,
    half* D,
    float scale,
    const int consumers,
    int* queue,
    int &index,
    int &p_state,
    int &role_id,
    const int dropped_rows,
    const int dropped_cols,
    const int n,
    const int k,
    const int k_blocks
) {
    if constexpr (OP_M == 8) {
        consume_tiles_sparse_16x16x64<A_LANES, B_LANES, QSIZE, OPS>(A_buffer, B_buffer, D, scale, consumers, queue, index, p_state,
                                                                 role_id, dropped_rows, dropped_cols, n, k, k_blocks);
    } else if constexpr (OP_M == 16) {
        consume_tiles_dense_16x16x32<A_LANES, B_LANES, QSIZE, OPS>(A_buffer, B_buffer, D, scale, consumers, queue, index, p_state,
                                                                role_id, dropped_rows, dropped_cols, n, k, k_blocks);
    } else if constexpr (OP_M == 32) {
        consume_tiles_dense_32x32x16<A_LANES, B_LANES, QSIZE, OPS>(A_buffer, B_buffer, D, scale, consumers, queue, index, p_state,
                                                                role_id, dropped_rows, dropped_cols, n, k, k_blocks);
    }
}
