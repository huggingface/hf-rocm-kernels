#include "./4_half_tiles.cu"
#include "./n_full_tiles.cu"

template<int A_LANES, int QSIZE, int OP_M, int OPS>
void __device__ produce_A_tiles(
    const fp8* __restrict__ A,
    fp8* A_buffer,
    const int A_producers,
    int* A_queue,
    int &index,
    int &p_state,
    int &role_id,
    int k,
    const int k_blocks,
    int dropped_rows // number of rows that are out of bounds for this producer
) {
    if constexpr (OP_M == 8) {
        produce_4_half_tiles<A_LANES, QSIZE, OPS>(A, A_buffer, A_producers, A_queue, index, p_state, role_id,
                                                  dropped_rows, k, k_blocks);
    } else {
        static constexpr int OP_K = (32 * 16) / OP_M;
        produce_n_full_tiles<A_LANES, QSIZE, OP_K, OP_M, OPS, true>(A, A_buffer, A_producers, A_queue, index, p_state,
                                                                      role_id, dropped_rows, k, k_blocks);
    }
}

template<int B_LANES, int QSIZE, int OP_M, int OPS>
void __device__ produce_B_tiles(
    const fp8* __restrict__ B,
    fp8* B_buffer,
    const int B_producers,
    int* B_queue,
    int &index,
    int &p_state,
    int &role_id,
    int b_stride,
    const int k_blocks
) {
    static constexpr int OP_N = (OP_M == 32) ? 32 : 16;
    static constexpr int OP_K = 512 / OP_M;
    produce_n_full_tiles<B_LANES, QSIZE, OP_K, OP_N, OPS, false>(B, B_buffer, B_producers, B_queue, index, p_state,
                                                                 role_id, 0, // for B we offset tiles to not load OOB
                                                                 b_stride, k_blocks);
                                                                 // TODO: BENCHMARK W/ reuse for multi-row cases
}
