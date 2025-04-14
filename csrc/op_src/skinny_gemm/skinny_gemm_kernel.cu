#include "./consumers/consumers.cuh"
#include "./producers/producers.cuh"

// TODO: try uint16 for the p state

template<int A_LANES, int B_LANES, int QSIZE, int OP_M, int OPS>
void __global__ _skinny_gemm_kernel(
    const fp8* __restrict__ A,
    const fp8* __restrict__ B,
    half* __restrict__ D,
    const float* scale_tensor,
    const int m,
    const int n,
    const int k,
    const int b_stride,
    const int split_k,
    const int A_producers,
    const int B_producers,
    const int consumers
) {
    // Compile-time constants
    static constexpr int WARPTILE_M = OP_M * A_LANES; // is either 8, 16 or 32
    static constexpr int WARPTILE_N = (OP_M == 32 ? 32 : 16) * B_LANES;
    static constexpr int WARPTILE_K = (512 / OP_M) * OPS;

    // Initialize shared queue
    __shared__ int queue[(A_LANES + B_LANES) * QSIZE];
    if (threadIdx.x < (A_LANES + B_LANES) * QSIZE) {
        queue[threadIdx.x] = 0;
    }
    // Declare shared buffer
    __shared__ fp8 A_buffer[WARPTILE_M * WARPTILE_K * QSIZE];
    __shared__ fp8 B_buffer[WARPTILE_N * WARPTILE_K * QSIZE];
    __syncthreads();

    // Infer warp-specialization-related variables
    const int warp_id = get_warp_id();
    int role_id = (warp_id < A_producers) ? warp_id : (warp_id - A_producers);
        role_id = (warp_id < A_producers + B_producers) ? role_id : (role_id - B_producers);
    int index = role_id;
    int p_state = (warp_id >= A_producers + B_producers);

    // Tiles loop
    const int rows_of_tiles = CDIV(m, WARPTILE_M);
    const int tiles_per_row = CDIV(n, WARPTILE_N);
    const int total_tiles = rows_of_tiles * tiles_per_row * split_k;
    const int tiles_per_block = CDIV(total_tiles, CU);

    const int stop_tile = min(total_tiles, tiles_per_block * (blockIdx.x + 1));

    for (int tile = tiles_per_block * blockIdx.x; tile < stop_tile; tile++) {

        // Compute tile position
        int idx_m = tile % rows_of_tiles;
        int curr_m = idx_m * WARPTILE_M;

        int idx_nk = tile / rows_of_tiles;
        int curr_n = (idx_nk % tiles_per_row) * WARPTILE_N;
        int curr_k = (idx_nk / tiles_per_row) * WARPTILE_K * NUM_WARPTILE_K(k, split_k);

        // Compute tile's K blocks (number of blocks along the K axis)
        bool last_tile = (idx_nk / tiles_per_row) == (split_k - 1);
        int full_tiles = (k / WARPTILE_K);
        int k_blocks = last_tile ? full_tiles - (split_k - 1) * (full_tiles / split_k) : (full_tiles / split_k);

        // Account for column overflow
        int dropped_rows = max(0, curr_m + WARPTILE_M - m);
        int dropped_cols = max(0, curr_n + WARPTILE_N - n);
        curr_n -= dropped_cols; // make sure we are in the bounds of B

        // A producer warp
        if (warp_id < A_producers) {
            produce_A_tiles<A_LANES, QSIZE, OP_M, OPS>(
                A  + curr_m * k + curr_k,
                &A_buffer[0],
                A_producers,
                &queue[0],
                index, p_state, role_id,
                k, k_blocks,
                dropped_rows
            );
        }
        // B producer warp
        else if (warp_id < A_producers + B_producers) {
            // TODO: investigate the reuse parameter forB (true = faster but goes wrong because no sc1)
            produce_B_tiles<B_LANES, QSIZE, OP_M, OPS>(
                B + curr_n * b_stride + curr_k,
                &B_buffer[0],
                B_producers,
                &queue[A_LANES * QSIZE],
                index, p_state, role_id,
                b_stride, k_blocks
            );
        }
        // Consumers warp
        else if (warp_id < A_producers + B_producers + consumers) {
            consume_tiles<A_LANES, B_LANES, QSIZE, OP_M, OPS>(
                &A_buffer[0],
                &B_buffer[0],
                D + curr_m * n + curr_n,
                scale_tensor[0],
                consumers,
                &queue[0],
                index, p_state, role_id,
                dropped_rows, dropped_cols,
                n, k, k_blocks
            );
        }
    }
}
