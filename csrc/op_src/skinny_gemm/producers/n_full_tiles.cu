#include "./../utils.cuh"

// WARNING / TODO : this producer always skips cache
template<
    int LANES,
    int QSIZE,
    int OP_K,  // This is the size of the operation in the contiguous axis
    int OP_AD, // This is the size of the operation in the strided axis
    int OPS,
    bool REUSE
>
void __device__ produce_n_full_tiles(
    const fp8* __restrict__ source,
    fp8* buffer,
    const int producers,
    int* queue,
    int &index,
    int &p_state,
    int &role_id,
    int dropped_ad, // number of rows or columns that are out of bounds for this producer
    const int stride_ad,
    const int k_blocks
) {
    // Compile-time constants
    static constexpr int E_PER_THREAD = 16;
    static constexpr int E_PER_BANK = 4;
    static constexpr int WARPTILE_K = OP_K * OPS;

    static constexpr int TILES_PER_LOADS = (WARPSIZE * E_PER_THREAD) / (OP_K * OP_AD);
    static constexpr int LOADS = OPS / TILES_PER_LOADS;
    static constexpr int THREADS_PER_LD = (OP_K * TILES_PER_LOADS) / E_PER_THREAD;
    static constexpr int THREADS_PER_AD = WARPSIZE / THREADS_PER_LD;

    // Infer ids
    const int lane_id = get_lane_id();

    // Infer thread position in source
    int curr_ld, curr_ad;
    if constexpr (OP_K == 16) {
        curr_ld = (lane_id / THREADS_PER_AD) * E_PER_THREAD;
        curr_ad = lane_id % THREADS_PER_AD;
    } else {
        curr_ld = (lane_id % THREADS_PER_LD) * E_PER_THREAD;
        curr_ad = lane_id / THREADS_PER_LD;
    }
    dropped_ad += curr_ad;

    // Relocate thread in source
    source += curr_ad * stride_ad;
    source += curr_ld;

    // Relocate thread in buffer
    buffer += curr_ad * E_PER_BANK;
    if constexpr (OP_K == 16) {
        buffer += (curr_ld / OP_K) * NB_BANKS * E_PER_BANK * 4;
    } else {
        buffer += (curr_ld / 32) * NB_BANKS * E_PER_BANK * 4;
        buffer += (curr_ld % 32 >= 16) ? 16 * E_PER_BANK : 0;
    }

    // Prepare registers
    fp8 reg[LOADS][E_PER_THREAD];

    // K-wise loop
    const fp8* src;
    fp8* buf;
    int b = role_id;

    while (b < LANES * k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE * LANES) ? QSIZE * LANES : 0;

        // Relocate in source, accounting for dropped rows or cols
        int offset_ad = (b % LANES) * OP_AD;
        offset_ad = (offset_ad + dropped_ad >= OP_AD * LANES) ? 0 : offset_ad;
        src = source + (b / LANES) * WARPTILE_K + offset_ad * stride_ad;

        // Relocate in buffer without accounting for drops
        buf = buffer + index * (OP_AD * WARPTILE_K);

        // Start loading all data
        load_from_gmem_to_reg_no_waitcnt<LOADS, REUSE, OP_K>(src, reg);

        // Wait for buffer to be consumed
        while (queue[index] != p_state) {
            asm volatile("s_sleep 0");
        }

        // Fill N-tile buffer
        #pragma unroll
        for (int load = 0; load < LOADS; load++) {

            // Wait for tile to be loaded in registers
            switch (LOADS - load - 1) {
                case 0: asm volatile("s_waitcnt vmcnt(0)"); break;
                case 1: asm volatile("s_waitcnt vmcnt(1)"); break;
                case 2: asm volatile("s_waitcnt vmcnt(2)"); break;
                case 3: asm volatile("s_waitcnt vmcnt(3)"); break;
            } // TODO: find a way to factor this out

            // Place tile in shared memory
            #pragma unroll
            for (int line = 0; line < E_PER_THREAD / E_PER_BANK; line++) {
                #pragma unroll
                for (int elem = 0; elem < E_PER_BANK; elem++) {
                    buf[line * (NB_BANKS * E_PER_BANK) + elem] = reg[load][E_PER_BANK * line + elem];
                }
            }
            buf += WARPSIZE * E_PER_THREAD;
        }

        // Debug: check reg00 of lane0
        // if (lane_id == 0) {
        //     printf("threadIdx.x: %d, blockIdx.x: %d, LANES: %d, role_id: %d, b: %d, index: %d, p_state: %d, reg[0][0]: %f\n",
        //             threadIdx.x,     blockIdx.x,     LANES,     role_id,     b,     index,     p_state,     DB_FP8_TO_FP32(reg[0][0]));
        // }

        // Mark buffer as filled
        queue[index] = p_state + 1;

        // Update loop variables
        index += producers;
        p_state = (index >= QSIZE * LANES) ? (p_state + 2) : p_state;
        b += producers;
    }

    // Bring warps back in order
    role_id = b - (LANES * k_blocks);
}
