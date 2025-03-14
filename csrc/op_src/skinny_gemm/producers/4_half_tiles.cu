#include "./../core.cu"

// WARNING / TODO : tiles always hit cache AND are swizzled
template<int PRODUCERS, int B_LANES, int QSIZE>
void __device__ produce_4_half_tiles(
    const fp8* __restrict__ src,
    fp8* buffer,
    int* queue,
    int &index,
    int &p_state,
    int &role_id,
    const int dropped_ad,
    const int k,
    const int k_blocks
) {
    // Compile-time constants
    static constexpr int E_PER_THREAD = 16;
    static constexpr int E_PER_BANK = 4;
    static constexpr int OP_M = 8;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 64;
    static constexpr int WARPTILE_M = OP_M;
    static constexpr int WARPTILE_N = OP_N * B_LANES;
    static constexpr int WARPTILE_K = OP_K * OPS;

    static constexpr int THREADS_PER_LD = 8;

    // Infer ids
    const int lane_id = get_lane_id();

    // Infer thread position in source
    const int curr_ld = (lane_id % THREADS_PER_LD) * E_PER_THREAD;
    const int curr_ad = lane_id / THREADS_PER_LD;

    // Relocate thread in source (and queue for B producers) // WARNING: currently assume m == 8
    src += (curr_ad + dropped_ad > WARPTILE_M - 1) ? 0 : curr_ad * k;
    src += curr_ld;
    src += (OPS == 1 ? 2 : 1) * role_id * WARPTILE_K;

    // Relocate thread in buffer
    buffer += E_PER_BANK * curr_ad;
    buffer += (curr_ld / 64) * 32 * E_PER_BANK * 4;
    buffer += (curr_ld % 64) * 2;

    // Prepare registers
    fp8_4 reg0[E_PER_THREAD / 4];
    fp8_4 reg1[E_PER_THREAD / 4];
    fp8_4 swap_reg[E_PER_THREAD / 4];

    // K-wise loop
    fp8_4* buf;
    int b = (OPS == 1 ? 2 : 1) * role_id;

    while (b < k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE) ? QSIZE : 0;
        buf = reinterpret_cast<fp8_4*>(buffer) + index * (WARPTILE_M * WARPTILE_K / 4);

        // Load from gmem to reg
        asm volatile(
            "global_load_dwordx4 %0, %2, off offset:0  \n\t"
            "global_load_dwordx4 %1, %2, off offset:128\n\t"
            : "=v"(reg0), "=v"(reg1)
            : "v"(src)
        );

        // Wait for buffer to be consumed
        while (queue[2 * B_LANES * index] != p_state) {
            asm volatile("s_sleep 0");
        }

        // Store in smem from reg
        asm volatile(
            "s_waitcnt vmcnt(1)\n\t"
            "v_pack_b32_f16 %0, %4, %5, op_sel:[0,0]\n\t"
            "v_pack_b32_f16 %1, %4, %5, op_sel:[1,1]\n\t"
            "v_pack_b32_f16 %2, %6, %7, op_sel:[0,0]\n\t"
            "v_pack_b32_f16 %3, %6, %7, op_sel:[1,1]\n\t"
            "v_swap_b32 %0, %4\n\t"
            "v_swap_b32 %1, %6\n\t"
            "v_swap_b32 %2, %5\n\t"
            "v_swap_b32 %3, %7"
            : "=v"(swap_reg[0]), "=v"(swap_reg[1]), "=v"(swap_reg[2]), "=v"(swap_reg[3])
            : "v"(reg0[0]), "v"(reg0[1]), "v"(reg0[2]), "v"(reg0[3])
        );
        #pragma unroll
        for (int line = 0; line < 4; line++) {
            buf[line * 32] = reg0[line];
        }

        // Store in smem from reg
        asm volatile(
            "s_waitcnt vmcnt(0)\n\t"
            "v_pack_b32_f16 %0, %4, %5, op_sel:[0,0]\n\t"
            "v_pack_b32_f16 %1, %4, %5, op_sel:[1,1]\n\t"
            "v_pack_b32_f16 %2, %6, %7, op_sel:[0,0]\n\t"
            "v_pack_b32_f16 %3, %6, %7, op_sel:[1,1]\n\t"
            "v_swap_b32 %0, %4\n\t"
            "v_swap_b32 %1, %6\n\t"
            "v_swap_b32 %2, %5\n\t"
            "v_swap_b32 %3, %7"
            : "=v"(swap_reg[0]), "=v"(swap_reg[1]), "=v"(swap_reg[2]), "=v"(swap_reg[3])
            : "v"(reg1[0]), "v"(reg1[1]), "v"(reg1[2]), "v"(reg1[3])
        );
        #pragma unroll
        for (int line = 0; line < 4; line++) {
            buf[line * 32 + OP_K * OP_M / 2] = reg1[line];
        }

        // Mark buffer as filled
        queue[2 * B_LANES * index] = p_state + 1;

        // Advance
        src += WARPTILE_K * PRODUCERS;

        // Update index
        index += PRODUCERS;
        p_state = (index >= QSIZE) ? (p_state + 2) : p_state;
        b += PRODUCERS;
    }

    // Bring warps back in order
    role_id = (b - k_blocks); // WARNING: not sure about this (and it's late)
}
