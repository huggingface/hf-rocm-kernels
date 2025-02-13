#include "./core.cu"

template<int A_PRODUCERS, int B_LANES, int QSIZE>
void __device__ _tsr_A_producer(
    const fp8* __restrict__ src,
    fp8* buffer,
    int* queue,
    int &index,
    int &p_state,
    int &role_id,
    const int dropped_rows,
    const int k,
    const int k_blocks
) {
    static constexpr int elems_per_thread = 16;
    static constexpr int threads_per_ld = 8;

    // Infer ids
    const int thread_id = threadIdx.x % WARPSIZE;

    // Infer thread position in source
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread in source (and queue for B producers) // WARNING: currently assume m == 8
    src += (curr_ad + dropped_rows > WARPTILE_M - 1) ? 0 : curr_ad * k;
    src += curr_ld;
    src += (OPS == 1 ? 2 : 1) * role_id * WARPTILE_K;

    // Relocate thread in buffer
    buffer += (E_P_BANK * curr_ad) + (curr_ld % 64) * 2 + (curr_ld / 64) * (32*E_P_BANK * 4);
    
    // Prepare registers
    fp8_4 reg0[elems_per_thread / 4];
    fp8_4 reg1[elems_per_thread / 4];
    fp8_4 swap_reg[elems_per_thread / 4];

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
        src += WARPTILE_K * A_PRODUCERS;

        // Update index
        index += A_PRODUCERS;
        p_state = (index >= QSIZE) ? (p_state+2) : p_state;
        b += A_PRODUCERS;
    }

    // Bring warps back in order
    role_id = (b - k_blocks); // WARNING: not sure about this (and it's late)
}

template<int B_PRODUCERS, int B_LANES, int QSIZE>
void __device__ _tsr_B_producer(
    const fp8* __restrict__ source,
    fp8* buffer,
    int* queue,
    int &index,
    int &p_state,
    int &role_id,
    const int k,
    const int k_blocks
) {
    static constexpr int elems_per_thread = 16;
    static constexpr int threads_per_ld = 4;

    // Infer ids
    const int thread_id = threadIdx.x % WARPSIZE;

    // Infer thread position in source
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread in source (and queue for B producers)
    source += curr_ad * k;
    source += curr_ld;

    // Relocate thread in buffer
    buffer += (E_P_BANK * curr_ad) + ((curr_ld % 32 == 16) ? (16*E_P_BANK) : 0) + (curr_ld / 32) * (32*E_P_BANK * 4);

//     // Prepare registers
//     fp8 reg[elems_per_thread];

//     // K-wise loop
//     const fp8* src;
//     fp8* buf;
//     int b = role_id;

//     while (b < B_LANES * k_blocks) {

//         // Account for cyclic queue
//         index -= (index >= QSIZE * B_LANES) ? QSIZE * B_LANES : 0;
//         src = source + (b / B_LANES) * WARPTILE_K + (b % B_LANES) * OP_N * k;
//         buf = buffer + index * (OP_N * WARPTILE_K);

//         #pragma unroll
//         for (int op = 0; op < OPS; op++) {

//             // Load from gmem to reg
//             asm volatile("global_load_dwordx4 %0, %1, off\n\t" : "=v"(reg) : "v"(src));
            
//             if (op == 0) {
//                 // Wait for buffer to be consumed
//                 while (queue[2 * index] != p_state) {
//                     asm volatile("s_sleep 0");
//                 }
//             }

//             // Make sure load is finished
//             asm volatile("s_waitcnt vmcnt(0)");
//             // Store in smem from reg
//             #pragma unroll
//             for (int i = 0; i < 4; i++) {
//                 #pragma unroll
//                 for (int j = 0; j < 4; j++) {
//                     buf[32*4*i + j] = reg[4*i + j];
//                 }
//             }

//             if (op < OPS - 1) {
//                 src += OP_K;
//                 buf += OP_K * OP_N;
//             }
//         }

//         // Mark buffer as filled
//         queue[2 * index] = 4 + p_state;

//         // Update index
//         index += B_PRODUCERS;
//         p_state = (index >= QSIZE * B_LANES) ? ((p_state + 1) % 4) : p_state;
//         b += B_PRODUCERS;
//     }

//     // Bring warps back in order
//     role_id = b - (B_LANES * k_blocks);
// }

// // Prepare registers
//     fp8 reg0[elems_per_thread];
//     fp8 reg1[elems_per_thread];

//     // K-wise loop
//     const fp8* src;
//     fp8* buf;
//     int b = role_id;

//     while (b < B_LANES * k_blocks) {

//         // Account for cyclic queue
//         index -= (index >= QSIZE * B_LANES) ? QSIZE * B_LANES : 0;
//         src = source + (b / B_LANES) * WARPTILE_K + (b % B_LANES) * OP_N * k;
//         buf = buffer + index * (OP_N * WARPTILE_K);

//         asm volatile(
//             "global_load_dwordx4 %0, %2, off offset:0\n\t" 
//             "global_load_dwordx4 %1, %2, off offset:64\n\t" 
//             : "=v"(reg0), "=v"(reg1)
//             : "v"(src)
//         );

//         while (queue[2 * index] != p_state) {
//             asm volatile("s_sleep 0");
//         }

//         asm volatile("s_waitcnt vmcnt(1)");
//         #pragma unroll
//         for (int i = 0; i < 4; i++) {
//             #pragma unroll
//             for (int j = 0; j < 4; j++) {
//                 buf[NB_BANKS * E_P_BANK * i + j] = reg0[E_P_BANK * i + j];
//             }
//         }
//         asm volatile("s_waitcnt vmcnt(0)");
//         #pragma unroll
//         for (int i = 0; i < 4; i++) {
//             #pragma unroll
//             for (int j = 0; j < 4; j++) {
//                 buf[NB_BANKS * E_P_BANK * i + j + OP_K * OP_N * 1] = reg1[E_P_BANK * i + j];
//             }
//         }

//         // Mark buffer as filled
//         queue[2 * index] = 4 + p_state;

//         // Update index
//         index += B_PRODUCERS;
//         p_state = (index >= QSIZE * B_LANES) ? ((p_state + 1) % 4) : p_state;
//         b += B_PRODUCERS;
//     }

//     // Bring warps back in order
//     role_id = b - (B_LANES * k_blocks);
// }


    // Prepare registers
    fp8 reg0[elems_per_thread];
    fp8 reg1[elems_per_thread];
    fp8 reg2[elems_per_thread];
    fp8 reg3[elems_per_thread];

    // K-wise loop
    const fp8* src;
    fp8* buf;
    int b = role_id;

    while (b < B_LANES * k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE * B_LANES) ? QSIZE * B_LANES : 0;
        src = source + (b / B_LANES) * WARPTILE_K + (b % B_LANES) * OP_N * k;
        buf = buffer + index * (OP_N * WARPTILE_K);

        asm volatile(
            "global_load_dwordx4 %0, %4, off offset:0   sc0 sc1 nt\n\t" 
            "global_load_dwordx4 %1, %4, off offset:64  sc0 sc1 nt\n\t" 
            "global_load_dwordx4 %2, %4, off offset:128 sc0 sc1 nt\n\t" 
            "global_load_dwordx4 %3, %4, off offset:192 sc0 sc1 nt\n\t"
            : "=v"(reg0), "=v"(reg1), "=v"(reg2), "=v"(reg3)
            : "v"(src)
        );

        while (queue[2 * index] != p_state) {
            asm volatile("s_sleep 0");
        }

        asm volatile("s_waitcnt vmcnt(3)");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                buf[NB_BANKS * E_P_BANK * i + j] = reg0[E_P_BANK * i + j];
            }
        }
        asm volatile("s_waitcnt vmcnt(2)");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                buf[NB_BANKS * E_P_BANK * i + j + OP_K * OP_N * 1] = reg1[E_P_BANK * i + j];
            }
        }
        asm volatile("s_waitcnt vmcnt(1)");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                buf[NB_BANKS * E_P_BANK * i + j + OP_K * OP_N * 2] = reg2[E_P_BANK * i + j];
            }
        }
        asm volatile("s_waitcnt vmcnt(0)");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                buf[NB_BANKS * E_P_BANK * i + j + OP_K * OP_N * 3] = reg3[E_P_BANK * i + j];
            }
        }

        // Mark buffer as filled
        queue[2 * index] = p_state + 1;

        // Update index
        index += B_PRODUCERS;
        p_state = (index >= QSIZE * B_LANES) ? (p_state+2) : p_state;
        b += B_PRODUCERS;
    }

    // Bring warps back in order
    role_id = b - (B_LANES * k_blocks);
}
