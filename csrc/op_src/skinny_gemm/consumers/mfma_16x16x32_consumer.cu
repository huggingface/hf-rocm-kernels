template<int A_LANES, int B_LANES, int QSIZE, int OPS>
void __device__ consume_tiles_dense_16x16x32(
    fp8* A_buffer,
    fp8* B_buffer,
    half* D,
    float scale,
    const int consumers,
    int* a_queue,
    int &index,
    int &p_state,
    int &role_id,
    const int dropped_rows,
    const int dropped_cols,
    const int n,
    const int k,
    const int k_blocks
) {
    // Compile-time constants
    static constexpr int E_PER_BANK = 4;
    static constexpr int OP_M = 16;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 32;
    static constexpr int WARPTILE_M = OP_M * A_LANES;
    static constexpr int WARPTILE_N = OP_N * B_LANES;
    static constexpr int WARPTILE_K = OP_K * OPS;

    // Compute thread position
    const int lane_id = get_lane_id();

    // Relocate in buffers
    A_buffer += (lane_id % 16) * E_PER_BANK + (lane_id / 32) * 16 * E_PER_BANK;
    A_buffer += (lane_id % 32 >= 16) ? NB_BANKS * E_PER_BANK * 2 : 0;

    B_buffer += (lane_id % 16) * E_PER_BANK + (lane_id / 32) * 16 * E_PER_BANK;
    B_buffer += (lane_id % 32 >= 16) ? NB_BANKS * E_PER_BANK * 2 : 0;

    // Declare input registers
    fp8x8 reg_A[A_LANES][OPS];
    fp8x8 reg_B[OPS];

    // Initialize output registers
    f32x4 reg_D[A_LANES][B_LANES];
    #pragma unroll
    for (int a_lane = 0; a_lane < A_LANES; a_lane++) {
        #pragma unroll
        for (int b_lane = 0; b_lane < B_LANES; b_lane++) {
            reg_D[a_lane][b_lane][0] = 0.0f;
            reg_D[a_lane][b_lane][1] = 0.0f;
            reg_D[a_lane][b_lane][2] = 0.0f;
            reg_D[a_lane][b_lane][3] = 0.0f;
        }
    }

    // K-wise loop
    int b = role_id;
    int* b_queue = a_queue + A_LANES * QSIZE;

    while (b < k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE) ? QSIZE : 0;
        fp8* A_offs_buff = A_buffer + index * (WARPTILE_M * WARPTILE_K);
        fp8* B_offs_buff = B_buffer + index * (WARPTILE_N * WARPTILE_K);

        // Go through all B lanes
        #pragma unroll
        for (int b_lane = 0; b_lane < B_LANES; b_lane++) {

            // Wait for B buffer to be filled
            while (b_queue[B_LANES * index + b_lane] != p_state) {
                asm volatile("s_sleep 0");
            }
            // Load B buffer
            #pragma unroll
            for (int op = 0; op < OPS; op++) {
                consumer_smem_to_reg(B_offs_buff + (b_lane * OP_N * WARPTILE_K) + (op * OP_N * OP_K), reg_B[op]);
            }
            // Mark B buffer as consumed (wait for last B lane)
            // if (b_lane == B_LANES - 1) { TODO: restore this in some way
            asm volatile("s_waitcnt lgkmcnt(0)");
            // }
            b_queue[B_LANES * index + b_lane] = p_state + 1;

            // Go through all a lanes
            #pragma unroll
            for (int a_lane = 0; a_lane < A_LANES; a_lane++) {

                // If this is the first B lane, fill A registers
                if (b_lane == 0) {
                    // Wait for A buffer to be filled
                    while (a_queue[A_LANES * index + a_lane] != p_state) {
                        asm volatile("s_sleep 0");
                    }
                     // Load A buffer
                    #pragma unroll
                    for (int op = 0; op < OPS; op++) {
                        consumer_smem_to_reg(A_offs_buff + (a_lane * OP_M * WARPTILE_K) + (op * OP_M * OP_K), reg_A[a_lane][op]);
                    }
                    // Mark A buffer as consumed (wait only if there is not other B lane than B lane 0)
                    // NOTE: not sure this is needed, but then again who uses B_lanes == 1
                    if constexpr (B_LANES == 1) {
                        asm volatile("s_waitcnt lgkmcnt(0)");
                    }
                    a_queue[A_LANES * index + a_lane] = p_state + 1;
                }

                // Consume registers
                #pragma unroll
                for (int op = 0; op < OPS; op++) {
                    reg_D[a_lane][b_lane] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                        reinterpret_cast<long>(reg_A[a_lane][op]),
                        reinterpret_cast<long>(reg_B[op]),
                        reg_D[a_lane][b_lane],
                        0, // src2
                        0, // cbsz
                        0  // abid
                    );
                }
            }
        }
        // Update index
        index += consumers;
        p_state = (index >= QSIZE) ? (p_state + 2) : p_state;
        b += consumers;
    }

    // Bring warps back in order
    role_id = b - k_blocks;

    // Infer the current column in D
    int out_n = 2 * ((lane_id % 16) / 2);

    // Prepare swapping variables
    int id_to_swap = 1 - lane_id % 2;
    int src_lane = lane_id + 1 - 2 * (lane_id % 2);

    // Loop over all a lanes
    #pragma unroll
    for (int a_lane = 0; a_lane < A_LANES; a_lane++) {

        // Loop over all b lanes
        #pragma unroll
        for (int b_lane = 0; b_lane < B_LANES; b_lane++) {

            // Scaling
            float final_scale = (out_n + b_lane * OP_N) >= dropped_cols ? scale : 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                reg_D[a_lane][b_lane][j] *= final_scale;
            }

            // Swapping
            #pragma unroll
            for (int j = 0; j < 4; j+=2) {
                reg_D[a_lane][b_lane][id_to_swap + j] = __shfl(reg_D[a_lane][b_lane][id_to_swap + j], src_lane);
            }
        }

        // Infer the starting row in D
        int out_m0 = a_lane * OP_M + (lane_id / 16) * 4 + (lane_id % 2);
        __half2* D_packed = reinterpret_cast<__half2*>(D + (out_m0 * n + out_n));

        // Loop over all rows
        #pragma unroll
        for (int row = 0; row < 2; row++) {

            // Quit if we are in dropped rows territory
            if (out_m0 + dropped_rows >= WARPTILE_M - 2 * row) { return; }

            // Out lane by lane
            __half2 x;
            #pragma unroll
            for (int b_lane = 0; b_lane < B_LANES; b_lane++) {
                x.x = __float2half(reg_D[a_lane][b_lane][2 * row]); // TODO: packed conversion
                x.y = __float2half(reg_D[a_lane][b_lane][2 * row + 1]);
                asm volatile("global_atomic_pk_add_f16 %0, %1, off\n\t" : : "v"(&D_packed[b_lane * (OP_N / 2)]), "v"(x));
            }

            // Advance two rows (one in packed)
            D_packed += n;
        }
    }

    // Debug: input register look-up
    // if (threadIdx.x % WARPSIZE == 0) {
    //     for (int a_lane = 0; a_lane < A_LANES; a_lane++) {
    //         for (int op = 0; op < OPS; op++) {
    //             printf("Op %d: ", op);
    //             for (int i = 0; i < 16; i++) {
    //                 printf("%f, ", (float)__hip_cvt_fp8_to_halfraw(reg_A[a_lane][op][i], __HIP_E4M3_FNUZ).data);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    // Debug: bank look-up
    // if (threadIdx.x % WARPSIZE == 0) {
    //     for (int bank = 0; bank < NB_BANKS; bank++) {
    //         for (int line = 0; line < 16; line++) {
    //             printf("B %d / L %d: ", bank, line);
    //             for (int elem = 0; elem < E_PER_BANK; elem++) {
    //                 int index = bank * (E_PER_BANK * NB_BANKS) + line * E_PER_BANK + elem;
    //                 float x = __hip_cvt_fp8_to_halfraw(A_buffer[index], __HIP_E4M3_FNUZ).data;
    //                 printf("%f, ", x);
    //             }
    //             printf("\n");
    //         }
    //     }
    //     printf("\n");
    // }
}
