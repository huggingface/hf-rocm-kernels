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
) {
    // Compile-time constants
    static constexpr int E_PER_BANK = 4;
    static constexpr int OP_M = 16;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 32;
    static constexpr int WARPTILE_M = OP_M;
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
    fp8x8 reg_A[OPS];
    fp8x8 reg_B[OPS];

    // Initialize output registers
    f32x4 reg_D[B_LANES];
    #pragma unroll
    for (int i = 0; i < B_LANES; i++) {
        reg_D[i][0] = 0.0f;
        reg_D[i][1] = 0.0f;
        reg_D[i][2] = 0.0f;
        reg_D[i][3] = 0.0f;
    }

    // K-wise loop
    int b = role_id;
    while (b < k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE) ? QSIZE : 0;
        fp8* A_offs_buff = A_buffer + index * (WARPTILE_M * WARPTILE_K);
        fp8* B_offs_buff = B_buffer + index * (WARPTILE_N * WARPTILE_K);

        // Wait for A buffer to be filled
        while (queue[2 * B_LANES * index] != p_state) {
            asm volatile("s_sleep 0");
        }
        // Load A buffer
        #pragma unroll
        for (int op = 0; op < OPS; op++) {
            consumer_smem_to_reg(A_offs_buff + (op * OP_M * OP_K), reg_A[op]);
        }
        // Mark A buffer as consumed
        queue[2 * B_LANES * index] = p_state + 1;

        // Go through each lanes
        #pragma unroll
        for (int lane = 0; lane < B_LANES; lane++) {

            // Wait for B buffer to be filled
            while (queue[2 * (B_LANES * index + lane) + 1] != p_state) {
                asm volatile("s_sleep 0");
            }
            // Load B buffer
            #pragma unroll
            for (int op = 0; op < OPS; op++) {
                consumer_smem_to_reg(B_offs_buff + (lane * OP_N * WARPTILE_K) + (op * OP_N * OP_K), reg_B[op]);
            }
            // Mark B buffer as consumed
            queue[2 * (B_LANES * index + lane) + 1] = p_state + 1;

            // Consume registers
            #pragma unroll
            for (int op = 0; op < OPS; op++) {
                reg_D[lane] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                    reinterpret_cast<long>(reg_A[op]),
                    reinterpret_cast<long>(reg_B[op]),
                    reg_D[lane],
                    0, // src2
                    0, // cbsz
                    0  // abid
                );
            }
        }

        // Update index
        index += CONSUMERS;
        p_state = (index >= QSIZE) ? (p_state + 2) : p_state;
        b += CONSUMERS;
    }

    // Bring warps back in order
    role_id = b - k_blocks;

    // Infer the current column in D
    int out_n = 2 * ((lane_id % 16) / 2);

    // Finalize registers so that each threads hold 2 consecutive results in memory
    float final_scale;
    int id_to_swap = 1 - lane_id % 2;
    int src_lane = lane_id + 1 - 2 * (lane_id % 2);

    #pragma unroll
    for (int i = 0; i < B_LANES; i++) {

        // Scaling
        final_scale = (out_n + i * OP_N) >= dropped_cols ? scale : 0.0f;
        reg_D[i][0] *= final_scale;
        reg_D[i][1] *= final_scale;
        reg_D[i][2] *= final_scale;
        reg_D[i][3] *= final_scale;

        // Swapping
        reg_D[i][id_to_swap    ] = __shfl(reg_D[i][id_to_swap    ], src_lane);
        reg_D[i][id_to_swap + 2] = __shfl(reg_D[i][id_to_swap + 2], src_lane);
    }

    // Infer the current row in D
    int out_m = (lane_id / 16) * 4 + (lane_id % 2);

    // Quit if we are in dropped rows territory
    if (out_m + dropped_rows >= WARPTILE_M) { return; }
    // Relocate on D
    __half2* D_ = reinterpret_cast<__half2*>(D + (out_m * n + out_n));

    // Out lane by lane for the first exit row
    __half2 x;
    #pragma unroll
    for (int lane = 0; lane < B_LANES; lane++) {
        x.x = reg_D[lane][0];
        x.y = reg_D[lane][1];
        asm volatile("global_atomic_pk_add_f16 %0, %1, off\n\t" : : "v"(&D_[lane * (OP_N / 2)]), "v"(x));
    }

    // Quit if the second exit row is in dropped rows territory
    if (out_m + 2 + dropped_rows >= WARPTILE_M) { return; }
    // Advance to the second exit row (which is two rows after the first one)
    D_ += n;

    // Out lane by lane for the second exit row
    #pragma unroll
    for (int lane = 0; lane < B_LANES; lane++) {
        x.x = reg_D[lane][2];
        x.y = reg_D[lane][3];
        asm volatile("global_atomic_pk_add_f16 %0, %1, off\n\t" : : "v"(&D_[lane * (OP_N / 2)]), "v"(x));
    }
}



// TODO: non-atomic exit path if split-k is equal to 1

// Disabled: if D is of type float
// // Relocate on D
// D += (out_m * n + out_n);

// // Out lane by lane
// #pragma unroll
// for (int i = 0; i < B_LANES; i++) {
//     atomicAdd(&D[0 + i*OP_N], reg_D[i][0]);
//     atomicAdd(&D[1 + i*OP_N], reg_D[i][1]);
// }
