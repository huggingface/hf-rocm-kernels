template<int A_LANES, int B_LANES, int QSIZE, int OPS>
void __device__ consume_tiles_sparse_16x16x64(
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
    static constexpr int OP_M = 8;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 64;
    static constexpr int WARPTILE_M = OP_M;
    static constexpr int WARPTILE_N = OP_N * B_LANES;
    static constexpr int WARPTILE_K = OP_K * OPS;

    // Compute thread position and sparsity indices
    const int lane_id = get_lane_id();
    const int sparsity_indices = (threadIdx.x % 2) ? 0x0000EEEE : 0x00004444;

    // Relocate in buffers
    A_buffer += (lane_id / 2) * E_PER_BANK;
    A_buffer += (lane_id % 2) * E_PER_BANK * 2 * NB_BANKS;

    B_buffer += (lane_id % 32) * E_PER_BANK;
    B_buffer += (lane_id / 32) * E_PER_BANK * 4 * NB_BANKS;

    // Declare input registers
    fp8x8 reg_A[A_LANES][OPS];
    fp8x16 reg_B[OPS];

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

        // Load A registers
        #pragma unroll
        for (int a_lane = 0; a_lane < A_LANES; a_lane++) {
            // Wait for A buffer to be filled
            while (a_queue[A_LANES * index + a_lane] != p_state) {
                asm volatile("s_sleep 0");
            }
            // Load A buffer
            #pragma unroll
            for (int op = 0; op < OPS; op++) {
                consumer_smem_to_reg(A_offs_buff + (a_lane * OP_M * WARPTILE_K) + (op * OP_M * OP_K), reg_A[a_lane][op]);
            }
            // Mark A buffer as consumed
            // DEBUG: disabled this because not needed in practice // asm volatile("s_waitcnt lgkmcnt(0)");
            a_queue[A_LANES * index + a_lane] = p_state + 1;
        }

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
            // Mark B buffer as consumed
            // DEBUG: disabled this because not needed in practice // asm volatile("s_waitcnt lgkmcnt(0)");
            b_queue[B_LANES * index + b_lane] = p_state + 1;

            // Go through all A lanes
            #pragma unroll
            for (int a_lane = 0; a_lane < A_LANES; a_lane++) {

                // Consume registers
                #pragma unroll
                for (int op = 0; op < OPS; op++) {
                    reg_D[a_lane][b_lane] = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
                        reinterpret_cast<fp8_4x2>(reg_A[a_lane][op]),
                        reinterpret_cast<fp8_4x4>(reg_B[op]),
                        reg_D[a_lane][b_lane],
                        sparsity_indices, // src2
                        7, 1 // cbsz, abid
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

            // Fusing
            reg_D[a_lane][b_lane][0] = reg_D[a_lane][b_lane][0] + reg_D[a_lane][b_lane][1];
            reg_D[a_lane][b_lane][1] = reg_D[a_lane][b_lane][2] + reg_D[a_lane][b_lane][3];

            // Scaling
            float final_scale = (out_n + b_lane * OP_N) >= dropped_cols ? scale : 0.0f;
            reg_D[a_lane][b_lane][0] *= final_scale;
            reg_D[a_lane][b_lane][1] *= final_scale;

            // Swapping
            reg_D[a_lane][b_lane][id_to_swap] = __shfl(reg_D[a_lane][b_lane][id_to_swap], src_lane);
        }

        // Infer the current row in D
        int out_m = a_lane * OP_M + (lane_id / 16) * 2 + (lane_id % 2);
        __half2* D_ = reinterpret_cast<__half2*>(D + out_m * n + out_n);

        // If we are in dropped rows territory, we can return now
        if (out_m + dropped_rows >= WARPTILE_M) { return ; }

        // Out lane by lane
        __half2 x;
        #pragma unroll
        for (int b_lane = 0; b_lane < B_LANES; b_lane++) {
            x.x = __float2half(reg_D[a_lane][b_lane][0]); // TODO: packed conversion
            x.y = __float2half(reg_D[a_lane][b_lane][1]);
            asm volatile("global_atomic_pk_add_f16 %0, %1, off\n\t" : : "v"(&D_[b_lane * OP_N / 2]), "v"(x));
        }
    }

}



    // Disabled: if D is of type float
    // // Relocate on D
    // D += (out_m * n + out_n);

    // // Out lane by lane
    // #pragma unroll
    // for (int i = 0; i < B_LANES; i++) {
    //     atomicAdd(&D[0 + i*OP_N], reg_D[i][0]);
    //     atomicAdd(&D[1 + i*OP_N], reg_D[i][1]);
    // }
