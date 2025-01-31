#include "./core.cu"

void inline __device__ consumer_smem_to_reg8(fp8* buffer, fp8x8 &reg) 
{
    // 32 bits load from the current bank
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg[i] = buffer[i];
    }
    // 32 bits load from the same bank, hopefully extension of the first load
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg[4 + i] = buffer[i + 32*E_P_BANK];
    }
}

void inline __device__ consumer_smem_to_reg16(fp8* buffer, fp8x16 &reg) 
{
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i     ] = buffer[i                   ]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i +  4] = buffer[i +     32*E_P_BANK]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i +  8] = buffer[i + 2 * 32*E_P_BANK]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i + 12] = buffer[i + 3 * 32*E_P_BANK]; }
}

template<int CONSUMERS, int B_LANES, int QSIZE>
void __device__ _tsr_consumer(
    fp8* A_buffer,
    fp8* B_buffer,
    half* D,
    float scale,
    uint8* queue,
    int &index,
    uint8 &p_state,
    int &role_id,
    const int n,
    const int dropped_rows,
    const int dropped_cols,
    const int k,
    const int k_blocks
) {
    // Compute thread position
    const int thread_id = threadIdx.x % WARPSIZE;
    A_buffer += (thread_id / 2) * E_P_BANK + (threadIdx.x % 2) * 32*E_P_BANK * 2;
    B_buffer += (thread_id % 32) * 4 + (thread_id / 32) * 32*E_P_BANK * 4;
    const int sparsity_indices = (threadIdx.x % 2) ? 0x0000EEEE : 0x00004444;

    // Declare input registers
    fp8x8 reg_A[OPS];
    fp8x16 reg_B[OPS];

    // Initialize output registers
    f32x4 reg_D[B_LANES];
    #pragma unroll
    for (int i = 0; i < (B_LANES); i++) {
        reg_D[i][0] = 0.0f; reg_D[i][1] = 0.0f; reg_D[i][2] = 0.0f; reg_D[i][3] = 0.0f;
    }

    // K-wise loop
    fp8 *A_offs_buff, *B_offs_buff;
    int b = role_id;

    while (b < k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE) ? QSIZE : 0;
        A_offs_buff = A_buffer + index * (WARPTILE_M * WARPTILE_K);
        B_offs_buff = B_buffer + index * ((OP_N * B_LANES) * WARPTILE_K);

        // Wait for A buffer to be filled
        while (queue[2 * B_LANES * index] != p_state) {
            asm volatile("s_sleep 0");
        }
        // Load A buffer
        #pragma unroll
        for (int op = 0; op < OPS; op++) {
            consumer_smem_to_reg8(A_offs_buff + (op * OP_M * OP_K), reg_A[op]);
        }
        // Mark A buffer as consumed
        queue[2 * B_LANES * index] = p_state + 32;

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
                consumer_smem_to_reg16(B_offs_buff + (lane * OP_N * WARPTILE_K) + (op * OP_N * OP_K), reg_B[op]);
            }
            // Mark B buffer as consumed
            queue[2 * (B_LANES * index + lane) + 1] = p_state + 32;

            // Consume registers
            #pragma unroll
            for (int op = 0; op < OPS; op++) {
                reg_D[lane] = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
                    reinterpret_cast<fp8_4x2>(reg_A[op]),
                    reinterpret_cast<fp8_4x4>(reg_B[op]),
                    reg_D[lane], 
                    sparsity_indices, // src2
                    7, 1 // cbsz, abid
                );
            }
        }

        // Update index
        index += CONSUMERS;
        p_state = (index >= QSIZE) ? p_state + 64 : p_state;
        b += CONSUMERS;
    }

    // Bring warps back in order
    role_id = b - k_blocks;

    // Infer the current column in D
    int out_n = 2 * ((thread_id % 16) / 2);

    // Finalize registers so that each threads hold 2 consecutive results in memory
    float final_scale;
    int id_to_swap = 1 - threadIdx.x % 2;
    int src_lane = thread_id + 1 - 2 * (thread_id % 2);

    #pragma unroll
    for (int i = 0; i < B_LANES; i++) {

        // Fusing
        reg_D[i][0] = reg_D[i][0] + reg_D[i][1];
        reg_D[i][1] = reg_D[i][2] + reg_D[i][3];

        // Scaling
        final_scale = (out_n + i * OP_N) >= dropped_cols ? scale : 0.0f;
        reg_D[i][0] *= final_scale;
        reg_D[i][1] *= final_scale;

        // Swapping 
        reg_D[i][id_to_swap] = __shfl(reg_D[i][id_to_swap], src_lane);
    }

    // Infer the current row in D
    int out_m = (thread_id / 16) * 2 + (thread_id % 2);

    // If we are in dropped rows territory, we can return now
    if (out_m + dropped_rows > WARPTILE_M -1) {
        return ;
    }

    // Relocate on D
    __half2* D_ = reinterpret_cast<__half2*>(D) + (out_m * n + out_n) / 2;

    // Out lane by lane
    __half2 x;

    #pragma unroll
    for (int i = 0; i < B_LANES; i++) {
        
        // Form the packed f16
        x.x = reg_D[i][0];
        x.y = reg_D[i][1];

        asm volatile(
            "global_atomic_pk_add_f16 %0, %1, off\n\t" : : "v"(&D_[i * OP_N / 2]), "v"(x)
        );
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
}
