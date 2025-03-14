#pragma once
#include "./core.cu"


//----------------------------------------------------------------------------------------------------------------------
template<typename T>
void inline __device__ consumer_smem_to_reg(fp8* buffer, T &reg);

template<>
void inline __device__ consumer_smem_to_reg(fp8* buffer, fp8x8 &reg)
{
    static constexpr int E_PER_BANK = 4;
    #pragma unroll
    for (int i = 0; i < E_PER_BANK; i++) { reg[i + 0 * E_PER_BANK] = buffer[i + 0 * NB_BANKS * E_PER_BANK]; }
    #pragma unroll
    for (int i = 0; i < E_PER_BANK; i++) { reg[i + 1 * E_PER_BANK] = buffer[i + 1 * NB_BANKS * E_PER_BANK]; }
}

template<>
void inline __device__ consumer_smem_to_reg(fp8* buffer, fp8x16 &reg)
{
    static constexpr int E_PER_BANK = 4;
    #pragma unroll
    for (int i = 0; i < E_PER_BANK; i++) { reg[i + 0 * E_PER_BANK] = buffer[i + 0 * NB_BANKS * E_PER_BANK]; }
    #pragma unroll
    for (int i = 0; i < E_PER_BANK; i++) { reg[i + 1 * E_PER_BANK] = buffer[i + 1 * NB_BANKS * E_PER_BANK]; }
    #pragma unroll
    for (int i = 0; i < E_PER_BANK; i++) { reg[i + 2 * E_PER_BANK] = buffer[i + 2 * NB_BANKS * E_PER_BANK]; }
    #pragma unroll
    for (int i = 0; i < E_PER_BANK; i++) { reg[i + 3 * E_PER_BANK] = buffer[i + 3 * NB_BANKS * E_PER_BANK]; }
}


//----------------------------------------------------------------------------------------------------------------------
template <int LOADS, bool REUSE>
void inline __device__ load_from_gmem_to_reg_no_waitcnt(const fp8* &src, fp8 reg[LOADS][16]) {
    if constexpr (LOADS == 1) {
        if constexpr (REUSE) {
            asm volatile(
                "global_load_dwordx4 %0, %1, off offset:0  \n\t"
                : "=v"(reg[0]) : "v"(src)
            );
        } else {
            asm volatile(
                "global_load_dwordx4 %0, %1, off offset:0  sc0 sc1 nt  \n\t"
                : "=v"(reg[0]): "v"(src)
            );
        }
    }
    if constexpr (LOADS == 2) {
        if constexpr (REUSE) {
            asm volatile(
                "global_load_dwordx4 %0, %2, off offset:0   \n\t"
                "global_load_dwordx4 %1, %2, off offset:64  \n\t"
                : "=v"(reg[0]), "=v"(reg[1]) : "v"(src)
            );
        } else {
            asm volatile(
                "global_load_dwordx4 %0, %2, off offset:0   sc0 sc1 nt  \n\t"
                "global_load_dwordx4 %1, %2, off offset:64  sc0 sc1 nt  \n\t"
                : "=v"(reg[0]), "=v"(reg[1]) : "v"(src)
            );
        }
    }
    if constexpr (LOADS == 4) {
        if constexpr (REUSE) {
            asm volatile(
                "global_load_dwordx4 %0, %4, off offset:0    \n\t"
                "global_load_dwordx4 %1, %4, off offset:64   \n\t"
                "global_load_dwordx4 %2, %4, off offset:128  \n\t"
                "global_load_dwordx4 %3, %4, off offset:192  \n\t"
                : "=v"(reg[0]), "=v"(reg[1]), "=v"(reg[2]), "=v"(reg[3]) : "v"(src)
            );
        } else {
            asm volatile(
                "global_load_dwordx4 %0, %4, off offset:0    sc0 sc1 nt  \n\t"
                "global_load_dwordx4 %1, %4, off offset:64   sc0 sc1 nt  \n\t"
                "global_load_dwordx4 %2, %4, off offset:128  sc0 sc1 nt  \n\t"
                "global_load_dwordx4 %3, %4, off offset:192  sc0 sc1 nt  \n\t"
                : "=v"(reg[0]), "=v"(reg[1]), "=v"(reg[2]), "=v"(reg[3]) : "v"(src)
            );
        }
    }
}
