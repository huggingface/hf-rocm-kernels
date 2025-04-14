#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>

void increment(torch::Tensor& x);

void residual_rms(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& scale_tensor,
                  double epsilon, torch::Tensor& output, torch::Tensor& next_buffer, int64_t num_threads,
                  bool force_scalar);

void swiglu(torch::Tensor& gate_up, torch::Tensor& scale_tensor, torch::Tensor& output, torch::Tensor& next_buffer,
            int64_t num_threads, bool force_scalar);

int64_t skinny_gemm_tb(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D,
    torch::Tensor& scale_tensor,
    int64_t split_k,
    int64_t A_producers,
    int64_t B_producers,
    int64_t consumers,
    int64_t a_lanes,
    int64_t b_lanes,
    int64_t qsize,
    int64_t op_m,
    int64_t ops
);
