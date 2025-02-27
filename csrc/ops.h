#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>

void increment(torch::Tensor& x);

void residual_rms(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& scale_tensor,
                  double epsilon, torch::Tensor& output, torch::Tensor& next_buffer, int64_t num_threads,
                  bool force_scalar);

void swiglu(torch::Tensor& gate_up_proj, torch::Tensor& swiglu_out, torch::Tensor& scale_tensor,
            torch::Tensor& next_buffer, int64_t mode);

void skinny_gemm(torch::Tensor& A, torch::Tensor& B, torch::Tensor& D, torch::Tensor& scale_tensor, int64_t b_lanes,
                 int64_t split_k);
