#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>

void increment(torch::Tensor& x);

void residual_rms(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& output,
                  double epsilon, double scale, int64_t mode, int64_t num_threads);

void sparse_k(torch::Tensor& A, torch::Tensor& B, torch::Tensor& D, int64_t W);
