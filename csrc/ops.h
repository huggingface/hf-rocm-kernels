#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>

void increment(torch::Tensor& x);

void residual_rms(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& scale_tensor, 
                  double epsilon, torch::Tensor& output, torch::Tensor& next_buffer, int64_t mode, int64_t num_threads);
