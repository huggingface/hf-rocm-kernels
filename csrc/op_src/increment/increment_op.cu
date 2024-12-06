#include <torch/all.h>

void increment(torch::Tensor& x) {
    x += 1;
}
