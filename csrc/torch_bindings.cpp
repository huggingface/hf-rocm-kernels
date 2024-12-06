#include <torch/all.h>
#include <torch/library.h>

#include "ops.h"
#include "utils/registration.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

// Pay close attention to whether or not your tensor is modified inplace during
// the operation: if it is the case, then add a ! after "Tensor"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // Increment operator
    ops.def("increment(Tensor! x) -> ()");
    ops.impl("increment", torch::kCUDA, &increment);

    // Residual + RMS operator
    ops.def("residual_rms(Tensor input, Tensor! residual, Tensor weight, Tensor! output, float epsilon, float scale, int mode, int num_threads) -> ()");
    ops.impl("residual_rms", torch::kCUDA, &residual_rms);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
