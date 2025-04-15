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
    ops.def(
        "residual_rms(Tensor input, Tensor! residual, Tensor weight, Tensor scale_tensor, float epsilon, Tensor! "
        "output, Tensor! next_buffer, int num_threads, bool force_scalar) -> ()");
    ops.impl("residual_rms", torch::kCUDA, &residual_rms);

    // Swiglu
    ops.def(
        "swiglu(Tensor gate_up, Tensor scale_tensor, Tensor! output, Tensor! next_buffer, int num_threads, "
        "bool force_scalar) -> ()");
    ops.impl("swiglu", torch::kCUDA, &swiglu);

    // Skinny GEMM
    ops.def(
        "skinny_gemm_tb(Tensor A, Tensor B, Tensor! D, Tensor scale_tensor, int split_k, int A_producers, "
        "int B_producers, int consumers, int a_lanes, int b_lanes, int qsize, int op_m, int ops) -> int");
    ops.impl("skinny_gemm_tb", torch::kCUDA, &skinny_gemm_tb);

    // Fused skinny GEMM allreduce
    // Initialization
    ops.def("all_reduce_init(int rank, int worldSize, int port, Tensor comms_buff_A, Tensor comms_buff_B) -> int");
    ops.impl("all_reduce_init", torch::kCUDA, &all_reduce_init);
    // Calling
    ops.def("all_reduce(int allreduce_engine_ptr, Tensor A, Tensor B, Tensor D, Tensor scale_tensor, int b_lanes, int split_k, int n_blocks, int compute_warps, bool is_capturing) -> Tensor");
    ops.impl("all_reduce", torch::kCUDA, &all_reduce);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
