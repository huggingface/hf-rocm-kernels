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

TORCH_LIBRARY_IMPL(hfrk, CUDA, m) {
    // Increment operator
    m.def("increment(Tensor! x) -> ()");
    m.impl("increment", increment);

    // Residual + RMS operator
    m.def(
        "residual_rms(Tensor input, Tensor! residual, Tensor weight, Tensor scale_tensor, float epsilon, Tensor! "
        "output, Tensor! next_buffer, int num_threads, bool force_scalar) -> ()");
    m.impl("residual_rms", &residual_rms);

    // Swiglu
    m.def(
        "swiglu(Tensor gate_up, Tensor scale_tensor, Tensor! output, Tensor! next_buffer, int num_threads, "
        "bool force_scalar) -> ()");
    m.impl("swiglu", &swiglu);

    // Skinny GEMM
    m.def("skinny_gemm(Tensor A, Tensor B, Tensor! D, Tensor scale_tensor, int b_lanes, int split_k) -> ()");
    m.impl("skinny_gemm", &skinny_gemm);


    // Fused skinny GEMM allreduce
    // Initialization
    m.def("all_reduce_init(int rank, int, worldSize, int port, Tensor comms_buff_A, Tensor comms_buff_B) -> int");
    m.impl("all_reduce_init", &all_reduce_init);
    // Calling
    m.def("all_reduce(int allreduce_engine_ptr, Tensor A, Tensor B, Tensor D, Tensor scale_tensor, int b_lanes, int split_k, bool is_capturing) -> Tensor");
    m.impl("all_reduce", &all_reduce);
}

REGISTER_EXTENSION(hfrk)
