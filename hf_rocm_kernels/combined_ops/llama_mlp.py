from torch import Tensor
import torch
from typing import Tuple, Optional

from hf_rocm_kernels import residual_rms, skinny_gemm, swiglu

def llama_mlp_combined_ops(
    attn_output: Tensor,
    residual: Tensor,
    rms_weight: Tensor,
    rms_epsilon: float,
    gate_up_weight: Tensor,
    gate_up_input_scale: Tensor,
    gate_up_output_scale: Tensor,
    down_weight: Tensor,
    down_input_scale: Tensor,
    down_output_scale: Tensor,
    gate_up_gemm_hp: Tuple[Optional[int], Optional[int]] = (None, None),
    down_gemm_hp: Tuple[Optional[int], Optional[int]] = (None, None),
) -> Tuple[Tensor, Tensor]:

    # RMS norm
    gate_up_proj = torch.empty(
        size=(attn_output.size(0), gate_up_weight.size(1)), dtype=torch.float16, device=attn_output.device
    )
    normalized, residual = residual_rms(
        input=attn_output,
        residual=residual,
        weight=rms_weight,
        epsilon=rms_epsilon,
        scale_tensor=gate_up_input_scale,
        next_buffer=gate_up_proj,
    )

    # Gate + up projection
    skinny_gemm(
        skinny_a=normalized,
        b=gate_up_weight,
        scale_tensor=gate_up_output_scale,
        output=gate_up_proj,
        split_k=gate_up_gemm_hp[0],
        b_lanes=gate_up_gemm_hp[1],
    )

    # Swiglu
    mlp_output = torch.empty(
        size=(attn_output.size(0), down_weight.size(1)), dtype=torch.float16, device=attn_output.device
    )
    swiglu_out = swiglu(gate_up_proj=gate_up_proj, scale_tensor=down_input_scale, next_buffer=mlp_output)

    # Down projection
    skinny_gemm(
        skinny_a=swiglu_out,
        b=down_weight,
        scale_tensor=down_output_scale,
        output=mlp_output,
        split_k=down_gemm_hp[0],
        b_lanes=down_gemm_hp[1],
    )
    return mlp_output, residual
