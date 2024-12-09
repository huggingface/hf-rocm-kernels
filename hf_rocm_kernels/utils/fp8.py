from typing import Tuple
import torch
from torch import Tensor


def fp8_quantize(
    x_full_precision: Tensor,
    scale: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Quantizes a tensor (x_full_precision) according to a tensor-wise (scale) to float8_e4m3fnuz format. This function is
    meant to mimic the behavior of TGI and thus was inspired by it:
    https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/layers/fp8.py
    For reference on dtypes: https://onnx.ai/onnx/technical/float8.html
    """
    # Scale and clamp in full precision
    finfo = torch.finfo(torch.float8_e4m3fn)
    x_quantized = (x_full_precision * scale.reciprocal()).clamp(min=finfo.min, max=finfo.max)
    # Convert to float8_e4m3fn format, without removing signed zeros
    x_quantized = x_quantized.to(torch.float8_e4m3fn)
    # Remove signed zeros, which correspond to NaNs in float8_e4m3fnuz format
    weight_as_int8 = x_quantized.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    mask = weight_as_int8 == ROCM_FP8_NAN_AS_INT
    weight_as_int8[mask] = 0
    x_quantized = weight_as_int8.view(torch.float8_e4m3fnuz)
    # For the same bits representation, e4m3fnuz value is half of the e4m3fn value, so we should double the scaling 
    # factor to get the same dequantized value.
    return x_quantized, scale * 2.0
