from typing import Tuple, Optional
from torch import Tensor
import torch

from hf_rocm_kernels.utils.fp8 import fp8_quantize

def reference_swiglu(
    gate_up_proj: Tensor,
    scale_tensor: Tensor,
    next_buffer: Optional[Tensor],
) -> Tensor:
    """Reference for the swiglu operation. Check its docstring for more details."""
    # Type checking
    assert gate_up_proj.dtype == torch.float16, f"Expected torch.float16 but got {gate_up_proj.dtype = }"
    if next_buffer is not None:
        assert next_buffer.dtype == torch.float16, f"Expected torch.float16 but got {next_buffer.dtype = }"
    # SwiGLU
    gate_up_proj = gate_up_proj.view(-1, 2, gate_up_proj.size(1) // 2).float() 
    # NOTE: In TGI, there is no .float(), but we do this to avoid repeat conversions in our kernel which would only 
    #       lessen the final precision
    swiglu_out = torch.nn.functional.silu(gate_up_proj[:, 0]) * gate_up_proj[:, 1]
    # Convert to fp8
    qinput, _ = fp8_quantize(swiglu_out, scale_tensor)
    # Zero-init the next buffer
    if next_buffer is not None:
        next_buffer.fill_(0)
    return qinput


def generate_swiglu_data(
    rows: int, hidden_dim: int, buffer_cols: int = 0, seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Generates random inputs for the swiglu operation. The generated input's shape is determined by (rows) and (cols),
    and one can pass a (seed) to ensure repeatability. Also generates an empty buffer if (buffer_cols) is set to 
    non-zero."""
    if seed is not None:
        torch.manual_seed(seed)
    gate_up_proj = torch.normal(0, 1, size=(rows, 2*hidden_dim), device="cuda", dtype=torch.float16)
    scale_tensor = torch.rand(size=(1,), device="cuda", dtype=torch.float32).mul(2).add(1)
    next_buffer = None if buffer_cols == 0 else torch.empty((rows, buffer_cols), device="cuda", dtype=torch.float16)
    return gate_up_proj, scale_tensor, next_buffer
