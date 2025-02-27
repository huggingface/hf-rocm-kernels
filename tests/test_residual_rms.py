from typing import Tuple
import pytest
import torch

from hf_rocm_kernels.operators.residual_rms import residual_rms, reference_residual_rms, generate_residual_rms_data
from hf_rocm_kernels.utils.testing import compare_x_with_ref


def _test_residual_rms(
    rows: int, cols: int, buffer_cols: int, dtype: torch.dtype, force_scalar: bool, verbose: bool
) -> Tuple[Tuple[float, float, float], ...]:
    """Test for the residual_rms operation. Can be either called with (verbose) flag on, in which case there will be a 
    lot of text displayed, which is good for debugging, or with (verbose) turned off, which is good for pytest."""
    # Generate data
    input, residual, weight, epsilon, scale_tensor, next_buffer = generate_residual_rms_data(
        rows=rows, 
        cols=cols, 
        buffer_cols=buffer_cols, 
        dtype=dtype, 
        seed=0
    )
    # Compute operation outputs
    qinput, attn_res = residual_rms(
        input=input.clone(), 
        residual=residual.clone(), 
        weight=weight, 
        epsilon=epsilon,
        scale_tensor=scale_tensor, 
        next_buffer=next_buffer, 
        force_scalar=force_scalar
    )
    # Check next buffer if there is one
    if next_buffer is not None:
        assert next_buffer.sum() == 0, next_buffer
    # Compute reference outputs
    ref_qinput, ref_attn_res, ref_scale = reference_residual_rms(
        input=input, 
        residual=residual, 
        weight=weight, 
        epsilon=epsilon, 
        scale_tensor=(None if scale_tensor is None else scale_tensor.div(2)), 
        next_buffer=next_buffer
    )
    # Crunch error metrics on each output and maybe display them
    return (
        compare_x_with_ref(qinput.float(), ref_qinput.float(), "qinput" if verbose else None),
        compare_x_with_ref(attn_res.float(), ref_attn_res.float(), "attn_res" if verbose else None),
    )

@pytest.mark.parametrize("force_scalar", [False, True])
@pytest.mark.parametrize("buffer_cols", [0, 32, 57, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float8_e4m3fnuz])
@pytest.mark.parametrize("cols", [8, 24, 128, 512, 4096, 16384])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16, 32, 64, 128, 256])
def test_residual_rms(
    rows: int, 
    cols: int,
    buffer_cols: int,
    dtype: torch.dtype,
    force_scalar: bool,
    atol: float = 5e-2,
    rtol: float = 0.15,
    ctol: int = 0.5,
) -> None:
    """Pytested version of the residual_rms test. Threshold are not final."""
    if buffer_cols > 0 and dtype != torch.float8_e4m3fnuz:
        return
    (max_error_qinput, max_relaive_error_qinput, changes_qinput), (max_error_res, _, _) = (
        _test_residual_rms(rows, cols, buffer_cols, dtype, force_scalar, verbose=False)
    )
    assert max_error_res == 0, f"Residual are not equal: {max_error_res = }"
    assert max_error_qinput < atol, f"Absolute tolerance is not respected: {max_error_qinput = } >= {atol}"
    percent_changes = 100 * changes_qinput / (rows * cols)
    assert percent_changes < ctol, f"Too many changes: {percent_changes = } >= {ctol}"
    assert max_relaive_error_qinput < rtol, f"Relative tolerance is not met: {max_relaive_error_qinput = } >= {rtol}"


if __name__ == "__main__":

    nb_toks = 1
    hidden_size = 8
    dtype = torch.float8_e4m3fnuz
    buffer_cols = 32
    force_scalar = False

    _test_residual_rms(nb_toks, hidden_size, buffer_cols, dtype, force_scalar, verbose=True)
