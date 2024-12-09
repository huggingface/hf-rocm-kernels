from typing import Tuple
import pytest

from hf_rocm_kernels.operators.residual_rms import residual_rms, reference_residual_rms, generate_residual_rms_data
from hf_rocm_kernels.utils.testing import compare_x_with_ref


def _test_residual_rms(rows: int, cols: int, mode: int, verbose: bool) -> Tuple[Tuple[float, float, float], ...]:
    """Test for the residual_rms operation. Can be either called with (verbose) flag on, in which case there will be a 
    lot of text displayed, which is good for debugging, or with (verbose) turned off, which is good for pytest."""
    # Generate data
    input, residual, weights, epsilon, scale = generate_residual_rms_data(rows, cols, seed=0)
    # Compute operation outputs
    qinput, attn_res = residual_rms(input, residual.clone(), weights, epsilon, scale.item(), mode)
    scale_ = scale * 2
    # Compute reference outputs
    ref_qinput, ref_attn_res, ref_scale = reference_residual_rms(input, residual, weights, epsilon, scale)
    # Crunch error metrics on each output and maybe display them 
    return (
        compare_x_with_ref(qinput.float(), ref_qinput.float(), "qinput" if verbose else None),
        compare_x_with_ref(attn_res.float(), ref_attn_res.float(), "attn_res" if verbose else None),
        compare_x_with_ref(scale_, ref_scale, "scale" if verbose else None),
    )

@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("cols", [8, 24, 128, 512, 4096, 16384])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16, 32, 64, 128, 256])
def test_residual_rms(
    rows: int, 
    cols: int,
    mode: int,
    atol: float = 2e-2,
    rtol: float = 0.15,
    ctol: int = 10,
) -> None:
    """Pytested version of the residual_rms test. Threshold are not final."""
    (max_error_qinput, max_relaive_error_qinput, changes_qinput), (max_error_res, _, _), (max_error_scale, _, _) = (
        _test_residual_rms(rows, cols, mode, verbose=False)
    )
    assert max_error_res == 0
    assert max_error_scale == 0
    assert (max_error_qinput < atol) and (changes_qinput < ctol)
    assert max_relaive_error_qinput < rtol


if __name__ == "__main__":

    NB_TOKENS = 1
    HIDDEN_SIZE = 24
    MODE = 1

    print("Test for residual_rms:", f"{NB_TOKENS = }", f"{HIDDEN_SIZE = }", f"{MODE = }", sep="\n\t")
    _test_residual_rms(NB_TOKENS, HIDDEN_SIZE, MODE, verbose=True)
