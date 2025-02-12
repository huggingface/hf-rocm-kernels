from typing import Tuple
import pytest

from hf_rocm_kernels.utils.testing import compare_x_with_ref
from hf_rocm_kernels.operators.swiglu import swiglu, reference_swiglu, generate_swiglu_data


def _test_swiglu(rows: int, hidden_dim: int, buffer_cols: int, mode: int, verbose: bool) -> Tuple[float, float, float]:
    """Test for the swiglu operation. Can be either called with (verbose) flag on, in which case there will be a 
    lot of text displayed, which is good for debugging, or with (verbose) turned off, which is good for pytest."""
    # Generate data
    gate_up, scale_tensor, next_buffer = generate_swiglu_data(rows, hidden_dim, buffer_cols, seed=0)
    # Compute operation outputs
    swiglu_out = swiglu(gate_up, scale_tensor.mul(2), next_buffer, mode)
    if next_buffer is not None:
        assert next_buffer.sum() == 0, next_buffer.sum()
    # Compute reference outputs
    ref_swiglu_out = reference_swiglu(gate_up, scale_tensor, next_buffer)
    # Crunch error metrics on each output and maybe display them 
    return compare_x_with_ref(swiglu_out.float(), ref_swiglu_out.float(), "swiglu_out" if verbose else None)

@pytest.mark.parametrize("mode", [0, 1]) # 4])
@pytest.mark.parametrize("buffer_cols", [0, 1024, 16384])
@pytest.mark.parametrize("hidden_dim", [8, 24, 128, 512, 4096, 6656])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16, 32, 64, 128, 256])
def test_swiglu(
    rows: int, 
    hidden_dim: int,
    buffer_cols: int,
    mode: int,
    atol: float = 2e-2,
    rtol: float = 0.15,
    ctol: int = 10,
) -> None:
    """Pytested version of the swiglu test. Threshold are not final."""
    max_error, max_relative_error, changes = _test_swiglu(rows, hidden_dim, buffer_cols, mode, verbose=False)
    assert max_error < atol
    assert max_relative_error < rtol
    assert changes < ctol


if __name__ == "__main__":

    NB_TOKENS = 8
    HIDDEN_SIZE = 16384
    BUFFER_COLS = 0
    MODE = 1

    print("Test for residual_rms:", f"{NB_TOKENS = }", f"{HIDDEN_SIZE = }", f"{MODE = }", sep="\n\t")
    _test_swiglu(NB_TOKENS, HIDDEN_SIZE, BUFFER_COLS, MODE, verbose=True)
