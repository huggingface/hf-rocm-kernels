from typing import Tuple
import pytest
import matplotlib.pyplot as plt

from hf_rocm_kernels.utils.testing import compare_x_with_ref
from hf_rocm_kernels.operators.swiglu import swiglu, reference_swiglu, generate_swiglu_data


def _test_swiglu(rows: int, hidden_dim: int, buffer_cols: int, force_scalar: bool, verbose: bool) -> Tuple[float, float, float]:
    """Test for the swiglu operation. Can be either called with (verbose) flag on, in which case there will be a 
    lot of text displayed, which is good for debugging, or with (verbose) turned off, which is good for pytest."""
    # Generate data
    gate_up, scale_tensor, next_buffer = generate_swiglu_data(rows, hidden_dim, buffer_cols, seed=0)
    # Compute operation outputs
    swiglu_out = swiglu(gate_up, scale_tensor.mul(2), next_buffer, force_scalar=force_scalar)
    if next_buffer is not None:
        assert next_buffer.sum() == 0, next_buffer.sum()
    # Compute reference outputs
    ref_swiglu_out = reference_swiglu(gate_up, scale_tensor, next_buffer)
    # Crunch error metrics on each output and maybe display them 
    return compare_x_with_ref(swiglu_out.float(), ref_swiglu_out.float(), "swiglu_out" if verbose else None)

@pytest.mark.parametrize("force_scalar", [True, False])
@pytest.mark.parametrize("buffer_cols", [0, 1024, 16384])
@pytest.mark.parametrize("hidden_dim", [8, 24, 128, 512, 4096, 6656])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 1024, 2048]) # errors for 1024 but small ones
def test_swiglu(
    rows: int,
    hidden_dim: int,
    buffer_cols: int,
    force_scalar: bool,
    atol: float = 1e-2,
    rtol: float = 0.15,
    ctol: int = 10,
) -> None:
    """Pytested version of the swiglu test. Threshold are not final."""
    max_error, max_relative_error, changes = _test_swiglu(rows, hidden_dim, buffer_cols, force_scalar, verbose=False)
    assert max_error < atol
    assert max_relative_error < rtol
    assert changes < ctol


if __name__ == "__main__":

    NB_TOKENS = 2144
    HIDDEN_SIZE = 6656
    BUFFER_COLS = 0
    FORCE_SCALAR = False

    # Generate data
    gate_up, scale_tensor, next_buffer = generate_swiglu_data(NB_TOKENS, HIDDEN_SIZE, BUFFER_COLS, seed=0)

    # Get outputs
    swiglu_out = swiglu(gate_up_proj=gate_up, scale_tensor=scale_tensor.mul(2), next_buffer=next_buffer, force_scalar=FORCE_SCALAR)
    ref_swiglu_out = reference_swiglu(gate_up, scale_tensor, next_buffer)

    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7.5))

    im1 = ax1.matshow(swiglu_out.float().numpy(force=True))
    ax1.set_title("Our Implementation")
    im2 = ax2.matshow(ref_swiglu_out.float().numpy(force=True))
    ax2.set_title("Reference")
    abs_diff = (swiglu_out.float() - ref_swiglu_out.float()).abs().numpy(force=True)
    im3 = ax3.matshow(abs_diff)
    ax3.set_title("Absolute Difference")

    fig.tight_layout()
    fig.savefig('__test__.png')

    # Print the first two rows of each tensor with their names
    # print("Swiglu out:")
    # print(swiglu_out[:2])
    # print("\nRef swiglu out:")
    # print(ref_swiglu_out[:2])
    # print("\nAbs diff:")
    # print(abs_diff[:2])

    print("Test for swiglu:", f"{NB_TOKENS = }", f"{HIDDEN_SIZE = }", f"{BUFFER_COLS = }", f"{FORCE_SCALAR = }", sep="\n\t")
    _test_swiglu(NB_TOKENS, HIDDEN_SIZE, BUFFER_COLS, FORCE_SCALAR, verbose=True)
