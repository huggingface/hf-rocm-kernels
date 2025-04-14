from typing import Tuple
import pytest
import torch

from hf_rocm_kernels.operators.residual_rms import residual_rms, reference_residual_rms, generate_residual_rms_data, precise_residual_rms
from hf_rocm_kernels.utils.testing import compare_x_with_ref, compare_x_and_ref_to_precise


def _test_residual_rms(
    rows: int, cols: int, buffer_cols: int, dtype: torch.dtype, force_scalar: bool, verbose: bool
) -> Tuple[float, ...]:
    """Test for the residual_rms operation. Can be either called with (verbose) flag on, in which case there will be a
    lot of text displayed, which is good for debugging, or with (verbose) turned off, which is good for pytest."""
    # Generate data
    input, residual, weight, epsilon, scale_tensor, next_buffer = generate_residual_rms_data(
        rows=rows, cols=cols, buffer_cols=buffer_cols, dtype=dtype, seed=0)
    # Compute operation outputs
    qinput, attn_res = residual_rms(input=input.clone(), residual=residual.clone(), weight=weight, epsilon=epsilon,
        scale_tensor=scale_tensor, next_buffer=next_buffer, force_scalar=force_scalar)
    # Compute reference outputs
    ref_qinput, ref_attn_res, _ = reference_residual_rms(input=input.clone(), residual=residual.clone(), weight=weight,
        epsilon=epsilon, scale_tensor=(None if scale_tensor is None else scale_tensor.div(2)))

    # Hard check: if there is a next buffer, it should be zero
    if next_buffer is not None:
        assert next_buffer.sum() == 0, next_buffer
    # Hard check: residuals should be equal
    assert (attn_res - ref_attn_res).abs().max() == 0

    # Compute precise outputs
    precise_qinput, _ = precise_residual_rms(input=input.clone(), residual=residual.clone(),
        weight=weight, epsilon=epsilon, scale_tensor=(None if scale_tensor is None else scale_tensor.div(2)))
    # Crunch error metrics on each output and maybe display them
    delta_x_max, delta_ref_max, delta_x_mean, delta_ref_mean, regression_percentage, regression_max, regression_mean \
        = compare_x_and_ref_to_precise(qinput.float(), ref_qinput.float(), precise_qinput.float())
    if verbose:
        print(
            f"Max delta x: {delta_x_max}, max delta ref: {delta_ref_max}\n"
            f"Mean delta x: {delta_x_mean}, mean delta ref: {delta_ref_mean}\n"
            f"Regression percentage: {regression_percentage:.2f}%, "
            f"regression max: {regression_max}, regression mean: {regression_mean}"
        )
    return (
        delta_x_max, delta_ref_max,
        delta_x_mean, delta_ref_mean,
        regression_percentage, regression_max, regression_mean
    )

@pytest.mark.parametrize("force_scalar", [False, True])
@pytest.mark.parametrize("dtype, buffer_cols", [
    (torch.float16, 0),
    (torch.float8_e4m3fnuz, 0),
    (torch.float8_e4m3fnuz, 32),
    (torch.float8_e4m3fnuz, 1024),
])
@pytest.mark.parametrize("cols", [8, 24, 128, 512, 4096, 16384])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16, 32, 64, 128, 256])
def test_residual_rms(
    rows: int,
    cols: int,
    buffer_cols: int,
    dtype: torch.dtype,
    force_scalar: bool,
    mean_tolerance: float = 1e-6,
    max_regression_percentage: float = 0.1,
    max_regression: float = 0.25,
    max_regression_mean: float = 1e-4,
) -> None:
    """Pytested version of the residual_rms test. Threshold are not final."""
    delta_x_max, delta_ref_max, delta_x_mean, delta_ref_mean, regression_percentage, regression_max, regression_mean \
        = _test_residual_rms(rows, cols, buffer_cols, dtype, force_scalar, verbose=False)

    # Check deltas with fp64 precision
    assert delta_x_max <= delta_ref_max, f"Max delta x: {delta_x_max} is bigger than max delta ref: {delta_ref_max}"
    assert delta_x_mean <= delta_ref_mean + mean_tolerance, \
        f"Mean delta x: {delta_x_mean} is bigger than mean delta ref: {delta_ref_mean} + {mean_tolerance}"

    # Check regression metrics
    assert regression_percentage < max_regression_percentage, \
        f"Regression percentage: {regression_percentage} >= {max_regression_percentage}"
    assert regression_max <= max_regression, \
        f"Regression max: {regression_max} >= {max_regression}"
    assert regression_mean < max_regression_mean, \
        f"Regression mean: {regression_mean} >= {max_regression_mean}"

if __name__ == "__main__":

    nb_toks = 256
    hidden_size = 16384
    dtype = torch.float8_e4m3fnuz
    buffer_cols = 1024
    force_scalar = False

    max_delta_x, max_delta_ref, error_percentage, error_mean, qinput, ref_qinput, precise_qinput = _test_residual_rms(nb_toks, hidden_size, buffer_cols, dtype, force_scalar, verbose=True)
    print(f"Max delta x: {max_delta_x}, max delta ref: {max_delta_ref}, "
          f"error percentage: {error_percentage} ({error_percentage * 100}%), error mean: {error_mean}")

    # Plot the delta between output, ref and precise
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1)
    axs[0].matshow((qinput.float() - precise_qinput.float()).abs().numpy(force=True))
    axs[1].matshow((ref_qinput.float() - precise_qinput.float()).abs().numpy(force=True))
    axs[2].matshow((qinput.float() - ref_qinput.float()).abs().numpy(force=True))
    plt.savefig("residual_rms.png")

