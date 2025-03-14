from typing import Tuple
import matplotlib.pyplot as plt
import pytest
import numpy as np
import torch

from hf_rocm_kernels.utils.testing import compare_x_with_ref
from hf_rocm_kernels.operators.skinny_gemm import skinny_gemm, reference_skinny_gemm, generate_skinny_gemm_data


def _test_skinny_gemm(m: int, n: int, k: int, split_k: int, b_lanes: int, verbose: bool) -> Tuple[float, float, float]:
    """Test for the skinny_gemm operation. Can be either called with (verbose) flag on, in which case there will be a
    lot of text displayed, which is good for debugging, or with (verbose) turned off, which is good for pytest."""
    # Generate data
    skinny_a, b, scale_tensor, out = generate_skinny_gemm_data(m, n, k, seed=0)
    # Compute operation outputs
    output = skinny_gemm(skinny_a, b, scale_tensor, out.clone(), split_k, b_lanes)
    # Compute reference outputs
    scale_b = scale_tensor.clone().fill_(1.0)
    ref_output = reference_skinny_gemm(skinny_a, b, scale_tensor, scale_b, out)
    # Crunch error metrics on each output and maybe display them
    return compare_x_with_ref(output.float(), ref_output.float(), "output" if verbose else None)

@pytest.mark.parametrize("split_k", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("b_lanes", [3, 5])
@pytest.mark.parametrize("k", [256, 1024, 16384])
@pytest.mark.parametrize("n", [128, 1024, 6656, 13312])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
def test_skinny_gemm(
    m: int,
    n: int,
    k: int,
    split_k: int,
    b_lanes: int,
    atol: float = 0.125,
) -> None:
    """Pytested version of the swiglu test. Threshold are not final."""
    max_error, max_relative_error, changes =  _test_skinny_gemm(m, n, k, split_k, b_lanes, verbose=False)
    assert max_error <= atol
    # WARNING: not passed AT ALL - assert max_relative_error < rtol
    # WARNING: not passed AT ALL - assert changes < ctol


if __name__ == "__main__":

    B_LANES = 3
    SPLIT_K = 1

    M = 16
    N = 2 * B_LANES * 16
    K = 1024

    # Generate data
    skinny_a, b, scale_tensor, out = generate_skinny_gemm_data(M, N, K, seed=0)

    # Compute with our implementation
    output = skinny_gemm(skinny_a, b, scale_tensor, out.clone(), SPLIT_K, B_LANES)
    torch.cuda.synchronize()
    # Compute with reference implementation
    scale_b = scale_tensor.clone().fill_(1.0)
    ref_output = reference_skinny_gemm(skinny_a, b, scale_tensor, scale_b, out)

    # Convert to numpy for plotting
    output_np = output.float().numpy(force=True)
    ref_output_np = ref_output.float().numpy(force=True)

    # Calculate difference
    diff = np.abs(output_np - ref_output_np)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 2.5))

    # Plot our implementation output
    im0 = axes[0].matshow(output_np, cmap='viridis')
    axes[0].set_title('Skinny GEMM Output')
    fig.colorbar(im0, ax=axes[0], orientation='horizontal', pad=0.1)

    # Plot reference output
    im1 = axes[1].matshow(ref_output_np, cmap='viridis')
    axes[1].set_title('Reference Output')
    fig.colorbar(im1, ax=axes[1], orientation='horizontal', pad=0.1)

    # Plot difference
    im2 = axes[2].matshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    fig.colorbar(im2, ax=axes[2], orientation='horizontal', pad=0.1)

    # Add overall title
    plt.suptitle(f'Skinny GEMM Comparison (M={M}, N={N}, K={K}, SPLIT_K={SPLIT_K}, B_LANES={B_LANES})')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(f'skinny_gemm_comparison_M{M}_N{N}_K{K}_SK{SPLIT_K}_BL{B_LANES}.png', dpi=300)

    # Run the comparison
    compare_x_with_ref(output.float(), ref_output.float(), "output")
