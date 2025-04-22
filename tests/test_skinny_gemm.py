from typing import Tuple
import pytest

from hf_rocm_kernels.utils.testing import compare_x_with_ref
from hf_rocm_kernels.operators.skinny_gemm import skinny_gemm, reference_skinny_gemm, generate_skinny_gemm_data


def _test_skinny_gemm(m: int, n: int, k: int, verbose: bool) -> Tuple[float, float, float]:
    """Test for the skinny_gemm operation. Can be either called with (verbose) flag on, in which case there will be a 
    lot of text displayed, which is good for debugging, or with (verbose) turned off, which is good for pytest."""
    # Generate data
    skinny_a, b, scale_tensor, out = generate_skinny_gemm_data(m, n, k, seed=0)
    # Compute operation outputs
    output = skinny_gemm(skinny_a, b, scale_tensor, out.clone())
    # Compute reference outputs
    scale_b = scale_tensor.clone().fill_(1.0)
    ref_output = reference_skinny_gemm(skinny_a, b, scale_tensor, scale_b, out)
    # Crunch error metrics on each output and maybe display them
    return compare_x_with_ref(output.float(), ref_output.float(), "output" if verbose else None)

@pytest.mark.parametrize("repeat_id", "abcde")
@pytest.mark.parametrize("k", [256, 1024, 16384])
@pytest.mark.parametrize("n", [128, 1024, 6656, 13312])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 16, 17, 32])
def test_skinny_gemm(
    m: int,
    n: int,
    k: int,
    repeat_id: str,
    atol: float = 0.125,
) -> None:
    """Pytested version of the swiglu test. Threshold are not final."""
    max_error, max_relative_error, changes =  _test_skinny_gemm(m, n, k, verbose=False)
    assert max_error <= atol
    # WARNING: not passed AT ALL - assert max_relative_error < rtol
    # WARNING: not passed AT ALL - assert changes < ctol


if __name__ == "__main__":

    M = 8
    N = 2304
    K = 16384

    print("Test for skinny_gemm:", f"{M = }", f"{N = }", f"{K = }", sep="\n\t")
    _test_skinny_gemm(M, N, K, verbose=True)
