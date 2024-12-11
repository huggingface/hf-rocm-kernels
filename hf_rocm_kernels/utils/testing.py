from typing import Optional, Tuple
from torch import Tensor


def compare_x_with_ref(x: Tensor, ref: Tensor, name: Optional[str] = None) -> Tuple[float, float, float]:
    """Compares a given tensor (x) with a reference tensor (ref) and returns their max absolute difference, their max
    relative difference and the number of elements that differ between the two. The function displays a lot of info if a
    (name) is given for the tensors compared."""
    # Check shapes
    if x.shape != ref.shape:
        raise ValueError(
            f"Cannot compare tensor of shape {x.shape} with reference of shape {ref.shape} (name: {name})."
        )
    # Compute metrics
    delta_x = (ref.float() - x.float()).abs()
    max_delta = delta_x.max().item()
    relative_delta = delta_x.div(ref.float())[ref.float() != 0]
    if relative_delta.numel() == 0:
        max_relative_delta = float("inf")
    else:
        max_relative_delta = relative_delta.max().item()
    mean_delta = delta_x.mean().item()
    nb_changed_coeffs = delta_x.not_equal(0).int().sum().item()
    percent_changed_coeffs = delta_x.not_equal(0).float().mean().mul(100).item()
    # If the variables have a name, display something
    if name is not None:
        if max_delta == 0:
            print(f"All good for {name}")
        else:
            print(f"Not good for {name}")
            # Print the start of both tensors
            print(f"\n#-- Fragment of {name} --#")
            viz = 4 # you can increase this when debugging
            print("Reference:", ref.flatten().tolist()[:viz], "...", ref.flatten().tolist()[-viz:])
            print("Computed: ", x.flatten().tolist()[:viz], "...", x.flatten().tolist()[-viz:])
            # Print their absolute max difference
            print(f"\n#-- Delta of {name} --#")
            print("Max absolute diff:", max_delta)
            print("Max relative diff:", max_relative_delta)
            # Print their mean absolute difference
            print("Mean absolute diff:", mean_delta)
            # Print the number of different coefficients
            print("Number of changes:", nb_changed_coeffs, f"out of {delta_x.numel()} ({percent_changed_coeffs:.5f} %)")
            # Print the number of NANS
            x_nans, ref_nans = map(lambda t: t.isnan().int().sum().item(), (x, ref))
            if max(x_nans, ref_nans) == 0:
                print(f"\n#-- No NANs in either of {name} --#\n")
            else:
                print(f"\n#-- NANs of {name} --#\nReference: {ref_nans}\nComputed:{x_nans}\n")
    # Return the relevant metrics
    return max_delta, max_relative_delta, nb_changed_coeffs
