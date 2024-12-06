import torch

from .binding import _increment


def increment(x: torch.Tensor) -> None:
    """Adds 1 to the given GPU-resident tensor inplace."""
    assert x.device.type == "cuda", f"Expected x.device to be of type cuda but got {x.device.type = }"
    _increment(x)
