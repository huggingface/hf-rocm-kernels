import torch
import torch.utils

from hf_rocm_kernels import increment


if __name__ == "__main__":

    x = torch.rand(size=(128,), device="cuda")
    y = x.clone()
    increment(x)
    assert x.eq(y + 1).all()
