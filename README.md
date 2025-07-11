<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 🤗 ROCm Kernels
Optimized ML kernels for ROCm devices.

## Installation

To install this package, you first need to find [your GPU's architecture](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html). For instance, __if you are using a MI300 GPU, your architecture would be `gfx942`__. This package has only been tested on the `gfx942` architecture.  
Then specify the architecture you are running on when calling `install.sh`: this will set the `PYTORCH_ROCM_ARCH` environment variable and install the project.

```bash
./install.sh gfx942
````


## Adding a new operator

### Writing the source files

Imagine we want to add a simple increment operator. We first create the `increment` directory in `csrc/op_src`. This directory should contains the source files for our operator, which in our case is a file named `increment_op.cu` containing:

```
#include <torch/all.h>

void increment(torch::Tensor& x) {
    x += 1;
}
```

### Adding to CMake

We add this file to the list of files CMake builds, located in `CMakeLists.txt`. You can look for the line `# This is the list of operator files to build.` and add the operator source file:

```
# This is the list of operator files to build.
set(KERNEL_SRC
  "csrc/torch_bindings.cpp"
  "csrc/op_src/increment/increment_op.cu"
)
```

### Binding python and C

We need to create the binding to our C operator in python. We first declare our operator in `csrc/ops.h` (*), so we had this line at the end of the file:

```
void increment(torch::Tensor& x);
```

Then, we create a torch binding in C++ by adding the operator in the `csrc/torch_bindings.cpp` file. Inside the `TORCH_LIBRARY_EXPAND` scope, we add the lines:

```
// Increment operator
ops.def("increment(Tensor! x) -> ()");
ops.impl("increment", torch::kCUDA, &increment);
```

Notice the `!` after the `Tensor` type: this means that the tensor `x` is modified by the operator we just declared. When passing a tensor that is not modified by the operator, you can drop the `!`.

Finally, we add the python-side of the binding. We create the directory `increment` in `hf_rocm_kernels/operators` and in it the file `binding.py` containing:

```
import torch

import hf_rocm_kernels._HFRK_C


def _increment(x: torch.Tensor) -> None:
    torch.ops._HFRK_C.increment(x)
```

Ideally, we then create a user-friendly version of the operator with checks (eg. assert tensors are on device and in the right layout) and documentation, as was done in `hf_rocm_kernels/increment/wrapped.py`. That version should be the one exposed in `hf_rocm_kernels/__init__.py`.



### Side notes

(*) When passing scalar arguments to an operator, use types `int64_t` (corresponds to python `int`) and `double` (python `float`) for the C operator.


### VLLM Integration

The HF ROCm kernels are integrated into VLLM. For detailed instructions and usage information, see the [HFRK integration README](https://github.com/remi-or/vllm/blob/patch_hfrk/HFRK_readme.md).
