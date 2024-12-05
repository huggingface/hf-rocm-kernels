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
Set the specific architecture you are running on with the `PYTORCH_ROCM_ARCH` environment variable.
For example, for a MI300 GPU, you would set `PYTORCH_ROCM_ARCH=gfx942`.
You can find a list of supported architectures [here](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html).

```bash
pip install uv
uv sync
```
