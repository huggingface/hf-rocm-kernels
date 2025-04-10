from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules=[
    CUDAExtension(
        'hfrk', [
            'csrc/op_src/fused_gemm_ar/allreduce.cu',
            'csrc/op_src/increment/increment_op.cu',
            'csrc/op_src/residual_rms/residual_rms_dispatch.cu',
            'csrc/op_src/skinny_gemm/skinny_gemm.cu',
            'csrc/op_src/swiglu/swiglu_dispatch.cu'
        ],
        include_dirs=['/usr/local/mscclpp/include', "/opt/ompi/include"],  # Adjust this path
        library_dirs=['/usr/local/mscclpp/lib', "/opt/ompi/lib", "/usr/local/lib"],      # Adjust this path
        libraries=['mscclpp', 'mpi'],
        extra_compile_args=['--offload-arch=gfx942', '-U__HIP_NO_HALF_CONVERSIONS__', '-U__HIP_NO_HALF_OPERATORS__'],
    )
]

setup(
    name="hf-rocm-kernels",
    version="0.0.0",
    description=("PyTorch kernels for Hugging Face models on ROCm devices."),
    long_description="<some markdown file>",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("benchmarks", "csrc", "docs", "examples", "tests*")),
    python_requires=">=3.9",
    ext_modules=ext_modules,
    extras_require={},
    cmdclass={
        'build_ext': BuildExtension
    },
    package_data={"hf_rocm_kernels": ["py.typed", "*.so"]},
    entry_points={
        "console_scripts": [],
    },
)
