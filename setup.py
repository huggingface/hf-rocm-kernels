import importlib.util
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Dict

import torch
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

# Environment variables
TARGET_DEVICE = "rocm"
MAX_JOBS = os.getenv("MAX_JOBS", None)
CMAKE_BUILD_TYPE = os.getenv("CMAKE_BUILD_TYPE")  # Available options: ["Debug", "Release", "RelWithDebInfo"]
VERBOSE = bool(int(os.getenv("VERBOSE", "0")))
USE_PRECOMPILED = False

if not sys.platform.startswith("linux"):
    logger.warning(
        "hf_rocm_kernels only supports Linux platform (including WSL). "
        "Building on %s, "
        "so hf_rocm_kernels may not be able to run correctly",
        sys.platform,
    )
    TARGET_DEVICE = "empty"


def is_ninja_available() -> bool:
    return which("ninja") is not None


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: Dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = None

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DCMAKE_MODULE_PATH=/usr/lib64/cmake/hip",
            "-DTARGET_DEVICE={}".format(TARGET_DEVICE),
        ]

        if VERBOSE:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ["-DPYTHON_EXECUTABLE={}".format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ["-DPYTHON_PATH={}".format(":".join(sys.path))]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ["-DNVCC_THREADS={}".format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    @staticmethod
    def clean_target_name(s: str) -> str:
        return remove_prefix(s, "hf_rocm_kernels.")

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []
        target_name = self.clean_target_name

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            prefix = outdir
            for i in range(ext.name.count(".")):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                "cmake",
                "--install",
                ".",
                "--prefix",
                prefix,
                "--component",
                target_name(ext.name),
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        # First, run the standard build_ext command to compile the extensions
        super().run()


def _no_device() -> bool:
    return TARGET_DEVICE == "empty"


def _is_hip() -> bool:
    return (TARGET_DEVICE == "cuda" or TARGET_DEVICE == "rocm") and torch.version.hip is not None


def _is_cpu() -> bool:
    return TARGET_DEVICE == "cpu"


def _build_custom_ops() -> bool:
    return _is_hip() or _is_cpu()


def get_hipcc_rocm_version():
    # Run the hipcc --version command
    result = subprocess.run(
        ["hipcc", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Check if the command was executed successfully
    if result.returncode != 0:
        print("Error running 'hipcc --version'")
        return None

    # Extract the version using a regular expression
    match = re.search(r"HIP version: (\S+)", result.stdout)
    if match:
        # Return the version string
        return match.group(1)
    else:
        print("Could not find HIP version in the output")
        return None


### SCRIPT #############################################################################################################

ext_modules = []

# Check op_src's names are small enough (<= 6 characters) otherwise they won't link
# Don't ask why I don't know
root = os.path.dirname(__file__)
for dir_name in os.listdir(os.path.join(root, "csrc", "op_src")):
    # TODO: with only the "increment" operator, this test did not pass but linking did.
    # So skipping this test for now, turn it back on if linking fails.
    continue
    assert len(dir_name) <= 6 or dir_name.startswith((".", "__")), f"{dir_name = } too long for op_src"


if _build_custom_ops():
    ext_modules.append(CMakeExtension(name="hf_rocm_kernels._HFRK_C"))

package_data = {"hf_rocm_kernels": ["py.typed"]}
if USE_PRECOMPILED:
    ext_modules = []
    package_data["hf_rocm_kernels"].append("*.so")

if _no_device():
    ext_modules = []

setup(
    name="hf-rocm-kernels",
    version="0.0.0",
    description=("PyTorch kernels for Hugging Face models on ROCm devices."),
    long_description="<some markdown file>",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("benchmarks", "csrc", "docs", "examples", "tests*")),
    python_requires=">=3.8",
    ext_modules=ext_modules,
    extras_require={},
    cmdclass={"build_ext": cmake_build_ext} if len(ext_modules) > 0 else {},
    package_data=package_data,
    entry_points={
        "console_scripts": [],
    },
)
