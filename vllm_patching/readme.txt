This folder contains the files needed to patch VLLM with custom kernels for mi300. 

It contains two files that replace some parts of the VLLM code: "llama.py" and "loader.py". You can compare the files to
the original ones to see the changes, but mostly:
    - in loader.py, we add a "combined_scale" attribute to modules that have an "input_scale" and "weight_scale" 
        attribute, with "combined_scale" being the product of the two.
    - in llama.py, we replace the default swiglu and residual_rms kernels with the ones from the "hf_rocm_kernels" 
        package. We also replace the default GEMM kernel with the skinny version when is appropriate (batch size <= 8).

There is also a "benchmarking_launcher.py" file that contains a modified version of the "benchmark_launcher.py", that
allows to run benchmarks with multiple batch sizes in the same run. It also add a short sanity check in the form of 5 
prompts to complete and print.

To use these patched files, you can do the following:
    1. Install hf_rocm_kernels as described in the README.md file in the "hf_rocm_kernels" folder.
    2. Run the "patch_vllm.sh" script.


