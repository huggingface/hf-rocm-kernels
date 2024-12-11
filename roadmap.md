# Roadmap

## New kernels

- A skinny GEMM that uses sparse instruction to increase throughput. `(high prio)`
- A swiglu epilogue kernel meant to go between an up/gate projector and a down projector in a Llama layer. The epilogue 
    would have the option to be run on the result of a non-reduced split K GEMM. `(medium prio)`
- A compute and communicate kernel for row-TP GEMMs. `(low prio)`

## Existing kernels

### Residual rms

- A new version that leverages shared memory to avoid the global load in the second loop. `(medium prio)`
- Check the kernel performance using rocm tools. `(low prio)`
