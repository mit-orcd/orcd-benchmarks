
# chris-openmp-gpu

This directory contains example commands and scripts for building and running OpenMP GPU benchmarks using LLVM and Apptainer on NVIDIA hardware.

## Usage

1. **Build the Apptainer container**  
   *(Optional, only if you need to build the container image)*  
   ```bash
   apptainer build --fakeroot rocky8-llvm.sif rocky8-llvm.def
   ```

2. **Load required modules**  
   ```bash
   module load apptainer
   module load cuda
   ```

3. **Start an Apptainer shell with CUDA and custom path bindings**  
   ```bash
   apptainer shell --nv --env PATH="${PATH}" \
       $(echo $PATH | tr ':' '\n' | grep -v /home/cnh/bin | grep -v /usr/bin | awk '{printf "--bind %s ", $1}') \
       --bind /orcd/software/ rocky8-llvm.sif
   ```

4. **Compile and run the OpenMP GPU bandwidth benchmark**  
   ```bash
   clang -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -o d2d_bandwidth omp_device_to_device_bandwidth.c
   ./d2d_bandwidth
   ```

## Files

- `commands`: Example commands for container setup and running benchmarks.
- `omp_device_to_device_bandwidth.c`: C source file for the device-to-device bandwidth benchmark.

## Requirements

- Apptainer/Singularity
- CUDA-enabled NVIDIA GPU
- LLVM (with OpenMP GPU offload support)

## Reference

This setup is used for benchmarking OpenMP GPU offloading using LLVM on Rocky Linux 8.

