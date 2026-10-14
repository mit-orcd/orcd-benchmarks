# gpu-burn-r8 — installation

gpu-burn is built from the source in this dir: `gpu_burn-drv.cpp`,
`compare.cu` and the `Makefile`. These are a copy of upstream.

- Upstream: <https://github.com/wilicc/gpu-burn> (<http://wili.cc/blog/gpu-burn.html>)

## Build

The working `gpu_burn` was built with gcc 12.2.0 and the CUDA 12.4 toolkit in
`/orcd/software/core/001/pkg/cuda/12.4`. The `CUDAPATH` hard-coded in the
`Makefile` is an old Rocky-8 spack path that no longer works, so override it on
the command line:

```bash
cd /orcd/data/orcd/022/benchmarks/gpu-burn-r8
module load gcc/12.2.0
make clean
make CUDAPATH=/orcd/software/core/001/pkg/cuda/12.4 COMPUTE=80
# -> gpu_burn and compare.ptx
```

- `COMPUTE=80` builds `compare.ptx` for `compute_80`. The driver JIT-compiles
  it for newer GPUs (L40S, H100/H200, B200), so one build covers all of them.
- The binary has an rpath to `cuda/12.4/lib64`, so jobs need no `module load`.
- To use a newer toolkit, use e.g. `CUDAPATH=/orcd/software/core/001/pkg/cuda/12.9.1`.

`gpu_burn` loads `compare.ptx` from the current directory at run time, so run
it from this dir. `run/run.sh` does a `cd` here first.

## Check

```bash
srun -t 10 -n 2 --gres=gpu:1 -p mit_normal_gpu --pty ./gpu_burn 30
# ends with: GPU 0: OK
```

## Optional: container

```bash
make image COMPUTE=80      # docker build using the upstream Dockerfile (needs docker)
```
