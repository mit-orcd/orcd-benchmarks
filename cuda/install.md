# cuda — installation

The programs in `src/` are single-file CUDA codes built with `nvcc`. Nothing is
downloaded. `cuda-examples/`, `cuda-test/`, `cuSolver/`, `github-cuda-ex/` and
`gpu-harvard-workshop/` are reference material and are not part of the
benchmark.

**The prebuilt binaries in `src/` are corrupt.** Their contents are zero bytes
(`file` reports `data`), so you must rebuild everything before running.

## Toolchain

```bash
module load cuda/12.9.1
```

`src/run_job.sh` runs `module load cuda/12.4.0`. That module now lives in the
deprecated tree and loads only after `module load deprecated-modules`.
Either add that line before it, or change it to `module load cuda/12.9.1`. nvcc links cudart
statically, but the cuBLAS programs load `libcublas` at run time, so keep the
module load in the job script.

## Build

Build on a node with the target GPU, or pass the right `-arch`: `sm_80` A100,
`sm_89` L40S, `sm_90` H100/H200, `sm_100` B200 (CUDA ≥ 12.8). The `_h100` and
`_l40s` suffixes mean "built for that GPU".

```bash
cd /orcd/data/orcd/022/benchmarks/cuda/src
ARCH=sm_90       # sm_89 for L40S
SUF=h100         # l40s for L40S
TMPD=$(mktemp -d -p ~/orcd/scratch)

# Naive matmul: single precision (MatMult) and double precision (MatMult_d), 10000 x 10000, 16x16 thread block
nvcc -O3 -arch=$ARCH MatMult.cu   -o MatMult
nvcc -O3 -arch=$ARCH MatMult_d.cu -o MatMult_d_$SUF

# Thread-block variants. The sizes are #defines, so make an edited copy
for t in "16 16" "16 32"; do
  set -- $t
  sed -e "s/#define numThreads_x .*/#define numThreads_x $1/" \
      -e "s/#define numThreads_y .*/#define numThreads_y $2/" MatMult.cu   > $TMPD/mm.cu
  nvcc -O3 -arch=$ARCH $TMPD/mm.cu -o MatMult_${1}x${2}
  sed -e "s/#define numThreads_x .*/#define numThreads_x $1/" \
      -e "s/#define numThreads_y .*/#define numThreads_y $2/" MatMult_d.cu > $TMPD/mmd.cu
  nvcc -O3 -arch=$ARCH $TMPD/mmd.cu -o MatMult_d_${SUF}_${1}x${2}
done
rm -rf $TMPD

# cuBLAS matmul (single precision, double precision)
nvcc -O3 -arch=$ARCH matmul_cuBLAS.cu   -o matmul_cuBLAS            -lcublas
nvcc -O3 -arch=$ARCH matmul_cuBLAS_d.cu -o matmul_cuBLAS_d_$SUF     -lcublas

# Small examples
nvcc -O3 -arch=$ARCH SAXPY.cu       -o SAXPY
nvcc -O3 -arch=$ARCH dot.cu         -o dot
nvcc -O3 -arch=$ARCH CalculatePi.cu -o CalculatePi
nvcc -O3 -arch=$ARCH HelloWorld.cu  -o HelloWorld
```

The `*_small` binaries were built from the same sources with a smaller `dim`
(edit `#define dim` the same way).

## Output dirs

`run_job.sh` writes to `../out_files/<precision>/<partition>/` relative to
`src/`, and does not create those dirs:

```bash
mkdir -p /orcd/data/orcd/022/benchmarks/cuda/out_files/{single,double}_precision/<partition>
```

## Check

```bash
srun -t 10 --gres=gpu:1 -p mit_normal_gpu --pty ./HelloWorld
```
