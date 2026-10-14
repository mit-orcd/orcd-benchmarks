# nccl-tests — installation

This dir is a clone of NVIDIA nccl-tests (`src/`, `Makefile`), built with
`make MPI=1` into a separate `build-<stack>/` dir for each compiler/MPI stack.
The CUDA toolkit and NCCL come from NVHPC.

- Source: <https://github.com/NVIDIA/nccl-tests>
- Its `.git` pack file is corrupt here, so re-clone if you need git history.

Build on a compute node. With `MPI=1` the build does not need a GPU, but it
does need the module stack you will run with.

## Fresh clone (only if the source is missing)

```bash
cd /orcd/data/orcd/022/benchmarks
git clone https://github.com/NVIDIA/nccl-tests.git nccl-tests-new   # copy src/ and Makefile from here
```

## Build used by `run/` (L40S / H100 / H200): `build-nvhpc-24.5-ompi-5.0.8`

```bash
cd /orcd/data/orcd/022/benchmarks/nccl-tests
module purge
module load nvhpc/24.5
module load openmpi/5.0.8        # with nvhpc loaded this is /orcd/software/core/001/pkg/openmpi/5.0.8-nvhpc-24.5
MPI_HOME=$(dirname $(dirname $(which mpicc)))
make clean
make -j 8 MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$NVHPC_ROOT/cuda NCCL_HOME=$NVHPC_ROOT/comm_libs/nccl
mv build build-nvhpc-24.5-ompi-5.0.8
```

`run/env.sh` sets `BUILD_DIR=../build-nvhpc-24.5-ompi-5.0.8` and loads the same
two modules. The notes record that this was the only combination that gave
full inter-node bandwidth: about 15–23 GB/s, against about 1.9 GB/s with
nvhpc's bundled MPI.

## B200 build (CUDA 13): `build-nvhpc-26.1`

```bash
./build-nvhpc-26.1.sh
```

This loads `nvhpc/26.1` only and uses its bundled HPC-X Open MPI
(`comm_libs/13.1/hpcx/hpcx-2.25.1/ompi`). It builds for `sm_90` + `sm_100`
(plus `compute_100` PTX) and is used by `b200-nodes`. `build-utuntu-nvhpc-26.1.sh`
does the same for the Ubuntu B200 image (`b200-ubuntu`).

## Other stacks

| Script | Result |
|---|---|
| `build.sh` | `build-nvhpc-24.5-ompi-5.0.6` (needs the retired `openmpi/5.0.6` module) |
| `build-nvhpc.sh` | `build-nvhpc-24.5`, nvhpc's bundled MPI |
| `build-ompi.sh` | `build-nvhpc-24.5-ompi-4.1.7-avx2`, for the AVX2-only nodes; build on such a node |

Each script `rm -rf`s its own build dir before building. `notes.install`
records older builds (nvhpc 23.3, Rocky-8 Open MPI 4.1.4).

## Check

```bash
salloc -N 1 -n 1 --gres=gpu:2 -p mit_normal_gpu -t 15
cd /orcd/data/orcd/022/benchmarks/nccl-tests/run && source env.sh
mpirun -np 1 $BUILD_DIR/sendrecv_perf -b 1M -e 64M -f 4 -g 2
```
