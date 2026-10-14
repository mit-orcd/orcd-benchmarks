# b200-ubuntu — installation

The Ubuntu B200 nodes (node5700/node5701) have **no Lmod and no
`/orcd/software` mount**, so the NVHPC SDK is installed by hand on the shared
filesystem. Nothing is built in this dir.

> These steps record the 2026-08 Ubuntu 24.04 setup. The nodes have since been
> reinstalled with Rocky 8. To rerun there, use the `../b200-nodes` toolchain
> (`module load nvhpc/26.1`, `build-nvhpc-26.1`).

| Piece | Location | Used by |
|---|---|---|
| NVHPC 26.1 SDK | `/orcd/data/orcd/022/benchmarks/nvhpc/Linux_x86_64/26.1` | `run-nccl-*.sh` |
| nccl-tests (CUDA 12.9 build) | `../nccl-tests/build-utuntu-nvhpc-26.1` | `run-nccl-*.sh` |
| gpu-fryer image | `../gpu-fryer/gpu-fryer_1.1.0.sif` | `run-gpu-fryer.sh` |
| apptainer | `/usr/bin/apptainer` on the nodes | `run-gpu-fryer.sh` |

## 1. NVHPC 26.1 SDK (16 GB tarball, ~40 GB installed)

```bash
cd /orcd/data/orcd/022/benchmarks/nvhpc
wget https://developer.download.nvidia.com/hpc-sdk/26.1/nvhpc_2026_261_Linux_x86_64_cuda_multi.tar.gz
tar xpzf nvhpc_2026_261_Linux_x86_64_cuda_multi.tar.gz
NVHPC_SILENT=true \
NVHPC_INSTALL_DIR=/orcd/data/orcd/022/benchmarks/nvhpc \
NVHPC_INSTALL_TYPE=single \
  ./nvhpc_2026_261_Linux_x86_64_cuda_multi/install
ls /orcd/data/orcd/022/benchmarks/nvhpc/Linux_x86_64/26.1/comm_libs/12.9/{nccl,hpcx}
```

The `cuda_multi` bundle includes CUDA 12.9 and 13.1. These nodes run driver
570.211.01, which supports CUDA 12.8 at most, so everything here uses the
**12.9** flavour: `cuda/12.9`, `comm_libs/12.9/hpcx/latest/ompi` (HPC-X Open MPI
4.1.9) and `comm_libs/12.9/nccl` (NCCL 2.29.2). CUDA 13 would need an r580+
driver. After the install you can delete the tarball and the unpacked
installer dir.

## 2. nccl-tests against that SDK

Run this on node5700 or node5701:

```bash
ssh node5700
cd /orcd/data/orcd/022/benchmarks/nccl-tests
./build-utuntu-nvhpc-26.1.sh
```

The script exports `PATH` and `LD_LIBRARY_PATH` for the 12.9 flavour and runs:

```bash
make -j MPI=1 MPI_HOME=$NVHPC_HOME/comm_libs/12.9/hpcx/latest/ompi \
     CUDA_HOME=$NVHPC_HOME/cuda/12.9 NCCL_HOME=$NVHPC_HOME/comm_libs/12.9/nccl \
     NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_100,code=compute_100" \
     BUILDDIR=$PWD/build-utuntu-nvhpc-26.1
```

It ends with an `ldd` check, in which no library may be `not found`.

## 3. gpu-fryer image

```bash
cd /orcd/data/orcd/022/benchmarks/gpu-fryer
apptainer pull gpu-fryer_1.1.0.sif docker://ghcr.io/huggingface/gpu-fryer:1.1.0
```

On Ubuntu, `/lib64` holds only the loader, so `run-gpu-fryer.sh` does not
bind-mount it. It points gpu-fryer at the NVML that `--nv` injects
(`/.singularity.d/libs/libnvidia-ml.so.1`).

## 4. Admin prerequisites for 2-node NCCL

`run-nccl-2node.sh` launches over ssh with `mpirun`. This needs passwordless
ssh between node5700 and node5701 in both directions, plus the IB/RDMA settings
listed in `ubuntu-nccl.md` and `admin-nccl-notes.md`.
