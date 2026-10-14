# dgx-nodes — installation

The DGX H100 nodes (node1700, node1800, …; see `notes`) run **Ubuntu 24.04**,
without Lmod or `/orcd/software`. The NCCL tests therefore use the private
NVHPC 26.1 SDK on the shared filesystem and a separate nccl-tests build made on
an Ubuntu node. Nothing is built in this dir.

| Piece | Location |
|---|---|
| NVHPC 26.1 SDK (CUDA 12.9 flavour) | `/orcd/data/orcd/022/benchmarks/nvhpc/Linux_x86_64/26.1` |
| nccl-tests, H100 build | `../nccl-tests/build-songhan-utuntu-nvhpc-26.1` |
| Run-time env | `env-ubuntu.sh` (sourced by `job-nccl-*.sh`) |

## 1. NVHPC 26.1 SDK (shared with `b200-ubuntu`; skip if `../nvhpc/Linux_x86_64/26.1` exists)

```bash
cd /orcd/data/orcd/022/benchmarks/nvhpc
wget https://developer.download.nvidia.com/hpc-sdk/26.1/nvhpc_2026_261_Linux_x86_64_cuda_multi.tar.gz
tar xpzf nvhpc_2026_261_Linux_x86_64_cuda_multi.tar.gz
NVHPC_SILENT=true \
NVHPC_INSTALL_DIR=/orcd/data/orcd/022/benchmarks/nvhpc \
NVHPC_INSTALL_TYPE=single \
  ./nvhpc_2026_261_Linux_x86_64_cuda_multi/install
```

## 2. nccl-tests for H100 (build on an Ubuntu DGX node)

The login node runs RHEL, whose glibc and rdma-core differ, so build on the DGX
node itself:

```bash
ssh node1800 /orcd/data/orcd/022/benchmarks/nccl-tests/build-songhan-utuntu-nvhpc-26.1.sh
```

The script uses the CUDA 12.9 flavour (`cuda/12.9`, `comm_libs/12.9/hpcx/latest/ompi`,
`comm_libs/12.9/nccl`), even though driver 590 could run CUDA 13, so that the
build matches `env-ubuntu.sh`. It runs:

```bash
make -j 16 MPI=1 MPI_HOME=$NVHPC_HOME/comm_libs/12.9/hpcx/latest/ompi \
     CUDA_HOME=$NVHPC_HOME/cuda/12.9 NCCL_HOME=$NVHPC_HOME/comm_libs/12.9/nccl \
     NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90" \
     BUILDDIR=$PWD/build-songhan-utuntu-nvhpc-26.1
```

## 3. Run-time environment

There is nothing to install. `env-ubuntu.sh` exports the same SDK paths and
stops if the build is missing. It keeps the MPI bootstrap on plain TCP
(`--mca pml ob1 --mca btl tcp,self`, UCC/HCOLL off) over `ETH_IF=ens6f0np0`,
because the HPC-X UCX/UCC components can crash in `MPI_Init` on these nodes.
NCCL itself still uses NVLink and InfiniBand. If a node names its routable NIC
differently, set `ETH_IF`.

`ib-check/` (job-ibtest.sh, job-tune.sh) needs nothing extra; it uses the same build and env.
