#!/bin/bash
# Toolchain for the *Ubuntu* nccl-tests build, sourced by job-nccl-*.sh.
#
# The DGX H100 nodes in ./notes run Ubuntu 24.04, so they use
# nccl-tests/build-utuntu-nvhpc-26.1 (gencode sm_90 + sm_100), not the Rocky 8
# build that nccl-tests/run/env.sh selects.
#
# No `module load` here: that build links against the NVHPC 26.1 SDK installed
# on the shared filesystem under ../nvhpc, so the same absolute paths used at
# build time are exported at run time. CUDA 12.9 flavour is used throughout
# (cuda + hpcx + nccl), matching build-utuntu-nvhpc-26.1.sh.

NCCL_DIR=/orcd/data/orcd/022/benchmarks/nccl-tests
BUILD_DIR=${BUILD_DIR:-$NCCL_DIR/build-songhan-utuntu-nvhpc-26.1}

NVHPC_HOME=/orcd/data/orcd/022/benchmarks/nvhpc/Linux_x86_64/26.1
CUDA_VER=12.9
MPI_HOME=$NVHPC_HOME/comm_libs/$CUDA_VER/hpcx/latest/ompi   # HPC-X OpenMPI
NCCL_LIB=$NVHPC_HOME/comm_libs/$CUDA_VER/nccl/lib           # NCCL 2.29.2

export PATH=$NVHPC_HOME/cuda/$CUDA_VER/bin:$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$NCCL_LIB:$NVHPC_HOME/cuda/$CUDA_VER/lib64:${LD_LIBRARY_PATH:-}

if [ ! -x "$BUILD_DIR/sendrecv_perf" ]; then
   echo "nccl-tests not built for Ubuntu: $BUILD_DIR" >&2
   echo "Build it with: $NCCL_DIR/build-utuntu-nvhpc-26.1.sh" >&2
   exit 1
fi

# MPI is only used to exchange the small NCCL unique-id at start-up; the GPU
# data path is NCCL. The HPC-X UCC/UCX collective components can crash in
# MPI_Init on these nodes, so keep the bootstrap on plain TCP.
ETH_IF=${ETH_IF:-ens6f0np0}   # routable 10.1.x inband NIC (ibp41s0f0 is IPoIB)
MPI_FLAGS="--mca pml ob1 --mca btl tcp,self \
   --mca coll_ucc_enable 0 --mca coll_hcoll_enable 0 \
   --mca btl_openib_warn_no_device_params_found 0 \
   --mca btl_tcp_if_include $ETH_IF --mca oob_tcp_if_include $ETH_IF"

MIN_SIZE=${MIN_SIZE:-1M}
MAX_SIZE=${MAX_SIZE:-16G}
FACTOR=${FACTOR:-4}

# short collective name -> nccl-tests binary
declare -A BIN=(
   [sendrecv]=sendrecv_perf
   [allreduce]=all_reduce_perf
   [allgather]=all_gather_perf
   [reducescatter]=reduce_scatter_perf
   [reduce]=reduce_perf
   [broadcast]=broadcast_perf
   [alltoall]=alltoall_perf
   [gather]=gather_perf
   [scatter]=scatter_perf
   [hypercube]=hypercube_perf
)
ALL_ORDER="sendrecv allreduce allgather reducescatter reduce broadcast alltoall gather scatter hypercube"

# select_programs <comma/space separated list or "all">  -> fills PROGRAMS[]
select_programs() {
   PROGRAMS=()
   for tok in ${1//,/ }; do
      key=$(echo "$tok" | tr 'A-Z' 'a-z' | tr -d '_-')
      if [ "$key" = "all" ]; then
         for k in $ALL_ORDER; do PROGRAMS+=("${BIN[$k]}"); done
         continue
      fi
      if [ -n "${BIN[$key]:-}" ]; then
         PROGRAMS+=("${BIN[$key]}")
      else
         echo "Unknown collective: '$tok' (known: ${!BIN[*]} all)" >&2
         return 1
      fi
   done
}
