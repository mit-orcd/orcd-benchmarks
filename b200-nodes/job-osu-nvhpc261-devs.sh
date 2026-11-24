#!/bin/bash
#SBATCH -J osu-nvhpc-devs
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --gpus-per-node=b200:1
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /orcd/data/orcd/022/benchmarks/b200-nodes/out-osu-ib/nvhpc-26.1/devs-%j.out
#
# Confirm which IB HCAs UCX selects for inter-node traffic with the CUDA-aware
# HPC-X OpenMPI in nvhpc/26.1, default settings (no UCX_NET_DEVICES).
# UCX_PROTO_INFO=y prints, per endpoint, the protocol / transport / device
# chosen for each message-size range. Host (H H) and GPU (D D) buffers.

B=/orcd/data/orcd/022/benchmarks
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
module purge >/dev/null 2>&1
module load nvhpc/26.1
NVHPC_HOME=/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1
HPCX=$NVHPC_HOME/comm_libs/13.1/hpcx/hpcx-2.25.1
export PATH=$HPCX/ompi/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ompi/lib:$LD_LIBRARY_PATH
OSU=$B/mpi-p2p/install/r8-nvhpc-26.1/libexec/osu-micro-benchmarks/mpi/pt2pt

echo "date: $(date '+%F %T %Z')  job: $SLURM_JOB_ID  nodes: $SLURM_JOB_NODELIST"
for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    srun -N1 -n1 -w "$n" --overlap bash -c 'echo "gpu $(hostname): $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader)"' < /dev/null
done
srun -N1 -n1 --overlap bash -c "ldd $OSU/osu_bw | grep -E 'libucp|libmpi'" < /dev/null
$HPCX/ucx/bin/ucx_info -v 2>&1 | head -2

export UCX_PROTO_INFO=y
for buf in "H H" "D D"; do
    da=""; [ "$buf" = "D D" ] && da="-d cuda"
    for t in osu_latency osu_bw; do
        echo "===== $t $buf (default settings, UCX_PROTO_INFO=y)"
        timeout 300 mpirun -np 2 --map-by ppr:1:node --bind-to none \
            -x LD_LIBRARY_PATH -x PATH -x UCX_PROTO_INFO \
            "$OSU/$t" -m 1:4194304 -i 200 -x 20 $da $buf < /dev/null 2>&1
        echo "===== RC $t $buf $?"
    done
done
echo "done: $(date '+%F %T %Z')"
