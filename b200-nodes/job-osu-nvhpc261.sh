#!/bin/bash
#SBATCH -J osu-nvhpc261
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --gpus-per-node=b200:1
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -o /orcd/data/orcd/022/benchmarks/b200-nodes/out-osu-ib/nvhpc-26.1/osu-nvhpc261-%j.out
#
# CUDA-aware OSU MPI point-to-point (osu_latency / osu_bw / osu_bibw) with the
# HPC-X OpenMPI 4.1.9a1 bundled in nvhpc/26.1, on any two mit_testing (B200)
# nodes Slurm picks: 1 rank (1 core) + 1 B200 per node. Buffers in host memory
# (H H) and in GPU memory (D D, `-d cuda`). Each with default settings, with
# UCX_NET_DEVICES=mlx5_4:1, and with UCX limited to the 8 NDR HCAs (UCX can then
# pick the HCA closest to the GPU). --bind-to none: HPC-X mpirun otherwise
# binds to CPU 0,112, outside the 1-core Slurm allocation, and aborts. analyze-osu-ompi508.py writes the report.

B=/orcd/data/orcd/022/benchmarks
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
module purge >/dev/null 2>&1
module load nvhpc/26.1
NVHPC_HOME=/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1
OMPI_HOME=$NVHPC_HOME/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi
export PATH=$OMPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$OMPI_HOME/lib:$LD_LIBRARY_PATH
OSU=$B/mpi-p2p/install/r8-nvhpc-26.1/libexec/osu-micro-benchmarks/mpi/pt2pt
NDR=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_13:1,mlx5_14:1,mlx5_15:1

echo "date: $(date '+%F %T %Z')"
echo "job: $SLURM_JOB_ID nodes: $SLURM_JOB_NODELIST"
for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    echo "node: $n others=$(squeue -h -w "$n" -o %i | grep -vc "^$SLURM_JOB_ID\$")"
    srun -N1 -n1 -w "$n" --overlap bash -c 'echo "gpu $(hostname): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | tr "\n" " ")"' < /dev/null
done
echo "mpirun: $(which mpirun)"

# name | modulepath | build | env ("-" = default settings) | OSU buffer args
STACKS=(
"nvhpc-HH|-|r8-nvhpc-26.1|-|"
"nvhpc-HH|-|r8-nvhpc-26.1|UCX_NET_DEVICES=mlx5_4:1|"
"nvhpc-DD|-|r8-nvhpc-26.1|-|-d cuda D D"
"nvhpc-DD|-|r8-nvhpc-26.1|UCX_NET_DEVICES=mlx5_4:1|-d cuda D D"
"nvhpc-DD|-|r8-nvhpc-26.1|UCX_NET_DEVICES=$NDR|-d cuda D D"
)

for s in "${STACKS[@]}"; do
    IFS='|' read -r name mpath build envv bufargs <<< "$s"
    (
    echo "===== STACK $name modulepath=$mpath build=$build env=$envv"
    xe=""
    if [ "$envv" != "-" ]; then export "$envv"; xe="-x ${envv%%=*}"; fi
    mpirun --version 2>&1 < /dev/null | head -1
    for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
        miss=$(srun -N1 -n1 -w "$n" --overlap bash -c "ldd $OSU/osu_latency | awk '/not found/{print \$1}' | tr '\n' ' '" < /dev/null)
        echo "preflight $n: ${miss:-OK}"
    done
    for t in osu_latency osu_bw osu_bibw; do
        echo "----- TEST $t"
        args="$bufargs"; [ "$t" != osu_latency ] && args="-m 1:16777216 $bufargs"
        timeout 300 mpirun -np 2 --map-by ppr:1:node --bind-to none -x LD_LIBRARY_PATH -x PATH $xe \
            "$OSU/$t" $args < /dev/null 2>&1
        echo "----- RC $t $?"
    done
    )
done

python3 $B/b200-nodes/analyze-osu-ompi508.py nvhpc-26.1 "CUDA-aware OSU MPI p2p on B200: nvhpc/26.1 HPC-X OpenMPI 4.1.9a1"
echo "done: $(date '+%F %T %Z')"
