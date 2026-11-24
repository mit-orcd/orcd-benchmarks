#!/bin/bash
#SBATCH -J osu-ucxdev
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /orcd/data/orcd/022/benchmarks/b200-nodes/out-osu-ib/ucx-net-devices/osu-ucxdev-%j.out
#
# OSU MPI point-to-point (osu_latency / osu_bw / osu_bibw) with the default
# openmpi/4.1.4 module on any two mit_testing (B200) nodes Slurm picks, 1 rank
# (1 core) per node: once with default settings (control) and once with
# export UCX_NET_DEVICES=mlx5_4:1 (UCX limited to one NDR HCA, so it does not
# use the node-local HCAs mlx5_0-3). Report: out-osu-ib/ucx-net-devices/report.md

B=/orcd/data/orcd/022/benchmarks
OUT=$B/b200-nodes/out-osu-ib/ompi-5.0.8
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1

echo "date: $(date '+%F %T %Z')"
echo "job: $SLURM_JOB_ID nodes: $SLURM_JOB_NODELIST"
for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    echo "node: $n others=$(squeue -h -w "$n" -o %i | grep -vc "^$SLURM_JOB_ID\$")"
done

# name | modulepath | OSU build dir | extra env ("-" = default settings)
STACKS=(
"ompi414|-|r8-4.1.4|-"
"ompi414|-|r8-4.1.4|UCX_NET_DEVICES=mlx5_4:1"
)

for s in "${STACKS[@]}"; do
    IFS='|' read -r name mpath build envv <<< "$s"
    (
    echo "===== STACK $name modulepath=$mpath build=$build env=$envv"
    module purge >/dev/null 2>&1
    module load gcc/12.2.0
    [ "$mpath" != "-" ] && module use "$mpath"
    module load openmpi/4.1.4
    [ "$envv" != "-" ] && export "$envv"
    echo "mpirun: $(which mpirun)"
    mpirun --version 2>&1 < /dev/null | head -1
    OSU=$B/mpi-p2p/install/$build/libexec/osu-micro-benchmarks/mpi/pt2pt
    for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
        miss=$(srun -N1 -n1 -w "$n" --overlap bash -c "ldd $OSU/osu_latency | awk '/not found/{print \$1}' | tr '\n' ' '" < /dev/null)
        echo "preflight $n: ${miss:-OK}"
    done
    for t in osu_latency osu_bw osu_bibw; do
        echo "----- TEST $t"
        args=""; [ "$t" != osu_latency ] && args="-m 1:16777216"
        timeout 300 mpirun -np 2 --map-by ppr:1:node "$OSU/$t" $args < /dev/null 2>&1
        echo "----- RC $t $?"
    done
    )
done

python3 $B/b200-nodes/analyze-osu-ompi508.py ucx-net-devices "OSU MPI p2p on B200: openmpi/4.1.4 with UCX_NET_DEVICES=mlx5_4:1"
echo "done: $(date '+%F %T %Z')"
