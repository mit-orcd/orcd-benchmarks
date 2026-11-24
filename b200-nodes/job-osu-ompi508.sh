#!/bin/bash
#SBATCH -J osu-ompi508
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /orcd/data/orcd/022/benchmarks/b200-nodes/out-osu-ib/ompi-5.0.8/osu-ompi508-%j.out
#
# OSU MPI point-to-point (osu_latency / osu_bw / osu_bibw) with the two
# different modules that are both called openmpi/5.0.8, on whatever two
# mit_testing (B200) nodes Slurm picks. 1 rank (1 core) per node, default
# settings (no UCX / MCA tuning). Both modules run back to back on the same
# node pair, each with default settings and with PMIX_MCA_gds=hash (the
# workaround PMIx prints for its gds/shmem2 attach error). analyze-osu-ompi508.py writes the report at the end.

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
"spack|/orcd/software/core/001/spack/modulefiles/gcc/12.2.0|r8-spack-ompi-5.0.8|-"
"spack|/orcd/software/core/001/spack/modulefiles/gcc/12.2.0|r8-spack-ompi-5.0.8|PMIX_MCA_gds=hash"
"core|/orcd/software/core/001/modulefiles|r8-core-ompi-5.0.8|-"
"core|/orcd/software/core/001/modulefiles|r8-core-ompi-5.0.8|PMIX_MCA_gds=hash"
)

for s in "${STACKS[@]}"; do
    IFS='|' read -r name mpath build envv <<< "$s"
    (
    echo "===== STACK $name modulepath=$mpath build=$build env=$envv"
    module purge >/dev/null 2>&1
    module load gcc/12.2.0
    module use "$mpath"
    module load openmpi/5.0.8
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

python3 $B/b200-nodes/analyze-osu-ompi508.py
echo "done: $(date '+%F %T %Z')"
