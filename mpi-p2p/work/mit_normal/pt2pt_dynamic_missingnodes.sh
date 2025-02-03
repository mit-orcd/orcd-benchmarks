#!/bin/bash

nodes=(1615 1619 1624)

for i in ${!nodes[@]}; do
        node1=${nodes[$i]}
        node2=${nodes[$i+1]}

        sbatch << EOF
#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -t 10
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB   # mem per node
#SBATCH --constraint=rocky8
#SBATCH -w node[$node1,$node2] 

#SBATCH -o out_files_copy/out.%N-%J

source ./env.sh $1 $2

#
echo "number of nodes = \$SLURM_NNODES"

# -n --> SLURM_NTASKS
echo "total number of tasks = \$SLURM_NTASKS"

# Turn off hyperthreads
echo "number of tasks per core = \$SLURM_NTASKS_PER_CORE"

# ntasks-per-node --> SLURM_NTASKS_PER_NODE

# -c --> SLURM_CPUS_PER_TASK
echo "number of cores per task = \$SLURM_CPUS_PER_TASK"

#
echo "total number of virutal cores (hyperthreads) per node = \$SLURM_CPUS_ON_NODE"
echo "total memory per node = \$SLURM_MEM_PER_NODE"


echo "--- mpirun ---"
which mpirun
mpirun -n \$SLURM_NTASKS hostname
echo "--- srun ---"
srun hostname
echo "--- osu_bw ---"
mpirun -n \$SLURM_NTASKS osu_bw
#mpirun --mca pml ucx -n \$SLURM_NTASKS osu_bw
echo "--- osu_latency ---"
mpirun -n \$SLURM_NTASKS osu_latency
#mpirun --mca pml ucx -n \$SLURM_NTASKS osu_latency

EOF

    
done

