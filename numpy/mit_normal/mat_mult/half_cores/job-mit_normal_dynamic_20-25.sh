#!/bin/bash

nodes=(1620 1621 1622 1623 1625) # 1624 is drained

for node in "${nodes[@]}"; do
    sbatch << EOF
#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 96
#SBATCH -t 60
#SBATCH -w node$node
#SBATCH -o out_files/%N-%J.out
#SBATCH --exclusive

hostname

module load miniforge/24.3.0-0
conda activate benchmark

for i in 1 2 4 8 16 48 72 96 144 192
do 
  export OMP_NUM_THREADS=\$i
  echo "====== Run with \$OMP_NUM_THREADS threads. ======"
  time python ../../mat_mult.py 
done

EOF

done

