#!/bin/bash

nodes=(1620 1621 1622 1623 1624 1625)

for node in "${nodes[@]}"; do
    sbatch << EOF
#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 192
#SBATCH -t 60
#SBATCH -w node$node
#SBATCH -o out_files/new/%N.out
#SBATCH --exclusive

hostname

module load miniforge/24.3.0-0
conda activate benchmark

for i in 1 2 4 8 16 96 144 192 288 384
do 
  export OMP_NUM_THREADS=\$i
  echo "====== Run with \$OMP_NUM_THREADS threads. ======"
  time python ../mat_mult.py 
done

EOF

done

