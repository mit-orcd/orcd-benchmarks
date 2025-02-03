#!/bin/bash

nodes=(1600 1601 1602 1603 1604 1605 1606 1607 1608 1609 1610 1611 1612 1613 1614 1615 1616 1617 1618 1619)

for node in "${nodes[@]}"; do
    sbatch << EOF
#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 96
#SBATCH -t 60
#SBATCH -w node$node
#SBATCH -o out_files/new/%N.out
#SBATCH --exclusive

hostname

module load miniforge/24.3.0-0
conda activate benchmark

for i in 1 2 4 8 16 48 72 96 144 192
do 
  export OMP_NUM_THREADS=\$i
  echo "====== Run with \$OMP_NUM_THREADS threads. ======"
  time python ../mat_mult.py 
done

EOF

done
