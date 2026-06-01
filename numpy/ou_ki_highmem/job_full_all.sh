#!/bin/bash
#Author: justinwz

nodes=(9806 9807 9810 9811)

for i in ${!nodes[@]}; do
	node=${nodes[i]}
	host=node$node
	sbatch << EOF
#!/bin/bash
#SBATCH -t 30
#SBATCH -p ou_ki_highmem
#SBATCH -n 96
#SBATCH -N 1
#SBATCH -w node$node
#SBATCH -o full_cores/out_full.%N-%J

for j in 1 2 4 8 16 48 72 96 144 196
do
     export OMP_NUM_THREADS=\$j
     echo "Ran with OMP_NUM_THREADS=\$OMP_NUM_THREADS"
     python mat_mult.py
done

EOF

done

