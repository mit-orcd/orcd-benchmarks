#!/bin/bash
#Author: justinwz

# nodes=(9800 9801 9802 9803 9804 9805 9808 9089)
nodes=(9809) # problematic - keep getting invalid node name

for i in ${!nodes[@]}; do
	node=${nodes[i]}
	host=node$node
	sbatch << EOF
#!/bin/bash
#SBATCH -t 30
#SBATCH -p ou_ki
#SBATCH -n 48 
#SBATCH -N 1
#SBATCH -w node$node
#SBATCH -o half_cores/out_half.%N-%J

hostname

for j in 1 2 4 8 16 24 36 48 72 96
do
     export OMP_NUM_THREADS=\$j
     echo "Ran with OMP_NUM_THREADS=\$OMP_NUM_THREADS"
     python mat_mult.py
done

EOF

done

