#!/bin/bash
#SBATCH --output=/dev/null
# script to run cuda jobs lazily
# specify the nodes you want to run the job on
# specify the partition that the nodes are on (must be the same paritition)
# specify the executables you want to run (separates single/double precision into separate folders)
# change other parameters if needed
# author: justinwz@mit.edu

nodelist=(2501)
partition="pi_linaresr"

for i in ${!nodelist[@]}; do
	host=node${nodelist[i]}
	sbatch << EOF
#!/bin/bash
#SBATCH -p "${partition}"
#SBATCH -w "${host}"
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o test.out
#SBATCH -t 500
#SBATCH --mem=100G

module load cuda/12.4.0

echo "Running job on $host with parition $partition"
exe32p=("HelloWorld") # list executables for single precision programs you want to run
exe64p=("HelloWorld") # list executables for double precision programs you want to run

for i in \${!exe32p[@]}; do
    exe=\${exe32p[i]}
    echo "Running \${exe} single precision, ${host}"
    ./\${exe} > ../out_files/single_precision/${partition}/\${exe}_${host}_${SLURM_JOB_ID}.out
done

for i in \${!exe64p[@]}; do
	exe=\${exe64p[i]}
	echo "Running \${exe} double precision ${host}"
	./\${exe} > ../out_files/double_precision/${partition}/\${exe}_${host}_${SLURM_JOB_ID}.out
done
EOF

done


