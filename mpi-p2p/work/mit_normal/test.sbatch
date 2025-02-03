#!/bin/bash

node_1=$1
node_2=$2

#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p $1
#SBATCH -o test.out
#SBATCH -e test.err

echo "Test successful!"

