#!/bin/bash
#SBATCH -t 200
#SBATCH -n 2
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

./build-nvhpc.sh r8 nvhpc 26.1

