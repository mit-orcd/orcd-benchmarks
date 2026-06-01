#!/bin/bash
#SBATCH -J bw-imagenet
#SBATCH -p mit_normal
#SBATCH -t 60
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=30GB
#SBATCH -o out/%x-%N-%j

module load miniforge/24.3.0-0
source activate torch

python bw-imagenet.py 

