#!/bin/bash
#SBATCH -J bw-llama
#SBATCH -p mit_normal
#SBATCH -q unlimited
#SBATCH -t 02:00:00  # 6-00:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=10GB
#SBATCH -o out-llama/%x-%N-%j

module load miniforge/24.3.0-0
source activate torch

#file="/orcd/datasets/001/shaohao-staging/data-llama/llama_tokenized_data/wikipedia_tokenized.pt"
#python bw-llama.py $file

python bw-llama.py $1

