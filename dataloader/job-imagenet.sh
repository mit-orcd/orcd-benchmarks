#!/bin/bash
#SBATCH -J bw-imagenet
#SBATCH -p mit_normal # mit_normal_gpu  # mit_normal
#SBATCH -q unlimited
#SBATCH -t 06:00:00  # 6-00:00:00
#SBATCH -N 1
#SBATCH -n 64  # 64   # 96 
#SBATCH --mem=80GB   # 128GB
#SBATCH -o out-imagenet/%x-%N-%j
#  #SBATCH -x node[1620-1625]
#SBATCH --exclusive

module load miniforge/24.3.0-0
source activate torch

#dir="/orcd/datasets/001/imagenet/images_complete/ilsvrc"
dir="/orcd/scratch/orcd/008/shaohao/imagenet/images_complete/ilsvrc"
#dir="/orcd/scratch/orcd/012/shaohao/imagenet/images_complete/ilsvrc"
#dir="/orcd/scratch/orcd/013/shaohao/imagenet/images_complete/ilsvrc"

ls -d $dir
python bw-imagenet.py $dir

