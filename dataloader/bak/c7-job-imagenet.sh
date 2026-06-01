#!/bin/bash
#SBATCH -J bw-imagenet
#SBATCH -p sched_mit_wvanrees  # sched_mit_psfc
#SBATCH -t 06:00:00  # 6-00:00:00
#SBATCH -N 1
#SBATCH -n 32  # 64   # 96 
#SBATCH --mem=30GB   # 128GB
#SBATCH -o out-test/%x-%N-%j

module load miniforge/24.3.0-0
source activate torch

#dir="/orcd/datasets/001/imagenet/images_complete/ilsvrc"
#dir="/orcd/scratch/orcd/008/shaohao/imagenet/images_complete/ilsvrc"
#dir="/orcd/scratch/orcd/012/shaohao/imagenet/images_complete/ilsvrc"
dir="/orcd/scratch/orcd/013/shaohao/imagenet/images_complete/ilsvrc"

ls -d $dir
python bw-imagenet.py $dir

