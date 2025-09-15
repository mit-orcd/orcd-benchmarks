#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 60
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4  
#SBATCH --mem=50GB

node=`hostname`

mkdir $1
./gpu_burn -tc 300 >$1/out.tc.$node
./gpu_burn 300 >$1/out.sp.$node
./gpu_burn -d 300 >$1/out.dp.$node

