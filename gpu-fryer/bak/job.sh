#!/bin/bash
#SBATACH -p mit_normal_gpu
#SBATACH -q unlimited
#SBATACH -t 60
#SBATACH -n 2
#SBATACH --mem=30GB
#SBATACH --gres=gpu:h200:1
#SBATACH -o output/%N-%x-%J.out

module load apptainer/1.4.2
singularity exec --nv -B /lib64:/home/shaohao/lib64 gpu-fryer_1.1.0.sif \
	    gpu-fryer --nvml-lib-path /home/shaohao/lib64/libnvidia-ml.so.1 300

