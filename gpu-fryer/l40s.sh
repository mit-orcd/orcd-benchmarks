#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -q unlimited
#SBATCH -t 01:00:00
#SBATCH --mem=30GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o output/%N-%J.out

module load apptainer/1.4.2
#module load nvhpc/24.5
which singularity

SING_CMD="singularity exec --nv -B /lib64:/home/shaohao/lib64 gpu-fryer_1.1.0.sif "
FLAGS="--nvml-lib-path /home/shaohao/lib64/libnvidia-ml.so.1"
ELAPSE="300"

echo "Number of GPUs = $SLURM_GPUS $SLURM_GPUS_ON_NODE"

echo "======== Run with fp32 =========="
$SING_CMD gpu-fryer --use-fp32 $FLAGS $ELAPSE
echo "======== Run with bf16 =========="
$SING_CMD gpu-fryer --use-bf16 $FLAGS $ELAPSE
echo "======== Run with fp8  =========="
$SING_CMD gpu-fryer --use-fp8 $FLAGS $ELAPSE

#singularity exec --nv -B /lib64:/home/shaohao/lib64 gpu-fryer_1.1.0.sif \
#	    gpu-fryer --nvml-lib-path /home/shaohao/lib64/libnvidia-ml.so.1 300

