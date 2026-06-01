#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -q unlimited
#SBATCH -t 06:00:00
#SBATCH -n 2
#SBATCH --mem=30GB

module load apptainer
#apptainer pull pytorch_24.01-py3.sif docker://nvcr.io/nvidia/pytorch:24.01-py3
#apptainer pull pytorch_24.07-py3.sif docker://nvcr.io/nvidia/pytorch:24.07-py3
apptainer pull pytorch_26.02-py3.sif docker://nvcr.io/nvidia/pytorch:26.02-py3

