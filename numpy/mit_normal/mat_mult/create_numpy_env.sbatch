#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p mit_normal

module load miniforge/24.3.0-0

conda create -n benchmark numpy

