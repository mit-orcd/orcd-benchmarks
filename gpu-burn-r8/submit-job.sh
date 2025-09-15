#!/bin/bash

partition=pi_mghassem
res=orcd_testing
flag="--reservation=$res -p $partition"

#sbatch $flag --gres=gpu:10 -w node3800 1node.sh $partition
sbatch $flag --gres=gpu:10 -w node3801 1node.sh $partition
#sbatch $flag --gres=gpu:7 -w node3802 1node.sh $partition
#sbatch $flag --gres=gpu:8 -w node3900 1node.sh $partition
#sbatch $flag --gres=gpu:5 -w node3901 1node.sh $partition
sbatch $flag --gres=gpu:9 -w node3902 1node.sh $partition


