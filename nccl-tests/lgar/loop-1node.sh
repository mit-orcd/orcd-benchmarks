#!/bin/bash
#for node in `cat ../../nodes/nodes-mit_normal_gpu-1`
for node in `cat ../../nodes/nodes-mit_normal_gpu-h200s`
do
   echo $node
   sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w $node 1node.sh
done

