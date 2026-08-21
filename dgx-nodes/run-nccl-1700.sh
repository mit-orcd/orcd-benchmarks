#!/bin/bash
# NCCL benchmarks on node170[0,1] (excluded from run-nccl.sh because they are
# not in mit_testing). These nodes belong to pi_songhan / sched_system_all /
# ou_orcd_everything -- ou_orcd_everything is used here as the general-access
# partition (AllowAccounts=ALL, AllowQos=normal,unlimited).
#
# Same job scripts as run-nccl.sh (job-nccl-1node.sh / job-nccl-2node.sh,
# env-ubuntu.sh / build-utuntu-nvhpc-26.1 -- both nodes are Ubuntu 24.04).

nodes=(1700 1701)

partition=ou_orcd_everything
reservation=none
qos=normal

gpu_type=h100
gpus=8
gpus_2node=1

collectives=all

out_1node=out-1node
out_2node=out-2node
mkdir -p $out_1node $out_2node

flags="-p $partition -q $qos --exclusive"
if [[ "$reservation" != "none" ]]; then
   flags="$flags --reservation=$reservation"
fi

echo "########## nodes: ${nodes[@]} , partition: $partition #########"

for n in "${nodes[@]}"
do
   host=node$n
   echo "----- 1-node NCCL on $host -----"
   sbatch $flags --gres=gpu:$gpu_type:$gpus -w $host \
          -o $out_1node/%x-%N-%J job-nccl-1node.sh $gpus $collectives
done

host1=node${nodes[0]}
host2=node${nodes[1]}
echo "----- 2-node NCCL on $host1,$host2 -----"
sbatch $flags --gpus-per-node=$gpu_type:$gpus_2node -w $host1,$host2 \
       -o $out_2node/%x-%N-%J job-nccl-2node.sh $gpus_2node $collectives
