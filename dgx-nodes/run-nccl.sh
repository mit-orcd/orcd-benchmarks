#!/bin/bash
# NCCL benchmarks on the Ubuntu DGX H100 nodes listed in ./notes
#   - 1-node NCCL (NVLink)      on every node
#   - 2-node NCCL (InfiniBand)  on each pair, i.e. the nodes inside one [] in
#                               notes -- no cross-pairs between the groups
#
# Same shape as ../all-bench/run-dgx*.sh (set the variables, then submit), but
# it submits the local job-nccl-*.sh instead of calling nccl-tests/run/run.sh:
# those scripts source nccl-tests/run/env.sh, which selects the Rocky 8 build,
# while these nodes are Ubuntu 24.04 and need build-utuntu-nvhpc-26.1
# (see ./env-ubuntu.sh).
#
# Usage: ./run-nccl.sh

# node pairs: 2-node jobs are submitted per group, 1-node jobs per node
groups=(
   "1800 1801"
   "2700 2701"
   "2800 2801"
)
# node170[0,1] are skipped: they are in pi_songhan, not mit_testing

partition=mit_testing
reservation=none  # none #orcd_testing  #  WareWulf_testing
qos=normal        # normal   # unlimited

# only for GPU nodes
gpu_type=h100     # l40s  # a100 # h100 # h200 # b200
gpus=8            # GPUs per node for the 1-node test
gpus_2node=1      # GPUs per node for the 2-node test

collectives=all   # all, or e.g. "sendrecv,allreduce" (see env-ubuntu.sh)

run_1node=yes     # yes / no
run_2node=yes     # yes / no

out_1node=out-1node
out_2node=out-2node
mkdir -p $out_1node $out_2node

flags="-p $partition -q $qos --exclusive"
if [[ "$reservation" != "none" ]]; then
   flags="$flags --reservation=$reservation"
fi

for group in "${groups[@]}"
do
    nodes=($group)
    echo "########## nodes: ${nodes[@]} #########"

    # 1-node NCCL: one independent job per node, all GPUs of the node
    if [[ "$run_1node" == "yes" ]]; then
       for n in "${nodes[@]}"
       do
          host=node$n
          echo "----- 1-node NCCL on $host -----"
          sbatch $flags --gres=gpu:$gpu_type:$gpus -w $host \
                 -o $out_1node/%x-%N-%J job-nccl-1node.sh $gpus $collectives
       done
    fi

    # 2-node NCCL: one job on the pair
    if [[ "$run_2node" == "yes" ]]; then
       host1=node${nodes[0]}
       host2=node${nodes[1]}
       echo "----- 2-node NCCL on $host1,$host2 -----"
       sbatch $flags --gpus-per-node=$gpu_type:$gpus_2node -w $host1,$host2 \
              -o $out_2node/%x-%N-%J job-nccl-2node.sh $gpus_2node $collectives
    fi
done
