#!/bin/bash
# Run a set of benchmarks on a set of nodes (submits Slurm jobs, returns immediately).
#
# Usage: ./run-all.sh [bench ...]
#   no argument : run the default all_bench list set below
#   bench ...   : run only these, e.g.  ./run-all.sh gpu-burn-r8 nccl-tests
#
# Benchmarks:
#   openmp mpi-calc-pi mpi-p2p gpu-burn-r8 nvidia-hpc-benchmarks nccl-tests
#       -> <bench>/run/run.sh (+ run-2node.sh) with the 7 args below
#   megatron-lm -> Megatron-LM/job.sh, 1 node per node + 1 job per node pair (l40s, h200 only)
#   b200-nodes  -> gpu-fryer, nccl 1-node/2-node, megatron 1-node/2-node on B200 nodes
#                  (partition mit_testing is fixed in its scripts)
#   b200-kimi   -> b200-kimi/chain.sh: Kimi-K3 vLLM serving chain, 1-2 B200 nodes, ~45 min;
#                  uses its own nodes/reservation from b200-kimi/common/env.sh

# for all nodes
nodes="3511 3512"          # node numbers, nodeNNNN
partition=mit_normal_gpu   # pi_mshoulde # pi_qmqi # pi_mshoulde pg_tata #ou_sloan_gpu  # mit_normal # mit_normal_gpu # mit_testing (b200)
reservation=none #orcd_testing  #none #orcd_testing  #  WareWulf_testing
qos=unlimited  # normal   # unlimited
cpus=48  # 48  # 40  #96  # 40  # 88  # all cores on a CPU  node, substract reserved cores on a GPU node

# only for GPU nodes
gpu_type=l40s # l40s  # a100 #h100 # h200 # l40s # b200
gpus=4  #2  #8  #4

#all_bench="openmp"
#all_bench="openmp mpi-calc-pi"  # single CPU node
#all_bench="mpi-calc-pi"  # single CPU node
#all_bench="openmp mpi-calc-pi mpi-p2p"  # two or more CPU nodes
#all_bench="mpi-p2p"
#all_bench="openmp mpi-calc-pi gpu-burn-r8 nvidia-hpc-benchmarks nccl-test"  # single GPU node
#all_bench="gpu-burn-r8 nccl-tests"   # L40S GPU nodes
#all_bench="gpu-burn-r8 nvidia-hpc-benchmarks nccl-tests openmp mpi-calc-pi megatron-lm"   # H200 GPU nodes
#all_bench="nvidia-hpc-benchmarks"
#all_bench="b200-nodes"   # B200 nodes, e.g. nodes="5500 5502"
#all_bench="b200-kimi"    # B200 Kimi-K3 serving, nodes set in b200-kimi/common/env.sh
all_bench="nccl-tests"

[ $# -gt 0 ] && all_bench="$*"

root_dir=/orcd/data/orcd/022/benchmarks

hosts=(); for n in $nodes; do hosts+=(node$n); done
pairs=()   # every node pair i<j, as nodeA,nodeB
for ((i=0; i<${#hosts[@]}; i++)); do
    for ((j=i+1; j<${#hosts[@]}; j++)); do pairs+=("${hosts[i]},${hosts[j]}"); done
done
resv_flag=""; [ "$reservation" != "none" ] && resv_flag="--reservation=$reservation"

for bench in $all_bench
do
    echo "########## $bench #########"
    case $bench in
    megatron-lm)
        if [ "$gpu_type" != "l40s" ] && [ "$gpu_type" != "h200" ]; then
            echo "megatron-lm supports gpu_type l40s or h200 only (use b200-nodes for B200); skipped"
            continue
        fi
        cd $root_dir/megatron-lm/Megatron-LM
        mkdir -p output
        for host in "${hosts[@]}"; do   # 1 node, global batch = 128 x GPUs
            sbatch -p $partition -q $qos $resv_flag -w $host -N 1 -n 1 \
                   --gpus-per-node=$gpu_type:$gpus job.sh $((128 * gpus))
        done
        for pair in "${pairs[@]}"; do   # 2 nodes
            sbatch -p $partition -q $qos $resv_flag -w $pair -N 2 -n 2 \
                   --gpus-per-node=$gpu_type:$gpus job.sh $((256 * gpus))
        done
        ;;
    b200-nodes)
        cd $root_dir/b200-nodes
        ./job-gpu-fryer.sh "${hosts[*]}"
        ./job-nccl-1node.sh "${hosts[*]}" sendrecv $gpus
        for host in "${hosts[@]}"; do ./job-megatron-1node.sh $host $gpus; done
        for pair in "${pairs[@]}"; do
            sbatch -w $pair job-nccl-2node.sh sendrecv $gpus
            ./job-megatron-2node.sh $pair $gpus
        done
        ;;
    b200-kimi)
        cd $root_dir/b200-kimi
        ./chain.sh
        ;;
    *)
        cd $root_dir/$bench/run
        ./run.sh "$nodes" $partition $reservation $qos $cpus $gpu_type $gpus  # use double quote for multiple words
        if [ -f "run-2node.sh" ]; then
           ./run-2node.sh "$nodes" $partition $reservation $qos $cpus $gpu_type $gpus
        fi
        ;;
    esac
done
