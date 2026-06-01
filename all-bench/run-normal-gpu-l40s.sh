# for all nodes
#nodes="3202 3203 3206 3207 3402 3403 3404 3405 3406 3407 3408 3502 3503 3504 3505"  #"4205 4206"  #"3511 3512"
#nodes="3505 3506 3507 3508"
nodes="3202 3203 3204"
partition=mit_normal_gpu  
reservation=none #monthly_maint #orcd_testing  #none  #  WareWulf_testing
qos=unlimited  # normal   # unlimited
cpus=48  # 48  # 40  #96  # 40  # 88  # all cores on a CPU  node, substract reserved cores on a GPU node

# only for GPU nodes
gpu_type=l40s # l40s  # a100 #h100 # h200 # l40s
gpus=4  #2  #8  #4

#all_bench="openmp"
#all_bench="openmp mpi-calc-pi"  # single CPU node
#all_bench="mpi-calc-pi"  # single CPU node
#all_bench="openmp mpi-calc-pi mpi-p2p"  # two or more CPU nodes
#all_bench="mpi-p2p" 
#all_bench="openmp mpi-calc-pi gpu-burn-r8 nvidia-hpc-benchmarks nccl-test"  # single GPU node
#all_bench="gpu-burn-r8 nccl-tests"   # L40S GPU nodes
all_bench="nccl-tests"  
#all_bench="nccl-tests gpu-burn-r8 openmp mpi-calc-pi mpi-p2p" 

root_dir=/orcd/data/orcd/022/benchmarks

for bench in $all_bench
do
    echo "########## $bench #########"
    cd $root_dir/$bench/run
    ./run.sh "$nodes" $partition $reservation $qos $cpus $gpu_type $gpus  # use double quote for multiple words
    if [ -f "run-2node.sh" ]; then
       ./run-2node.sh "$nodes" $partition $reservation $qos $cpus $gpu_type $gpus 
    fi
done

#    jid1=`sbatch job.sh $nodes $partition $reservation $qos $cpus | awk '{ print $4 }'`
#    jid2=`sbatch --dependency=afterok:$jid1 get-results.sh $partition $lines | awk '{ print $4 }'`
