# for all nodes
nodes="3618"  
partition=pi_zhang_f
reservation=orcd_testing # ""  #orcd_testing  #  WareWulf_testing
qos=normal   # unlimited
cpus=192  # 48  # 40  #96  # 40  # 88  # all cores on a CPU  node, substract reserved cores on a GPU node

# only for GPU nodes
gpu_type= # h100 # h200 # l40s
gpus= #2  #8  #4

all_bench="openmp mpi-calc-pi"  # single CPU node
#all_bench="openmp mpi-calc-pi mpi-p2p"  # two or more CPU nodes
#all_bench="mpi-p2p" 
#all_bench="gpu-burn-r8 nccl-tests"   # L40S GPU nodes
#all_bench="gpu-burn-r8 nvidia-hpc-benchmarks nccl-tests"   # H200 GPU nodes
#all_bench="gpu-burn-r8 nvidia-hpc-benchmarks nccl-tests openmp mpi-calc-pi"   # H200 GPU nodes
#all_bench="nvidia-hpc-benchmarks"    
#all_bench="nccl-tests" 

root_dir=/orcd/data/orcd/002/benchmarks

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
