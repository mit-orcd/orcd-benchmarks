# for all nodes
nodes="3200 3201"
partition=mit_normal_gpu # pi_mshoulde # pi_qmqi # pi_mshoulde pg_tata #ou_sloan_gpu  # mit_normal # mit_normal_gpu
reservation="" #orcd_testing # ""  #orcd_testing  #  WareWulf_testing
qos=unlimited # normal   # unlimited
cpus=120  # 40  #96  # 40  # 88  # all cores on a CPU  node, substract reserved cores on a GPU node

# only for GPU nodes
gpu_type=h200 # h200 # l40s
gpus=8  #8  #4

#all_bench="openmp mpi-calc-pi"  # single CPU node
#all_bench="openmp mpi-calc-pi mpi-p2p"  # two or more CPU nodes
#all_bench="mpi-p2p" 
#all_bench="gpu-burn-r8 nccl-tests"   # L40S GPU nodes
#all_bench="gpu-burn-r8 nvidia-hpc-benchmarks nccl-tests"   # H200 GPU nodes
#all_bench="gpu-burn-r8 nvidia-hpc-benchmarks nccl-tests openmp mpi-calc-pi"   # H200 GPU nodes
#all_bench="nvidia-hpc-benchmarks"    
all_bench="nccl-tests" 

root_dir=/orcd/data/orcd/002/benchmarks

for bench in $all_bench
do
    echo "########## $bench #########"
    cd $root_dir/$bench/run
    #./run.sh "$nodes" $partition $reservation $qos $cpus $gpu_type $gpus  # use double quote for multiple words
    if [ -f "run-2node.sh" ]; then
       ./run-2node.sh "$nodes" $partition $reservation $qos $cpus $gpu_type $gpus 
    fi
done

#    jid1=`sbatch job.sh $nodes $partition $reservation $qos $cpus | awk '{ print $4 }'`
#    jid2=`sbatch --dependency=afterok:$jid1 get-results.sh $partition $lines | awk '{ print $4 }'`
