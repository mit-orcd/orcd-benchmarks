nodes="4300 4301"
partition=pg_tata #ou_sloan_gpu  # mit_normal # mit_normal_gpu
reservation=orcd_testing # ""  #orcd_testing  #  WareWulf_testing
qos=normal   # unlimited
cpus=88
gpu_type=h200 # l40s
gpus=8  #4

#all_bench="openmp mpi-calc-pi" 
#all_bench="mpi-p2p" 
#all_bench="gpu-burn-r8 nccl-tests" 
#all_bench="gpu-burn-r8" 
all_bench="nccl-tests" 
#all_bench="nvidia-hpc-benchmarks" 
#all_bench="nccl-tests nvidia-hpc-benchmarks" 

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
