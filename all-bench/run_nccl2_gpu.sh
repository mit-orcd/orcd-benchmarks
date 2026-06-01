# for all nodes
nodes="5201 5102"
partition=mit_testing 
reservation=none #orcd_testing  #none #orcd_testing  #  WareWulf_testing
qos=normal  # normal   # unlimited
cpus=48  # 48  # 40  #96  # 40  # 88  # all cores on a CPU  node, substract reserved cores on a GPU node

# only for GPU nodes
gpu_type=h200 # l40s  # a100 #h100 # h200 # l40s
gpus=4  #2  #8  #4

#all_bench="openmp mpi-calc-pi"  # single CPU node
#all_bench="mpi-p2p" 
#all_bench="openmp mpi-calc-pi gpu-burn-r8 nvidia-hpc-benchmarks nccl-test"  # single GPU node
#all_bench="gpu-burn-r8 nccl-tests"   # RTX GPU nodes
#all_bench="openmp mpi-calc-pi gpu-burn-r8 nccl-tests"   # L40S GPU nodes
#all_bench="nccl-tests gpu-burn-r8"  # nvidia-hpc-benchmarks"   # a100 GPU nodes
#all_bench="nccl-tests gpu-burn-r8 nvidia-hpc-benchmarks openmp mpi-calc-pi"   # H200 GPU nodes
#all_bench="nccl-tests nvidia-hpc-benchmarks gpu-burn-r8 mpi-p2p"   # H200 GPU nodes
#all_bench="nvidia-hpc-benchmarks"    
all_bench="nccl-tests" 

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
