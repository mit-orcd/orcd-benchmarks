partition=mit_normal_gpu
lines=2  # how many jobs you want to get results from + 1
gpu_type=l40s  # h200 # h100

#all_bench="openmp mpi-calc-pi mpi-p2p gpu-burn-r8 nccl-tests"  # all for L40S
all_bench="openmp mpi-calc-pi gpu-burn-r8 nccl-tests"  # single L40s node

root_dir=/orcd/data/orcd/022/benchmarks

for bench in $all_bench
do
    echo "########## $bench #########"
    cd $root_dir/$bench/run
    ./get-results.sh $partition $lines $gpu_type
    if [ -f "get-results-2node.sh" ]; then
       ./get-results-2node.sh $partition $lines $gpu_type
    fi
done


# backup

