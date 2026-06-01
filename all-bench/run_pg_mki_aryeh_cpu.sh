# for all nodes
nodes="4701 4702" 
partition=mit_testing
reservation=none #orcd_testing  #none #orcd_testing  #  WareWulf_testing
qos=normal  # normal   # unlimited
cpus=96  # 48  # 40  #96  # 40  # 88  # all cores on a CPU  node, substract reserved cores on a GPU node

# only for GPU nodes
gpu_type=none #l40s # l40s  # a100 #h100 # h200 # l40s
gpus=none # 4  #2  #8  #4

#all_bench="openmp"
#all_bench="openmp mpi-calc-pi"  # single CPU node
#all_bench="mpi-calc-pi"  # single CPU node
#all_bench="openmp mpi-calc-pi mpi-p2p"  # two or more CPU nodes
all_bench="mpi-p2p" 

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
