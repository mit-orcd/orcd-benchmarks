# for all nodes
nodes="4509 4510 4511 4512 4513 4514"
partition=ou_ki   # pi_mshoulde # pi_qmqi # pi_mshoulde pg_tata #ou_sloan_gpu  # mit_normal # mit_normal_gpu
reservation=orcd_testing  #orcd_testing  # none #none #orcd_testing  #  WareWulf_testing
qos=normal  #unlimited  # normal   # unlimited
cpus=96 #64  # 48  # 40  #96  # 40  # 88  # all cores on a CPU  node, substract reserved cores on a GPU node

# only for GPU nodes
gpu_type=none #h100 # l40s  # a100 #h100 # h200 # l40s
gpus=none  #2  #8  #4

#all_bench="openmp"
#all_bench="openmp mpi-calc-pi"  # single CPU node
#all_bench="mpi-calc-pi"  # single CPU node
#all_bench="mpi-p2p openmp mpi-calc-pi"  # two or more CPU nodes
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

#    jid1=`sbatch job.sh $nodes $partition $reservation $qos $cpus | awk '{ print $4 }'`
#    jid2=`sbatch --dependency=afterok:$jid1 get-results.sh $partition $lines | awk '{ print $4 }'`
