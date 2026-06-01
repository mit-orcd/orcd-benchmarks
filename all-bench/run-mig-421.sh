# for all nodes
nodes="3401" 
partition=mit_normal_gpu   # pi_mshoulde # pi_qmqi # pi_mshoulde pg_tata #ou_sloan_gpu  # mit_normal # mit_normal_gpu
reservation=orcd_testing  #none #orcd_testing  #  WareWulf_testing
qos=unlimited  # normal   # unlimited
cpus=120  # 48  # 40  #96  # 40  # 88  # all cores on a CPU  node, substract reserved cores on a GPU node

# only for GPU nodes
gpu_type=$1  #h200_1g.18gb # l40s  # a100 #h100 # h200 # l40s
gpus=8  #56 # 1 #2  #8  #4

all_bench="nvidia-hpc-benchmarks"    

root_dir=/orcd/data/orcd/022/benchmarks

for bench in $all_bench
do
    echo "########## $bench #########"
    cd $root_dir/$bench/run
    ./submit-mig.sh "$nodes" $partition $reservation $qos $cpus $gpu_type $gpus  # use double quote for multiple words
done
