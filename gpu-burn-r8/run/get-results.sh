partition=$1  # pg_tata #mit_normal_gpu
N_lines=$2 
gpu_type=$3

root_dir=/orcd/data/orcd/002/benchmarks/gpu-burn-r8
dir=$root_dir/$partition/output-$gpu_type

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "================================"
  echo "$file"
  grep "100.0%"  $dir/$file
done

