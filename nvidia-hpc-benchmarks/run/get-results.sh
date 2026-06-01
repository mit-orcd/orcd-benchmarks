partition=$1  # pg_tata #mit_normal_gpu
N_lines=$(( $2 + 1 ))
gpu_type=$3
root_dir=/orcd/data/orcd/022/benchmarks/nvidia-hpc-benchmarks
dir=$root_dir/$partition/output-$gpu_type

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "================================"
  echo "GFLOPs of $file"
  echo "n*GPUs    per GPU"
  grep WC0 $dir/$file | awk '{print $7, $9}'
  #grep "GEMM GFLOPS" $dir/$file 
  grep "LU GFLOPS" -B 1 $dir/$file 
done

# grep WC0 mit_normal_gpu/output/node3100-3496959.out | awk '{print $7, $9}'
