partition=$1  
N_lines=$(( $2 *3 + 1 ))   # $2 
echo $N_line
gpu_type=$3

root_dir=/orcd/data/orcd/022/benchmarks/gpu-burn-r8
dir=$root_dir/$partition/output-$gpu_type

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "================================"
  echo "$file"
  grep "100.0%"  $dir/$file | awk -F "100.0%" '{print $2}' 
done

