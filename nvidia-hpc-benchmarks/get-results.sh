partition=mit_normal_gpu
dir=$partition/output
N_lines=10

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "================================"
  echo "GFLOPs of $file"
  echo "n*GPUs    per GPU"
  grep WC0 $dir/$file | awk '{print $7, $9}'
done

# grep WC0 mit_normal_gpu/output/node3100-3496959.out | awk '{print $7, $9}'
