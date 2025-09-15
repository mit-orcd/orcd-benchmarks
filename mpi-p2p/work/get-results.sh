partition=mit_normal
dir=$partition/output
N_lines=30

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "================================"
  echo "Bandwidth (MB/s) and Avg Latency(us) of $file"
  grep 4194304 $dir/$file |awk '{print $2}'
done

