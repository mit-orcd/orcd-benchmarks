partition=$1  # pg_tata #mit_normal_gpu
N_lines=$2  # 5

dir=../work/$partition/output

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "================================"
  echo "Bandwidth (MB/s) and Avg Latency(us) of $file"
  grep 4194304 $dir/$file |awk '{print $2}'
done

