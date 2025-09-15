partition=pi_mshoulde # pg_tata #mit_normal_gpu
gputype=l40s
dir=$partition/output-$gputype
N_lines=10 #50

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "================================"
  echo "$file"
  grep "100.0%"  $dir/$file
done

#grep -B 1  Killing  mit_normal_gpu/output-l40s/std32_node*
