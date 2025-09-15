partition=pi_mshoulde  # pg_tata #mit_normal_gpu
dir=$partition/output
N_lines=5

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "================================"
  echo "$file"
  grep -e "THREADS" -e "time"  $dir/$file
done

