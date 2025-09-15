mode=mnode  # 1node  # mnode
dir=output-$mode
N_lines=10

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "================================"
  echo "Mode=$mode,  Output=$file"
  grep -e "directory" -e "IOPS" $dir/$file
done

