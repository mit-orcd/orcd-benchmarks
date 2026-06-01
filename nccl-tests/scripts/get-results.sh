mode=2node-2gpu  # 1node   # 2node-2gpu

# partition=mit_normal_gpu
#dir="../$partition/out-$mode"
dir="./out-$mode"

N_lines=3

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "========================================================================================="
  echo $file
  #grep -e "_perf" -e "4294967296" $dir/$file
  #grep -e "_perf" -e "Rank" -e "4294967296" $dir/$file  # get all communications
  grep -e "sendrecv_perf" -A 23 $dir/$file | grep -e "sendrecv_perf" -e "Rank" -e "4294967296"  # only get sendrecv
done

