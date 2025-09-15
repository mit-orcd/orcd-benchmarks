partition=$1  # pg_tata #mit_normal_gpu
N_lines=$2  # 5

dir_1node=../$partition/out-1node
dir_2node=../$partition/out-2node

for dir in $dir_1node $dir_2node
do
   echo "^^^^^^^ $dir ^^^^^^^^^^"
   for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
   do
     echo "========================================================================================="
     echo $file
     #grep -e "_perf" -e "Rank" -e "4294967296" $dir/$file  # get all communications
     grep -e "sendrecv_perf" -A 23 $dir/$file | grep -e "sendrecv_perf" -e "Rank" -e "4294967296"  # only get sendrecv
   done
done

