mode=imagenet # llama  # imagenet  # llama
dir=out-$mode
N_lines=10

for file in `ls -lt $dir |head -n $N_lines |awk '{print $9}'`
do
  echo "================================"
  echo "Mode=$mode,  Output=$file"
  grep "num_workers" $dir/$file |grep GB
  #grep "num_workers=96" $dir/$file
  #grep "num_workers=0" $dir/$file |grep GB
done

