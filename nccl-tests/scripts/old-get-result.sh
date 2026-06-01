#dir="../mit_normal_gpu/out-1node"
dir="../mit_normal_gpu/out-2node-2gpu"

for file in `ls $dir`
do
     echo $file
     #sed -n '29,30p' $dir/$file 
     sed -n '26,27p' $dir/$file 
done
