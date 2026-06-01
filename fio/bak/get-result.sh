bw_1node="results/bw.1node" 
bw_mnode="results/bw.mnode"
nodefile1="results/nodes.mit_normal"
nodefile2="results/nodes.mit_normal_gpu"
out1="results/bw.1node-mit_normal"
out2="results/bw.1node-mit_normal_gpu"
out3="results/bw.mnode-mit_normal"
out4="results/bw.mnode-mit_normal_gpu"

grep -e "directory" -e "IOPS" output-1node/* > $bw_1node
grep -e "directory" -e "IOPS" output-mnode/* > $bw_mnode

rm $out1 $out2 $out3 $out4
touch $out1 $out2 $out3 $out4

for node in `cat $nodefile1`
do 
   grep $node $bw_1node >> $out1
   grep $node $bw_mnode >> $out3
done

for node in `cat $nodefile2`
do 
   grep $node $bw_1node >> $out2
   grep $node $bw_mnode >> $out4
done


