
#file="../nodes/nodes-mit_normal-all"
#flag="-p mit_normal -q unlimited --exclusive"
flag="-p mit_normal_gpu -q unlimited --exclusive"
type=scratch  # data

#for i in 002 005 006 007 008
# for i in 008 007 006 005 002
#for i in 001 002   # for data
for i in 004 009 010 011 012 013
do
   sbatch $flag job-1node.sh $type $i
done


