
# use --exclusive to ensure 1 job per node only. 
flag1="-p mit_normal -q unlimited --exclusive"
flag2="-p mit_normal_gpu -q unlimited --exclusive"
type=scratch

#for i in 002 005 006 007 008
#for i in 008 007 006 005 002
#for i in 006 008 007 002 005
#for i in 006 008 007 002 005
#for i in 007 006 005 008 002
for i in 009 010 011 012 013
do
   sbatch $flag1 job-1node.sh $type $i
   sbatch $flag2 job-1node.sh $type $i
done


