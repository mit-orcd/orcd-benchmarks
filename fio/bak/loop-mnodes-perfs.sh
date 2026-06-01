
flag1="-p mit_normal -q unlimited --array=1-8 --exclusive"
flag2="-p mit_normal_gpu -q unlimited --array=1-8 --exclusive"
type=scratch

i=006  # 002 005 006 007 008
j=005
sbatch $flag1 job-mnode.sh $type $i
sbatch $flag2 job-mnode.sh $type $j


