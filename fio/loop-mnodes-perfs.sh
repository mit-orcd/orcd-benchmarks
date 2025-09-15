# use --exclusive to ensure 1 job per node only. 
N_jobs=16  # 32 # 16
flag1="-p mit_normal -q unlimited --array=1-$N_jobs --exclusive"
flag2="-p mit_normal_gpu -q unlimited --array=1-$N_jobs --exclusive"
type=scratch # data # scratch

i=004 # 011 # 004 #009  #001  # 002 005 006 007 008
j=013 # 012 # 013 #010  # 002
sbatch $flag1 job-mnode.sh $type $i
sbatch $flag2 job-mnode.sh $type $j


