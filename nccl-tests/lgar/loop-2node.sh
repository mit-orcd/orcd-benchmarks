#!/bin/bash
file="../../nodes/nodes-mit_normal_gpu-l40s"
flags="--reservation=orcd_testing -p mit_normal_gpu -q unlimited"
script="2nodes-2gpus.sh"

#submit the first job
job_id_current=`sbatch $flags -w node3404,node3506 $script`
echo $job_id_current

#submit the following jobs with depenendy
for node1 in `cat $file`
do
   echo "== $node1"
   for node2 in `cat $file`
   do
     echo "  -- $node2"
     echo "     ## $job_id_current"
     #job_id=`sbatch $flags -w $node1,$node2 --dependency=afterok:$job_id $script`
     job_id_next=`sbatch $flags -w $node1,$node2 --dependency=afterany:$job_id_current $script`
     job_id_current=$job_id_next
   done
done

