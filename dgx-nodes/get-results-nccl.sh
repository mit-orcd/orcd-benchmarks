#!/bin/bash
# Collect the NCCL results submitted by ./run-nccl.sh, same reporting as
# nccl-tests/run/get-results.sh (the largest message size, 4 GB, of sendrecv).
#
# Usage: ./get-results-nccl.sh [n_jobs]
#   n_jobs: how many of the most recent job outputs per directory (default 12)

N_lines=$(( ${1:-12} + 1 ))

for dir in out-1node out-2node
do
   echo "^^^^^^^ $dir ^^^^^^^^^^"
   [ -d "$dir" ] || { echo "(no such directory yet)"; continue; }
   for file in `ls -lt $dir | head -n $N_lines | awk '{print $9}'`
   do
     echo "========================================================================================="
     echo $file
     grep -e "^node" $dir/$file
     grep -e "sendrecv_perf" -A 45 $dir/$file | grep -e "sendrecv_perf" -e "4294967296"
   done
done
