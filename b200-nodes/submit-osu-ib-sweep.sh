#!/bin/bash
# Sweep job-osu-ib-pair.sh over every pair of the given B200 nodes, as one Slurm
# job array (one task per pair). The array is submitted held, each task is then
# pinned to its own node pair (scontrol update ReqNodeList, which splits that
# task out of the array record), and the array is released. No dependencies:
# tasks start as soon as their two nodes have room; job-osu-ib-pair.sh's node
# locks keep two tasks from measuring on the same node at once.
#
# Usage: ./submit-osu-ib-sweep.sh "<node list>" [pairs to skip, e.g. "a+b c+d"]
#   Writes out-osu-ib/pairs.txt (all pairs, read by analyze-osu-ib.py).
DIR=/orcd/data/orcd/022/benchmarks/b200-nodes
cd "$DIR" || exit 1
NODES=($(echo ${1//,/ } | tr ' ' '\n' | sort -u))
SKIP=" ${2:-} "
mkdir -p out-osu-ib slurm-logs

: > out-osu-ib/pairs.txt
PAIRS=()
for ((i=0; i<${#NODES[@]}; i++)); do
   for ((j=i+1; j<${#NODES[@]}; j++)); do
      a=${NODES[i]}; b=${NODES[j]}
      echo "$a $b" >> out-osu-ib/pairs.txt
      [[ "$SKIP" == *" $a+$b "* || "$SKIP" == *" $b+$a "* ]] && continue
      PAIRS+=("$a,$b")
   done
done
n=${#PAIRS[@]}
[ "$n" -gt 0 ] || { echo "no pairs to run"; exit 1; }

aid=$(sbatch --parsable --hold --array=1-$n%10 job-osu-ib-pair.sh) || exit 1
for ((k=1; k<=n; k++)); do
   scontrol update JobId=${aid}_$k ReqNodeList=${PAIRS[k-1]} || echo "update failed: ${aid}_$k"
   echo "${aid}_$k ${PAIRS[k-1]}"
done | tee out-osu-ib/sweep-$aid.tasks
scontrol release $aid
echo "array $aid: $n pairs submitted (task -> pair list in out-osu-ib/sweep-$aid.tasks)"
