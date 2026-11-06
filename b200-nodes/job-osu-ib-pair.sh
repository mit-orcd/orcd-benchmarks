#!/bin/bash
# OSU MPI point-to-point + raw InfiniBand (perftest) between two B200 nodes.
#
#   OSU  : osu_latency, osu_bw, osu_bibw, 1 rank per node, CPU host memory,
#          default openmpi/4.1.4 module (gcc/12.2.0), OSU 7.5-1 build in
#          ../mpi-p2p/install/r8-4.1.4. No MCA/UCX tuning: Open MPI picks the
#          transport and the NIC itself.
#   raw IB: ib_write_bw (64 MiB, host mem -> host mem) and ib_write_lat (2 B)
#          on mlx5_4, same rail and message size as job-ibwrite-2node.sh.
#
# Submit one pair:
#     sbatch -w nodeA,nodeB job-osu-ib-pair.sh
# Sweep over many pairs: see submit-osu-ib-sweep.sh (a job array).
#
# The job is not --exclusive (B200 nodes are rarely fully idle), so other
# users' jobs may share the nodes and their traffic can add noise. Two of OUR
# pair jobs never measure on the same node at once: each job takes a node-local
# flock on both of its nodes, in sorted node order (no deadlock), before testing.
#
# Each job writes out-osu-ib/osu-ib-<nodeA>-<nodeB>-<jobid>.out, then runs
# analyze-osu-ib.py, which regenerates out-osu-ib/summary.md from every job
# that has finished so far.
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 30
#SBATCH -J osu-ib
#SBATCH -o slurm-logs/%x-%A_%a.out

set -u
DIR=/orcd/data/orcd/022/benchmarks/b200-nodes
cd "$DIR" || exit 1
export SLURM_OVERLAP=1          # lock steps, perftest steps and mpirun's daemons share the allocation

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort))
A=${NODES[0]}; B=${NODES[1]}
JID=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
JID=${JID:-$SLURM_JOB_ID}
OUT=out-osu-ib/osu-ib-$A-$B-$JID.out
RAW=out-osu-ib/raw/$JID
mkdir -p "$RAW"
exec > "$OUT" 2>&1

DEV=mlx5_4
SIZE=$((64*1024*1024))
OSU=/orcd/data/orcd/022/benchmarks/mpi-p2p/install/r8-4.1.4/libexec/osu-micro-benchmarks/mpi/pt2pt

echo "=== info"
echo "date    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "job     : $JID"
echo "nodes   : $A $B"

# ------------------------------------------------------------- preflight check
# Every node must see this directory (the OSU build and the output live here).
# 2026-11-03: node5800-c1 had /orcd/data/orcd/022 not mounted (autofs) and the
# first smoke test hung; now such a pair fails at once and says why.
BAD=0
for n in "$A" "$B"; do
   if ! srun -N1 -n1 -w "$n" test -x "$OSU/osu_latency"; then
      echo "preflight: FAIL $n cannot see $OSU (is /orcd/data/orcd/022 mounted?)"
      BAD=1
   fi
done
if [ $BAD = 1 ]; then
   echo "=== done $(date '+%Y-%m-%d %H:%M:%S') (aborted in preflight)"
   python3 analyze-osu-ib.py > /dev/null 2>&1
   exit 1
fi

# ------------------------------------------------------------------ node locks
# The LOCKED marker goes to the step's stdout, which this batch script writes on
# its own node, so it does not depend on the remote node's shared filesystem.
LOCKPIDS=()
trap 'kill ${LOCKPIDS[*]} 2>/dev/null' EXIT
for n in "$A" "$B"; do
   f=$RAW/lock-$n
   srun -N1 -n1 -w "$n" bash -c \
      'exec 9>/dev/shm/osu-ib-'"$USER"'.lock; flock 9; echo LOCKED; exec sleep 86400' > "$f" &
   LOCKPIDS+=($!)
   t0=$(date +%s)
   until [ -s "$f" ]; do
      if [ $(( $(date +%s) - t0 )) -gt 900 ]; then
         echo "lock    : FAIL no lock on $n after 900 s"
         echo "=== done $(date '+%Y-%m-%d %H:%M:%S') (aborted waiting for lock)"
         python3 analyze-osu-ib.py > /dev/null 2>&1
         exit 1
      fi
      sleep 2
   done
   echo "lock    : $n after $(( $(date +%s) - t0 )) s"
done

for n in "$A" "$B"; do
   srun -N1 -n1 -w "$n" bash -c 'echo "host    : $(hostname) | $(cat /etc/redhat-release) | $(uname -r) | $(ibv_devinfo -d '"$DEV"' 2>/dev/null | awk "/state:/{print \$2; exit}") '"$DEV"'"'
done
echo "sharing : $(squeue -h -w "$A,$B" -o '%u' | sort -u | grep -vxc "$USER") other user(s) with jobs on these nodes"

# --------------------------------------------------------------------------- OSU
module purge
module load gcc/12.2.0
module load openmpi/4.1.4
echo "mpirun  : $(which mpirun)"
echo "ompi    : $(ompi_info --parsable 2>/dev/null | grep -m1 '^ompi:version:full' | cut -d: -f4)"

for b in osu_latency osu_bw osu_bibw; do
   echo "=== $b"
   timeout 300 mpirun -np 2 --map-by ppr:1:node "$OSU/$b"
   echo "exit    : $?"
done

# ------------------------------------------------------------------------ raw IB
ibpair () {   # ibpair <label> <binary> <port> <args...>
   local label=$1 bin=$2 port=$3; shift 3
   echo "=== $label"
   srun -N1 -n1 -w "$A" timeout 120 $bin -d $DEV -p $port -F "$@" > "$RAW/$label.srv" 2>&1 &
   local spid=$!
   sleep 5
   srun -N1 -n1 -w "$B" timeout 120 $bin -d $DEV -p $port -F "$@" "$A" > "$RAW/$label.cli" 2>&1
   echo "exit    : $?"
   wait $spid 2>/dev/null
   cat "$RAW/$label.cli"
}
ibpair ib_write_bw  ib_write_bw  18615 --report_gbits -s $SIZE -n 200
ibpair ib_write_lat ib_write_lat 18616 -n 10000

echo "=== done $(date '+%Y-%m-%d %H:%M:%S')"
kill ${LOCKPIDS[*]} 2>/dev/null
python3 analyze-osu-ib.py > /dev/null 2>&1 || echo "analyze-osu-ib.py failed"
