#!/bin/bash
# Does a given MPI launch method work? Runs a tiny 2-rank osu_latency (1..8 B)
# between two nodes, one rank per node, with the requested launcher.
# Used by check-cluster (mpi domain); also runnable by hand.
#
# launcher-test.sh <method> <nodeA> <nodeB> [partition] [build]
#   method:    mpirun | pmix | pmi2
#                mpirun -> sbatch --wait job running `mpirun -n 2` (salloc can't
#                          get its allocation back to the login node, so mpirun
#                          is tested inside a batch job, as in run.sh)
#                pmix   -> srun --mpi=pmix osu_latency
#                pmi2   -> srun --mpi=pmi2 osu_latency
#   partition: default ou_orcd_everything
#   build:     default r8-4.1.4 (openmpi/4.1.4)
#
# Last line (machine-readable):
#   LAUNCH_RESULT method=<m> works=<yes|no|untested> lat_us=<8 B latency|NA>
# works=untested means Slurm never started the job (couldn't allocate), which
# says nothing about the launcher itself.

method=$1
nodeA=$2
nodeB=$3
partition=${4:-ou_orcd_everything}
build=${5:-r8-4.1.4}

if [[ -z "$method" || -z "$nodeA" || -z "$nodeB" ]]; then
    echo "usage: $0 <mpirun|pmix|pmi2> <nodeA> <nodeB> [partition] [build]" >&2
    exit 2
fi

type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
ompi_version=${build##*-}
module load gcc/12.2.0 openmpi/"$ompi_version" 2>/dev/null
if ! command -v mpirun >/dev/null 2>&1; then
    echo "module load openmpi/$ompi_version FAILED"
    echo "LAUNCH_RESULT method=$method works=no lat_us=NA"
    exit 90
fi

OSU=/orcd/data/orcd/022/benchmarks/mpi-p2p/install/${build}/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency
OSU_ARGS="-m 1:8 -i 100 -x 10"
# Same short, non-exclusive 2-node request for every method.
COMMON=(-p "$partition" -w "${nodeA},${nodeB}" -N 2 --ntasks-per-node=1
        --mem=2G -t 00:03:00 -J mpi-launch-check)

case "$method" in
    mpirun)
        # batch output must be on a filesystem the compute nodes can write
        outdir=/home/shaohao/benchmarks/mpi-p2p/work/launch-check
        mkdir -p "$outdir"
        outfile=$outdir/mpirun-${nodeA}-${nodeB}-$$.out
        # --deadline: Slurm cancels it itself if it can't start in time, so a
        # busy node never leaves a job queued behind us. Slurm drops the job
        # once start + time limit would pass the deadline, so the deadline must
        # be the time limit (2 min here) plus how long it may wait to start.
        jobid=$(sbatch --wait --parsable "${COMMON[@]}" -t 00:02:00 \
                    --deadline=now+5minutes \
                    -o "$outfile" \
                    --wrap "module load gcc/12.2.0 openmpi/$ompi_version; \
                            mpirun -n 2 --map-by ppr:1:node $OSU $OSU_ARGS" 2>&1)
        rc=$?
        out=$(cat "$outfile" 2>/dev/null)
        state=$(sacct -n -X -P -j "${jobid%%;*}" -o State 2>/dev/null | head -1)
        rm -f "$outfile"
        echo "sbatch job ${jobid} state=${state:-unknown} rc=$rc"
        ;;
    pmix|pmi2)
        out=$(srun "${COMMON[@]}" --immediate=120 --mpi="$method" $OSU $OSU_ARGS 2>&1)
        rc=$?
        state=""
        ;;
    *)
        echo "unknown method: $method" >&2
        exit 2
        ;;
esac
echo "$out"

lat=$(echo "$out" | awk '$1==8 {print $2; exit}')
if [ $rc -eq 0 ] && [ -n "$lat" ]; then
    works=yes
elif echo "$out $state" | grep -qiE "unable to allocate resources|requested nodes are busy|DEADLINE|^CANCELLED"; then
    works=untested
elif [ "$method" = mpirun ] && [ -z "$out" ] && [ "$state" != "FAILED" ] && [ "$state" != "COMPLETED" ]; then
    works=untested    # job never ran (e.g. still pending when cancelled)
else
    works=no
fi
echo "LAUNCH_RESULT method=$method works=$works lat_us=${lat:-NA}"
[ "$works" = yes ]
