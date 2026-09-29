#!/bin/bash
# MPI launcher matrix: for each OpenMPI stack, does each launch method work?
# Meant to run INSIDE a 2-node allocation, one task per node, e.g.
#   sbatch -p <partition> -N 2 --ntasks-per-node=1 launcher-matrix.sh <stack>...
# (check-cluster's mpi domain submits one such job per partition).
#
# Each <stack> argument is  key|modulepath|module|osu_build|prefix
#   key        short name used in the result lines, e.g. spack-4.1.4
#   modulepath module path the module is loaded from (`module use`), needed
#              because two different modules are both called openmpi/5.0.8;
#              "none" = no module exists: use <prefix>/bin and <prefix>/lib
#              directly (e.g. spack's openmpi/4.1.4/zahpnmk)
#   module     e.g. openmpi/5.0.8 ("-" when modulepath is none)
#   osu_build  OSU install tree under ../install, or "none" if there is none
#   prefix     the OpenMPI install the module must resolve to (checked, so a
#              wrong same-named module can never be tested by mistake)
#
# Methods, each a 2-rank osu_latency (1..8 B), one rank per node:
#   mpirun -> mpirun -n 2 --map-by ppr:1:node      (OpenMPI's own launcher;
#             its daemons are the PMIx server, so it doesn't need Slurm's pmix)
#   pmix   -> srun -n 2 --mpi=pmix                 (needs Slurm's pmix plugin)
#   pmi2   -> srun -n 2 --mpi=pmi2                 (needs Slurm's pmi2 plugin
#             AND an MPI built with PMI-2 support -- OpenMPI 5 has none)
#
# Output, one line per case:
#   MATRIX_RESULT stack=<key> method=<m> works=<yes|no> lat_us=<x|NA> reason=<text>
# plus MATRIX_INFO lines (nodes, OS, Slurm's MPI plugins).

CASE_TIMEOUT=${CASE_TIMEOUT:-60}
# RANKS_PER_NODE=2 (with a 1-node allocation) tests both ranks on ONE node
RPN=${RANKS_PER_NODE:-1}
INSTALL_ROOT=/orcd/data/orcd/022/benchmarks/mpi-p2p/install
OSU_ARGS="-m 1:8 -i 100 -x 10"

type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1

. /etc/os-release 2>/dev/null
echo "MATRIX_INFO nodes=${SLURM_JOB_NODELIST:-?} ranks_per_node=$RPN os=${ID:-?}${VERSION_ID:-} partition=${SLURM_JOB_PARTITION:-?}"
echo "MATRIX_INFO slurm_mpi_plugins=$(srun --mpi=list 2>&1 | awk 'NR>1 {print $1}' | tr '\n' ',' | sed 's/,$//')"

# why a case failed: the meaningful error lines, not the benchmark table
_reason() {
    grep -vE '^#|^[0-9]+ |^\s*$|^-+$|queued and waiting|has been allocated' <<<"$1" \
        | sed 's/[[:space:]]\+/ /g' | tail -3 | paste -sd'|' | cut -c1-300
}

for spec in "$@"; do
    IFS='|' read -r key modulepath module build want_prefix <<<"$spec"
    (
        module purge >/dev/null 2>&1
        module load gcc/12.2.0 >/dev/null 2>&1
        if [ "$modulepath" = none ]; then
            export PATH=$want_prefix/bin:$PATH
            export LD_LIBRARY_PATH=$want_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            export OPAL_PREFIX=$want_prefix
        else
            module use "$modulepath"
            module load "$module" >/dev/null 2>&1
        fi
        mpirun_path=$(command -v mpirun)
        setup_err=""
        prefix=$(dirname "$(dirname "${mpirun_path:-/x/x}")")
        if [ -z "$mpirun_path" ] || { [ -n "$want_prefix" ] && [ "$prefix" != "$want_prefix" ]; }; then
            setup_err="module load $module from $modulepath gave mpirun=${mpirun_path:-none}, expected $want_prefix/bin/mpirun"
        else
            # libmpi's own dependencies, checked on BOTH nodes (they can differ:
            # /opt/mellanox/hcoll is on some nodes and not others)
            missing=$(srun -n 2 --ntasks-per-node=$RPN bash -c \
                "ldd $prefix/lib/libmpi.so 2>/dev/null | awk '/not found/ {printf \"%s \", \$1}' \
                 | sed \"s/^./\$(hostname -s): &/\"" 2>/dev/null | sort -u | paste -sd';')
            if [ -n "$missing" ]; then
                setup_err="libmpi.so dependencies not installed on $missing -- no MPI program can run there with this module"
            elif [ "$build" = none ] || [ ! -x "$INSTALL_ROOT/$build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency" ]; then
                setup_err="no OSU build for this module ($build)"
            fi
        fi
        OSU=$INSTALL_ROOT/$build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency

        for method in mpirun pmix pmi2; do
            if [ -n "$setup_err" ]; then
                echo "MATRIX_RESULT stack=$key method=$method works=no lat_us=NA reason=$setup_err"
                continue
            fi
            case $method in
                mpirun) cmd=(mpirun -n 2 --map-by ppr:$RPN:node "$OSU" $OSU_ARGS) ;;
                *)      cmd=(srun -n 2 --ntasks-per-node=$RPN --mpi="$method" "$OSU" $OSU_ARGS) ;;
            esac
            tmpf=$(mktemp)
            timeout -k 10 "$CASE_TIMEOUT" "${cmd[@]}" > "$tmpf" 2>&1
            rc=$?
            out=$(tr -d '\000' < "$tmpf"); rm -f "$tmpf"
            lat=$(awk '$1==8 {print $2; exit}' <<<"$out")
            if [ $rc -eq 0 ] && [ -n "$lat" ]; then
                echo "MATRIX_RESULT stack=$key method=$method works=yes lat_us=$lat reason="
            else
                if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
                    why="hung: no result after ${CASE_TIMEOUT}s"
                else
                    why=$(_reason "$out")
                    if grep -q "requires exactly two processes" <<<"$out"; then
                        why="each rank started as its own 1-process MPI job (no PMI wire-up; OpenMPI 5 has no PMI-2 support) -- This test requires exactly two processes"
                    fi
                    [ -n "$lat" ] || why="${why:-exit code $rc, no benchmark output}"
                fi
                echo "MATRIX_RESULT stack=$key method=$method works=no lat_us=NA reason=${why:-exit code $rc}"
            fi
        done
    )
done
