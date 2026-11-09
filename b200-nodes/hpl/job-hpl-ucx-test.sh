#!/bin/bash
# Launch test for NVIDIA HPL with the container's own MPI + UCX (Open MPI 4.1.7 with CUDA, UCX 1.18,
# hpc-benchmarks_25.04.sif): prints the IB / CUDA setup, then a small HPL (3% of HBM) per UCX variant.
#   sbatch -N 1 --ntasks-per-node=2 job-hpl-ucx-test.sh 2  <dir>           # shared memory + CUDA, one node
#   sbatch -N 2 --ntasks-per-node=8 job-hpl-ucx-test.sh 16 <dir> [variant] # IB between nodes
# Writes "<G> <variant> <rc> <tflops> <passed> <tls lines>" to <dir>/results.txt.
#SBATCH -p mit_testing
#SBATCH --gpus-per-node=b200:8
#SBATCH --exclude=node5500-c1,node5802-c1,node5800-c1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 00:40:00
#SBATCH -J hpl-ucx-test
#SBATCH -o out-hpl/%x-%j.out
set -uo pipefail
G=${1:?ngpus}; DIR=${2:?dir}; ONLY=${3:-}
SIF=/orcd/data/orcd/022/benchmarks/nvidia-hpc-benchmarks/hpc-benchmarks_25.04.sif
NB=1024; HBM=191500000000
export TMPDIR=$HOME/orcd/scratch/tmp APPTAINER_TMPDIR=$HOME/orcd/scratch/tmp
mkdir -p "$TMPDIR" "$DIR"; cd "$DIR" || exit 2
module load apptainer 2>/dev/null
case $G in 2) P=2 Q=1 ;; 8) P=4 Q=2 ;; 16) P=4 Q=4 ;; *) echo "G=$G?"; exit 2 ;; esac
PPN=$(( G > 8 ? 8 : G )); CPT=$(( ${SLURM_CPUS_ON_NODE:-224} / PPN ))
N=$(awk -v g=$G -v h=$HBM -v m=$((NB*P*Q)) 'BEGIN{n=sqrt(0.03*h*g/8); printf "%d", int(n/m)*m}')
DAT=$DIR/HPL-g$G.dat
cat >"$DAT" <<EOF
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
$N           Ns
1            # of NBs
$NB          NBs
1            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
$P           Ps
$Q           Qs
16.0         threshold
1            # of panel fact
0            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
2            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
0            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
3            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
1            SWAP (0=bin-exch,1=long,2=mix)
192          swapping threshold
1            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
0            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
EOF
D=$DIR/diag.$SLURM_JOB_ID.txt
{ echo "nodes=$SLURM_JOB_NODELIST N=$N G=$G $(date -Iseconds)"; apptainer --version
  ulimit -l; ls /dev/infiniband 2>&1 | head -20; nvidia-smi topo -m 2>&1 | head -30; lscpu | grep -E 'NUMA node|^CPU\(s\)'
  apptainer exec --nv "$SIF" bash -c '/opt/hpcx/ucx/bin/ucx_info -d 2>/dev/null | grep -E "Transport:|Device:" | paste - - | sort -u; /opt/hpcx/ompi/bin/ompi_info --parsable --all 2>/dev/null | grep -m1 mpi_built_with_cuda_support:value'
  for d in /sys/class/infiniband/*; do echo "$d $(cat $d/ports/1/state 2>/dev/null) $(cat $d/ports/1/rate 2>/dev/null) numa=$(cat $d/device/numa_node 2>/dev/null) $(cat $d/ports/1/link_layer 2>/dev/null)"; done
} >"$D" 2>&1

run() {  # name  apptainer-flags  [K=V ...]
  local name=$1 aflags=$2; shift 2
  [[ -n $ONLY && $ONLY != "$name" ]] && return 0
  local log=$DIR/g$G.$name.$SLURM_JOB_ID.log
  echo "# $name flags=[$aflags] env=[$*] $(date -Iseconds)" >"$log"
  ( for kv in "$@"; do export "$kv"; done
    export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-warn}
    timeout 420 srun --mpi=pmi2 --cpu-bind=none -n "$G" --ntasks-per-node="$PPN" -c "$CPT" \
       apptainer exec --nv $aflags -B "$DIR" "$SIF" /workspace/hpl.sh --dat "$DAT" >>"$log" 2>&1 )
  local rc=$?
  local gf; gf=$(awk '$1 ~ /^W[RC]/ && $2 ~ /^[0-9]+$/ {print $7; exit}' "$log")
  local ok=$(( $(grep -c PASSED "$log") > 0 && $(grep -c FAILED "$log") == 0 ? 1 : 0 ))
  echo "$G $name rc=$rc gflops=${gf:-0} passed=$ok job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST" >>"$DIR/results.txt"
}
# A: keep the default transports, only stop UCX using /proc/<pid>/fd for shared memory
run A "" UCX_POSIX_USE_PROC_LINK=n
# B: A plus explicit transports (IB rc, shared memory, CUDA)
run B "" UCX_POSIX_USE_PROC_LINK=n UCX_TLS=rc,sm,self,cuda_copy,cuda_ipc,gdr_copy
# C: ranks share one apptainer namespace, default UCX
run C "--sharens" UCX_LOG_LEVEL=warn
# D: no shared memory at all, inside the node over IB and CUDA IPC
run D "" UCX_TLS=rc,self,cuda_copy,cuda_ipc,gdr_copy
# E: D without gdr_copy (not in the container), only the 400 Gb NDR cards (mlx5_0-3 are a separate 100 Gb HDR network)
run E "" UCX_TLS=rc,self,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_13:1,mlx5_14:1,mlx5_15:1
echo "done $(date -Iseconds)"
