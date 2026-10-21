#!/bin/bash
# NVIDIA HPL (FP64 Linpack) on G B200 GPUs, one MPI rank per GPU, from the NGC HPC-Benchmarks
# container (../../nvidia-hpc-benchmarks/hpc-benchmarks_25.04.sif: CUDA 12.8, xhpl has sm_100).
# Same rules as rocHPL on MI355X (../../amd-ubuntu/rochpl/run_rochpl.sh), so the two compare:
#   N fills 75% of the usable HBM per GPU (178 GiB on B200), retry at 60% if that run fails;
#   the same N twice in one HPL.dat, the fastest run is reported; residual check must pass.
#   NB and P x Q are NVIDIA's own defaults (NB 1024, P >= Q as in the container's sample decks).
# Submitted by submit-hpl-b200.sh (one job per GPU count; G = 1..8 on one node, 16 on two):
#   sbatch -N 1 --ntasks-per-node=G job-hpl-b200.sh G <run dir>
#   sbatch -N 2 --ntasks-per-node=8 job-hpl-b200.sh 16 <run dir>
#   sbatch -N 1 --ntasks-per-node=1 job-hpl-b200.sh 1 <run dir> smoke   # launch check first
# Appends "G P Q N NB time_s tflops passed mem_frac nodes jobid" to <run dir>/summary.txt.
#SBATCH -p mit_testing
#SBATCH --gpus-per-node=b200:8
#SBATCH --exclude=node5500-c1,node5802-c1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 01:00:00
#SBATCH -J hpl-b200
#SBATCH -o out-hpl/%x-%j.out
set -uo pipefail
G=${1:?usage: sbatch job-hpl-b200.sh <ngpus> <run dir>}
RUN=${2:?run dir}
MODE=${3:-full}             # smoke: one small run (3% of HBM) into smoke.txt, to check the launch
SIF=/orcd/data/orcd/022/benchmarks/nvidia-hpc-benchmarks/hpc-benchmarks_25.04.sif
NB=1024
HBM=191500000000            # 178.35 GiB usable per B200 (nvidia-smi)
mkdir -p "$RUN"
export TMPDIR=$HOME/orcd/scratch/tmp APPTAINER_TMPDIR=$HOME/orcd/scratch/tmp
mkdir -p "$TMPDIR"
module load apptainer 2>/dev/null

case $G in
  1) P=1 Q=1 ;; 2) P=2 Q=1 ;; 3) P=3 Q=1 ;; 4) P=2 Q=2 ;; 5) P=5 Q=1 ;;
  6) P=3 Q=2 ;; 7) P=7 Q=1 ;; 8) P=4 Q=2 ;; 16) P=4 Q=4 ;;
  *) echo "unsupported G=$G"; exit 2 ;;
esac
PPN=$(( G > 8 ? 8 : G ))
CPT=$(( ${SLURM_CPUS_ON_NODE:-224} / PPN ))
echo "hpl-b200 G=$G P=$P Q=$Q nodes=$SLURM_JOB_NODELIST job=$SLURM_JOB_ID $(date -Iseconds)"
nvidia-smi --query-gpu=index,name,memory.total,power.limit,clocks.max.sm --format=csv,noheader | head -8

run_hpl() {   # $1 = mem_frac
  local frac=$1 N dat log t gf ok
  N=$(awk -v g=$G -v f=$frac -v h=$HBM -v m=$((NB*P*Q)) 'BEGIN{n=sqrt(f*h*g/8); printf "%d", int(n/m)*m}')
  dat=$RUN/HPL-g${G}-N${N}.dat
  log=$RUN/hpl_g${G}_N${N}_NB${NB}.$SLURM_JOB_ID.log
  cat >"$dat" <<EOF
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
2            # of problems sizes (N)
$N $N        Ns
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
  echo "# N=$N mem_frac=$frac dat=$dat" | tee "$log"
  timeout 2700 srun --mpi=pmi2 --cpu-bind=none -n "$G" --ntasks-per-node="$PPN" -c "$CPT" \
      apptainer exec --nv -B "$RUN" "$SIF" /workspace/hpl.sh --dat "$dat" >>"$log" 2>&1
  echo "# rc=$?" >>"$log"
  # result rows: "WC0..  N  NB  P  Q  Time  Gflops (per GPU)" -> keep the fastest
  read -r t gf < <(awk '$1 ~ /^W[RC]/ && $2 ~ /^[0-9]+$/ {if ($7+0 > best) {best=$7+0; tm=$6}} END {print tm, best}' "$log")
  ok=$(( $(grep -c 'PASSED' "$log") > 0 && $(grep -c 'FAILED' "$log") == 0 ? 1 : 0 ))
  printf '%s %s %s %s %s %s %s %s %s %s %s\n' "$G" "$P" "$Q" "$N" "$NB" "${t:--}" \
      "$(awk -v g="${gf:-0}" 'BEGIN{printf "%.1f", g/1000}')" "$ok" "$frac" "$SLURM_JOB_NODELIST" "$SLURM_JOB_ID" >>"$RUN/$SUMF"
  grep -E '^W[RC]|PASSED|FAILED' "$log"
  (( ok == 1 ))
}

if [[ $MODE == smoke ]]; then
  SUMF=smoke.txt; run_hpl 0.03; rc=$?; echo "smoke rc=$rc $(date -Iseconds)"; exit $rc
fi
SUMF=summary.txt
run_hpl 0.75 || { echo "G=$G failed at mem_frac 0.75 -> retry at 0.6"; run_hpl 0.6; }
echo "done $(date -Iseconds)"
