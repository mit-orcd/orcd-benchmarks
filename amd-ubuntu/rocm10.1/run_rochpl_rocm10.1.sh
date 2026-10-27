#!/usr/bin/env bash
# ROCm 10.1 copy of ../rochpl/run_rochpl.sh: one rocHPL run (fp64 HPL) on G MI355X GPUs, one MPI rank per GPU.
# Same settings; only the stack differs: ROCm 10.1.0, UCX/Open MPI in amd-software/ompi-5.0.8-rocm10.1,
# rocHPL in rochpl/build-rocm10.1.
#   ./run_rochpl_rocm10.1.sh <G> [mem_frac]   G = 1..8 on this node, 16 = this node + PEER
# The caller holds the GPU lock(s) (run_rochpl_all.sh does).
# MPI: Open MPI 5.0.8 + UCX 1.18.0, both built with ROCm 10.1 (rochpl/setup/build-rochpl-rocm10.1.sh).
# rocHPL passes GPU buffers straight to MPI: XGMI inside a node (UCX rocm_ipc), RoCE between
# nodes (UCX rc over the ionic rails, GPU-direct). No RCCL.
# Problem size: N fills mem_frac (default 0.75) of the 288 GB HBM per GPU, N = multiple of NB*P*Q.
# Knobs (env): NB (512), IT (2: rocHPL --it, best run reported), FRAC_SPLIT (rocHPL -f, 0.3 = its default),
#   EXTRA_MPI (extra mpirun args), UCX_TLS / UCX_NET_DEVICES (override the defaults below),
#   PEER (node6101.inband), NET_IF (eno17695np0, MPI launch/OOB), GID (1 = RoCEv2 IPv4), OUT.
# Appends one line to $OUT/summary.txt:
#   G P Q N NB time_s tflops passed rc mem_frac log
set -uo pipefail
export ROCM_STACK=10.1   # amd-software/rocm-10.1.0, same as the build
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
R=/orcd/data/orcd/022/benchmarks/rochpl/build-rocm10.1
MPI=$SW/ompi-5.0.8-rocm10.1/openmpi
UCXD=$SW/ompi-5.0.8-rocm10.1/ucx
# our Open MPI/UCX before amd-software/openmpi/lib (no UCX/ROCm) from the node env; exported to every rank (-x)
export LD_LIBRARY_PATH=$MPI/lib:$UCXD/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}
G=${1:?usage: $0 <ngpus> [mem_frac]}
FRAC=${2:-0.75}
NB=${NB:-512}; IT=${IT:-2}; FRAC_SPLIT=${FRAC_SPLIT:-0.3}
PEER=${PEER:-node6101.inband}; NET_IF=${NET_IF:-eno17695np0}; GID=${GID:-1}
OUT=${OUT:?set OUT}
HERE=$(cd "$(dirname "$0")" && pwd)
HBM=288000000000

# process grid P x Q, P <= Q, as square as possible
case $G in
  1) P=1 Q=1 ;; 2) P=1 Q=2 ;; 3) P=1 Q=3 ;; 4) P=2 Q=2 ;; 5) P=1 Q=5 ;;
  6) P=2 Q=3 ;; 7) P=1 Q=7 ;; 8) P=2 Q=4 ;; 16) P=4 Q=4 ;;
  *) echo "unsupported G=$G" >&2; exit 2 ;;
esac
N=${N:-$(awk -v g=$G -v f=$FRAC -v h=$HBM -v m=$((NB*P*Q)) 'BEGIN{n=sqrt(f*h*g/8); printf "%d", int(n/m)*m}')}

MPIARGS=(--prefix "$MPI" -np "$G" --bind-to none --map-by "ppr:$(( G > 8 ? 8 : G )):node"
         --mca pml ucx --mca btl '^vader,tcp,openib,uct' --mca osc ucx
         -x PATH -x LD_LIBRARY_PATH -x UCX_IB_GID_INDEX="$GID" -x UCX_WARN_UNUSED_ENV_VARS=n)
# UCX transports. UD on the ionic NICs times out (even 1 rank, at MPI_Finalize), so never UD.
#   1 node : shared memory + ROCm IPC over XGMI only, no NIC.
#   2 nodes: + RC over the 8 RoCE rails. The rails are separate /27 subnets (rail r on node6100 only
#            reaches rail r on node6101), so every rank gets all 8 rails and UCX keeps the
#            same-subnet ones (UCX_IB_ROCE_LOCAL_SUBNET); TCP on $NET_IF only for connection setup.
if (( G > 8 )); then
  RAILS=$(printf 'ionic_%d:1,' 0 1 2 3 4 5 6 7)
  MPIARGS+=(--host "$(hostname -s).inband:8,$PEER:8" --mca plm_rsh_args "-o BatchMode=yes"
            --mca oob_tcp_if_include "$NET_IF"
            -x UCX_TLS="${UCX_TLS:-self,sm,rocm,rc_verbs,tcp}" -x UCX_NET_DEVICES="${UCX_NET_DEVICES:-$RAILS$NET_IF}"
            -x UCX_IB_ROCE_LOCAL_SUBNET=y)
else
  MPIARGS+=(-x UCX_TLS="${UCX_TLS:-self,sm,rocm}")
fi
read -r -a XTRA <<<"${EXTRA_MPI:-}"   # extra mpirun args, e.g. "-x UCX_PROTO_INFO=y"
MPIARGS+=("${XTRA[@]}")
LOG=$OUT/rochpl_g${G}_N${N}_NB${NB}.log
{ echo "# rocHPL G=$G P=$P Q=$Q N=$N NB=$NB it=$IT f=$FRAC_SPLIT mem_frac=$FRAC $(date -Iseconds)"
  echo "# rocm: $ROCM_PATH"
  echo "# hosts: $(hostname -s)${PEER:+ (+$PEER if G>8)}"
  echo "# mpirun ${MPIARGS[*]} $R/rocHPL/run_rochpl -P $P -Q $Q -N $N --NB $NB --it $IT -f $FRAC_SPLIT"
} >"$LOG"
# rocHPL writes its results to ./HPL.out: run in a per-run dir, then append HPL.out to the log
RUND=$OUT/run_g${G}_N${N}; mkdir -p "$RUND"; cd "$RUND" || exit 1; rm -f HPL.out
t0=$(date +%s)
timeout 3600 "$MPI/bin/mpirun" "${MPIARGS[@]}" \
    "$R/rocHPL/run_rochpl" -P "$P" -Q "$Q" -N "$N" --NB "$NB" --it "$IT" -f "$FRAC_SPLIT" >>"$LOG" 2>&1
rc=$?
{ echo "# ---- HPL.out"; cat HPL.out 2>/dev/null; } >>"$LOG"
echo "# rc=$rc wall=$(( $(date +%s) - t0 ))s" >>"$LOG"
# result rows: "WR..  N  NB  P  Q  Time  Gflops (per GPU)" -> keep the fastest
read -r t gf < <(awk '$1 ~ /^W[RC]/ && $2 ~ /^[0-9]+$/ {if ($7+0 > best) {best=$7+0; tm=$6}} END {print tm, best}' "$LOG")
passed=$(grep -c 'PASSED' "$LOG"); failed=$(grep -c 'FAILED' "$LOG")
tf=$(awk -v g="${gf:-0}" 'BEGIN{printf "%.1f", g/1000}')
ok=$(( passed > 0 && failed == 0 ? 1 : 0 ))
printf '%s %s %s %s %s %s %s %s %s %s %s\n' "$G" "$P" "$Q" "$N" "$NB" "${t:--}" "$tf" "$ok" "$rc" "$FRAC" "$LOG" >>"$OUT/summary.txt"
echo "G=$G P=$P Q=$Q N=$N: ${tf} TFLOPS time ${t:--}s passed=$ok rc=$rc"
(( ok == 1 && rc == 0 ))
