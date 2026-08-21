#!/usr/bin/env bash
# Submit a stage.
#
# Usage: ./submit.sh <probe|download|gate|1node|base> [extra sbatch args...]
#        ./submit.sh --alt <gate|1node|base>          # Rocky fallback nodes, no reservation
#
# The job scripts carry the reservation, account and node list from ./notes directly in
# their #SBATCH headers, so `sbatch job-kimi-base.sh` on its own already targets the
# right thing. This wrapper exists for two reasons: it prints what each stage costs, and
# `--alt` retargets a stage onto the Rocky nodes if the reserved pair turns out to be on
# a driver older than r580 (see README).
#
# There is no "run everything": each stage gates the next, and chaining them would defeat
# the point of the cheap gates.
set -uo pipefail
cd "$(dirname "$0")"
source common/env.sh

ALT=0
if [[ "${1:-}" == "--alt" ]]; then ALT=1; shift; fi
STAGE="${1:-}"; shift 2>/dev/null || true

usage() {
  cat <<USAGE
usage: $0 [--alt] <stage> [extra sbatch args...]

  probe     driver / HBM / model-mount verdict per node    ~1 min,  1 GPU each
  download  smoke-tier model only (Kimi-K3 is pre-staged)  ~5 min,  CPU only
  gate      image + GPU + IB + arch-registration gate     ~20 min, 1 node
  1node     Kimi-K3 TP8 on ONE node (expected to OOM)     <2 h,    1 node
  base      Kimi-K3 TP8 x PP2 on TWO nodes + analysis     3-4 h,   2 nodes

  --alt     retarget onto \$B200_NODES_ALT with no reservation

model (pre-staged, read-only, never downloaded):
  $MKIMI

from ./notes (baked into the #SBATCH headers):
  reservation : ${RESV:-<none>}
  account     : ${ACCT:-<none>}
  nodes       : $B200_NODES
  parallelism : 1node = TP8 x PP1 (8 GPUs) | base = TP8 x PP2 (16 GPUs)

fallback (--alt):
  nodes       : $B200_NODES_ALT   (no reservation, no account)
USAGE
}

[[ -n "$STAGE" ]] || { usage; exit 1; }

# Retarget a job script onto the fallback nodes by rewriting its #SBATCH header into a
# temp copy, rather than trying to clear an #SBATCH value from the command line. Slurm's
# behaviour when overriding a header reservation with an empty one is not something to
# discover on a 2-node allocation; deleting the lines outright is unambiguous.
retarget() {
  local src=$1 nodes=$2 tmp
  tmp=$(mktemp "$OUT_DIR/alt-$(basename "$src" .sh).XXXX.sh")
  sed -e '/^#SBATCH --reservation=/d' \
      -e '/^#SBATCH -A rres_acc_/d' \
      -e "s|^#SBATCH -w .*|#SBATCH -w $nodes|" "$src" >"$tmp"
  chmod +x "$tmp"
  echo "$tmp"
}

ALT_ARR=(${B200_NODES_ALT//,/ })

run() {  # run <script> <alt-nodes>
  local script=$1 altnodes=$2
  if [[ $ALT -eq 1 ]]; then
    script=$(retarget "$script" "$altnodes")
    echo "retargeted -> $script (nodes: $altnodes, no reservation)"
    grep -E '^#SBATCH (-w|-N|-p|--reservation|-A)' "$script" | sed 's/^/    /'
  fi
  set -x
  exec sbatch "$@" "$script"
}

case "$STAGE" in
  probe)    exec ./job-probe-drivers.sh ;;
  # No reservation: a CPU job must not consume the B200 reservation's walltime.
  download) set -x; exec sbatch "$@" download-kimi.sh ;;
  gate)     run job-gate-b200.sh  "${ALT_ARR[0]}" ;;
  1node)    run job-kimi-1node.sh "${ALT_ARR[0]}" ;;
  base)
    [[ ${#ALT_ARR[@]} -ge 2 || $ALT -eq 0 ]] || {
      echo "need 2 nodes in B200_NODES_ALT, have: $B200_NODES_ALT" >&2; exit 1; }
    run job-kimi-base.sh "$B200_NODES_ALT" ;;
  *) usage; exit 1 ;;
esac
