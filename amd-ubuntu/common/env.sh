#!/usr/bin/env bash
# Shared config for every part (A RVS, B rccl-tests, C Primus, D ATOM, megatron-ref).
# Source, don't execute. Ported from ../amd-benchmarks/amd-cloud/common/env.sh.
#
# Hosts: node6100.inband / node6101.inband -- 8x MI355X (gfx950) each, Ubuntu 24.04,
# ROCm 7.2.4 at /opt/rocm, ssh only (not in Slurm). Every script runs ON one of these nodes;
# launch from a login node with ../launch.sh. The nodes' $HOME is local and differs from the
# login nodes, so only absolute /orcd/data paths are used.
#
# Differences from amd-cloud that apply to every script here:
#   * No docker on these nodes: containers run with apptainer from SIFs in $SIF_DIR
#     (built by amd-software/setup/job-pull-sif.sh) through the common/bin/docker shim.
#   * All software/caches/weights live in ../amd-software (see amd-software/env.sh).
#   * HSA_OVERRIDE_GFX_VERSION is never set (native gfx950), same as amd-cloud.
source /orcd/data/orcd/022/benchmarks/amd-software/env.sh
export BENCH_ROOT=/orcd/data/orcd/022/benchmarks/amd-ubuntu
export REF_ROOT=/orcd/data/orcd/022/benchmarks/amd-benchmarks/amd-cloud   # amd-cloud MI355X results
# Per-node logs and results: both nodes run the same parts concurrently, so nothing they
# write may collide. The 2-node RCCL run writes under node6100's (or whichever launched it).
export HOSTNAME_S=$(hostname -s)
# RESULTS_SUBDIR separates runs of the same part with different images (atom/run_kimi_all.sh)
# or a different ROCm stack (ROCM_STACK=7.14 -> results/<node>/rocm7.14/).
[[ $ROCM_STACK == 7.14 && -z ${RESULTS_SUBDIR:-} ]] && export RESULTS_SUBDIR=rocm7.14
export LOG_ROOT=$BENCH_ROOT/logs/$HOSTNAME_S${RESULTS_SUBDIR:+/$RESULTS_SUBDIR}
export RESULTS=$BENCH_ROOT/results/$HOSTNAME_S${RESULTS_SUBDIR:+/$RESULTS_SUBDIR}
export RCCL_TESTS_DIR=$RCCL_TESTS/$RCCL_TESTS_BUILD          # MPI=0 binaries (single node)
export RCCL_TESTS_MPI_DIR=$RCCL_TESTS/$RCCL_TESTS_MPI_BUILD  # MPI=1 binaries (2 nodes)
export NGPU=8
mkdir -p "$LOG_ROOT"/{rvs,rccl,primus,atom,megatron-ref} "$RESULTS" 2>/dev/null

# RCCL env -- single node, XGMI only, IB off. Same as amd-cloud so numbers stay comparable.
rccl_env() {
  cat <<'EOF'
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=lo
NCCL_P2P_DISABLE=0
NCCL_SHM_DISABLE=0
RCCL_MSCCL_ENABLE=1
NCCL_PROTO=Simple,LL,LL128
NCCL_ALGO=Ring,Tree
NCCL_DEBUG=WARN
EOF
}

# ---- containers ---------------------------------------------------------------------
# No docker on these nodes. common/bin/docker is a shim that implements the docker subset the
# amd-cloud scripts use (run/-d/exec/ps/logs/stop/rm/image inspect) on top of apptainer and
# the SIFs in $SIF_DIR, so the ported scripts keep their docker calls unchanged.
export PATH="$BENCH_ROOT/common/bin:$PATH"
# amd-cloud passed GPU/IPC/ulimit flags via $(dgpu_args). apptainer exposes /dev/kfd,/dev/dri
# and uses host networking/IPC by default, so there is nothing to add.
dgpu_args() { :; }

# ---- guards ---------------------------------------------------------------------------
# The user must be able to open /dev/kfd (render group). Without it every GPU test reports
# "No supported GPUs" and produces garbage, so fail fast.
assert_gpu_access() {
  if [[ ! -r /dev/kfd || ! -w /dev/kfd ]]; then
    echo "ERROR: $(id -un) cannot open /dev/kfd on $HOSTNAME_S (needs render/video group)." >&2
    return 1
  fi
}

# Warn if someone else is using the GPUs.
assert_gpus_idle() {
  assert_gpu_access || exit 1
  local busy
  busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$' || true)
  [[ "${busy:-0}" -eq 0 ]] || { echo "WARNING: $busy GPU(s) busy — another workload is running"; }
}
