#!/usr/bin/env bash
# Shared config for the Kimi-K3 B200 benchmark. Source, don't execute.
#
# Mirrors ../amd-benchmarks/amd-cloud/common/env.sh in spirit so the two runs stay
# structurally comparable, but every path and tool differs: Engaging has Slurm +
# Apptainer, no Docker, and no ROCm.

export BENCH_ROOT=/orcd/data/orcd/022/benchmarks/b200-kimi
# The MI355X run this benchmark is compared against. Read-only: the analyzer reads
# its sweep JSON, server log and results/ from here and never writes into it.
export AMD_ROOT=/orcd/data/orcd/022/benchmarks/amd-benchmarks/amd-cloud

export MODELS=$BENCH_ROOT/models
export IMG_DIR=$BENCH_ROOT/imag
export LOG_ROOT=$BENCH_ROOT/logs
export RESULTS=$BENCH_ROOT/results
export OUT_DIR=$BENCH_ROOT/out          # sbatch -o lands here

# Kimi-K3 is ALREADY STAGED on the cluster -- do not download it.
#   /orcd/compute/orcd/025/models/Kimi-K3 -> safetensors/Kimi-K3/   (a symlink)
# Verified 2026-08-20: 96 safetensors shards, 1,560,936,091,448 bytes (1.42 TiB); the
# index declares 497,220 tensors across exactly 96 shards; quantization_config.format is
# "mxfp4-pack-quantized" under text_config -- the same checkpoint the MI355X run served.
#
# It belongs to another user (lincolnb, mode 644) on the shared NFS export
# fstor025.ib:/compute/orcd/025. Everything here treats it as strictly READ-ONLY: the
# container bind is :ro and no script ever writes inside it.
export MODEL_STORE="${MODEL_STORE:-/orcd/compute/orcd/025/models}"
export MKIMI="${MKIMI:-$MODEL_STORE/Kimi-K3}"
export MSMOKE="${MSMOKE:-$MODELS/Qwen3-8B-FP8}"

# Expected safetensors byte total, for the preflight check. Files other than the shards
# (tokenizer, *.py, assets) are excluded so the figure is stable regardless of which
# optional repo files were fetched.
export KIMI_SHARDS="${KIMI_SHARDS:-96}"
export KIMI_BYTES="${KIMI_BYTES:-1560936091448}"

# vLLM image. ATOM is ROCm-only, so the engine changes; the measurement path does not
# (`vllm bench serve` is the same code lineage as `atom.benchmarks.benchmark_serving`
# and writes the same JSON keys, which is what lets one analyzer read both runs).
export VLLM_TAG="${VLLM_TAG:-kimi-k3}"
export VLLM_SIF="${VLLM_SIF:-$IMG_DIR/vllm-openai_${VLLM_TAG}.sif}"
# Written by pull-image.sh only after a pull has fully completed and been verified.
# Its existence -- not the .sif's -- is what "the image is ready" means here.
export VLLM_SIF_MANIFEST="${VLLM_SIF_MANIFEST:-${VLLM_SIF}.manifest}"
export VLLM_SIF_LOCK="${VLLM_SIF_LOCK:-${VLLM_SIF}.lock}"

# Apptainer caches MUST NOT land in $HOME: it is at ~441/500 GB and a 15 GB image pull
# plus its layer cache would blow the quota mid-pull, leaving a corrupt .sif.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$BENCH_ROOT/.apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$BENCH_ROOT/.apptainer/tmp}"
export HF_HOME="${HF_HOME:-$BENCH_ROOT/.hf}"

export APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer/1.5.2}"

# --- python inside the container -------------------------------------------------
# The vllm:kimi-k3 image has NO `python`, only `python3` (/usr/bin/python3, 3.12.3).
# Discovered the hard way: the first gate run died with
#   FATAL: "python": executable file not found in $PATH
export PY_C="${PY_C:-python3}"

# Ray is NOT in this image (`Requires-Dist: ray` is absent and `import ray` fails), yet
# TP8 x PP2 across two nodes needs a distributed executor. Ray 2.57.0 is installed
# alongside it in $PYLIBS with --no-deps -- every one of its dependencies was already
# present in the container, so nothing gets shadowed. Verified: both
# vllm.v1.executor.ray_executor.RayDistributedExecutor and ray_executor_v2.RayExecutorV2
# import cleanly against it.
export PYLIBS="${PYLIBS:-$BENCH_ROOT/pylibs}"
# `pip install --target` does create pylibs/bin/ray, but its shebang points at whatever
# python resolved at install time. Invoking the module is shebang-proof.
export RAY_CLI="${RAY_CLI:-$PY_C -m ray.scripts.scripts}"

# --- Slurm ---------------------------------------------------------------------
export SLURM_PART="${SLURM_PART:-mit_testing}"

# The reservation from ./notes. Verified live 2026-08-20:
#   rres_joohye_2026-08-20_lj4j2ya3  ACTIVE 2026-08-20 13:41 -> 2026-08-27 13:41
#   Nodes=node5700-c1,node5701-c1   Partition=mit_testing
#   Accounts=rres_acc_joohye_2026-08-20_lj4j2ya3
export RESV="${RESV:-rres_joohye_2026-08-20_lj4j2ya3}"
export ACCT="${ACCT:-rres_acc_joohye_2026-08-20_lj4j2ya3}"
export B200_NODES="${B200_NODES:-node5700-c1,node5701-c1}"

# Fallback node set. Five B200 nodes exist on mit_testing:
#   node570[0-1]-c1  Ubuntu 24.04  -- the reserved pair above
#   node550[0-2]-c1  Rocky/EL10    -- free, no reservation needed
#
# WHY THIS MATTERS: as of the 2026-08-12 survey in ../b200-ubuntu/ubuntu-nccl.md the
# Ubuntu pair ran driver 570.211.01 / CUDA 12.9, and the Rocky nodes ran 590.48.01.
# The vllm:kimi-k3 image is a CUDA 13 (cu130) build with NO cu129 tag and needs r580+.
# If the reserved Ubuntu nodes are still on r570 the image will not run there and this
# benchmark has to move to the Rocky nodes. job-probe-drivers.sh settles it in minutes;
# do not skip it.
export B200_NODES_ALT="${B200_NODES_ALT:-node5500-c1,node5501-c1}"

# Assembled once so every job script asks for the allocation the same way.
slurm_args() {
  local a=(-p "$SLURM_PART")
  [[ -n "${RESV:-}" ]] && a+=(--reservation="$RESV")
  [[ -n "${ACCT:-}" ]] && a+=(-A "$ACCT")
  echo "${a[@]}"
}
export -f slurm_args

# --- run configuration ---------------------------------------------------------
# Arm A: matched to the MI355X run in ../amd-benchmarks/amd-cloud/results/kimi-k3-base.md
# so the only differences are hardware, engine and parallelism. Do not change these
# without also re-labelling the comparison as non-matched.
export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
export PROMPT_MULTIPLIER="${PROMPT_MULTIPLIER:-10}"   # num-prompts = conc x this
export KIMI_CONC="${KIMI_CONC:-1 2 4 8 16 32 64}"     # capped at max-num-seqs
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
export SEED="${SEED:-42}"

# TP8 x PP2 across the two nodes. NOT a tuning choice: the 1561 GB MXFP4 checkpoint
# does not fit in one node's 8 x 192 GB = 1538 GB of HBM -- 23 GB short on weights
# alone, ~150 GB short once KV pool, workspace and NCCL buffers are counted. The vLLM
# recipe reaches the same conclusion and marks TP8xPP2 the verified B200 layout.
export TP="${TP:-8}"
export PP="${PP:-2}"
# 0.90, not the recipe baseline 0.95: the flashinfer TRTLLM MXFP4 MoE kernel takes a
# ~1.6 GiB workspace outside vLLM's pool on the first forward, and at 0.95 a 180 GB
# B200 OOMs during warmup. Straight from the recipe's multi_node_tp_pp override.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

export PORT="${PORT:-8000}"
export SERVED_NAME="${SERVED_NAME:-Kimi-K3}"
export READY_TIMEOUT="${READY_TIMEOUT:-3600}"    # 1.42 TiB off NFS
export BENCH_TIMEOUT="${BENCH_TIMEOUT:-3600}"

# --- NCCL ----------------------------------------------------------------------
# Copied from ../b200-nodes/job-nccl-2node.sh, which is the configuration proven to
# connect at 8 GPUs/node on this fabric. Leaving NIC selection to NCCL works at 1
# GPU/node and FAILS to connect at 16 ranks, so the rails are pinned explicitly.
nccl_env() {
  cat <<'NCCLEOF'
NCCL_IB_DISABLE=0
NCCL_NET_GDR_LEVEL=2
NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15
NCCL_SOCKET_IFNAME=^lo,docker
NCCL_DEBUG=WARN
NCCL_CUMEM_ENABLE=1
NCCLEOF
}
export -f nccl_env

# apptainer exec args shared by every GPU step.
#   --nv          : inject the host driver + NVML
#   no --contain  : /dev/infiniband and the mlx5 uverbs devices must stay visible,
#                   otherwise NCCL silently falls back to TCP and the PP stage
#                   boundary runs at socket speed
apt_args() {
  # The model path is a SYMLINK into $MODEL_STORE/safetensors/, so binding the symlink
  # alone would leave a dangling link inside the container. Bind the store itself, which
  # covers both the link and its target -- and bind it READ-ONLY: it is another user's
  # data and nothing here has any business writing to it.
  #
  # PYTHONNOUSERSITE=1 is load-bearing, not hygiene. Apptainer maps $HOME by default, so
  # python auto-adds ~/.local/lib/python3.12/site-packages AHEAD of the container's own
  # dist-packages. That made `import transformers` resolve to the host's copy, which
  # imports tensorflow, which died on a missing opt_einsum -- inside a container that
  # ships a perfectly good transformers 5.14.1. Setting it makes the container's own
  # packages authoritative.
  #
  # PYTHONPATH is set explicitly (not appended) so the host's value -- which carried
  # /home/shaohao/VibeCodeHPC into sys.path -- cannot leak in either. $PYLIBS supplies
  # ray and nothing else.
  # PATH is pinned to the CONTAINER's own path. Apptainer prepends the host's PATH by
  # default (even under --cleanenv), which put Engaging's Spack gcc first -- and vLLM's
  # Triton JIT then compiled cuda_utils.c with
  #   /orcd/software/core/001/spack/pkg/gcc/12.2.0/.../bin/gcc
  # against the CONTAINER's headers, dying on
  #   /usr/include/stdlib.h:26: fatal error: bits/libc-header-start.h: No such file
  # because that Spack gcc knows nothing of Ubuntu's multiarch include layout. Model
  # inspection aborted before a single weight was read (job 20852177).
  # CC/CXX are pinned too, since Triton consults them before searching PATH.
  #
  # Caches go to node-local /tmp, never $HOME -- and HOME itself is redirected there.
  #
  # This is not just about the ~441/500 GB quota. FlashInfer JIT-builds the TRTLLM FP4
  # block-scale MoE kernel on first use (during vLLM's memory-profiling forward pass)
  # and guards its cache with a filelock. On NFS, fcntl.flock() returned
  #   OSError: [Errno 5] Input/output error
  # on 14 of 16 ranks and killed the 2-node run (job 20856604) after the weights had
  # already loaded -- 28 minutes in. flock over this NFS mount is not dependable, so
  # every JIT cache must live on node-local /tmp. XDG_CACHE_HOME alone was not enough:
  # FlashInfer derives its path from HOME, which still pointed at the NFS home.
  # --home, NOT --env HOME: apptainer explicitly refuses to override HOME via the
  # environment ("Overriding HOME environment variable with APPTAINERENV_HOME is not
  # permitted") and silently ignores it. This is the flag that actually moves HOME off
  # NFS, which is what keeps FlashInfer's filelock off a mount where flock returns EIO.
  # Local source patch, bind-mounted over the container's copy (the .sif is read-only).
  # patches/mamba_hybrid.py fixes an upstream vLLM crash on the Kimi-K3 KDA hybrid-state
  # path that killed the engine at concurrency >= 16:
  #   IndexError: index_fill_(): Expected dtype int64 for index.   (14 ranks)
  # See the LOCAL PATCH comment in that file for the two defects it addresses.
  local mh=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
  # Second local patch: cv2's native dlopen bootstrap segfaults non-deterministically
  # at high concurrency under this container -- a crash that no \`except ImportError\`
  # anywhere in the stack can catch, because a segfault is not a raiseable Python
  # exception. Reproduced across FOUR separate 2-node runs at three DIFFERENT import
  # sites (vllm.multimodal.video; mistral_common.imports.is_opencv_installed(), reached
  # via transformers' own tokenizer auto-detection on every bench-client startup; jobs
  # 20909041, 20912377, 20914469). Patching call sites one at a time does not scale --
  # replacing cv2/__init__.py itself with a clean, immediate ImportError fixes every
  # current and future call site at once, and restores the try/except guards already
  # written everywhere to their originally intended behaviour. This benchmark sends no
  # images or video (--dataset-name random, pure text), so cv2 is never functionally
  # needed. See patches/cv2_init_stub.py.
  local cv=/usr/local/lib/python3.12/dist-packages/cv2/__init__.py
  echo --nv \
    --home "/tmp/cthome-$USER" \
    --bind "$BENCH_ROOT/patches/mamba_hybrid.py":"$mh":ro \
    --bind "$BENCH_ROOT/patches/cv2_init_stub.py":"$cv":ro \
    --bind "$BENCH_ROOT" \
    --bind "$MODEL_STORE":"$MODEL_STORE":ro \
    --bind /orcd/data/orcd/022/benchmarks/amd-benchmarks \
    --env PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --env CC=/usr/bin/gcc \
    --env CXX=/usr/bin/g++ \
    --env TRITON_CACHE_DIR=/tmp/triton-cache-$USER \
    --env XDG_CACHE_HOME=/tmp/xdg-cache-$USER \
    --env FLASHINFER_CACHE_DIR=/tmp/flashinfer-$USER \
    --env FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer-$USER \
    --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE=1 \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONPATH="$PYLIBS"
}

# Resolve a hostname to a ROUTABLE IPv4 address.
#
# `getent hosts <node>` is not safe for this. On the login node it returns
# 10.1.57.64, but ON THE COMPUTE NODE it returned the IPv6 link-local
# fe80::a449:dfff:fe2e:3b60 -- which went straight into `ray start
# --node-ip-address`, and Ray then could not reach its own GCS:
#   Failed to connect to GCS at address [fe80::a449:dfff:fe2e:3b60]:6379
# That killed the 2-node run (job 20852178) during cluster bring-up.
#
# ahostsv4 forces the IPv4 family; the result is then validated as a dotted quad that is
# neither loopback nor link-local, with `ip route` as a last resort.
resolve_ipv4() {
  local host="${1:?resolve_ipv4 <hostname>}" ip=""
  ip=$(getent ahostsv4 "$host" 2>/dev/null | awk '{print $1; exit}')
  if ! [[ "$ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || [[ "$ip" == 127.* || "$ip" == 169.254.* ]]; then
    ip=$(getent hosts "$host" 2>/dev/null | awk '$1 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print $1; exit}')
  fi
  if ! [[ "$ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || [[ "$ip" == 127.* || "$ip" == 169.254.* ]]; then
    ip=$(ip -4 -o route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
  fi
  [[ "$ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] && [[ "$ip" != 127.* && "$ip" != 169.254.* ]] || return 1
  echo "$ip"
}
export -f resolve_ipv4
export -f apt_args

# --- image readiness -----------------------------------------------------------
# The image is pulled EXACTLY ONCE, by pull-image.sh, into $IMG_DIR. Nothing else ever
# pulls: a GPU job that discovers a missing image must fail immediately rather than
# spend allocation time fetching 15 GB, and two jobs must never race to write the same
# .sif. check_image is the single gate every consumer calls.
#
# Presence of the .sif is NOT the test. `apptainer pull` writes its destination in
# place, so an interrupted pull leaves a truncated file at the final path that a plain
# `-f` check happily accepts -- and the failure then surfaces as an inscrutable engine
# crash 20 minutes into a 2-node allocation. pull-image.sh instead builds to a temp
# name, verifies, atomically renames, and only then writes the manifest. So the manifest
# is the completion record, and the size check catches a .sif truncated afterwards.
check_image() {
  local sif man
  if [[ -n "${1:-}" ]]; then sif="$1"; man="$1.manifest"
  else                       sif="$VLLM_SIF"; man="$VLLM_SIF_MANIFEST"; fi

  if [[ ! -f "$sif" ]]; then
    echo "ERROR: image not found: $sif" >&2
    echo "       Pull it ONCE on the login node:  ./pull-image.sh" >&2
    return 1
  fi
  if [[ ! -f "$man" ]]; then
    echo "ERROR: $sif exists but has no manifest ($man)." >&2
    echo "       That means the pull did not complete -- the file is very likely a" >&2
    echo "       truncated download. Re-run ./pull-image.sh (it will discard and refetch)." >&2
    return 1
  fi
  local want have
  want=$(awk -F= '/^bytes=/{print $2}' "$man")
  have=$(stat -c %s "$sif" 2>/dev/null)
  if [[ -n "$want" && "$want" != "$have" ]]; then
    echo "ERROR: $sif is $have bytes, manifest says $want." >&2
    echo "       The image changed or was truncated after the pull." >&2
    echo "       Re-run ./pull-image.sh --force" >&2
    return 1
  fi
  return 0
}
export -f check_image

# --- model readiness ------------------------------------------------------------
# The checkpoint is pre-staged and read-only, so the only real questions are whether
# this node can see the export at all and whether the shard set is intact. Both are
# cheap metadata checks; neither reads a byte of the 1.42 TiB.
check_model() {
  local m="${1:-$MKIMI}"
  if [[ ! -d "$m" ]]; then
    echo "ERROR: model not found: $m" >&2
    echo "       Expected the pre-staged checkpoint. Is $MODEL_STORE mounted on this node?" >&2
    echo "       (login node has it via fstor025.ib:/compute/orcd/025)" >&2
    return 1
  fi
  local n
  n=$(ls -1 "$m"/*.safetensors 2>/dev/null | wc -l)
  if [[ "$n" -ne "$KIMI_SHARDS" ]]; then
    echo "ERROR: $m has $n safetensors shards, expected $KIMI_SHARDS" >&2
    return 1
  fi
  local b
  b=$(ls -lL "$m"/*.safetensors 2>/dev/null | awk '{t+=$5} END{print t+0}')
  if [[ "$b" != "$KIMI_BYTES" ]]; then
    echo "ERROR: $m safetensors total is $b bytes, expected $KIMI_BYTES" >&2
    echo "       The staged checkpoint changed. Do not benchmark against it silently." >&2
    return 1
  fi
  [[ -r "$m/config.json" ]] || { echo "ERROR: cannot read $m/config.json" >&2; return 1; }
  return 0
}
export -f check_model

mkdir -p "$LOG_ROOT" "$RESULTS" "$OUT_DIR" "$IMG_DIR" "$MODELS" \
         "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME" 2>/dev/null
