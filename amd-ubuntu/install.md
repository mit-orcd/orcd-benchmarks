# amd-ubuntu — installation

All software for the MI355X nodes (`node6100.inband`, `node6101.inband`: 8 ×
MI355X each, Ubuntu 24.04, ROCm 7.2.4 at `/opt/rocm`, ssh only, no Slurm, no
docker) is installed in
**`/orcd/data/orcd/022/benchmarks/amd-software`** (`$SW`), not here. That dir is
not on GitHub, so every step is written out below. Its `setup/*.sh` scripts do
the same steps and are idempotent.

The nodes have a node-local `$HOME`, so always use absolute `/orcd/data` paths.
`$SW/env.sh` defines every path. Source it instead of hard-coding paths.

## 0. Node access (once, admin or sudo)

```bash
# GPU access: user in render + video on both nodes (setup/grant-gpu-access.sh)
ssh node6100.inband 'sudo usermod -aG render,video $USER'
ssh node6101.inband 'sudo usermod -aG render,video $USER'

# passwordless ssh between the two nodes, for 2-node mpirun (node-local key on each)
ssh node6100.inband 'ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519'   # then add each node's
ssh node6101.inband 'ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519'   # .pub to the other's authorized_keys
```

Apptainer 1.5.4 is at `/usr/local/x86_64` on the nodes (unprivileged, user
namespaces enabled). `amd-ubuntu/common/bin/docker` is an apptainer shim, so the
ported scripts that call `docker` run unchanged.

## 1. Sources (`setup/clone-sources.sh`)

```bash
SW=/orcd/data/orcd/022/benchmarks/amd-software; mkdir -p $SW/src && cd $SW/src
git clone --depth=1 https://github.com/ROCm/rccl-tests.git rccl-tests
git clone --depth=1 https://github.com/AMD-AIG-AIMA/Primus.git Primus     # configs only; runs from its image
git clone --depth=1 https://github.com/ROCm/ATOM.git ATOM                 # recipes only; runs from its image
```

## 2. Open MPI 5.0.8, launcher for 2-node rccl-tests (`setup/build-openmpi.sh`)

Build in node-local `/tmp`, because configure on NFS is about 20× slower, then
install to `$SW/openmpi`:

```bash
ssh node6100.inband
SW=/orcd/data/orcd/022/benchmarks/amd-software; cd $SW/src
curl -fsSLO https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.8.tar.bz2
B=/tmp/$USER-openmpi-build; mkdir -p $B && tar xjf openmpi-5.0.8.tar.bz2 -C $B && cd $B/openmpi-5.0.8
./configure --prefix=$SW/openmpi --without-ucx --without-cuda --disable-mpi-fortran
make -j32 && make install
cd / && rm -rf $B
```

## 3. rccl-tests for gfx950, host ROCm 7.2.4 (`setup/build-rccl-tests.sh`)

```bash
SW=/orcd/data/orcd/022/benchmarks/amd-software; cd $SW/src/rccl-tests
make -j32 MPI=0 HIP_HOME=/opt/rocm GPU_TARGETS=gfx950 BUILDDIR=build
make -j32 MPI=1 MPI_HOME=$SW/openmpi HIP_HOME=/opt/rocm GPU_TARGETS=gfx950 BUILDDIR=build-mpi
```

## 4. Optional: ROCm 7.14 user space, the stack amd-cloud ran (`setup/install-rocm-7.14.sh`)

This step needs no root. It installs TheRock's gfx950 tarball, RVS built
against it, and rccl-tests built against it.

```bash
SW=/orcd/data/orcd/022/benchmarks/amd-software; R=$SW/rocm-7.14.0; mkdir -p $SW/downloads
curl -fL -C - -o $SW/downloads/therock-dist-linux-gfx950-dcgpu-7.14.0.tar.gz \
  https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx950-dcgpu-7.14.0.tar.gz
mkdir -p $R && tar -xzf $SW/downloads/therock-dist-linux-gfx950-dcgpu-7.14.0.tar.gz -C $R
#   (if the tarball has one top-level dir, move its contents up so $R/bin/hipcc exists)
$R/bin/hipcc --version | grep "clang version 23"

# RVS 1.7.8 at amd-cloud's commit
git clone https://github.com/ROCm/ROCmValidationSuite.git $SW/src/ROCmValidationSuite
cd $SW/src/ROCmValidationSuite && git checkout c82e4aeb2efdcca96ec9bf074e082eecf488ec8c && git submodule update --init --recursive
B=/tmp/$USER-rvs-build; cmake -S . -B $B -DROCM_PATH=$R -DCMAKE_PREFIX_PATH="$R;$R/lib/cmake" \
  -DCMAKE_C_COMPILER=$R/llvm/bin/clang -DCMAKE_CXX_COMPILER=$R/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=$SW/rvs-7.14.0 -DCPACK_PACKAGING_INSTALL_PREFIX=$SW/rvs-7.14.0
make -C $B -j16 && make -C $B install && rm -rf $B

# rccl-tests against 7.14
cd $SW/src/rccl-tests
make -j16 MPI=0 HIP_HOME=$R ROCM_PATH=$R NCCL_HOME=$R GPU_TARGETS=gfx950 BUILDDIR=build-rocm7.14
make -j16 MPI=1 MPI_HOME=$SW/openmpi HIP_HOME=$R ROCM_PATH=$R NCCL_HOME=$R GPU_TARGETS=gfx950 BUILDDIR=build-mpi-rocm7.14
```

To select it, `export ROCM_STACK=7.14` before `source $SW/env.sh`. That sets
`ROCM_PATH`, `HIP_PATH`, `HIP_CLANG_PATH`, `PATH`, `LD_LIBRARY_PATH` (including
`rocm-7.14.0/llvm/lib` for libomp), `RVS_BIN`, `RVS_CONF` and the rccl-tests
dirs. Without those variables, hipcc and cmake silently fall back to the
`/opt/rocm` clang. RVS for the default stack is the packaged `/opt/rocm/bin/rvs`.

## 5. Python venv for analysis and `hf` (`setup/make-venv.sh`)

The nodes lack `python3.12-venv`, so the venv is made with `uv`:

```bash
SW=/orcd/data/orcd/022/benchmarks/amd-software; mkdir -p $SW/bin
curl -fsSL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar xz -C $SW/bin --strip-components=1
export UV_CACHE_DIR=$SW/cache/uv
$SW/bin/uv venv -p /usr/bin/python3 $SW/venv
$SW/bin/uv pip install -p $SW/venv/bin/python matplotlib pandas tabulate "huggingface_hub[hf_xet]" requests
```

## 6. Models (`setup/download-models.sh`)

```bash
SW=/orcd/data/orcd/022/benchmarks/amd-software
$SW/venv/bin/hf download Qwen/Qwen3-8B-FP8 --local-dir $SW/models/Qwen3-8B-FP8 --max-workers 8                       # 8.8 GB, ATOM tier 1
$SW/venv/bin/hf download RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 --local-dir $SW/models/Llama-3.1-70B-Instruct-FP8 # 68 GB, tier 2
$SW/venv/bin/hf download Inferact/Kimi-K3-DSpark --local-dir $SW/models/Kimi-K3-DSpark                              # 6.6 GB, vLLM recipe draft model
```

**Kimi-K3 is never downloaded.** Copy the staged copy to node-local `/scratch`
on each node. The scripts always read `/scratch/Kimi-K3`.

```bash
ssh node6100.inband 'rsync -a /orcd/compute/orcd/025/models/Kimi-K3/ /scratch/Kimi-K3/'
ssh node6101.inband 'rsync -a /orcd/compute/orcd/025/models/Kimi-K3/ /scratch/Kimi-K3/'
```

## 7. Container images, `$SW/sif/` (`setup/build-sifs-on-nodes.sh`, `job-pull-sif.sh`)

Build on the AMD nodes with `apptainer build`, not `pull`: `pull` cannot limit
the mksquashfs thread count, and with too many threads mksquashfs dies with
`Failed to create thread`. Keep the layers in `/dev/shm` (about 5 min per
image). The amd-cloud ATOM images are pinned by digest, because `:latest` has
moved since.

```bash
SW=/orcd/data/orcd/022/benchmarks/amd-software; mkdir -p $SW/sif
build() {   # build <docker ref> <sif name>
  export APPTAINER_CACHEDIR=/dev/shm/$USER-sif/cache APPTAINER_TMPDIR=/dev/shm/$USER-sif/tmp
  mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
  apptainer build --force --mksquashfs-args "-processors 32" $SW/sif/$2 $1
  rm -rf /dev/shm/$USER-sif
}
build docker://rocm/primus:v26.5            primus-v26.5.sif          # Part C
build docker://rocm/megatron-lm:v26.1       megatron-lm-v26.1.sif     # megatron-ref
build docker://rocm/atom-dev@sha256:b750c5fbe4b79cee8c7466198ebfe63c29035cdb9133844ff3e946588e286b64 atom-dev-nightly_202608111555.sif  # Part D, Kimi-K3
build docker://rocm/atom-dev@sha256:f86e2bb21d86c5927cedf53e5f88d90e907b26d3494921eb1f8c1af7ff49b049 atom-dev-nightly_202608191459.sif  # Kimi-K3 isl4096
build docker://rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3 atom-dev-kimi-k3-mad.sif                       # Kimi-K3 MAD
build docker://vllm/vllm-openai-rocm@sha256:e76a953fa2c317e5e0913fab946887ae498629c8e57fb6b2dc633dadd4098b61 vllm-openai-rocm-nightly-rocm100.sif  # AMD vLLM recipe
```

The scripted version writes a `.manifest` (ref + digest) next to each SIF:

```bash
S=$SW/setup
ssh node6100.inband "setsid nohup bash $S/build-sifs-on-nodes.sh primus megatron atom-0811 vllm-rocm100 > $S/logs/build-sifs.node6100.log 2>&1 < /dev/null &"
ssh node6101.inband "setsid nohup bash $S/build-sifs-on-nodes.sh atom-mad atom-0819 > $S/logs/build-sifs.node6101.log 2>&1 < /dev/null &"
```

## All non-container steps in one go

Steps 1–3, 5 and 6 together:

```bash
ssh node6100.inband 'setsid nohup /orcd/data/orcd/022/benchmarks/amd-software/setup/setup-all.sh \
  > /orcd/data/orcd/022/benchmarks/amd-software/setup/logs/setup-all.log 2>&1 < /dev/null &'
cat /orcd/data/orcd/022/benchmarks/amd-software/setup/logs/STATUS.txt     # one line per finished step
```

## Check

```bash
source /orcd/data/orcd/022/benchmarks/amd-software/env.sh
$RCCL_TESTS/build/all_reduce_perf -b 8 -e 1G -f 2 -g 8
$MPI_HOME/bin/mpirun -np 16 --host node6100.inband:8,node6101.inband:8 $RCCL_TESTS/build-mpi/all_reduce_perf -b 8 -e 1G -f 2 -g 1
```
