# Plan — B200 benchmarks on the Ubuntu nodes (node5700, node5701)

Goal: reproduce the `../b200-nodes` benchmark set (gpu-fryer, NCCL, Megatron-LM)
on the two **Ubuntu 24.04** B200 nodes **node5700** and **node5701**, so results
are directly comparable to the Rocky-8 nodes (node5500/5501/5502).

Working dir: `/orcd/data/orcd/022/benchmarks/b200-utuntu` (this dir, on shared NFS).
Nothing in `../b200-nodes` or any other benchmark dir gets modified — new files
only, plus one *new* nccl-tests build directory.

---

## 1. Environment survey (already done on node5700)

| Item | node5700 (Ubuntu) | node5500/5502 (Rocky 8) |
|---|---|---|
| OS / kernel | Ubuntu 24.04.4, 6.8.0-124-generic | RHEL 8 / 10 |
| GPUs | 8× B200, 183 GB, driver **570.211.01** | 8× B200, driver 590.48.01 |
| Slurm | **absent** (no sbatch/sinfo) | `mit_testing` partition |
| Lmod / `module` | **absent** | modules for apptainer, nvhpc |
| `/orcd/software` | **not mounted** → no nvhpc/26.1, no host MPI, no nvcc | mounted |
| apptainer | `/usr/bin/apptainer` 1.4.x, FUSE mount works | module `apptainer/1.4.2` |
| docker | installed but **no socket permission** for this user | n/a |
| shared FS | `/orcd/data/orcd/022` (NFS) — scripts + output live here | same |
| `$HOME` | **local disk per node** (`/dev/md0`) — not shared | shared |
| InfiniBand | 16 `mlx5_*`; `mlx5_4,7,8,9,10,13,14,15` = 400 Gb/s NDR, `mlx5_0-3` = 100 Gb/s | same layout |
| node5701 | pings on `10.1.57.1` (`node5701.inband`); **ssh: Permission denied (publickey,password)** — no key pair in `~/.ssh` on node5700 | n/a |

Verified working on node5700 during the survey:
- `apptainer exec --nv gpu-fryer_1.1.0.sif gpu-fryer --use-fp32 --nvml-lib-path /.singularity.d/libs/libnvidia-ml.so.1 10` → ran clean, ~740 TFLOP/s per GPU, "All GPUs seem healthy".
- `pytorch_26.02-py3.sif` contains CUDA 13.1 (`nvcc`), OpenMPI (`/usr/local/mpi/bin/mpirun`), **NCCL 2.29.2 + `nccl.h`**, torch 2.11 seeing all 8 GPUs.

### Consequences (what must change vs. the Rocky-8 scripts)

1. **No Slurm** → every script is a local `run-*.sh`, executed after `ssh`ing to
   the node (the `job-*.sh` sbatch wrappers are dropped). A thin
   `run-both-nodes.sh` driver will fan out over ssh once ssh works.
2. **No `module load`** → drop all `module` calls; use `/usr/bin/apptainer`.
3. **No nvhpc / host MPI** → the prebuilt `nccl-tests/build-nvhpc-26.1` binaries
   **cannot run here** (`libcudart.so.13`, `libmpi.so.40`, `libnccl.so.2`,
   `libnvhpc*` all unresolved). NCCL tests must be **rebuilt inside the
   pytorch_26.02 container** into a new dir.
4. **gpu-fryer NVML path** → the Rocky script's `-B /lib64:/home/$USER/lib64`
   hack is wrong on Ubuntu (`/lib64` holds only the loader). Use the
   apptainer-injected library: `--nvml-lib-path /.singularity.d/libs/libnvidia-ml.so.1`.
   No bind needed. (Confirmed working.)
5. **`$HOME` is per-node** → all artifacts (containers, builds, outputs, caches)
   stay under `/orcd/data/orcd/022/...`; ssh keys must be installed on *each*
   node separately.
6. **No scheduler = no exclusivity** → each run script prints `nvidia-smi` first
   and aborts if another user's processes are on the GPUs.

---

## 2. Blocking prerequisite: ssh node5700 → node5701

Two-node NCCL and two-node Megatron both need passwordless ssh (torchrun
rendezvous / MPI remote launch), and the driver script needs it too.

Step 0 (needs the user, one password prompt or an admin):

```bash
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519      # on node5700
ssh-copy-id node5701                                  # needs password once
ssh node5701 hostname                                 # must succeed non-interactively
# and the reverse direction (MPI/torchrun can launch either way):
ssh node5701 'ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 && cat ~/.ssh/id_ed25519.pub' >> ~/.ssh/authorized_keys
```

If password login to node5701 is not available, everything single-node still
runs; the two-node items are deferred. **Single-node work does not wait on this.**

Also verify on node5701 (script `check-node.sh`): OS, driver version, 8 GPUs,
apptainer present, `/orcd/data/orcd/022` mounted, IB rails Active/400.

---

## 3. Files to create (all in this dir)

| File | Purpose |
|---|---|
| `check-node.sh` | one-shot env/health probe (OS, driver, GPUs, IB rates, apptainer, NFS) |
| `run-gpu-fryer.sh` | gpu-fryer fp32/bf16/fp8 stress, local, `[seconds]` (default 300) |
| `setup-nccl-tests.sh` | build nccl-tests inside `pytorch_26.02-py3.sif` → new build dir |
| `run-nccl-1node.sh` | intra-node NVLink collectives, `[collectives] [ngpus]` |
| `run-nccl-2node.sh` | inter-node IB collectives, `[collectives] [gpus_per_node]` |
| `megatron-cfg-1node.sh` | verbatim copy of `../b200-nodes/run-1node-b200.sh` (~7B GPT config) |
| `megatron-cfg-2node.sh` | verbatim copy of `../b200-nodes/run-2nodes-b200.sh` |
| `run-megatron-1node.sh` | container + GPU-count scan 1..8 on this node |
| `run-megatron-2node.sh` | torchrun c10d rendezvous across node5700+node5701 |
| `run-both-nodes.sh` | driver: ssh to each node and run the single-node suite |
| `analyze-*.py`, `md-to-pdf.py` | copies of the `../b200-nodes` analyzers (parsers are output-format based, so they should work unchanged; adjust only if the new output dirs differ) |
| `README.md` | usage, mirroring `../b200-nodes/README.md` but for the no-Slurm Ubuntu flow |
| `notes.md` | environment differences, issues hit, GDR/IB findings for these nodes |

Output dirs (same names as the Rocky-8 run, for analyzer reuse):
`out-gpu-fryer/`, `out-nccl-1node/`, `out-nccl-2node/`, `output-megatron/`.
Filenames carry `$(hostname)` + timestamp, so both nodes can write to the same
shared dir without collision.

---

## 4. Phase-by-phase

### Phase A — env check + gpu-fryer (no dependencies, start here)
1. Write `check-node.sh`, run on node5700 (and node5701 once ssh works).
2. Write `run-gpu-fryer.sh` (apptainer, `--nvml-lib-path /.singularity.d/libs/...`,
   no `/lib64` bind, no module).
3. Smoke test at 10 s per precision, then the real run: **300 s × {fp32, bf16, fp8}**
   ≈ 15–20 min per node. node5700 first, node5701 after ssh works.
4. `analyze-gpu-fryer.py` → `out-gpu-fryer/summary.md`.

Risk: low — already validated end-to-end for fp32 on node5700.

### Phase B — NCCL single node
1. `setup-nccl-tests.sh`: inside the container,
   `make -j MPI=1 MPI_HOME=/usr/local/mpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr`
   with `BUILDDIR=/orcd/data/orcd/022/benchmarks/nccl-tests/build-ubuntu-cuda13`
   (new dir — existing builds untouched). Blackwell SM: add `NVCC_GENCODE`
   for `sm_100` if the default gencode list rejects it.
2. `run-nccl-1node.sh`: same CLI as the Rocky version (`[collectives] [ngpus]`,
   names → binaries, `all` supported), but the mpirun/binaries run *inside*
   `apptainer exec --nv`, sizes `-b 1M -e 16G -f 4`, one rank driving `-g $NGPUS`.
3. Runs: `sendrecv` first as a smoke test, then `all` on 8 GPUs.
   ~10–20 min for the full collective sweep.
4. `analyze-nccl-1node.py` → `out-nccl-1node/summary.md`.

Risk: medium — the build is new. Fallback if the container build fails: build
nccl-tests with `MPI=0` (single-process, multi-GPU is all that single-node
needs) — costs nothing for Phase B, but blocks the MPI path in Phase C.

### Phase C — NCCL two nodes (needs ssh)
Inter-node launch without Slurm and without a host MPI. Primary approach:

- Run the container's `mpirun` on node5700 with an ssh **launch-agent wrapper**
  so the remote `orted` starts inside the same container on node5701:
  `mpirun --mca plm_rsh_agent /path/to/ssh-apptainer.sh -H node5700:N,node5701:N ...`
  where `ssh-apptainer.sh` does `ssh "$@"`-with-`apptainer exec --nv <sif>` around
  the remote command.
- NCCL env from the Rocky 2-node script:
  `NCCL_IB_DISABLE=0`, `NCCL_NET_GDR_LEVEL=2`,
  `NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15`,
  `NCCL_SOCKET_IFNAME=^lo,docker`.
- Runs: `sendrecv` at 1 GPU/node (matches the Rocky baseline), then 8 GPUs/node,
  then `allreduce`. `analyze-nccl-2node.py` → summary.

Fallbacks, in order, if the MPI-over-ssh launch fights us:
1. Ask an admin for host OpenMPI (`apt install openmpi-bin libopenmpi-dev`) on both nodes.
2. A torchrun/c10d-based NCCL bandwidth harness (allreduce + sendrecv, busbw
   computed with the standard nccl-tests formula) — same rendezvous mechanism
   Megatron 2-node already needs, so it is guaranteed to work; numbers are
   comparable in GB/s but should be labelled as "torch harness", not nccl-tests.

Also worth repeating here (it is what limited the Rocky nodes, see
`../b200-nodes/notes.md`): a `ib_write_bw --use_cuda=0` GPUDirect check between
node5700 and node5701 to see whether these Ubuntu nodes have the same ~18.5 GB/s
NIC-reads-from-GPU cap, plus a host-memory baseline. `ib_write_bw` availability
on Ubuntu to be confirmed (perftest may need installing; `ibstat`/`ibv_devinfo`
are present).

### Phase D — Megatron-LM
1. Copy the two config scripts verbatim into this dir (so this dir is
   self-contained and `../b200-nodes` stays untouched).
2. `run-megatron-1node.sh [ngpus...]`: `apptainer exec --nv --contain --cleanenv
   --bind /orcd/data/orcd/022/benchmarks/megatron-lm --bind <this dir>` running
   `megatron-cfg-1node.sh $N`, scanning N = 1..8 sequentially (no Slurm, so it is
   a loop, not parallel jobs). ~7B GPT, mock data, 100 iters — roughly
   10–25 min per GPU count → **2–4 h for a full 1..8 scan per node**. Default to
   the full scan; `run-megatron-1node.sh 8` for the quick single point.
3. `run-megatron-2node.sh [gpus_per_node]`: `megatron-cfg-2node.sh` via torchrun
   `--rdzv-backend=c10d --rdzv-endpoint=node5700:1234`, launched on node5700 and
   over ssh on node5701 (same container, same bind mounts, `$MASTER` = node5700's
   inband IP). Also needs `TORCH_EXTENSIONS_DIR`/`XDG_CACHE_HOME` pointed at a
   shared, per-node-distinct path since `$HOME` is local.
4. `analyze-megatron.py` → `output-megatron/summary.md` + scaling plot.

Risk: medium — the container path is proven on the Rocky nodes; the new parts
are the manual two-node launch and the local-`$HOME` cache dirs.

### Phase E — Report
- Per-benchmark `summary.md` + PDF via `md-to-pdf.py`.
- One comparison table Ubuntu (5700/5701) vs Rocky 8 (5500/5502): gpu-fryer
  TFLOP/s per precision, NCCL busbw intra/inter node, Megatron TFLOP/s per GPU
  at 1/2/4/8 GPUs and 2-node.
- `notes.md`: driver 570 vs 590, kernel/OS differences, IOMMU/ACS state on these
  nodes, whether the inter-node GDR cap reproduces.

---

## 5. Ordering and rough time

| Phase | Depends on | Wall clock |
|---|---|---|
| A env + gpu-fryer, node5700 | — | ~30 min |
| B NCCL build + 1-node, node5700 | A | ~1 h (build + sweep) |
| D1 Megatron 1-node scan, node5700 | — (can run after A/B, GPUs exclusive) | 2–4 h |
| ssh to node5701 | **user action** | minutes |
| A/B/D repeat on node5701 | ssh | ~3–5 h |
| C NCCL 2-node + IB/GDR checks | ssh, B | ~1 h |
| D2 Megatron 2-node | ssh, D1 | ~1 h |
| E report | all | ~30 min |

Serialization rule (carried over from the Rocky notes): **never run gpu-fryer and
NCCL/Megatron on the same node at the same time**; these nodes have no scheduler
enforcing exclusivity, so each script checks for foreign GPU processes first.

---

## 6. Open questions for the user

1. **node5701 access** — is there a password for `shaohao@node5701`, or a jump
   host / admin who can install the public key? Everything two-node depends on it.
2. **Megatron scan depth** — full 1..8 GPU scan per node (2–4 h each, matches the
   Rocky-8 data set) or only 8 GPUs (~20 min) for a quick comparison?
   *Plan assumes the full scan, since the point is comparability.*
3. **gpu-fryer duration** — keep 300 s per precision (Rocky default)? Assumed yes.
4. **perftest / `ib_write_bw`** — if it is missing on these Ubuntu nodes, is
   installing it (needs root) an option, or should the GDR check be skipped?
