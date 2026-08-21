# Plan — Kimi-K3 serving benchmark on B200 (Engaging)

Goal: reproduce, on Engaging's B200 nodes, the Kimi-K3 serving characterization that
`../amd-benchmarks/amd-cloud/atom` did on 8 × MI355X — so the two boxes can be compared
on the same 2.78 T-parameter frontier MoE.

Working dir: `/orcd/data/orcd/022/benchmarks/b200-kimi` (this dir). New files only.
Nothing in `../b200-nodes`, `../b200-ubuntu` or `../amd-benchmarks` is modified.

**Nothing here has been run. No job has been submitted.** This document is the plan only.

---

## 1. The constraint that shapes everything: it does not fit on one node

Measured from the HF repo, not assumed:

```
moonshotai/Kimi-K3   120 files, 96 safetensors shards
                     1,560,998,987,867 bytes = 1.561 TB = 1.420 TiB
                     2,779,931,837,184 params (U8 MXFP4 packed + 57.2 B BF16 + F32)
                     ungated, compressed-tensors "mxfp4-pack-quantized", group_size 32
```

| Box | HBM/GPU | GPUs/node | HBM/node | Holds the 1561 GB of weights? |
|---|---:|---:|---:|---|
| MI355X (the AMD run) | 288 GB | 8 | **2304 GB** | **yes** — 743 GB spare, ~57 GB/GPU of it used for KV |
| **B200 (here)** | **192 GB** (183,359 MiB) | **8** | **1538 GB** | **no — 23 GB short on weights alone** |
| B300 | 268 GB | 8 | 2144 GB | yes |

The 23 GB shortfall is the *optimistic* figure: it assumes 100% of HBM goes to weights.
Add the KV pool, activation workspace, NCCL buffers and the CUDA-graph pool — together
~15–20 GB per GPU — and a single B200 node is short by roughly **150–180 GB**.

The vLLM recipe says the same thing in its own words: *"The MXFP4 checkpoint (~1.68 TB of
weights) does not fit one 8xB200 node (1440 GB)."* (It quotes a pre-release weight estimate
and the conservative 180 GB/GPU convention; the measured checkpoint and nvidia-smi figures
above are tighter, and reach the same verdict.) The NVFP4 variant
(`RedHatAI/Kimi-K3-NVFP4`, `vram_minimum_gb: 1650`) does not rescue it either — there is no
single-node B200 configuration.

**Therefore the B200 run is a 2-node, 16-GPU run: TP8 within each node × PP2 across the
pair.** This is the recipe's `multi_node_tp_pp` strategy, listed `b200: verified`.

### What that does to the comparison

It is **not** an equal-hardware comparison and must never be printed as one:

| | MI355X | B200 |
|---|---|---|
| Nodes / GPUs | 1 / 8 | **2 / 16** |
| Parallelism | TP8, PP1, EP off | **TP8 × PP2**, EP off |
| Engine | ATOM (AMD's vLLM-like) | **vLLM** (ATOM is ROCm-only) |
| Critical-path fabric | XGMI only | XGMI-equivalent (NVLink) **+ IB between PP stages** |

Every headline number gets reported three ways — total tok/s, **tok/s per GPU**, and
**tok/s per node** — and the report leads with the capability statement (*MI355X serves this
model on one node; B200 needs two*) rather than with a raw throughput ratio.

---

## 2. What carries over from the ATOM work, and what does not

| ATOM artifact | Fate on B200 |
|---|---|
| `run_atom_server.sh` | **Rewritten.** ATOM is ROCm-only. Becomes a vLLM server launcher; keeps the refuse-don't-kill safety design and the readiness check (HTTP **and** real VRAM). |
| `run_atom_bench.sh` | **Ported almost verbatim.** `vllm bench serve` is the same code lineage as `atom.benchmarks.benchmark_serving` and writes the same JSON keys. Keeps the "0 completed ⇒ fatal" check. |
| `analyze_atom.py` | **Reused with edits.** It already carries a `Kimi-K3` entry in `MODEL_INFO`. Needs: B200 hardware facts, PP-awareness, and per-GPU/per-node normalization columns. |
| `run_part_d.sh` | **Replaced** by a Slurm driver (`job-kimi-b200.sh`). |
| `download_models.sh` | **Ported** — same `hf download` logic, different destination and run under Slurm rather than on the login node. |
| Docker | **Gone.** Engaging has no user Docker. Everything is Apptainer. |
| `rocm-smi` guards | **Gone.** Slurm gives us exclusive nodes; guards become `nvidia-smi` + allocation checks. |

---

## 3. Environment — what is known, what must be verified

Known from `sinfo` / `scontrol` and `../b200-nodes/notes.md`:

| Item | Value |
|---|---|
| Partition | `mit_testing` |
| B200 nodes | **5**: `node550[0-2]-c1` (Rocky/EL10, idle) and `node570[0-1]-c1` (Ubuntu, **reserved for us**) |
| Reservation | `rres_joohye_2026-08-20_lj4j2ya3`, ACTIVE 2026-08-20 → 2026-08-27, acct `rres_acc_joohye_2026-08-20_lj4j2ya3` |
| GRES | `--gpus-per-node=b200:8` |
| CPU / RAM | 224 cores, 2000 GB per node |
| OS | EL10, kernel 6.12 (reinstalled since the July notes) |
| Fabric | 8 rails/node @ 400 Gb/s NDR (`mlx5_4,7,8,9,10,13,14,15`); `iommu=off` |
| GPUDirect | **at line rate** — `ib_write_bw` GPU→GPU 395.5 Gb/s (2026-08-12 remeasure) |
| NCCL 2-node | sendrecv 48.4 GB/s per pair at 8 GPUs/node |
| Containers | `module load apptainer/1.5.2` (1.4.2, 1.1.9 also present) |
| Group storage | `/orcd/data/orcd/022`, 77 TB free — the only place with room for 1.42 TiB |
| `$HOME` | 441 / 500 GB used — **too full for weights, images, or apptainer cache** |
| `$SCRATCH` | 1 TB quota — **too small for the checkpoint** |

Must be verified in step 5.1 before anything expensive is committed:

1. **Driver ≥ r580.** The `vllm/vllm-openai:kimi-k3` image is a **CUDA 13 (cu130) build
   only** — there is no cu129 tag. July's notes recorded 590.48.01 on these nodes, which
   would be fine, but the nodes have been reinstalled since. If the driver is r575, the
   whole plan stops here and the fallback in §8 applies.
2. **`nvidia-smi` memory** — confirm 183 GB (the §1 arithmetic assumes it).
3. **Apptainer `--nv` + IB** inside the container: `nccl` sees the `mlx5` devices.
4. **`mlx5dv_reg_dmabuf_mr`** works. If engine init dies with errno 524, set
   `NCCL_DMABUF_ENABLE=0` and require `nvidia_peermem` (both documented in the recipe).
5. Whether two B200 nodes can be held **simultaneously** for ~4 h on `mit_testing`.

---

## 4. Engine and image

**vLLM**, image `vllm/vllm-openai:kimi-k3` (15.4 GB compressed on Docker Hub, pushed
2026-07-27). Mainline vLLM registers `KimiK3ForConditionalGeneration` and ships an
NVIDIA-specific K3 path (`vllm.models.kimi_k3.nvidia.*`); min version 0.27.1.

Pulled once to a `.sif` under this dir:

```
b200-kimi/imag/vllm-openai_kimi-k3.sif       # ~20-25 GB squashfs
```

with `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` forced onto `/orcd/data/orcd/022` — the
default is `~/.apptainer/cache`, and `$HOME` has ~59 GB left, which the pull would eat.
This mirrors how `../b200-nodes` stores `megatron-lm/imag/pytorch_26.02-py3.sif`.

### Launch configuration (from the vLLM recipe, not invented)

Blackwell baseline + the `multi_node_tp_pp` strategy overrides:

| Flag / env | Value | Why (recipe's reason) |
|---|---|---|
| `-tp 8 -pp 2` | 16 GPUs | the only layout that fits on B200 |
| `--gpu-memory-utilization` | **0.90**, not 0.95 | the flashinfer TRTLLM MXFP4 MoE kernel takes a ~1.6 GiB workspace *outside* vLLM's pool on the first forward; 0.95 OOMs a 180 GB B200 on warmup |
| `--max-num-batched-tokens` | 8192 | caps prefill chunks so one long request cannot OOM a pipeline stage |
| `--load-format` | `fastsafetensors` | 1.42 TiB off shared NFS |
| `--kv-cache-dtype` | `fp8` | matches the MI355X run |
| `--attention-config` | `{"use_prefill_query_quantization":true,"mla_prefill_backend":"flashinfer"}` | required companion to fp8 KV |
| `--no-enable-flashinfer-autotune` | — | Blackwell baseline |
| `--trust-remote-code` | — | custom `modeling_kimi_k3.py` |
| `NCCL_CUMEM_ENABLE=1` | — | strategy override |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 3600 | 1.42 TiB load off NFS |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 1800 | — |
| `VLLM_ALLREDUCE_USE_FLASHINFER=1` | — | Blackwell baseline |

Deliberately **not** set:

- `--speculative-config` (DSpark) — the recipe gates spec decoding **off** `multi_node_tp_pp`
  (does not compose with PP yet, vllm#50098). Turning it on would silently change what is
  being measured.
- `--enable-expert-parallel` — EP is off on MI355X too; keeping it off is what makes the two
  runs comparable. (On MI355X, EP additionally *fails to load* with MXFP4.)
- `--decode-context-parallel-size` — recipe-scoped to the single-node TP8 profile.
- `--moe-backend` — recipe says leave unset on B200 so vLLM auto-selects.
- The Blackwell baseline's `--max-model-len 1048576` and `--enable-prefix-caching` are
  **overridden** for the comparison arm (see §6).

---

## 5. Phases

### 5.1 Gate — cheap, no weights, no 2-node hold (~20 min, 1 node)

One short job on one B200 node: driver version, `nvidia-smi` memory, `torch.cuda`
device count, a BF16 4096³ matmul for correctness + a throughput floor, IB device
visibility inside the container. Same shape as `run_part_d.sh`'s stage 0, which gated on
*behaviour* rather than on an arch-list string. **Abort the plan here if the driver is
< r580.**

### 5.2 Two-node NCCL sanity (~15 min, 2 nodes)

Reuse `../b200-nodes/job-nccl-2node.sh` output if a recent run exists; otherwise one short
`all_reduce_perf` + `sendrecv_perf` across the pair, inside the vLLM image, to confirm the
container's NCCL takes the IB path (`GPU Direct RDMA (DMABUF) enabled` in `NCCL_DEBUG=INFO`)
and reaches the 48 GB/s already measured on this fabric. **PP2 puts IB in the per-token
critical path** — if this is degraded, the serving numbers are meaningless and we would be
re-measuring a fabric bug.

### 5.3 Weights (~1–3 h, no GPU)

`hf download moonshotai/Kimi-K3 --local-dir .../b200-kimi/models/Kimi-K3`, run as a
**batch job on a CPU partition**, not on the login node — 1.56 TB of network + NFS I/O is
exactly the kind of thing the shared-system guard rails say to keep off the login node.
Idempotent and resumable (`hf download` skips completed files), so an interruption is a
re-submit, not a restart. Verified afterwards by shard count (96) and total bytes
(1,560,998,987,867).

Storage decision: `/orcd/data/orcd/022/benchmarks/b200-kimi/models/` — 77 TB free.
**Not** `$HOME` (59 GB left), **not** `$SCRATCH` (1 TB quota).

### 5.4 Smoke tier (~30 min, 1 node)

A small ungated model (`Qwen/Qwen3-8B-FP8`, 8.9 GB — the same tier-1 model the AMD run used)
served TP1 through the full script stack: launcher → readiness check → sweep → analyzer.
This proves the Apptainer + Slurm + vLLM + benchmark-client path end to end for ~9 GB and
one GPU, so that the first time we spend a 2-node × 4-hour allocation it is not on a typo.

### 5.5 Headline: Kimi-K3, 2 nodes, TP8 × PP2 (~3–4 h, 2 nodes)

Comparison arm first (§6), then, if allocation time remains, the B200-native arm.

### 5.6 Analysis and report

`analyze-kimi-b200.py` → `results/kimi-k3-b200.{md,csv}`, plus a cross-box section against
`../amd-benchmarks/amd-cloud/results/kimi-k3.md`.

---

## 6. Run matrix

**Arm A — apples-to-apples with MI355X** (the point of the exercise). Match the ATOM run's
workload and admission settings exactly, so only hardware + engine + parallelism differ:

| Setting | Value | Matches MI355X? |
|---|---|---|
| ISL / OSL | 1024 / 1024, `--ignore-eos`, `--random-range-ratio 0.8` | yes |
| `--max-model-len` | 16384 | yes |
| `--max-num-seqs` | 64 | yes |
| KV dtype | fp8 | yes |
| Prefix caching | **off** | yes — and required: KDA recurrent state is per-request and cannot be rebuilt from the paged MLA cache, so prefix reuse would be *silently incorrect*, not merely slow |
| EP | off | yes |
| Concurrency | 1 2 4 8 16 32 64 | yes — capped at `max-num-seqs`; past it we would measure the request queue, not the engine |
| Parallelism | TP8 × PP2 | **no — unavoidable** (§1) |

**Arm B — B200-native** (what the box does when tuned for itself), only if time allows:
recipe defaults — prefix caching on, `--max-num-seqs` raised (256), longer `max-model-len`.
Reported separately and never mixed into Arm A's table.

The MI355X §3 finding says the bottleneck there is HBM bandwidth at ~29 %, and that
`max-num-seqs` is the lever (each expert sees only ~1.7 tokens at batch 64 — a GEMV that
cannot saturate HBM). Arm B is where that lever gets pulled on B200; B200's 8 TB/s HBM3e
against MI355X's 8 TB/s makes the comparison at higher batch the interesting one.

---

## 7. Files — written and staged

```
b200-kimi/
  plan.md                      # this file
  README.md                    # script documentation + the driver warning
  notes                        # reservation details (from you)
  chain.sh                     # submit the whole benchmark as one Slurm dependency chain
  submit.sh                    # submit any single stage
  job-pull-image.sh            # sbatch wrapper for the image pull
  job-summary.sh               # -> results/RUN-SUMMARY.md, afterany the main run
  common/env.sh                # paths, run config, NCCL env, apptainer args, slurm_args
  lib/kimi-run.sh              # shared body: preflight -> serve -> sweep -> analyze
  pull-image.sh                # apptainer pull -> imag/ ONCE: flock + atomic rename + manifest
  .gitignore                   # models/ and imag/ never get committed
  job-probe-drivers.sh         # driver/HBM verdict per node, ~1 min, 1 GPU
  download-kimi.sh             # sbatch: smoke model only -- Kimi-K3 is pre-staged
  job-gate-b200.sh             # 1 node: image/GPU/IB/matmul/arch gate
  job-kimi-1node.sh            # 1 node: TP8 attempt (expected OOM = the measurement)
  job-kimi-base.sh             # 2 nodes: TP8 x PP2 + automatic analysis
  run-vllm-server.sh           # ray cluster + vllm serve; HTTP *and* VRAM readiness
  run-vllm-bench.sh            # concurrency sweep via `vllm bench serve`
  stop-vllm-server.sh          # teardown, this job's processes only
  analyze-kimi-b200.py         # -> results/kimi-k3-base-b200.{md,csv}
  selftest-analyze.sh          # analyzer test on synthetic data + real MI355X baseline
  (Kimi-K3 is NOT here -- read in place from /orcd/compute/orcd/025/models/Kimi-K3)
  imag/*.sif                   # to be pulled
  logs/  out/  results/
```

All scripts are syntax-checked. `selftest-analyze.sh` passes, producing a ~400-line
report from synthetic B200 data and the real MI355X baseline.

### Multi-node launch mechanism

vLLM's default multiprocessing executor is single-node, so TP8 x PP2 across two nodes
needs Ray. `run-vllm-server.sh` stands it up inside the allocation:

```
sbatch -N 2 --ntasks-per-node=1 --exclusive --gpus-per-node=b200:8
  srun --overlap (node 0): apptainer -> ray start --head
  srun --overlap (node 1): apptainer -> ray start --address=<head>
  poll `ray.cluster_resources()['GPU']` until it reports 16
  srun --overlap (node 0): vllm serve ... -tp 8 -pp 2 --distributed-executor-backend ray
```

Decisions made rather than deferred:

- **Ray temp dir on node-local `/dev/shm`**, not the shared tree: a Unix socket path is
  capped at 107 bytes and Ray appends ~40 characters of session and socket names, and
  putting the object store on NFS would route Ray control traffic through the filesystem.
- **No `--contain`** on the apptainer invocations: `/dev/infiniband` and the mlx5 uverbs
  devices must stay visible, or NCCL silently falls back to TCP and the PP stage boundary
  runs at socket speed.
- **Poll for the GPU count**, never sleep a fixed interval — a worker that races the head
  dies immediately and vLLM then plans for the wrong world size.
- **`--overlap` on every `srun`** so the head, the worker and the server share one
  allocation.
- **`EXIT`/`TERM`/`INT` trap** so a walltime kill still tears down Ray rather than leaving
  16 GPUs held by orphans.
- **NCCL rails pinned** to `mlx5_4,7,8,9,10,13,14,15`, copied from
  `../b200-nodes/job-nccl-2node.sh` — the configuration proven to connect at 16 ranks.
  Leaving NIC selection to NCCL works at 1 GPU/node and fails at 8.

---

## 8. Risks, and what happens if each one bites

| Risk | Likelihood | Response |
|---|---|---|
| **Driver < r580 on the RESERVED nodes** (image is cu130-only) | **high — this is now the top risk.** `../b200-ubuntu/ubuntu-nccl.md` recorded node570[0-1]-c1 on **570.211.01 / CUDA 12.9** on 2026-08-12, and those are exactly the nodes `notes` reserves | Move to the Rocky nodes `node550[0-2]-c1` (idle, no reservation, last seen on 590.48.01): `./submit.sh --alt base`. Or ask ORCD to upgrade the reserved pair — but that same survey warns an upgrade *"would invalidate every result collected so far"* on those nodes. `./submit.sh probe` decides this in minutes and is the first step. |
| Cannot hold **2 B200 nodes** at once for ~4 h | low — five B200 nodes exist and two are reserved for us through 2026-08-27 | Fall back to the Rocky trio. There is no single-node fallback (§1). |
| Kimi-K3 support not actually in the pinned image | low — the tag exists and mainline registers the arch | Try `vllm/vllm-openai:nightly`; the recipe warns the K3 wheels are cu130-only. |
| Model load off NFS is very slow (1.42 TiB) | **high** | `fastsafetensors` is already in the flags; budget 20–40 min for first load; keep the server **up** across the whole sweep (the scripts already do — one server, many concurrency points). If NFS read is the wall, stage weights to node-local NVMe first — but `TmpDisk=0` in `scontrol`, so node-local scratch must be confirmed to exist before relying on it. |
| `mlx5dv_reg_dmabuf_mr` errno 524 at engine init | low — this kernel is 6.12 | `NCCL_DMABUF_ENABLE=0` + confirm `nvidia_peermem` loaded (recipe documents both). |
| PP2 over IB dominates and the numbers look bad | medium | That *is* a result, and an honest one: it is the measurable cost of the model not fitting in one node. Report the PP bubble and inter-stage transfer separately rather than burying it. |
| Walltime kill mid-sweep | medium | Per-concurrency JSON is written as each point completes, so a partial sweep is still analyzable; `STATE.txt` records where it stopped. Sweep ascending in concurrency so the cheap points land first. |
| Disk: image + weights ≈ 1.45 TiB | low | Group share has 77 TB free. Check again before 5.3; never write to `$HOME` or `$SCRATCH`. |

---

## 9. Shared-system conduct

- Every GPU-touching step is an **sbatch job**, never a login-node process.
- The 1.56 TB download is a batch job under `nice`, not a login-node `curl` loop.
- Jobs request exactly what they need (`-N 1` for the gate and the smoke tier; `-N 2` only
  for 5.2 and 5.5) and hold the two B200 nodes for the shortest window that gets the data.
- Walltimes are set to the estimate plus margin, not to the partition maximum, so the nodes
  return to the pool promptly.
- No `pkill`-style cleanup: teardown acts only on the Ray cluster and container instances
  this allocation created, keeping the ATOM scripts' refuse-don't-kill discipline.

---

## 10. Open questions — for you

1. **Driver on the reserved nodes.** `notes` reserves **node5700-c1 / node5701-c1**
   (verified: `rres_joohye_2026-08-20_lj4j2ya3`, ACTIVE through 2026-08-27). Those are
   the **Ubuntu** nodes, and `../b200-ubuntu/ubuntu-nccl.md` recorded them on driver
   **570.211.01 / CUDA 12.9** on 2026-08-12 — while the `vllm:kimi-k3` image is cu130
   and needs **r580+**. If that is still current the image will not run there.
   `./submit.sh probe` answers it in minutes. If it confirms r570, the choices are: move
   to the Rocky nodes (`./submit.sh --alt base` — `node550[0-2]-c1`, idle), ask ORCD to
   upgrade the pair (that survey warns an upgrade invalidates the Ubuntu results already
   collected), or build vLLM against cu129 (days).
2. **Walltime inside the reservation** — is a 2-node × ~4 h hold fine, or does it need
   coordinating with the reservation's owner?
3. **Node-local scratch** — `scontrol` reports `TmpDisk=0`, so 1.42 TiB is re-read from
   NFS on every server start. Budget 20–40 min per load.
4. **Arm B** (B200-native tuning) — in scope, or is arm A the deliverable?

## 11. Order of work

```
./submit.sh probe     (1 GPU/node, ~1 min)   <- settles the r580 question first
./pull-image.sh       (login node, ~20 min)
./submit.sh gate      (1 node, ~20 min)      <- image + GPU + IB + arch registration
./submit.sh download  (CPU job, ~5 min)      <- smoke model only; Kimi-K3 is pre-staged
./submit.sh 1node     (1 node, <2 h)         <- TP8 attempt; OOM here IS the result
./submit.sh base      (2 nodes, 3-4 h)       -> results/kimi-k3-base-b200.{md,csv}
```

Each stage gates the next, which is why there is no "run everything" target.
