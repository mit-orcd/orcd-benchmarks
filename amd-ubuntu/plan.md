# Plan: benchmarks on the AMD Ubuntu nodes (node6100, node6101)

Last updated: 2026-10-01 19:40 UTC

## Goal

Run every benchmark from [`../amd-benchmarks/amd-cloud`](../amd-benchmarks/amd-cloud) on the two
MIT AMD GPU nodes. Each node has 8 × MI355X (gfx950), Ubuntu 24.04.5 and ROCm 7.2.4. For Kimi-K3,
also run AMD's latest vLLM recipe and report what it improves.

Rules for this work (from `notes` and `change`):

- All AMD software and downloaded files go in [`../amd-software`](../amd-software), with downloads
  run in the background so they survive logout.
- Benchmark scripts, logs and results go here in `amd-ubuntu/`.
- The nodes are ssh only (not in Slurm). Their `$HOME` is node-local and differs from the login
  nodes, so every path is an absolute `/orcd/data/...` path.
- Kimi-K3 weights: always `/scratch/Kimi-K3` (node-local copy the user makes from
  `/orcd/compute/orcd/025/models/Kimi-K3`). Never download them.
- Kimi-K3 arm 1: the **same images as amd-cloud** (apple-to-apple). Arm 2: **AMD's latest recipe**,
  https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi355x.
- Every benchmark gets two md files: amd-ubuntu only (as if amd-cloud did not exist), and
  amd-ubuntu vs amd-cloud.
- Do not use sudo for installs. Exception approved by the user: adding `shaohao` to the
  `render`/`video` groups (`../amd-software/setup/grant-gpu-access.sh`).

---

## Done

### Host inventory (2026-10-01)

| Item | Finding |
|---|---|
| GPUs | 8 × MI355X OAM, gfx950. amdgpu 6.19.14 (same driver as amd-cloud), 2 × EPYC 9575F, 2.2 TiB RAM |
| ROCm | 7.2.4 at `/opt/rocm`: hipcc, RCCL, hipBLASLt, RVS with `conf/MI355X/` |
| Containers | apptainer 1.5.4 (unprivileged, `/usr/local/x86_64`). ✅ works since ~19:00 UTC (AppArmor userns restriction lifted) |
| NICs | 8 × AMD Pollara `ionic_0..7`, 400 Gb/s RoCEv2, rails 0-3 on NUMA 0, 4-7 on NUMA 1, IPv4 GID index 1 |
| Frontend NIC | `mlx5_0` / `eno17695np0` |
| Disks | `/` 98 GB, `/dev/shm` 1.2 TB, `/scratch` local (Kimi-K3 copy in progress by the user) |
| GPU access | ✅ since 2026-10-01 20:49 UTC: `shaohao` added to `render`,`video` with sudo (user-approved) by `setup/grant-gpu-access.sh` |

### Software in `../amd-software` (see its README)

| Item | Status |
|---|---|
| Open MPI 5.0.8, rccl-tests (gfx950, MPI and non-MPI), venv | ✅ |
| `models/Qwen3-8B-FP8` 8.8 GB, `models/Llama-3.1-70B-Instruct-FP8` 68 GB | ✅ |
| `models/Kimi-K3-DSpark` 6.6 GB (recipe's speculative-decoding draft) | ✅ |
| SIFs, built on the nodes (`setup/build-sifs-on-nodes.sh`, layers in `/dev/shm`), every digest in `sif/*.manifest` | ✅ all 9 below |

| SIF | Digest | Same as amd-cloud? | Used by |
|---|---|---|---|
| `primus-v26.5` | 3040bf42 | yes (tag unchanged since 07-23) | Part C |
| `megatron-lm-v26.1` | 4fc8808b | yes (amd-cloud pull log) | megatron-ref |
| `atom-dev-nightly_202608111555` | b750c5fb | yes: amd-cloud `:latest` 08-14..08-20 | Part D tiers 1-2, Kimi base/maxseqs/512/1024/2048/profile |
| `atom-dev-nightly_202608191459` | f86e2bb2 | yes: amd-cloud `:latest` after 08-20 07:40 | Kimi isl4096, repeats arm A |
| `atom-dev-kimi-k3-mad` | a2d017e4 | yes (amd-cloud pull log) | Kimi mad, single_stream, repeats arm B, ep_matched |
| `vllm-openai-rocm-nightly-rocm100` | e76a953f | — (recipe) | Kimi vLLM recipe. vLLM 0.30.1rc1.dev493, torch 2.12 + ROCm 10.0 |
| `atom-dev-latest` (= nightly_202610011450), `atom-dev-kimi_k3_agentic_0924` | 2344d7b9, 15ae4ea9 | — | optional `run_kimi_all.sh new` set (newest ATOM images), not scheduled |

How the amd-cloud digests were found: amd-cloud's `pull.log` / `STATE.txt` image ids (its docker
uses the containerd store, so the image id is the registry digest), matched to Docker Hub tags.

### Scripts written here

| Path | What |
|---|---|
| `common/env.sh` | Shared config; logs/results per node, `RESULTS_SUBDIR` for per-image-set Kimi output |
| `common/bin/docker` | docker → apptainer shim. ✅ Tested on node6101 with real images: foreground run, detached instance with a persistent 40 GB fuse-overlayfs overlay, exec, writes, stop/rm. `image inspect --format {{.Id}}` returns the registry digest. `rocm/atom-dev:<tag>` maps to `sif/atom-dev-<tag>.sif` |
| `launch.sh` | Runs a script on a node with `setsid nohup` (survives logout) |
| `run_all_node.sh` | Parts A → B → C → M → D on one node. D = ATOM tiers 1-2 on amd-cloud's image |
| `auto_run.sh` | Watcher (below) |
| `work-rocmval/`, `rccl-tests/`, `primus/`, `megatron-ref/` | Parts A, B (+2-node), C, M, ported. Analyzer system strings relabelled (ROCm 7.2.4, Ubuntu 24.04.5, "amd-ubuntu") |
| `atom/run_kimi_all.sh <cloud\|new>` | All ATOM Kimi-K3 experiments with the image set's exact images (table in its header); output `results/<node>/kimi-<set>/` |
| `atom/run_kimi_recipe.sh [C list]` | **New:** AMD vLLM recipe, one server per concurrency with the recipe's per-point settings (DSpark K=7..3 up to C=14, DCP 8 + CPU KV offload above), points 1 4 8 10 12 14 44 48 70 + 64 128 256; amd-cloud workload (random 1024/1024). `analyze_kimi_recipe.py` → `results/<node>/kimi-recipe/kimi-k3-recipe.{md,csv}` |
| `net/run_ib_bw.sh` | Host RDMA bandwidth per ionic rail, NUMA-bound |
| `report.py` | **New** (replaces `compare_amd_cloud.py`): one md per benchmark in `results/ubuntu/` (amd-ubuntu only, never mentions amd-cloud; Kimi: recipe vs best ATOM at each concurrency) and `results/vs-amd-cloud/` (side by side; Kimi: apple-to-apple same images, and recipe vs amd-cloud's best ATOM). Re-run by `auto_run.sh` after each stage |

### Results so far

- **Network fabric** (`results/ubuntu/net.md`): every ionic rail 392 Gb/s (98% of 400) alone; all 8
  at once ≈391 GB/s per direction (98%) **with NUMA binding**, 52% without.
- Everything on GPUs: pending GPU access.

### Running unattended

`auto_run.sh` on node6100 (pid 654423, restarted 2026-10-01 20:00 UTC). Each node runs **one job
at a time**; the two nodes run their single-node jobs in parallel. Per node:

1. `host-<node>`: as soon as that node has GPU access → `run_all_node.sh "A B"`.
2. `2node`: once both nodes finished step 1 and both are idle → `rccl-tests/run_part_b_2node.sh`
   (holds both nodes).
3. `ctr-<node>`: → `run_all_node.sh "C M D"`.
4. `kimi-<node>`: once that node's `/scratch/Kimi-K3` copy is complete → node6100
   `atom/run_kimi_all.sh cloud`, node6101 `atom/run_kimi_recipe.sh`.

Analysis is automatic: every part and every Kimi experiment runs its analyzer and then
`report.py` (`results/ubuntu/`, `results/vs-amd-cloud/`) when it finishes, and the watcher runs
`report.py` again after each job. Polls every 5 min for up to 14 days.

Log `logs/auto/auto_run.log` (one status line per change). Finished jobs:
`logs/auto/<job>.done`; the running job per node: `logs/auto/running-<node>` (a restarted watcher
re-attaches to it). Stop with `touch logs/auto/STOP` (running jobs continue), restart with
`./launch.sh node6100 auto_run.sh`.

---

## To do

### ROCm 7.14 (added 2026-10-01 21:16 UTC)

ROCm 7.14.0 user space (the ROCm amd-cloud ran) is installed without root in
`../amd-software/rocm-7.14.0`, with RVS built at amd-cloud's commit and rccl-tests rebuilt against
it (`../amd-software/setup/install-rocm-7.14.sh`). `with_rocm.sh 7.14 <script>` runs any host
benchmark on it; results go to `results/<node>/rocm7.14/`. The watcher runs them at the **end of
the queue**: `host714-<node>` (RVS + RCCL on 7.14) after that node's Kimi job, then `2node714`
(2-node RCCL on 7.14) once both nodes are free.
`report.py` writes `results/ubuntu/rocm.md` (7.2.4 vs 7.14 on the same nodes) and adds the 7.14
columns to `results/vs-amd-cloud/rvs.md` and `rccl.md` (the apple-to-apple comparison with
amd-cloud).

### Blockers

None. Benchmarks started 2026-10-01 20:50 UTC.

### Ours, once unblocked

- [ ] Part A smoke: `rvs -g` lists 8 GPUs; gst fp4/fp6 work with the packaged RVS.
- [ ] 2-node RCCL: confirm `NET/IB` on `ionic_*`; compare busbw with the 391 GB/s RDMA ceiling.
- [ ] Part C: Primus v26.5 CLI matches the amd-cloud flags.
- [ ] megatron-ref: is `HSA_OVERRIDE_GFX_VERSION=9.4.2` still needed?
- [ ] vLLM recipe: first point (C=1) — check the ROCm 10.0 userspace runs on the host's amdgpu
      6.19.14 driver, DSpark acceptance length in `server.log`, and that the CPU KV offload
      (1.8 TB of 2.2 TiB RAM at the recipe's 224.875 GB/rank) does not OOM; lower `OFFLOAD_BYTES`
      if it does.
- [ ] Read `results/ubuntu/*.md` and `results/vs-amd-cloud/*.md` after each stage; add a short
      written conclusion to each phase (no PDF).
- [ ] Optional: sync to the GitHub clone via `../sync-and-push.sh` (amd-ubuntu not in its whitelist).

### Known risks

- Host ROCm 7.2.4 vs 7.14 on amd-cloud (RVS, rccl-tests use host ROCm).
- vLLM recipe image is ROCm 10.0 userspace on a ROCm 7.14-era driver.
- The recipe was tuned for an agentic lane; we run amd-cloud's 1024/1024 random workload so the
  numbers line up with ATOM, so recipe numbers will differ from AMD's.
- Points 64/128/256 are beyond the recipe's matrix (same rule as 44+).
- `/orcd/data` is NFS: keep heavy build/temp I/O on node-local `/tmp` or `/dev/shm`.

---

## Rerun 2026-10-04/05 (fixes for the failed jobs)

Every GPU job holds the node-local lock `/dev/shm/shaohao-gpu.lock` (2-node: node6100's, then node6101's).

| Job | Root cause | Fix | Status |
|---|---|---|---|
| 2-node RCCL (7.2.4 and 7.14) | `run-rccl-2node.sh` line 43: bare `$EXTRA_ENV` under `set -u` → abort in 1 s. Also NCCL INFO lines polluted the result rows | `${EXTRA_ENV:-}`; per-rank NUMA binding (`rccl-tests/numa-bind.sh`); INFO only in a separate probe run; smoke must show `NET/IB` on ionic. `run_part_b_2node_both.sh` runs both stacks under one hold of both locks | Smoke 7.2.4: all_reduce 379 GB/s busbw at 1 GiB (97% of 391 GB/s RDMA), NET/IB on all 8 ionic rails. Full runs done 2026-10-05 02:49-03:06 UTC, both rc=0 (see note below) |
| ATOM tiers 1-2 (Part D) | Analysis failed after the first run | Fixed analyzer/run (`atom/run_part_d_locked.sh`) | node6100 rerun analyze rc=0; node6101 rerunning |
| megatron-ref | AITER JIT-builds `module_rope_general_fwd` into the image's package dir, which is on apptainer's 64 MB `--writable-tmpfs` → ENOSPC → all ranks crash. Fused rope then SIGSEGVs on gfx950 (amd-cloud never completed a full-flag run either) | `AITER_JIT_DIR=amd-software/cache/aiter-jit/megatron-lm-v26.1`; `ROPE_FUSION=0` default (`--no-rope-fusion`, same FLOPs) | node6101: 572.5 TF/s/GPU (teardown SIGSEGV after 50 iters is harmless). node6100 queued (`rerun_finish.sh`) |
| Kimi kimi_2048 | Not a bug: `--max-num-seqs 2048` needs a 107 GB per-request cache vs a 58 GB KV budget. amd-cloud failed identically (logs/atom/kimi_2048_20260820_073255) | None; reported as "does not fit" | Final |
| Kimi recipe | Points 1-14 and 256 rerun; C=14 worker died during warm-up | C=14 retried (`rerun_finish.sh` on node6101) | 11/12 points; C=14 queued |

`rerun_finish.sh` on node6100 runs `report.py` once all of the above have finished.

### 2-node RCCL result (2026-10-05)

Both stacks finished: 16 ranks (PPN=8) over 8 × ionic RoCEv2 rails with GPU-direct RDMA
(GDRDMA; 7.14's RCCL shows `NET/IB-CAST`, the AINIC RoCEv2 path), GPU r ↔ ionic_r, no socket fallback.
Busbw at 16 GiB, ROCm 7.2.4 / 7.14 in GB/s:

- all_reduce 380.9 / 372.6
- all_gather 376.7 / 372.3
- reduce_scatter 378.5 / 371.6
- broadcast 369.9 / 364.6

That is 95-97% of the measured 391 GB/s RDMA ceiling, close to single-node XGMI (386-394).

- **alltoall** (4 GiB): 89.6 / 77.1. Half of every rank's data crosses the fabric, so busbw tops out
  near 2 × 48 × 15/16 ≈ 91 GB/s. 7.2.4 is at that limit; 7.14 is 14% lower.
- **sendrecv**: 28.6 / 32.1. Only the rank 7→8 and 15→0 pairs cross nodes, and both are cross-rail
  (GPU7 → ionic_7 → ionic_0).
- **PPN scaling** (all_reduce, 7.2.4 / 7.14):
  - PPN=1: 48.7 / 48.5
  - PPN=2: 48.7 / 48.9, still one rail's worth
  - PPN=4: 160 / 177
  - PPN=8: 381 / 373
  
  At PPN=2 the ring crosses the nodes on only one rail. That is an RCCL ring choice, not a fault.

Results: `results/node6100{,/rocm7.14}/rccl_2node.md`, `results/ubuntu/rccl_2node.md`, and
`results/ubuntu/rocm.md` (2-node section). amd-cloud was single-node, so there is no 2-node comparison
in `results/vs-amd-cloud/`.

### Primus llama2-7B rerun (2026-10-05)
First run (10-02) failed at every N on both nodes: apex `fused_weight_gradient_mlp_cuda` JIT build hit ENOSPC on the 64 MB `--writable-tmpfs`. Fixed by `primus/rerun_megatron_llama.sh` (persistent per-node overlay `CTR_OVERLAY=primus-v26.5-<node>`, node GPU lock). All 16 runs OK; 8 GPUs: node6100 1,170.5, node6101 1,134.8 compute TF/s/GPU vs amd-cloud 1,135.2. Also fixed `report.py` megatron-ref parser (was reading the static Dell 790.4 row) and added hand-written analysis sections (`results/analysis/`).
