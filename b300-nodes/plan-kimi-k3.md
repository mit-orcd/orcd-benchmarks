# Plan: Kimi-K3 inference on one B300 node (8 × B300)

Written 2026-10-14. **Status: plan only, nothing submitted yet.**

Goal: run the Kimi-K3 tests of `../amd-ubuntu/results/mi355x-vs-b200.md` §7 on one B300 node (8 × B300, 288 GB each), tune vLLM for the best performance at each load, and add an **MI355X vs B300** comparison to that file. Job scripts, outputs and analysis md files go in `b300-nodes/kimi/`; only `mi355x-vs-b200.md` is edited in `amd-ubuntu`.

## 0. Blocker: the B300 node does not have 8 full GPUs right now

`scontrol show node node5900-c1` (2026-10-14):

- `Gres=gpu:b300:4(S:0),gpu:3g.135gb:8(S:1)`: only **4 full B300 GPUs**; the other 4 are split into MIG instances (8 × `3g.135gb`).
- Partitions: `sched_system_all, pi_rakhlin_rl, ou_orcd_everything`. It is **no longer in `mit_testing`**, where our earlier B300 jobs ran.
- Another user's job is running there (`pi_rakhlin_rl`, 1 × B300, 2-day limit).

Kimi-K3 needs **8 full GPUs**: 1,561 GB of weights; 4 × ≈270 GB usable = ≈1,080 GB is not enough. The MIG instances cannot be used, because vLLM tensor parallelism needs whole GPUs that talk to each other over NVLink.

**What is needed first:** MIG turned off on GPUs 4–7, and the whole node free for about one day: a reservation, or the node back in `mit_testing`. This changes a node another group is using, so it is a decision for the admins and the node's owners. I will not change the node's settings myself. The scripts can be written and checked before that. The monthly maintenance (2026-10-20 00:00–21:00) also blocks the node.

## 1. Does the model fit? Yes, on one node

| | per GPU | 8 GPUs |
|---|---:|---:|
| B300 HBM (usable, check with `nvidia-smi` in the gate job) | ≈270 GB | ≈2,160 GB |
| Kimi-K3 weights (MXFP4, TP8) | ≈195 GB | 1,561 GB |
| Left for KV cache + workspace at `--gpu-memory-utilization 0.90` | ≈45–50 GB | ≈380 GB |

- **TP8 on one node, no pipeline parallelism**, as on MI355X. That makes this comparison fairer than 16 × B200 vs 8 × MI355X: **8 GPUs vs 8 GPUs, same users, no per-GPU normalisation needed.**
- The KV cache is about half of MI355X's (≈90 GB/GPU there). This matters at 256–512 users and for 128K/1K. The gate job reports the real KV-cache token capacity, and the sweep caps `max-num-seqs` to what fits.
- Without pipeline parallelism, **DSpark speculative decoding is possible** (it was blocked on B200 by PP).

## 2. Software

- **Weights:** `/orcd/compute/orcd/025/models/Kimi-K3`, read only (bound `:ro`), never downloaded (same checkpoint as B200 and MI355X).
- **Engine:** vLLM. ATOM is ROCm only.
  - First choice: the existing B200 image `../b200-kimi/imag/vllm-openai_kimi-k3.sif` (`vllm/vllm-openai:kimi-k3`, pulled 2026-08-20).
  - B300 is compute capability 10.3 (sm_103). If the gate job shows that image has no sm_103 / FP4 MoE kernels, pull `vllm/vllm-openai:v0.29.0` (the same vLLM version as MI355X recipe 2) into `b300-nodes/kimi/imag/`. Apptainer cache and tmp go under `b300-nodes/kimi/.apptainer`, never `$HOME` or `/tmp`.
- **Node:** EL10. `module load apptainer` may fail, so fall back to the system `apptainer` (as `job-gpu-fryer.sh` does).
- **Reuse:** the server/bench/stop scripts and the analyzer from `../b200-kimi`, copied and edited here (single node: no Ray, PP = 1). `../b200-kimi` is not changed.

## 3. Tests (the same as §7 of mi355x-vs-b200.md)

All tests: Kimi-K3 MXFP4, fp8 KV cache, `vllm bench serve` random dataset, prompts = 10 × users, new prompts (a fresh `--seed`) for every run so the prefix cache cannot inflate results (see the prefix-cache test in `../amd-ubuntu/results/ubuntu/kimi-prefix-cache-test.md`). Metrics: output tok/s, total tok/s, TPOT (time per output token) and TTFT (time to first token) median and p99.

| # | Test | Loads (users) | Purpose |
|---|---|---|---|
| T0 | **Gate** (≈30 min) | — | 8 full GPUs visible, sm_103 kernels, weights readable, server starts, KV-cache capacity, 1 short request |
| T1 | **Base, ISL/OSL 1K/1K**: the B200 recipe on one node (TP8, `max-num-seqs` 64 / 256 / 512 by load, `--max-num-batched-tokens` 8192, prefix caching off) | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 | same settings as B200, so B300 vs B200 is the hardware |
| T2 | **Tuning, 1K/1K**: one server setting per job, the loads it is meant for (below) | as below | best B300 number at each load |
| T3 | **Long context, ISL/OSL 128K/1K**: base and best T2 setting | 1, 2, 4, 8, 16, 32, 64 (as many as the KV cache holds) | compare with §7.3 (AMD's tested numbers and our three MI355X recipes) |

T2 settings, chosen from what helped on B200 (`../b200-kimi/results/kimi-k3-improve-b200.md`) and on MI355X:

| Setting | Loads | Why |
|---|---|---|
| DSpark speculative decoding (as in MI355X vLLM recipe 1), 3 and 5 draft tokens | 1–16 | more tokens per weight read at low load; possible now without PP |
| Expert parallelism (`--enable-expert-parallel`) | 64–512 | gave 1.13x on B200 |
| `max-num-seqs` 128 / 256 / 512 and `--max-num-batched-tokens` 8192 / 16384 | 64–512 | gave 1.73x on B200 at 512 users |
| MoE kernel backend: FlashInfer TRT-LLM-gen vs CUTLASS (MXFP4 on Blackwell) | 8 and 128 | which FP4 MoE kernel is faster on sm_103 |
| `--gpu-memory-utilization` 0.90 / 0.92, `--async-scheduling` | best setting only | more KV cache, less CPU gap between steps |

The best result at each load (highest output tok/s, with TPOT reported next to it) is "B300 best". Settings that fail (out of memory, not supported) are recorded as failed, not retried without limit.

## 4. Jobs

All in `b300-nodes/kimi/`, single node, `--exclusive`, 8 × B300. Partition, account and reservation are set in one place (`kimi/env.sh`) once §0 is resolved.

| Script | Time limit | Runs |
|---|---:|---|
| `job-kimi-b300-gate.sh` | 1 h | T0 |
| `job-kimi-b300-base.sh` | 6 h | T1 |
| `job-kimi-b300-tune.sh <setting>` (one job per setting, ≈8 jobs) | 4 h each | T2 |
| `job-kimi-b300-long.sh <base\|best>` | 6 h each | T3 |
| `job-kimi-b300-post.sh` (CPU partition) | 30 min | analysis, md updates, sync and push |

- **Order:** the gate job first. Everything else depends on the gate passing (`afterok`), because a wrong image or missing GPUs would waste every later job. After that, base and tune jobs have no dependency on each other: Slurm runs them one at a time on the one node. The "best" long-context job waits for the tune jobs, since it needs their result. The post job runs `afterany` on all.
- Every job: start the server, check that it is ready (HTTP 200 and GPU memory loaded), run the loads, stop only its own server (trap on exit and on `scancel`), never `pkill`. Each load writes its JSON at once, so a job that hits its time limit keeps the loads it finished.
- Estimated node time: ≈20–30 h in total (the server loads 1.5 TB from NFS in ≈10–20 min per job).
- Submitted in the background with `sbatch`. Everything keeps running after logout, and job IDs are written to `kimi/logs/CHAIN.txt`.

## 5. Analysis and md files

- `kimi/analyze-kimi-b300.py`: reads the sweep JSONs and writes:
  - `b300-nodes/kimi/results/kimi-k3-b300.md` (+ `.csv`): base and tuned tables, the winning setting at each load, memory/KV capacity, long context. Each section starts with what is compared and which is faster.
  - **B300 vs B200** (16 × B200 from `../b200-kimi/results/`) in that file and one section in `COMPARISON-b200-vs-b300.md`.
- `../amd-ubuntu/results/mi355x-vs-b200.md`:
  - new **§7.5 "MI355X vs B300 (8 GPUs vs 8 GPUs)"**: throughput, TPOT and TTFT at the same users, MI355X best recipe vs B300 best, plus base vs base; 128K/1K vs §7.3;
  - one row in the Summary table; a short note in "Apple-to-apple?" (same GPU count, same model, different engine: ATOM/vLLM-ROCm vs vLLM-CUDA).
  - Rules for that file: spell out TPOT and TTFT at first mention, say what is compared and which is faster at the start of each section, MI355X numbers are ROCm 7.14 runs, keep the comment text short.
- The summary is written as each phase finishes (gate → base → tune → long), not all at the end.
- `../sync-and-push.sh` already includes `b300-nodes` and `amd-ubuntu`. The post job runs it, with the commit message ending in the Co-Authored-By line.

## 6. Risks

| Risk | What happens |
|---|---|
| Node not restored to 8 full GPUs (§0) | nothing can run; the scripts wait, unsubmitted |
| Image lacks sm_103 / FP4 MoE kernels | gate fails, then pull `v0.29.0` and run the gate again |
| Out of memory at 0.90 or at 512 users | lower to 0.88 / cap `max-num-seqs`; recorded in the md |
| DSpark not supported in the CUDA image | marked "not run" in T2 |
| Other users' jobs on the node | `--exclusive` waits for them; nothing is killed |
