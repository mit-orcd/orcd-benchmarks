# b200-kimi

## Introduction

A serving benchmark of the **Kimi-K3** model (2.78 T parameters, MXFP4, 1561 GB of
weights) on B200 nodes with vLLM. It repeats the MI355X serving run in
`../amd-benchmarks/amd-cloud/atom` and writes a report with the same structure,
plus a **B200 vs MI355X** comparison.

**Status: complete.** The full concurrency sweep (1 → 64) ran on 2 × 8 B200
(TP8 × PP2); peak **1,696 tok/s** at concurrency 64. The report is
`results/kimi-k3-base-b200.md`.

Why two nodes: one B200 node has 8 × 192 GB = 1538 GB of HBM, **23 GB less than
the weights alone**, before KV cache and workspace. The single-node run is kept
in the chain because its out-of-memory failure is itself the measurement. The
2-node layout is TP8 inside each node (NVLink) × PP2 across the pair (InfiniBand),
the vLLM recipe's `multi_node_tp_pp` strategy. MI355X (288 GB/GPU) fits the model
on one node, so the comparison is 16 B200 vs 8 MI355X and the report normalizes
per GPU and per node.

Measurement settings match the MI355X run: input 1024 / output 1024 tokens,
`max-num-seqs` 64, fp8 KV cache, concurrency 1 2 4 8 16 32 64. Two flags are
required for correct results, not tuning: `--gpu-memory-utilization 0.90`
(0.95 runs out of memory in warm-up) and `--no-enable-prefix-caching` (the
model's recurrent state cannot be rebuilt from the prefix cache).

## Installation

- **Image:** `./pull-image.sh` pulls `vllm/vllm-openai:kimi-k3` into
  `imag/vllm-openai_kimi-k3.sif` once (atomic, with a manifest; `--verify`,
  `--force`). Every other script checks the image and fails rather than pulling.
  It is a CUDA 13 build and needs driver r580+.
- **Model:** already on the cluster, read in place and never downloaded:
  `/orcd/compute/orcd/025/models/Kimi-K3` (another user's files, mounted read-only).
- **Ray:** not in the image; installed in `pylibs/` with `--no-deps`.
- **Config:** `common/env.sh` holds all paths, nodes, reservation and run
  settings. Caches go to this dir, never `$HOME`.

## Usage

### Automated, whole benchmark — `chain.sh`

```bash
./chain.sh                     # or: ../all-bench/run-all.sh b200-kimi
```

Submits one Slurm dependency chain and returns; it keeps running after logout:

```
pull    (CPU)      only if the image is missing
gate    (1 node)   image, GPU, InfiniBand, model registration
verify  (2 nodes)  Ray 16-GPU cluster and model inspection, no weight load
1node   (1 node)   TP8 x PP1, expected to run out of memory
base    (2 nodes)  TP8 x PP2: serve -> sweep 1..64 -> analysis   (after verify OK and 1node done)
summary (CPU)      results/RUN-SUMMARY.md, whatever the outcome
```

Job IDs are written to `logs/CHAIN.txt`.

**Before re-running:** the job headers still name reservation
`rres_joohye_2026-08-20_lj4j2ya3` on node5700-c1/5701-c1, which ended on
2026-08-27, so Slurm will refuse them. Use `./submit.sh --alt <stage>`, which
removes the reservation and moves the job to node5500-c1/5501-c1, or update the
headers and `common/env.sh`.

### Single stage — `submit.sh`

```bash
./submit.sh                    # list stages, nodes, reservation
./submit.sh probe              # driver / HBM / model visibility on each node, ~1 min
./submit.sh gate | 1node | base
./submit.sh --alt base         # same, on the fallback nodes without reservation
sbatch job-improve-b200.sh     # the four improvement options from the report, 2 nodes, ~2 h
```

| Script | Role |
|---|---|
| `job-gate-b200.sh`, `job-verify-2node.sh`, `job-kimi-1node.sh`, `job-kimi-base.sh` | the chain stages |
| `job-probe-drivers.sh`, `job-pull-image.sh`, `job-summary.sh`, `download-kimi.sh` | probe, image pull, summary, small test model only |
| `lib/kimi-run.sh` | shared body of the runs: preflight → serve → sweep → analyze |
| `run-vllm-server.sh`, `run-vllm-bench.sh`, `stop-vllm-server.sh` | start Ray + `vllm serve`, run `vllm bench serve`, stop only this job's server |

Safety built into the runs: nothing is `pkill`ed; a trap tears down the server on
walltime or `scancel`; the server counts as ready only with HTTP 200 **and**
loaded GPU memory; a sweep point with zero completed requests is fatal.

## Analysis

`job-kimi-base.sh` runs the analyzer itself, so the report exists when the job
ends:

| File | Content |
|---|---|
| `results/RUN-SUMMARY.md` | stage outcomes, hardware, key finding |
| `results/kimi-k3-base-b200.md` / `.csv` | the report: compute, memory, bottleneck, communication, B200 vs MI355X (§6) |
| `results/kimi-k3-improve-b200.md` / `.csv` | results of `job-improve-b200.sh` (`analyze-improve-b200.py`) |

To regenerate the report from a saved run without re-running:

```bash
source common/env.sh; module load apptainer/1.5.2
B=$(ls -dt logs/kimi_base_* | head -1)
apptainer exec $(apt_args) "$VLLM_SIF" $PY_C analyze-kimi-b200.py \
  --sweep "$B/sweep" --server-log "$B/server/vllm_server.log" \
  --model-config "$MKIMI/config.json" --run-dir "$B" \
  --tp 8 --pp 2 --isl 1024 --osl 1024 --max-num-seqs 64 --kv-dtype fp8 \
  --hbm-mib 183359 --weight-bytes 1560936091448 -o results
```

`analyze-kimi-b200.py` reads the B200 sweep JSON, the vLLM server log, the model
`config.json`, and the MI355X run's own logs in `../amd-benchmarks/amd-cloud/logs/atom/`,
and applies the same formulas to both systems. Its parameter count matches the
checkpoint exactly (2,722,740,830,208 MoE parameters); it warns if that changes.
`./selftest-analyze.sh` runs it on synthetic data plus the real MI355X baseline;
run it before changing the analyzer.

Background: `plan.md`, `kimi-k3-research.md`.
