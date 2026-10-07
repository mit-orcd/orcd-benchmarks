# AMD MI355X Ubuntu nodes (node6100, node6101)

## Introduction

The benchmark suite from [`../amd-benchmarks/amd-cloud`](../amd-benchmarks/amd-cloud), ported
to the two MIT AMD GPU nodes. Each node has 8 × MI355X (gfx950), Ubuntu 24.04 and ROCm 7.2.4.
The nodes are reachable by ssh only (not in Slurm), and their home directory differs from the
login nodes. The amd-cloud results (one 8 × MI355X node, ROCm 7.14) are the comparison baseline.

| Part | Dir | What it measures | Runs on |
|---|---|---|---|
| A | `work-rocmval/` | RVS: gst TFLOPS sweep (9 precisions × 1-8 GPUs), HBM/PCIe/XGMI/power health, levels 1-5 | host (packaged RVS) |
| B | `rccl-tests/` | RCCL collectives over XGMI, N=2..8, plus a 5-config knob sweep | host |
| B-2node | `rccl-tests/` | **New:** RCCL across both nodes (16 ranks) over 8 × 400G ionic RoCE | host + Open MPI |
| Net | `net/` | **New:** host RDMA bandwidth per ionic rail, alone and all 8 at once (`ib_write_bw`) | host |
| C | `primus/` | Primus GEMM/attention/RCCL microbenches and Megatron-LM llama2-7B BF16 | `rocm/primus:v26.5` |
| M | `megatron-ref/` | Megatron-LM GPT-15.6B, the Dell/B200-comparable config | `rocm/megatron-lm:v26.1` |
| D | `atom/` | ATOM LLM serving: Qwen3-8B TP1, Llama-3.1-70B TP8 | `rocm/atom-dev` (amd-cloud's digest) |
| K | `atom/` | Kimi-K3 TP8: all amd-cloud ATOM experiments on **amd-cloud's exact images**, plus **AMD's latest vLLM recipe** | `rocm/atom-dev` (3 digests), `vllm/vllm-openai-rocm:nightly-rocm100` |

Differences from amd-cloud:

- **No docker.** Containers run under apptainer from SIFs. `common/bin/docker` is a shim that
  implements the docker subset the amd-cloud scripts use (`run [-d]`, `exec`, `ps`, `logs`,
  `stop`, `rm`, `image inspect`) on apptainer instances, so the ported scripts are unchanged
  apart from paths. Detached servers get a persistent writable overlay for AITER JIT builds.
- **RVS** is the packaged `/opt/rocm/bin/rvs` on host ROCm 7.2.4. RVS and rccl-tests also run a
  second time on **ROCm 7.14** (amd-cloud's version, installed without root in
  `../amd-software/rocm-7.14.0`): `./launch.sh <node> with_rocm.sh 7.14 run_all_node.sh "A B"`.
  Results in `results/<node>/rocm7.14/`, compared in `results/ubuntu/rocm.md`.
- **All software and weights** are in [`../amd-software`](../amd-software). Logs and results are
  **per node**: `logs/<node>/`, `results/<node>/`.
- **2-node RCCL** is new, since amd-cloud had a single node.

## Installation

Everything is installed by `../amd-software/setup/` (see [its README](../amd-software/README.md)):
Open MPI, the rccl-tests builds, the venv, the models and the four SIFs. Nothing here needs
building.

**Blocking prerequisites, waiting on the admin:**

1. `shaohao` must be added to the `render`/`video` groups on both nodes. Until then every GPU
   test fails at the `assert_gpu_access` guard.
2. ~~Apptainer~~: works since 2026-10-01 19:00 UTC.
3. Kimi-K3 weights must be copied to node-local `/scratch/Kimi-K3` on each node (the user is doing
   this). Scripts always use that path.

## Usage

Everything runs **on an AMD node**. `launch.sh` starts a script there detached, so it survives
logout. The launcher log goes to `logs/launch/`.

```bash
cd /orcd/data/orcd/022/benchmarks/amd-ubuntu
./launch.sh node6100 run_all_node.sh          # A -> B -> C -> M -> D on node6100
./launch.sh node6101 run_all_node.sh          # same on node6101, concurrently
./launch.sh node6100 rccl-tests/run_part_b_2node.sh   # after both are idle: 2-node RCCL
./launch.sh node6100 atom/run_kimi_all.sh cloud    # Kimi-K3, all ATOM experiments, amd-cloud's images
./launch.sh node6101 atom/run_kimi_recipe.sh        # Kimi-K3, AMD's vLLM recipe (per-concurrency servers)
```

`run_all_node.sh "A B"` runs a subset. Parts whose prerequisite is missing (SIF, apptainer,
rccl-tests build) are skipped, not failed, so A and B can run before the container work is
unblocked.

Single parts, each a self-contained driver with a GPU guard, stages and analysis:

| Script | Stages |
|---|---|
| `work-rocmval/run_part_a.sh` | smoke → gst sweep N=1..8 × 9 precisions → health modules → `analyze_rvs.py` |
| `rccl-tests/run_part_b.sh` | smoke → 10 collectives × N=2..8 → config sweep → analysis + plot |
| `rccl-tests/run_part_b_2node.sh` | smoke → 6 collectives × 16 ranks, 8 B–16 GB → PPN 1/2/4 scaling → `analyze_rccl_2node.py` |
| `primus/run_part_c.sh` | gemm gate → microbench sweep → Megatron llama2-7B N=1..8 → `generate_report.py` |
| `megatron-ref/run_megatron_ref.sh [N]` | GPT-15.6B, Dell's exact flags, `HSA_OVERRIDE_GFX_VERSION=9.4.2` as in amd-cloud |
| `atom/run_part_d.sh [tier1\|tier2\|tier3\|all]` | functional gate → per tier: server → concurrency sweep 1..256 → stop → `analyze_atom.py` |
| `atom/run_kimi_all.sh <cloud\|new> [exps]` | base (tier 3), maxseqs, mad, 512/1024/2048, isl4096, single_stream, repeats, ep_matched, profile; each on the image amd-cloud used for it (`cloud`) or the newest ATOM images (`new`) |
| `atom/run_kimi_recipe.sh [C list]` | AMD vLLM recipe (recipes.vllm.ai, 2026-09-25): one server per concurrency with the recipe's settings, C = 1 4 8 10 12 14 44 48 70 64 128 256, random 1024/1024 |

Lower-level scripts (`run_tflops.sh`, `run_rvs_health.sh`, `run-rccl-all.sh`,
`run-rccl-configs.sh`, `run-rccl-2node.sh`, `run_atom_server.sh` + `run_atom_bench.sh` +
`stop_atom_server.sh`, and so on) take the same env knobs as on amd-cloud and are documented in
their headers.

2-node RCCL settings (`run-rccl-2node.sh`): `NCCL_IB_HCA=ionic_0..7`, `NCCL_IB_GID_INDEX=1`
(RoCEv2 IPv4), bootstrap on `eno17695np0`, and one MPI rank per GPU via
`$MPI_HOME/bin/mpirun`. Override them with `IB_HCA`, `GID`, `NET_IF` and `EXTRA_ENV`.

## Analysis

Each driver ends by running its analysis into `results/<node>/`:

| Part | Output | Compare against |
|---|---|---|
| A | `rvs_tflops.{md,csv}` | `../amd-benchmarks/amd-cloud/results/rvs_tflops.md` |
| B | `rccl.{md,csv}`, `rccl_busbw.png` | `amd-cloud/results/rccl.md` |
| B-2node | `rccl_2node.{md,csv}`, `rccl_2node_busbw.png` | line rate: 8 × 400 Gb/s = 400 GB/s per node |
| Net | `ib_bw.md` | line rate, and the 2-node RCCL busbw |
| C | `PRIMUS_REPORT.md` | `amd-cloud/results/PRIMUS_REPORT.md` |
| M | updates `PRIMUS_REPORT.md` §1.2 | Dell MI355X 790.4, B200 986.0 TFLOP/s/GPU |
| D | `atom.{md,csv}` | `amd-cloud/results/atom.md` |
| K | `kimi-cloud/kimi-k3-*.{md,csv}`, `kimi-recipe/kimi-k3-recipe.{md,csv}` | amd-cloud `kimi-k3-*.md` |

Raw logs (`STATE.txt` per driver run, per-test logs) are in `logs/<node>/<part>/`.

`report.py` turns those into two md files per benchmark, re-run by `auto_run.sh` after every stage:

- [`results/ubuntu/`](results/ubuntu/README.md): amd-ubuntu only, as if no other system existed
  (absolute numbers, node6100 vs node6101, scaling; Kimi-K3: what the vLLM recipe improves over
  ATOM at each concurrency).
- [`results/vs-amd-cloud/`](results/vs-amd-cloud/README.md): side by side with amd-cloud (Kimi-K3:
  same image digests apple-to-apple, and the recipe vs amd-cloud's best ATOM result).
  Start with [`results/vs-amd-cloud/SUMMARY.md`](results/vs-amd-cloud/SUMMARY.md): one section per
  benchmark on what is the same and what differs. It is hand-written in
  `results/analysis/vs-amd-cloud/SUMMARY.md` and copied by `report.py`, like the "Analysis"
  sections of the other reports (`results/analysis/<set>/<report>.md`).

The per-node analyzer reports in `results/<node>/` are the ported amd-cloud analyzers; their system
strings are relabelled but some fixed narrative still describes the amd-cloud campaign, so read the
numbers there and the conclusions in `results/ubuntu/`.

Things to expect relative to amd-cloud:

- ROCm 7.2.4 on the host vs 7.14 there. RVS and rccl-tests use host ROCm, so small
  differences can come from the stack rather than the hardware.
- The container images carry their own ROCm, so Parts C, M and D are less sensitive to this.
- Both nodes should agree with each other within noise. A node-to-node gap in Part A or B
  points to hardware or config, not software.

## Status

- All software, models and container images are in place; image digests match amd-cloud for
  Primus, Megatron-LM and all three ATOM images. The docker shim is tested with the real images.
- **Network fabric is done**: ≈391 GB/s per direction over 8 rails (98% of line rate, NUMA-bound).
- **No GPU benchmark has run yet**: GPU access (render group) is the only blocker; Kimi-K3 also
  waits for the `/scratch/Kimi-K3` copy on each node.
- `auto_run.sh` is running on node6100 and starts everything as blockers clear; see
  [`plan.md`](plan.md). Log `logs/auto/auto_run.log`, stop with `touch logs/auto/STOP`.
