# b300-nodes

## Introduction

Single-node benchmarks for the NVIDIA **B300** node **node5900-c1** (8 × B300
SXM6 AC, NVLink 5 / NVSwitch, NDR InfiniBand), Slurm partition **`mit_testing`**,
run as a direct comparison with the B200 nodes in `../b200-nodes`. The node runs
EL10 (the B200 nodes run Rocky 8), so `job-probe.sh` checks modules, containers,
GPUs and NICs first.

| Benchmark | What it measures | Same as `../b200-nodes`? |
|---|---|---|
| gpu-fryer | Sustained TFLOP/s per GPU, fp32 / bf16 / fp8, 300 s each | **Same** image and settings |
| NCCL 1-node | Collective bus bandwidth over NVLink, 8 GPUs, all collectives | **Same** test and build (`build-nvhpc-26.1`) |
| ib_write_bw 1-node | GPUDirect RDMA bandwidth between two rails of the node | **Same** test; NICs picked from `nvidia-smi topo -m` |
| Megatron-LM 1-node | GPT pre-training TFLOP/s per GPU, 1..8 GPUs | **Same** ~7B model as the B200 reference |
| Megatron-LM max sweep | Best TFLOP/s per GPU over a tuning grid, B300 **and** B200 | New here; each GPU type keeps its own best config |

2-node tests are not included (one B300 node). The max sweep grid, 8 GPUs, seq
4096, distributed optimizer: 5B (24 layers, hidden 4096) × micro 4/8/16 ×
recompute none/selective/full, and 13B (40 layers, hidden 5120) × micro 2/4/8 ×
recompute none/full, each in bf16 and fp8. Out-of-memory points at the top of
the grid are expected.

## Installation

Nothing to build. The scripts use prebuilt pieces from the other benchmark dirs:

- NCCL: `../nccl-tests/build-nvhpc-26.1` (module `nvhpc/26.1`)
- gpu-fryer: `../gpu-fryer/gpu-fryer_1.1.0.sif`
- Megatron-LM: `../megatron-lm/Megatron-LM` and `../megatron-lm/imag/pytorch_26.02-py3.sif`

Containers run with module `apptainer/1.4.2`, or the system apptainer /
singularity if the module does not load on EL10. gpu-fryer falls back to
`--unsquash` where FUSE is not allowed.

## Usage

### Automated, many runs — `submit-all.sh`

```bash
./submit-all.sh
```

Submits everything in one go and returns: probe, gpu-fryer (300 s), NCCL 1-node
(all collectives), ib_write_bw, Megatron-LM 1-node (1..8 GPUs), and the max sweep
on B300 (node5900-c1) and on B200 (any free node). The benchmark jobs have no
dependencies; they are `--exclusive` on the one B300 node, so Slurm runs them
back to back. After each group, a small CPU job (`mit_normal`) runs
`analyze-all.sh`, so the summaries fill in as results arrive; a final analysis
job runs after everything. The submission log is in `slurm-logs/submit-all-*.log`.

### Single runs — scripts in this dir

The `job-*.sh` scripts submit to Slurm; the `run-*.sh` scripts run directly
after `ssh` to the node (the job scripts call them).

```bash
sbatch job-probe.sh                                # environment check, ~1 min
./job-gpu-fryer.sh [nodes] [seconds]               # default node5900-c1, 300 s
./job-nccl-1node.sh [nodes] [collectives] [ngpus]  # default sendrecv, all GPUs
sbatch job-ibwrite-1node.sh                        # CLI_DEV= / SRV_DEV= to pick NICs
./job-megatron-1node.sh [node] [ngpus]             # no ngpus: scan 1..8, one job each
./job-megatron-max-sweep.sh b300|b200 [node|any]   # the full tuning grid, one job per point
./job-megatron-1node-max.sh [node] [ngpus|scan] [bf16|fp8]   # one ~5B max-throughput run
```

Collectives: `sendrecv allreduce allgather reducescatter reduce broadcast
alltoall gather scatter hypercube`, a comma list, or `all`.
`GPU_TYPE=b200 ./job-megatron-1node-max.sh any 8` runs the same thing on a B200 node.

| Run script | Called by |
|---|---|
| `run-gpu-fryer.sh`, `run-nccl-1node.sh` | `job-gpu-fryer.sh`, `job-nccl-1node.sh` |
| `run-1node-b300.sh` | `job-megatron-1node.sh` (~7B reference model) |
| `run-1node-max-tuned.sh` | `job-megatron-max-sweep.sh` (one grid point) |
| `run-1node-b300-max.sh` | `job-megatron-1node-max.sh` (~5B, full recompute) |

Slurm output goes to `slurm-logs/`.

## Analysis

```bash
./analyze-all.sh     # re-runs every analyzer on the results so far, then the comparison
```

| Analyzer | Reads | Writes |
|---|---|---|
| `analyze-gpu-fryer.py` | `out-gpu-fryer/` | `out-gpu-fryer/summary.md` + speed-up SVG |
| `analyze-nccl-1node.py` | `out-nccl-1node/` | `out-nccl-1node/summary.md` |
| `analyze-megatron.py` | `output-megatron/` | `output-megatron/summary.md` + scaling SVG |
| `compare-b200-b300.py` | the above, `out-ibwrite/`, `output-max-sweep/`, and `../b200-nodes/` results | `COMPARISON-b200-vs-b300.md` |

`COMPARISON-b200-vs-b300.md` is the main report: B300 / B200 ratios for each
benchmark (B200 = mean over the B200 nodes, newest run per node), the
ib_write_bw size sweep, and the best max-sweep point per GPU type and precision.
Missing data shows as `—`, so it can be regenerated at any time.

Figure of merit: TFLOP/s per GPU (gpu-fryer, Megatron-LM), bus bandwidth
`busbw` in GB/s (NCCL), Gb/s (ib_write_bw).
