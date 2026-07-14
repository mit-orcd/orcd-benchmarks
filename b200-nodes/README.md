# B200 node benchmarks

Benchmarks for the B200 test nodes **node5500** and **node5502** (8× NVIDIA B200
each, NVLink 5.0 / NVSwitch intra-node, NDR 400 Gb/s InfiniBand inter-node).
Slurm partition: **`mit_testing`**.

No installation needed — every script loads its own modules
(`nvhpc/26.1` for NCCL, `apptainer/1.4.2` for gpu-fryer and Megatron) and uses
prebuilt binaries / SIF images. Run everything as your normal user.

Two ways to run each benchmark:
- **Slurm** (recommended): `sbatch` or a `./job-*.sh` wrapper — no manual ssh.
- **Local**: `ssh node5500` first, then `./run-*.sh` on the node.

---

## 1. gpu-fryer (single-node GPU stress / TFLOP·s)

Runs fp32, bf16, fp8 stress (default 300 s each) on all 8 GPUs.

```bash
# Slurm: one job per node (default: both nodes, 300 s)
./job-gpu-fryer.sh                     # node5500 + node5502
./job-gpu-fryer.sh node5500 600        # node5500 only, 600 s/precision

# Local: ssh to the node first
ssh node5500
./run-gpu-fryer.sh [seconds]           # default 300
```

Output → `out-gpu-fryer/`.  Analyze:

```bash
./analyze-gpu-fryer.py                 # writes out-gpu-fryer/summary.md
./md-to-pdf.py out-gpu-fryer/summary.md
```

---

## 2. NCCL collective bandwidth (nccl-tests)

Collective names: `sendrecv allreduce allgather reducescatter reduce broadcast
alltoall gather scatter hypercube` — pass one, a comma list, or `all`.
Figure of merit is **busbw** (GB/s).

### Single node (intra-node NVLink)

```bash
# Slurm: one job per node
./job-nccl-1node.sh                              # both nodes, sendrecv, all GPUs
./job-nccl-1node.sh node5500 all                 # node5500, every collective
./job-nccl-1node.sh node5500,node5502 all 8      # both, all collectives, 8 GPUs

# Local
ssh node5500
./run-nccl-1node.sh [collectives] [ngpus]        # default: sendrecv, all GPUs
```

Output → `out-nccl-1node/`.  Analyze:

```bash
./analyze-nccl-1node.py                # writes out-nccl-1node/summary.md
./md-to-pdf.py out-nccl-1node/summary.md
```

### Two nodes (inter-node InfiniBand)

`job-nccl-2node.sh` is an sbatch script pinned to node5500 + node5502.

```bash
sbatch job-nccl-2node.sh [collectives] [gpus_per_node]   # default: sendrecv, 1
sbatch job-nccl-2node.sh all 8                            # all collectives, 8 GPUs/node
```

Output → `out-nccl-2node/`.  Analyze:

```bash
./analyze-nccl-2node.py                # writes out-nccl-2node/summary.md
./md-to-pdf.py out-nccl-2node/summary.md
```

---

## 3. Megatron-LM (GPT pretrain, throughput scan)

Scripts live in `/orcd/data/orcd/022/benchmarks/megatron-lm/Megatron-LM/`.
Each wrapper scans GPUs-per-node **1…8** by default (one job per count) and runs
the `pytorch_26.02` container. Global batch scales with GPU count (weak scaling).

```bash
cd /orcd/data/orcd/022/benchmarks/megatron-lm/Megatron-LM

# Single node
./job-megatron-1node.sh                  # node5500, scan 1..8
./job-megatron-1node.sh node5502 4       # node5502, just 4 GPUs

# Two nodes
./job-megatron-2node.sh                          # node5500,node5502, scan 1..8/node
./job-megatron-2node.sh node5500,node5502 8      # both nodes, 8 GPUs/node
```

Output → `output/` (throughput on the `--log-throughput` lines, every 20 iters).

---

## Notes

- Never run gpu-fryer and NCCL on the same node at the same time.
- Slurm stdout for the `./job-*.sh` wrappers goes to `slurm-logs/`; the parsed
  benchmark data goes to the `out-*/` dirs.
- `md-to-pdf.py` uses `reportlab` (already installed for this user).
- Inter-node GPU RDMA is currently capped (~18.5 GB/s) on these nodes — see
  `notes.md` for the GPUDirect / nvidia_peermem investigation.
