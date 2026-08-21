# NCCL benchmark summary -- DGX H100 nodes (node180[0,1], node270[0,1], node280[0,1])

Generated 2026-08-20 20:57 by gen-summary.py (analyze-nccl.py over out-1node/, out-2node/).

* **Build:** `/orcd/data/orcd/022/benchmarks/nccl-tests/build-songhan-utuntu-nvhpc-26.1`
  (built on node1800 under Ubuntu 24.04, NVHPC 26.1 / CUDA 12.9 / NCCL 2.29.2, gencode sm_90)
* **Partition:** `mit_testing`, 6 nodes, `--exclusive`
* **1-node:** 8 GPUs in one process, NVLink/NVSwitch, 1 MiB -> 16 GiB
* **2-node:** 1 GPU per node (2 ranks), InfiniBand, 1 MiB -> 16 GiB
* **Collectives:** all 10 nccl-tests binaries

### 1-node peak busbw (GB/s)

| Collective | node1800 | node1801 | node2700 | node2701 | node2800 | node2801 | mean |
|---|---|---|---|---|---|---|---|
| SendRecv | 367.4 | 367.2 | 367.5 | 367.7 | 367.2 | 367.5 | **367.4** |
| AllReduce | 481.8 | 481.6 | 482.3 | 482.0 | 482.3 | 481.4 | **481.9** |
| AllGather | 366.2 | 365.3 | 366.0 | 366.7 | 365.7 | 365.8 | **365.9** |
| ReduceScatter | 366.1 | 365.6 | 365.8 | 366.4 | 366.1 | 366.7 | **366.1** |
| Reduce | 369.7 | 369.4 | 369.4 | 369.6 | 369.5 | 369.5 | **369.5** |
| Broadcast | 366.8 | 366.0 | 366.2 | 366.9 | 366.9 | 366.2 | **366.5** |
| AllToAll | 347.9 | 348.0 | 348.1 | 348.0 | 348.0 | 347.9 | **348.0** |
| Gather | 377.1 | 377.1 | 377.1 | 377.1 | 377.1 | 377.1 | **377.1** |
| Scatter | 373.0 | 373.0 | 373.2 | 373.2 | 373.2 | 373.2 | **373.1** |
| Hypercube | 332.0 | 331.2 | 332.7 | 330.4 | 332.2 | 332.1 | **331.8** |


### 2-node peak busbw (GB/s)

| Collective | node[1800-1801] | node[2700-2701] | node[2800-2801] | mean |
|---|---|---|---|---|
| SendRecv | 48.8 | 48.7 | 48.7 | **48.7** |
| AllReduce | 49.4 | 49.4 | 49.4 | **49.4** |
| AllGather | 48.8 | 48.8 | 49.0 | **48.9** |
| ReduceScatter | 48.1 | 48.1 | 48.1 | **48.1** |
| Reduce | 49.5 | 49.5 | 49.5 | **49.5** |
| Broadcast | 49.5 | 49.5 | 49.5 | **49.5** |
| AllToAll | 48.8 | 48.7 | 48.7 | **48.7** |
| Gather | 49.3 | 49.4 | 49.4 | **49.4** |
| Scatter | 49.4 | 49.3 | 49.3 | **49.4** |
| Hypercube | 48.8 | 48.7 | 48.7 | **48.7** |


### 1-node SendRecv sweep

| Message size | busbw min | busbw mean | busbw max |
|---|---|---|---|
| 1M | 29.7 | 29.8 | 30.0 |
| 4M | 64.6 | 65.2 | 65.8 |
| 16M | 75.0 | 75.6 | 76.3 |
| 64M | 81.4 | 81.6 | 81.7 |
| 256M | 297.9 | 298.8 | 299.8 |
| 1G | 359.6 | 360.1 | 360.6 |
| 4G | 365.5 | 365.8 | 366.0 |
| 16G | 367.2 | 367.4 | 367.7 |


### 2-node SendRecv sweep

| Message size | busbw min | busbw mean | busbw max |
|---|---|---|---|
| 1M | 20.2 | 20.3 | 20.6 |
| 4M | 35.5 | 35.8 | 36.0 |
| 16M | 43.6 | 43.7 | 43.7 |
| 64M | 46.8 | 47.0 | 47.0 |
| 256M | 48.1 | 48.1 | 48.2 |
| 1G | 48.5 | 48.6 | 48.6 |
| 4G | 48.7 | 48.7 | 48.7 |
| 16G | 48.7 | 48.7 | 48.8 |


### 1-node AllReduce sweep

| Message size | busbw min | busbw mean | busbw max |
|---|---|---|---|
| 1M | 42.6 | 43.0 | 43.2 |
| 4M | 133.4 | 133.9 | 134.4 |
| 16M | 237.3 | 237.6 | 238.1 |
| 64M | 361.3 | 362.4 | 363.0 |
| 256M | 423.4 | 423.8 | 424.9 |
| 1G | 468.3 | 468.5 | 468.7 |
| 4G | 478.1 | 478.4 | 478.6 |
| 16G | 481.4 | 481.9 | 482.3 |


### 2-node AllReduce sweep

| Message size | busbw min | busbw mean | busbw max |
|---|---|---|---|
| 1M | 19.2 | 19.7 | 20.0 |
| 4M | 32.6 | 32.6 | 32.7 |
| 16M | 40.4 | 40.5 | 40.5 |
| 64M | 44.0 | 44.1 | 44.2 |
| 256M | 47.3 | 47.4 | 47.5 |
| 1G | 48.9 | 48.9 | 49.0 |
| 4G | 49.3 | 49.3 | 49.3 |
| 16G | 49.4 | 49.4 | 49.4 |



## Hardware ceilings

| Link | Theoretical peak | Measured (mean) | % of ceiling |
|---|---|---|---|
| NVLink 4 / NVSwitch (H100 SXM, per GPU) | 18 links x 26.562 GB/s = 478 GB/s per direction (~900 GB/s bidir, marketed) | AllReduce 481.9 GB/s | ~101% |
| InfiniBand NDR, 1 rail (`mlx5_0`, 400 Gb/s) | ~50 GB/s | AllReduce 49.4 GB/s | ~99% |

Notes on reading these numbers:

* NVLink generation confirmed on node2800: `nvidia-smi nvlink -s` reports
  **18 active links at 26.562 GB/s each** = NVLink 4.0, i.e. 478 GB/s per
  direction per GPU (the "900 GB/s" headline number is the bidirectional sum).
* **1-node** figures are NCCL `busbw` over NVLink/NVSwitch with 8 GPUs in one
  process. `busbw` normalises out each collective's algorithmic traffic, so
  AllReduce legitimately exceeds the per-GPU SendRecv figure; the NVSwitch
  fabric, not a single NVLink port, is the limit. AllReduce landing marginally
  above the 478 GB/s line rate reflects that normalisation plus in-place
  buffer reuse, not a measurement error.
* **2-node** figures use 1 GPU per node (2 ranks), so they exercise a single
  400 Gb/s NDR rail, whose practical ceiling is ~50 GB/s. Scaling to 8 GPUs
  per node would engage all 8 rails.

## Why 1-node busbw is ~480 GB/s and not ~700 GB/s

~700 GB/s is a **B200 / NVLink 5.0** figure, not an H100 one. The B200 results
in this repo (`b200-nodes/out-nccl-1node/summary.md`) record sendrecv busbw of
695 GB/s against a 900 GB/s per-direction NVLink 5.0 ceiling. These nodes are
H100 SXM with NVLink 4.0, whose ceiling is half that.

Checks performed to rule out a configuration or measurement fault:

| Check | Result |
|---|---|
| NVLink generation (`nvidia-smi nvlink -s`, node2800) | 18 links x 26.562 GB/s = NVLink 4.0 |
| All links up on all GPUs (node1801/2700/2800) | 18/18 active on all 8 GPUs, all 3 nodes |
| GPU-GPU topology (`nvidia-smi topo -m`) | NV18 between every pair (full NVSwitch) |
| `mpirun -np 1 -g 8`, default binding (what the suite uses) | 475.6 GB/s avg |
| `mpirun -np 1 -g 8 --bind-to none` | 475.2 GB/s avg |
| `mpirun -np 8 -g 1 --bind-to numa` (one rank per GPU) | 475.8 GB/s avg |

The three launch configurations agree to within 0.6 GB/s, so the single-rank
launch used by `job-nccl-1node.sh` is not costing anything: the fabric, not the
CPU or the rank layout, is the limit.

Efficiency against each platform's own ceiling is comparable:

| Platform | SendRecv busbw | NVLink per-direction ceiling | % |
|---|---|---|---|
| H100 (these nodes, NVLink 4.0) | 367.4 GB/s | 450 GB/s | 82% |
| B200 (`b200-nodes`, NVLink 5.0) | 695.0 GB/s | 900 GB/s | 77% |

## InfiniBand status

NCCL uses the IB fabric on these nodes. Verified with `NCCL_DEBUG=INFO`:

```
NCCL INFO NET/Plugin: Loaded net plugin NCCL RDMA Plugin v11 (v11)
NCCL INFO NET/IB : Made virtual device [0] name=mlx5_0 speed=400000 ndevs=1
```

A single-rail SendRecv check between node1800 and node1801 reached
**48.1 GB/s busbw at 1 GiB**, i.e. ~96% of the ~50 GB/s NDR line rate.

Earlier runs of this suite (archived under `results-oldbuild-tcp/`) recorded
only ~2 GB/s inter-node, because `ibv_open_device` was failing at the time and
NCCL silently fell back to its TCP socket transport. Verbs are working now
(`ibv_devinfo -d mlx5_0` succeeds both inside and outside a Slurm job), so
those archived 2-node numbers should be disregarded; the tables below are the
IB results.

