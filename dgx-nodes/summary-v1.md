# NCCL benchmark summary -- DGX H100 nodes (node180[0,1], node270[0,1], node280[0,1])

Generated 2026-08-20 16:14 by gen-summary.py (analyze-nccl.py over out-1node/, out-2node/).

Jobs: 1-node NCCL (NVLink/NVSwitch, 8 GPUs) on each of the 6 nodes; 2-node
NCCL (inter-node) on each bracketed pair (180[0,1], 270[0,1], 280[0,1]).
node170[0,1] excluded (not in mit_testing). Ubuntu 24.04 build
(build-utuntu-nvhpc-26.1, NVHPC 26.1 / CUDA 12.9 / NCCL 2.29.2).

### 1-node peak busbw (GB/s)

| Collective | node1800 | node1801 | node2700 | node2701 | node2800 | node2801 | mean |
|---|---|---|---|---|---|---|---|
| SendRecv | 367.3 | 367.5 | 367.1 | 368.0 | 367.1 | 365.6 | **367.1** |
| AllReduce | 482.0 | 481.7 | 481.9 | 481.7 | 482.3 | 481.6 | **481.9** |
| AllGather | 365.9 | 364.7 | 365.5 | 366.7 | 365.9 | 284.9 | **352.2** |
| ReduceScatter | 366.5 | 365.7 | 365.7 | 365.8 | 366.4 | 366.4 | **366.1** |
| Reduce | 369.7 | 369.6 | 369.7 | 369.7 | 369.7 | 369.7 | **369.7** |
| Broadcast | 366.0 | 365.3 | 366.4 | 366.8 | 366.3 | 366.2 | **366.2** |
| AllToAll | 348.0 | 347.9 | 348.0 | 347.9 | 348.0 | 348.1 | **348.0** |
| Gather | 377.1 | 377.1 | 377.1 | 377.1 | 377.1 | 377.1 | **377.1** |
| Scatter | 373.2 | 373.2 | 373.3 | 373.2 | 373.2 | 373.3 | **373.2** |
| Hypercube | 332.6 | 331.7 | 331.6 | 332.7 | 332.5 | 332.2 | **332.2** |


### 2-node peak busbw (GB/s)

| Collective | node[1800-1801] | node[2700-2701] | node[2800-2801] | mean |
|---|---|---|---|---|
| SendRecv | 2.0 | 2.0 | 2.0 | **2.0** |
| AllReduce | 2.0 | 2.0 | 2.0 | **2.0** |
| AllGather | 2.1 | 2.1 | 2.1 | **2.1** |
| ReduceScatter | 2.0 | 2.1 | 2.0 | **2.0** |
| Reduce | 2.9 | 2.9 | 2.9 | **2.9** |
| Broadcast | 2.9 | 2.9 | 2.9 | **2.9** |
| AllToAll | 2.0 | 2.1 | 2.0 | **2.0** |
| Gather | 2.9 | 2.9 | 2.9 | **2.9** |
| Scatter | 2.9 | 2.9 | 2.9 | **2.9** |
| Hypercube | 2.0 | 2.0 | 2.0 | **2.0** |


### 1-node SendRecv sweep

| Message size | busbw min | busbw mean | busbw max |
|---|---|---|---|
| 1M | 29.5 | 29.8 | 30.0 |
| 4M | 64.5 | 64.8 | 65.2 |
| 16M | 74.9 | 75.6 | 76.0 |
| 64M | 81.3 | 81.4 | 81.5 |
| 256M | 297.1 | 298.0 | 299.0 |
| 1G | 359.6 | 360.2 | 360.5 |
| 4G | 365.6 | 365.8 | 366.1 |
| 16G | 367.1 | 367.4 | 368.0 |


### 2-node SendRecv sweep

| Message size | busbw min | busbw mean | busbw max |
|---|---|---|---|
| 1M | 1.6 | 1.6 | 1.7 |
| 4M | 1.9 | 1.9 | 1.9 |
| 16M | 2.0 | 2.0 | 2.0 |
| 64M | 2.0 | 2.0 | 2.0 |
| 256M | 1.9 | 2.0 | 2.0 |
| 1G | 1.9 | 2.0 | 2.0 |
| 4G | 2.0 | 2.0 | 2.0 |


### 1-node AllReduce sweep

| Message size | busbw min | busbw mean | busbw max |
|---|---|---|---|
| 1M | 42.5 | 42.7 | 43.1 |
| 4M | 133.3 | 133.8 | 134.2 |
| 16M | 236.1 | 236.6 | 237.0 |
| 64M | 360.9 | 362.2 | 362.8 |
| 256M | 422.6 | 423.4 | 424.0 |
| 1G | 468.2 | 468.5 | 468.9 |
| 4G | 478.0 | 478.2 | 478.5 |
| 16G | 481.6 | 481.9 | 482.3 |


### 2-node AllReduce sweep

| Message size | busbw min | busbw mean | busbw max |
|---|---|---|---|
| 1M | 1.4 | 1.4 | 1.4 |
| 4M | 1.9 | 1.9 | 1.9 |
| 16M | 2.0 | 2.0 | 2.0 |
| 64M | 2.0 | 2.0 | 2.0 |
| 256M | 2.0 | 2.0 | 2.0 |
| 1G | 2.0 | 2.0 | 2.0 |
| 4G | 2.0 | 2.0 | 2.0 |



## Hardware ceilings for reference

| Link | Theoretical peak | What it bounds |
|---|---|---|
| NVLink / NVSwitch (H100, per GPU) | ~900 GB/s | 1-node collectives (intra-node) |
| InfiniBand NDR (per HCA, x1) | ~50 GB/s (400 Gb/s) | 2-node collectives, if verbs worked |
| Ethernet fallback (measured path here) | ~10-25 GB/s typical for a single TCP stream | 2-node collectives on these nodes (verbs unavailable, see caveat below) |

**Caveat:** on all three node pairs tested, NCCL could not open any of the
12 IB HCAs (`ibv_open_device` fails, though `ibv_devinfo -l`/sysfs shows the
ports ACTIVE) -- this looks like a MOFED userspace/kernel mismatch on these
nodes rather than a cabling/switch problem. NCCL transparently fell back to
its TCP socket transport over the 10.1.x Ethernet NIC, so the 2-node numbers
below reflect Ethernet-fallback bandwidth, not the IB fabric's real ceiling.
Once verbs are fixed, expect the 2-node numbers to rise sharply toward the
~50 GB/s NDR ceiling.

