# SHARP on the MIT ORCD B200 nodes — why it is unavailable

**Verdict: a fabric-side software/configuration gap, not a hardware limitation and
not a node-side software problem.** The InfiniBand Aggregation Manager
(`sharp_am`) is not running on this subnet, so no SHARP job can be created.

Measured 2026-08-06 on node5500 / node5501 / node5502 (`mit_testing`), all three
node pairs, via `job-nccl-2node-sharp.sh`. Raw output and the generated tables
are in `out-nccl-2node-sharp/`.

---

## What was run

`job-nccl-2node-sharp.sh` runs NCCL `all_reduce` **twice inside one allocation**
so the comparison is on the same nodes, same NICs, same session:

- **Leg A — Ring:** `NCCL_COLLNET_ENABLE=0`
- **Leg B — SHARP:** `NCCL_COLLNET_ENABLE=1`,
  `NCCL_ALGO=CollNetChain,CollNetDirect`, `NCCL_PROTO=Simple`

Both legs share the same 8 NDR rails
(`NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15`),
`NCCL_NET_GDR_LEVEL=2`, 8 GPUs/node x 2 nodes, 1 MiB - 16 GiB.

## Result

| Node pair | Ring (GB/s) | SHARP (GB/s) | SHARP status |
|-----------|------------:|-------------:|--------------|
| node5500+node5501 | 239.1 | — | UNAVAILABLE (run aborted) |
| node5500+node5502 | 234.4 | — | UNAVAILABLE (run aborted) |
| node5501+node5502 | 233.5 | — | UNAVAILABLE (run aborted) |

The Ring legs ran cleanly and reproduce the standalone 2-node all_reduce numbers
(233-240 GB/s). The SHARP legs produced **no data at all** — they aborted rather
than silently falling back to Ring.

## Root cause — the evidence chain

Everything on the node side worked; the failure is at exactly one point:

```
NET/Plugin: Loaded collnet plugin SHARP (v11)          <- plugin loaded
16 coll channels, 16 collnet channels, ...             <- NCCL allocated CollNet channels
[SR]      error - no AM service record found(SA query)
[RDMA_SR] error - Error event recieved: RDMA_CM_EVENT_UNREACHABLE
[GENERAL] error - unable to connect to AM
[GENERAL] warn  - SHARPD_OP_CREATE_JOB failed with status: 52
ERROR No Aggregation Manager (sharp_am) detected in sharp_create_job.
ERROR sharp_create_job failed: No Aggregation Manager (sharp_am) detected(-52)
NCCL WARN NET/IB : SHARP coll init error: Cannot create SHARP job(-11)
```

NCCL honoured `NCCL_COLLNET_ENABLE=1`, loaded the SHARP plugin, and allocated 16
CollNet channels. The SHARP client library then queried the subnet administrator
for the Aggregation Manager service record and found none registered.

## Where the break is, layer by layer

| Layer | Status |
|---|---|
| NCCL SHARP/CollNet plugin on nodes | OK — present, loaded (v11) |
| SHARP client libs (HPC-X `libsharp_coll.so`) | OK — present, executed |
| NCCL config (`COLLNET_ENABLE`, `ALGO`, `PROTO`) | OK — accepted and applied |
| **`sharp_am` Aggregation Manager on the fabric** | **ABSENT — this is the failure** |
| Switch SHARP engines (hardware) | never reached |

`sharp_am` normally runs on the subnet manager / UFM host; it allocates SHARP
trees and reservations. Without it no SHARP job can be created regardless of what
the switches are capable of.

## Does the hardware support SHARP?

**Almost certainly yes.** The HCA on these nodes reports:

```
CA type:          MT4129        (ConnectX-7)
Firmware version: 28.49.1120
Rate:             400           (NDR)
Link layer:       InfiniBand
```

NDR is implemented only by **NVIDIA Quantum-2** switch silicon (MQM9700/9790
family), and Quantum-2 carries **SHARPv3** aggregation engines in the ASIC as a
standard part of the chip. A genuine 400 Gb/s NDR fabric therefore has
SHARP-capable switches by construction. This is consistent with the MIT
aicr-benchmarks reference cluster, which measured SHARP working (357 GB/s) on the
same hardware generation.

**Switch-side software is unverified.** SHARP must also be enabled in switch
firmware and configured by the subnet manager, which could not be checked from a
compute node — unprivileged subnet-management queries are blocked here:

```
smpquery: Can't open SMI UMAD port (Input/output error)
```

Reading the switch model, firmware, and SHARP configuration needs root or access
to the SM/UFM host.

## Questions for the InfiniBand admins

1. Is `sharp_am` running anywhere on this subnet (typically the UFM / SM host)?
   That is the immediate blocker.
2. Is SHARP enabled in the subnet manager configuration, with aggregation trees
   provisioned for the `mit_testing` nodes?
3. Is `sharpd` running on the compute nodes, and may they reserve a SHARP tree?

This may be deliberate — many NDR clusters run without SHARP enabled, and this is
a testing partition. The point is that it is a configuration decision someone can
reverse, not a hardware limit.

## Why it is worth enabling

Ring `all_reduce` measures **240 GB/s against a 400 GB/s fabric ceiling (~60%)**,
and that shortfall is precisely the two-pass penalty SHARP removes: Ring
all_reduce is reduce_scatter followed by all_gather, so the ring fills and drains
twice and pays the phase-transition latency. SHARP collapses this into a single
in-switch reduction.

It is the **largest remaining headroom in the inter-node results** — every other
ring collective already runs at 92-96% of the fabric ceiling — and it directly
gates multi-node DDP gradient sync. The reference cluster measured a **2.2x**
gain (163 -> 357 GB/s). Because our fabric is faster, the fixed phase-transition
latency is a *larger* fraction of our runtime (our all_reduce is 65% of our
all_gather, versus 78% on the reference), so the potential gain here is if
anything greater.

## Reproducing

```bash
sbatch -w node5500,node5501 job-nccl-2node-sharp.sh 8   # per node pair
./analyze-nccl-sharp.py                                 # -> out-nccl-2node-sharp/summary.md
```

`analyze-nccl-sharp.py` distinguishes three outcomes explicitly, so a null result
is never ambiguous: **engaged**, **FELL BACK to Ring** (NCCL does this *silently*
— identical numbers in both legs), and **UNAVAILABLE (run aborted)**, each
reported with the causing log line.

---

*See also `out-nccl-2node/summary.md` (per-collective results against this
cluster's hardware ceiling) and `notes-aicr.md` (comparison with the AICR paper).*

---
env set up of sharp on aicr

depends_on("nvhpc/26.3")
setenv("SHARP_HOME","/apps/aicr/packages/nvhpc/26.3/7jhdyji/Linux_x86_64/26.3/comm_libs/13.1/hpcx/hpcx-2.25.1/sha
rp")
setenv("NCCL_PLUGIN_HOME","/apps/aicr/packages/nvhpc/26.3/7jhdyji/Linux_x86_64/26.3/comm_libs/13.1/hpcx/hpcx-2.25
.1/nccl_rdma_sharp_plugin")
setenv("CUDA_HOME","/apps/aicr/packages/nvhpc/26.3/7jhdyji/Linux_x86_64/26.3/cuda")
setenv("NCCL_HOME","/apps/aicr/packages/nvhpc/26.3/7jhdyji/Linux_x86_64/26.3/comm_libs/nccl")
prepend_path("LD_LIBRARY_PATH","/apps/aicr/packages/nvhpc/26.3/7jhdyji/Linux_x86_64/26.3/cuda/lib64")
prepend_path("LD_LIBRARY_PATH","/apps/aicr/packages/nvhpc/26.3/7jhdyji/Linux_x86_64/26.3/comm_libs/nccl/lib")
prepend_path("LD_LIBRARY_PATH","/apps/aicr/packages/nvhpc/26.3/7jhdyji/Linux_x86_64/26.3/comm_libs/13.1/hpcx/hpcx
-2.25.1/nccl_rdma_sharp_plugin/lib")
prepend_path("LD_LIBRARY_PATH","/apps/aicr/packages/nvhpc/26.3/7jhdyji/Linux_x86_64/26.3/comm_libs/13.1/hpcx/hpcx
-2.25.1/sharp/lib")
prepend_path{"LD_PRELOAD","/lib64/libnuma.so.1",delim=":"}
setenv("NCCL_COLLNET_ENABLE","1")
setenv("SHARP_COLL_LOCK_ON_COMM_INIT","1")
setenv("NCCL_IB_HCA","^mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_12")
setenv("NCCL_ALGO","allreduce:collnetchain,collnetdirect")

