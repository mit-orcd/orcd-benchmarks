# Notes — B200 node testing (node5500, node5502)

2026-07-13

## nvidia_peermem / GPUDirect RDMA check

Question: is `nvidia_peermem` set up correctly on node5500 and node5502?

**Answer: yes, the module is set up correctly — but GDR PCIe P2P performance is
degraded on both nodes, which is what limits the 2-node NCCL result.**

### Module status (both nodes)

- `nvidia_peermem` loaded on both nodes, driver version 590.48.01, correctly
  built for each node's kernel.
- MOFED 26.04 (`OFED-internal-26.04-0.8.6`) with the peer-memory API
  (`ib_register_peer_memory_client` present in kallsyms) on both.
- `/sys/kernel/mm/memory_peers/` does not exist on either node (not conclusive
  by itself on this MOFED version).
- Functional proof: `ib_write_bw --use_cuda=0` between the nodes works — GPU
  memory registers with the NIC and transfers run. GPUDirect RDMA is
  operational.
- Note: NCCL selects the newer **DMABUF** registration path anyway
  ("GPU Direct RDMA (DMABUF) enabled" in NCCL_DEBUG output), so
  `nvidia_peermem` is not on NCCL's critical path here. Installing it on
  node5500 did not change the NCCL sendrecv result (12.7 GB/s before and
  after).

### perftest measurements (mlx5_4, NDR 400 Gb/s, 64 MiB messages, RDMA write)

| Test | Bandwidth | Verdict |
|---|---:|---|
| host mem -> host mem | **379.5 Gb/s** (47.4 GB/s) | link is perfect (near NDR line rate) |
| NIC **reads from GPU** (node5500) | **147.6 Gb/s** (18.5 GB/s) | capped |
| NIC **reads from GPU** (node5502) | **147.7 Gb/s** (18.5 GB/s) | capped, identical |
| NIC **writes into GPU** (node5500) | **286.6 Gb/s** (35.8 GB/s) | partly degraded |

The GPU-read cap is symmetric across both nodes, so it is a common platform
configuration issue, not a single bad node.

> **Superseded.** These figures, and the two sections that follow them, were taken
> with the earlier system configuration (`iommu=pt intel_iommu=on`, EL8 / 4.18 on
> node5500). They no longer reproduce — see **2026-08-12** at the end of this file.

### Why this explains the NCCL 2-node result

NCCL sendrecv is bidirectional: each GPU simultaneously reads (TX) and writes
(RX) over its NIC. With the P2P read path capped at ~18.5 GB/s unidirectional,
~12.7 GB/s per direction under bidirectional contention is expected. The
reference cluster (b0029+b0030, results_b200.md) reaches 26.6 GB/s because its
P2P path runs at full speed.

### Suspects / next steps (platform level, for sysadmins)

1. **IOMMU** — both nodes boot `iommu=pt intel_iommu=on` (540 active IOMMU
   groups). The reference cluster explicitly required **`iommu=off`** to
   restore P2P throughput ("Fixed Issues" in results_b200.md). Top suspect,
   since the cap is identical on both nodes.
2. **ACS** — node5502 has `pci=disable_acs_redir=pci:1000:c030` on its kernel
   cmdline but **node5500 does not**; yet both show the same cap, so ACS
   redirect alone is not the whole story. Other ACS bits on the Broadcom
   PCIe switches may still be set — needs root `lspci -vvv` to verify
   (non-root cannot read ACSCtl).
3. **Node inconsistency** — node5500 is EL8 / kernel 4.18.0-553, node5502 is
   EL10 / kernel 6.12.0-211. Worth aligning.

### Network fabric (for reference)

Both nodes: 8 active GPU compute rails at **400 Gb/s 4X NDR**
(mlx5_4, 7, 8, 9, 10, 13, 14, 15); mlx5_0-3 are 100 Gb/s 2X HDR (HDR100)
secondary NICs not on the NCCL data path; mlx5_5, 6, 11, 12 down/unused.
The NCCL run bound to mlx5_4 on both nodes — the low bandwidth is not a
network-rate (HDR vs NDR) issue.

### How to reproduce the perftest checks

```bash
# server (node5500)
ib_write_bw -d mlx5_4 --use_cuda=0 --report_gbits -s $((64*1024*1024)) -n 200
# client (node5502); drop --use_cuda for host-memory baseline
ib_write_bw -d mlx5_4 --use_cuda=0 --report_gbits -s $((64*1024*1024)) -n 200 node5500
```

---

# 2026-08-12 — GPUDirect re-measured on both clusters

This retires the 2026-07-13 GPU-read cap recorded above. The NCCL results these figures
support are in `../b200-ubuntu/out-nccl-2node/summary.md` § 2.

**Bulk GPUDirect path, measured directly on both clusters** (`ib_write_bw`, mlx5_4, 64 MiB RDMA writes, 200 iters). The Rocky 8 column was measured on **2026-08-12** on the current system configuration (Slurm job 20306762 on node5501, `job-ibwrite-1node.sh`); the last column is what the same test returned on 2026-07-13, when the GPU-read path was capped:

| Test | Ubuntu 5700<->5701 | Rocky 8 node5501 | ratio | Rocky 8, 2026-07-13 |
|------|-------------------:|-----------------:|------:|--------------------:|
| host mem -> host mem | 380.8 Gb/s | 379.7 Gb/s | 1.00x | 379.5 Gb/s |
| **NIC reads from GPU** | **395.4 Gb/s** | **395.5 Gb/s** | **1.00x** | 147.6 Gb/s |
| **NIC writes into GPU** | 380.8 Gb/s | 380.5 Gb/s | 1.00x | 286.6 Gb/s |
| GPU -> GPU | not measured | 395.5 Gb/s | — | not measured |

**The bulk path is now a tie, and the old GPUDirect deficit is gone.** Rocky 8 reads from GPU memory at **395.5 Gb/s** — NDR line rate, within 0.1 Gb/s of Ubuntu — where 2026-07-13 measured 147.6; writes into GPU recovered from 286.6 to 380.5. `GPU -> GPU`, the pattern NCCL actually uses, is also at line rate. Two things about that cluster changed in between and either could account for it: it now boots **`iommu=off`** (0 IOMMU groups, where 2026-07-13 recorded `iommu=pt intel_iommu=on` with 540), and node5500/node5501 have been **reinstalled to EL10 / kernel 6.12** from the earlier EL8 / 4.18. The platform-level suspects previously raised for the cap — ACS redirect, PCIe topology, Relaxed Ordering, Max Payload Size — no longer have anything to explain.

**The NCCL data never supported the cap, and now agrees with perftest.** `sendrecv` at 8 GPUs/node on 2026-08-06 already reached 48.4 GB/s per pair on Rocky 8 — roughly 387 Gb/s out of GPU memory across one rail — which is arithmetically impossible if the NIC could only read from GPU at 147.6 Gb/s, and NCCL reported "GPU Direct RDMA (DMABUF) enabled" there, so it was not bypassing the GPU path. All three Rocky pairs agreed (47.7-49.7 GB/s vs Ubuntu's 48.8).

**Caveat on the re-measurement.** It is a **single-node, cross-rail** test — client on mlx5_4 + GPU0, server on mlx5_7 + GPU1, both PXB-adjacent pairs, traffic leaving the node and returning through the IB switch. The client's GPU -> PCIe switch -> NIC read path is identical to the inter-node case, which is what makes the "NIC reads from GPU" row comparable; the host-to-host row is a sanity check rather than a fabric measurement. A true 2-node reproduction (`job-ibwrite-2node.sh`) is queued and blocked on node availability — node5500 is held by another user's reservation until 2026-08-14 and node5502 has been `DOWN+NOT_RESPONDING` since 2026-08-10, leaving only one of the three Rocky nodes free.

## Small-message sweep (Rocky 8, node5501, job 20306762)

`ib_write_bw -a -n 1000 --use_cuda=0`. Message rate holds at **~5.3 Mpps** from 2 B to 4 KiB — about **190 ns per operation** — and bandwidth scales linearly with size across that whole range, so the rail is not the constraint there.

This is half of the test that would settle the NCCL small-message gap: the same sweep on node5700<->node5701, compared at the small sizes, attributes that gap to the IB stack or exonerates it. See `../b200-ubuntu/out-nccl-2node/summary.md` § 2.2.
