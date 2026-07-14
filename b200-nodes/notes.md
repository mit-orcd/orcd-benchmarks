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
