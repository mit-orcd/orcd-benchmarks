# AICR inter-node GDRDMA — findings from the Engaging counter-test

**Package prepared 2026-08-07 by MIT ORCD (Engaging cluster) for the AICR
administrators.** Everything here is measurement output plus read-only test
scripts. Nothing in this package changes any system setting.

---

## Conclusion of the Engaging test

**Engaging is healthy, and AICR's inter-node GPU RDMA collapse is a genuine
cluster defect — not a hardware limit.**

We ran the counter-test on two MIT Engaging B200 nodes (node5501 + node5502,
ConnectX-7 NDR400, rail `mlx5_4`). GPU-memory RDMA sustains **48.7 GB/s per
direction with both directions running simultaneously** — 97.3 GB/s total, ~97%
of the 50 GB/s NDR line rate. AICR measures **27.2 GB/s/dir** on the same
hardware generation.

| Measurement (GB/s per direction) | AICR | Engaging | Ratio |
|---|---:|---:|---:|
| host memory, unidirectional | 46.3 | 47.8 | 1.03x |
| host memory, bidirectional | 47.3 | 47.6 | 1.01x |
| GPU memory, **unidirectional** | 47.5 | 49.4 | 1.04x |
| **GPU memory, bidirectional** | **27.2** | **48.7** | **1.79x** |
| NCCL 2-node SendRecv | 26.6 | 49.7 | 1.87x |
| NCCL 2-node AllGather (node aggregate) | 218 | 383 | 1.76x |

Three things follow:

1. **The defect is narrow and specific.** Host memory is fine in both
   directions. GPU memory is fine *unidirectionally*. It collapses only when
   GPU-memory RDMA runs in **both directions at once** — which is exactly what
   every NCCL ring collective does. `27.2 x 8 rails = 218 GB/s` is precisely
   AICR's measured AllGather, so this microbenchmark gap *is* the collective gap.

2. **The "silicon-level wall" interpretation does not hold.** The model in
   `aicr_benchmarks_resubmit.pdf` Section IV B derives a ~26.7 GB/s per-direction
   ceiling from a claimed ~53.5 GB/s DMA budget *shared* between transmit and
   receive. PCIe Gen5 x16 is full duplex (~63 GB/s *each* way); Engaging reaches
   48.7 GB/s in both directions simultaneously on the same silicon. Details in
   `notes-aicr.md`.

3. **Relaxed ordering is not the lever, and the NVIDIA driver parameter is
   exonerated.** Toggling PCIe relaxed ordering changes nothing on either cluster
   (Engaging 48.7 vs 48.7, identical to 2 dp; AICR 32.0 vs 31.5).
   `EnablePCIERelaxedOrderingMode` is `0` on **both** clusters — Engaging reaches
   full rate with the same value, so that parameter discriminates nothing.

**Target for AICR: ~48-49 GB/s/dir on GPU bidirectional, which lifts NCCL 2-node
SendRecv from 26.6 to ~49 GB/s and AllGather from 218 to ~380 GB/s.**

After the eliminations above, the leading remaining hypothesis is the **Broadcom
PEX890xx Gen5 switch** between GPU and NIC — the only structural difference left
between the two platforms.

---

## Where to start

| Read this | For |
|---|---|
| **`aicr-2node-ib-test.md`** | **The tests to run on AICR**, ranked by how decisively each narrows the diagnosis. Start here. |
| **`aicr-nccl-2node-admin.md`** | What to fix: elimination table, prime suspect, diagnostic checklist, candidate fixes in order, and how to verify. |
| `RESULTS.md` | Full Engaging counter-test results, all verdicts, config diff. |
| `notes-aicr.md` | Why the Section IV B hardware-limit model is wrong, in detail. |

## One warning before you measure anything

**Verify rail affinity first.** Using a NIC that is NODE distance from the GPU
instead of its PIX/PXB partner gave us **18.6 GB/s instead of 49.4 GB/s** — a
2.6x error that mimics a hardware defect exactly, and it cost us a full round of
wrong conclusions before we caught it.

```bash
nvidia-smi topo -m      # confirm the rail is the GPU's PIX/PXB partner
```

Adding queue pairs does **not** compensate for wrong affinity (18.5 / 19.2 /
19.4 / 19.4 / 16.5 GB/s at q = 1 / 2 / 4 / 8 / 16 — see `eng-qp-*.out`).

AICR's healthy unidirectional figure (47.5 GB/s) suggests its rail *was*
correctly paired, so this is probably not AICR's issue — but the assumption has
never been checked, and it invalidates everything downstream if wrong.

## Files

**Documents**

| File | Contents |
|---|---|
| `aicr-2node-ib-test.md` | Tests to run on AICR (4, ranked) + fix verification |
| `aicr-nccl-2node-admin.md` | Remediation guide for administrators |
| `RESULTS.md` | Engaging counter-test results and verdicts |
| `notes-aicr.md` | Analysis of the Section IV B model |

**Scripts** (read-only, no root needed except where noted)

| File | Purpose |
|---|---|
| `run-engaging-check.sh` | The counter-test itself: Part A (PCIe duplex), Part B (config dump), Part C (RDMA matrix C1-C6). Run with `NIC_FORCE=<pxb_rail>`. |
| `pcie_duplex.cu` | CUDA source for Part A; compiled automatically by the above |
| `rail-affinity.sh` | Per-rail GPU RDMA sweep + `nvidia-smi topo -m` — use this to verify rail affinity |
| `analyze-engaging-check.py` | Generates `RESULTS.md` from the raw output; conversions and verdicts recomputed independently of the job script |

**Raw data**

| File | Contents |
|---|---|
| `eng-gdr-19884807.out` | **The valid counter-test run** (rail `mlx5_4`, PXB affinity) |
| `eng-gdr-19881928.out` | Earlier run on a NODE-distance rail — retained as a worked example of the affinity trap; **its C3/C4/C5 are invalid** |
| `eng-rail-19884230.out` | Per-rail sweep showing the PXB/NODE split (timed out after 3 of 8 rails; the split is nonetheless unambiguous) |
| `eng-qp-19883270.out` | Queue-pair sweep — negative result, retained so it is not re-tried |

## Known limitations of this package

Stated plainly so nothing is over-read:

- The rail sweep covered **3 of 8 rails** before hitting its wall-clock limit
  (`mlx5_4` 49.4, `mlx5_7` 18.6, `mlx5_8` 18.6 GB/s). Enough to establish the
  PXB/NODE split; not a complete table.
- The **direction-split reference** quoted in `aicr-2node-ib-test.md` (NIC reads
  from GPU 18.5 vs writes into GPU 35.8 GB/s) was measured on Engaging on
  2026-07-13 **while Engaging was itself misconfigured** (IOMMU enabled, SendRecv
  12.7 GB/s). It is a useful illustration of what an asymmetric read path looks
  like, but it is **not** a healthy-state baseline. We have not yet measured the
  direction split on healthy Engaging.
- Switch-side firmware and configuration on Engaging were not inspected;
  unprivileged subnet-management and PCIe capability queries are blocked here.

---

*Engaging platform: 8 x B200 per node, ConnectX-7 (MT4129, fw 28.49.1120), 8 x
NDR400 rails, `iommu=off`, `nvidia_peermem` loaded, PCIe Gen5 x16 throughout,
Mellanox MT2910 bridge chain GPU -> NIC. Measurements from node5501 + node5502 on
2026-08-07.*

*Contact: shaohao@mit.edu*
