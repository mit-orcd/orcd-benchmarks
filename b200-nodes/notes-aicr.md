# Why our inter-node NCCL results differ from the AICR paper (Section IV B)

Comparison of the 2-node NCCL results measured on the MIT ORCD B200 nodes
(node5500 / node5501 / node5502, 2026-08-06) against Section IV B,
*Inter-node (NDR InfiniBand)*, of `aicr_benchmarks_resubmit.pdf`
(Chen, Milechin & Hill — "Full-Stack Benchmarks of a Blackwell-Generation AI
Cluster"), whose numbers come from the MGHPCC AICR cluster (nodes b0029+b0030,
raw data in `~/data022/aicr-benchmarks/Benchmark_WG/nccl-tests/results_b200.md`).

**Conclusion: the AICR cluster has a real, fixable inter-node configuration
problem (option 3), and an incorrect hardware model in the paper (option 1)
masked it by "predicting" the degraded number. Option 2 — NCCL job setup — does
not explain the gap.**

---

## 1. The model in Section IV B is physically wrong

Section IV B argues:

> Although each NDR NIC is rated at 50 GB/s per direction, a B200 GPU reaches its
> NIC through a single 16-lane PCIe Gen5 port whose DMA engine has a fixed
> 53.5 GB/s HBM budget **shared** between transmit and receive; under simultaneous
> bidirectional traffic each direction collapses to half of that, about 26.7 GB/s,
> so the per-node aggregate for ring-style collectives is 8 x 26.7 ~ 214 GB/s per
> direction. [...] Send-Recv measures 26.6 GB/s, essentially 100% of the per-pair
> bidirectional limit, **a silicon-level wall that no NCCL tuning can overcome**.

PCIe is **full-duplex**. A Gen5 x16 link carries ~63 GB/s in *each* direction
simultaneously; there is no shared budget to halve.

The paper contradicts itself two sections later. Section IV D (RTX PRO 6000)
states PCIe Gen5 x16 is "~63 GB/s **per direction**" and reports RTX PRO 6000
Send-Recv at **37.4 GB/s per direction under bidirectional traffic** — already
40% above the supposed 26.7 GB/s wall, on a weaker CPU-relayed path.

## 2. Our measurements break the model cleanly

Converged busbw, 8 GPUs/node x 2 nodes (16 GPUs), 16 GiB messages:

| Collective | AICR (paper) | ours | ratio | paper's verdict |
|---|---:|---:|---:|---|
| Send-Recv | 26.6 | **49.7** | 1.87x | "100% of max, silicon wall" |
| Reduce | 201 | **384** | 1.91x | "94% of max" |
| All-Gather | 218 | **383** | 1.76x | "100% of max" |
| Reduce-Scatter | 218 | **382** | 1.75x | "100% of max" |
| Broadcast | 202 | **368** | 1.82x | "94% of max" |
| Scatter | 293 | **339** | 1.16x | "137%, unidirectional" |
| All-Reduce (Ring) | 170 | **240** | 1.41x | "79%, algorithm-limited" |
| All-to-All | 39.8 | **49.9** | 1.25x | "19%, algorithm-limited" |
| Gather | 90.5 | **95.4** | 1.05x | "42%, algorithm-limited" |

The pattern **inverts** the paper's model:

- Everything declared *hardware-saturated* (Send-Recv, All-Gather,
  Reduce-Scatter, Reduce, Broadcast) nearly **doubled**. If 214 GB/s were a real
  silicon wall, nothing could pass it — yet these land at 360–384 GB/s.
- Everything declared *algorithm-limited* barely moved (Gather 1.05x,
  All-to-All 1.25x). Those diagnoses were **correct**: NCCL's multi-NIC fan-in
  and non-pipelined N^2 exchange really are the constraint, and a faster fabric
  does not help them.

Sanity check on our own numbers: Send-Recv at 49.7 GB/s is **99% of the NDR line
rate** (400 Gb/s = 50 GB/s per direction) — a hard physical bound we approach but
never exceed. AICR's 26.6 GB/s is **53%** of it. That factor of ~2 is the whole
story.

## 3. What is likely misconfigured on AICR

26.6 GB/s is close to **PCIe Gen5 x8** effective throughput (32 GB/s theoretical,
~26–27 GB/s realistic). Ranked candidates, with the check for each:

1. **GPU<->NIC PCIe link negotiated at x8 instead of x16** — best numerical match.
   `nvidia-smi -q | grep -A3 "Link Width"`; `lspci -vv -s <nic> | grep LnkSta`
   (compare `LnkSta` against `LnkCap`).
2. **ACS redirect enabled** on the PCIe switch ports, forcing peer-to-peer TLPs up
   through the root complex. `lspci -vvv | grep -i acsctl`.
3. **IOMMU translating the P2P path.** `results_b200.md` ("Fixed Issues") says AICR
   required `iommu=off`; worth confirming it was actually in effect on the nodes
   benchmarked.
4. **GPU-NIC rail affinity** — if NCCL pairs a GPU with a NIC on the other socket,
   traffic crosses the CPU interconnect. `nvidia-smi topo -m` should show
   `PXB`/`PIX` for each GPU's chosen NIC, never `SYS`.

**Decisive test:** run `ib_write_bw` with `--use_cuda` and compare against the
host-memory case. If host->host reaches ~47 GB/s while GPU->GPU sits at ~26 GB/s,
the defect is in the P2P path — not the fabric, and definitively not silicon.

### We hit exactly this class of defect locally

Recorded in `notes.md` (2026-07-13), on these same MIT B200 nodes:

| perftest (mlx5_4, NDR 400 Gb/s, 64 MiB, RDMA write) | Bandwidth |
|---|---:|
| host mem -> host mem | 379.5 Gb/s (**47.4 GB/s**) — full line rate |
| NIC **reads from** GPU | 147.6 Gb/s (**18.5 GB/s**) — capped |
| NIC **writes into** GPU | 286.6 Gb/s (35.8 GB/s) — partly degraded |

That capped GPU-read path produced NCCL Send-Recv of only **12.7 GB/s**. The
suspects logged at the time were IOMMU (`iommu=pt intel_iommu=on`) and ACS
(`pci=disable_acs_redir`). It has since been resolved on node5500–5502 —
Send-Recv went **12.7 -> 49.7 GB/s** — so the remediation is known locally and
worth comparing against AICR's configuration.

## 4. Why it is not option 2 (NCCL job setup)

An NCCL setup error cannot halve raw Send-Recv, which is a plain point-to-point
transfer with almost no algorithmic surface to misconfigure.

NCCL *version* does matter, but only for the algorithm-limited collectives: ours
is **NCCL 2.29.2** (nvhpc 26.1 / CUDA 13.1) versus the paper's **nccl-tests
2.18.3** era. That plausibly accounts for part of the All-to-All (1.25x) and
All-Reduce (1.41x) differences — the two NCCL has since improved multi-NIC
pipelining for — but not the uniform ~1.9x on point-to-point and ring
collectives.

## 5. Implications for the paper

The measurements are fine; the interpretation and the derived `%max` column are
not. Every "100%" and "94%" in the inter-node half of Table III is computed
against a ceiling that is roughly half its true value. Recomputed against
~50 GB/s per pair (400 GB/s per node aggregate), AICR's Send-Recv is ~53% and
All-Gather ~55% — i.e. the cluster has a real inter-node problem rather than
being at silicon saturation.

Knock-on claims that need revisiting:

- "a silicon-level wall that no NCCL tuning can overcome" (Section IV B).
- The pipeline-parallel guidance in the discussion budgeting ~37 ms for a 1 GB
  activation transfer; on a healthy fabric this is ~20 ms.
- The framing of SHARP as *bypassing* a 214 GB/s bidirectional wall
  (Section IV C). SHARP's single-pass advantage over Ring All-Reduce is real and
  independent of this error, but the wall it is described as bypassing is not
  where the paper places it.

## Caveat

This diagnosis is based on the AICR numbers as published plus our contrasting
measurements; the AICR cluster was not inspected directly. The internal
contradiction (IV B vs IV D) and the inverted improvement pattern hold
regardless, but confirming the specific mechanism requires the
`lspci` / `nvidia-smi topo` / `ib_write_bw --use_cuda` checks listed above.

---

*Generated 2026-08-06 from the 2-node NCCL runs in `out-nccl-2node/summary.md`
(jobs 19791438 / 19791440 / 19791441, all three node pairs).*
