# Tests to run on AICR — narrowing the inter-node GDRDMA defect

Four tests, ranked by how decisively each narrows the diagnosis. None needs more
than `perftest`; only the `lspci` reads in test 4 need root. All are read-only —
nothing is reconfigured.

Context and the elimination table are in `aicr-nccl-2node-admin.md`. The defect
being chased: GPU-memory RDMA on AICR collapses to **27.2 GB/s per direction**
when both directions run at once, while Engaging reaches **48.7** on the same
hardware generation.

---

## 0. Verify rail affinity first — before trusting any measurement

```bash
nvidia-smi topo -m
```

Confirm the rail used in AICR's tests is the **PIX/PXB partner** of the GPU under
test, not a NODE-distance NIC.

We hit this trap on Engaging and it cost a full round of wrong conclusions: using
a NODE-distance rail instead of the GPU's PXB partner gave **18.6 GB/s instead of
49.4 GB/s** — a 2.6x error that mimics a hardware defect exactly. Adding queue
pairs does not compensate (18.5 / 19.2 / 19.4 / 19.4 / 16.5 GB/s at
q = 1 / 2 / 4 / 8 / 16).

AICR's healthy unidirectional C3 (47.5 GB/s) suggests its rail *was* correctly
paired, so this is probably not the issue — but the assumption has never been
checked, and it invalidates everything downstream if wrong.

---

## 1. Direction isolation — cheapest and most informative

**Question:** the bidirectional case collapses, but *which direction* fails?

Put GPU memory on only one side, so each half of the transfer is measured alone:

```bash
# NIC reads from GPU (GPU is the source)
ib_write_bw -d <pxb_rail> -s 8388608 -n 2000 -F --report_gbits --use_cuda=0 <server>

# NIC writes into GPU (GPU is the sink)
ib_write_bw -d <pxb_rail> -s 8388608 -n 2000 -F --report_gbits --use_cuda=0
```

**Reference — Engaging while it was broken** (`../notes.md`, 2026-07-13), same
rail, host memory at 47.4 GB/s for comparison:

| Path | Engaging (broken) |
|---|---:|
| host -> host | 47.4 GB/s |
| **NIC reads from GPU** | **18.5 GB/s** |
| NIC writes into GPU | 35.8 GB/s |

**How to read the AICR result**

| Outcome | Conclusion |
|---|---|
| Sharply asymmetric (read path much slower) | Mechanism is read-completion handling on the GPU->NIC path. Points directly at MaxReadRequest and switch read-completion credits — go to test 4. |
| Roughly symmetric | A different mechanism; read-completion credits are not it. |

Takes about a minute and halves the search space either way.

---

## 2. Switch-bypass test — settles the prime suspect outright

**Question:** is the Broadcom PEX890xx Gen5 switch responsible?

It is the only structural difference left between the clusters (AICR: AMD Turin
GPP root complex -> **Broadcom PEX890xx** -> B200 / ConnectX-7; Engaging:
Mellanox MT2910 bridge chain).

Check whether any GPU/NIC pair on an AICR node reaches its NIC **without**
traversing the switch:

```bash
lspci -t                 # topology tree
nvidia-smi topo -m       # GPU <-> NIC distances
```

If such a pair exists, measure it bidirectionally with GPU memory on both sides.

| Outcome | Conclusion |
|---|---|
| Bypass pair ~48 GB/s while switch-traversing pairs sit at ~27 | **Switch confirmed.** Fix is firmware/config on the PEX890xx. |
| Bypass pair also ~27 GB/s | Switch exonerated. Attention moves to the NIC or the AMD Turin root complex. |

**Highest-value test if the topology permits it** — it either confirms or kills
the leading hypothesis in one measurement.

---

## 3. Concurrency scaling — per-port vs switch-wide exhaustion

**Question:** is the limit per-link, or a shared switch resource saturating?

Run GPU bidirectional on **1 pair, then 2, then 4, then 8 pairs simultaneously**,
and watch per-pair bandwidth.

| Outcome | Conclusion |
|---|---|
| Per-pair stays ~27 regardless of pair count | A per-port / per-link limit. |
| Per-pair degrades as pairs are added | A **shared switch resource** (credits, buffers) saturating — the classic PCIe-switch signature, and strong corroboration for test 2. |

AICR's existing data is all single-pair, so this dimension is completely
unexplored.

---

## 4. PCIe parameter read + experiment (needs root)

```bash
# MaxPayload / MaxReadReq across GPU -> bridge(s) -> NIC
lspci -vvv -s <gpu_bdf>    | grep -E "DevCtl|MaxPayload|MaxReadReq|RlxdOrd|NoSnoop"
lspci -vvv -s <bridge_bdf> | grep -E "DevCtl|MaxPayload|MaxReadReq|RlxdOrd|NoSnoop"
lspci -vvv -s <nic_bdf>    | grep -E "DevCtl|MaxPayload|MaxReadReq|RlxdOrd|NoSnoop"

# Is ACS genuinely off on every bridge, including the PEX switch ports?
lspci -vvv | grep -i -B12 acsctl | grep -E "^[0-9a-f]|ACSCtl"

# ConnectX ordering / outstanding-read settings
mlxconfig -d <nic_bdf> q | grep -Ei "PCI_WR_ORDERING|ADVANCED_PCI|MAX_ACC_OUT_READ"
mlxfwmanager --query     # AICR 28.41.1000 vs Engaging 28.49.1120
```

A small `MaxReadReq` anywhere on the switch path throttles specifically the
NIC-reads-from-GPU direction — which test 1 will already have implicated or
cleared. Directly actionable if mismatched.

Note: **unprivileged `lspci -vv` silently omits the PCIe Express capability**, so
`ACSCtl` and `RlxdOrd` cannot be read without root. Their absence from an
unprivileged dump means "not visible", **not** "clean".

---

## What not to bother with

**Re-running `run-engaging-check.sh` unchanged on AICR.** AICR's C1-C6 already
exist and match what it would produce, so it adds nothing. The value is in tests
1-3 above, which probe dimensions the existing data does not cover.

Also already eliminated, with evidence, in `aicr-nccl-2node-admin.md`:
`EnablePCIERelaxedOrderingMode`, PCIe relaxed ordering generally, IOMMU, ACS,
`nvidia_peermem`, GPU PCIe endpoint health, and NCCL version/tuning.

---

## How to confirm a fix

Single command on the PXB-affinity rail, ~30 seconds:

```bash
# server
ib_write_bw -d <pxb_rail> -s 8388608 -n 2000 -F --report_gbits --use_cuda=0 -b
# client
ib_write_bw -d <pxb_rail> -s 8388608 -n 2000 -F --report_gbits --use_cuda=0 -b <server>
```

perftest reports bidirectional rows as the **sum of both directions** — divide
the Gb/s figure by 16 for GB/s per direction.

| Result | Meaning |
|---|---|
| ~**780 Gb/s** (48-49 GB/s/dir) | Fixed — matches Engaging |
| ~435 Gb/s (27 GB/s/dir) | Unchanged |

Then end to end with NCCL: 2-node SendRecv should move from 26.6 to ~49 GB/s, and
AllGather from 218 to ~380 GB/s.

---

*Engaging reference measurements: node5501 + node5502, 2026-08-07, rail `mlx5_4`
(PXB). Raw data and verdicts in `RESULTS.md`; remediation guidance in
`aicr-nccl-2node-admin.md`.*
