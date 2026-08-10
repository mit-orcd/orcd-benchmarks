# AICR: fixing inter-node NCCL to match Engaging

**For the AICR system administrators.** Everything below is derived from
measurements on both clusters; the supporting raw data is in this directory and
in `../out-nccl-2node/summary.md`.

---

## 1. The problem in one table

Both clusters are B200 + ConnectX-7 with 8 x NDR400 rails, `iommu=off`,
`nvidia_peermem` loaded. Same generation, same NIC, same driver parameter values.

| Measurement (GB/s per direction) | AICR | Engaging | Ratio |
|---|---:|---:|---:|
| host memory RDMA, unidirectional | 46.3 | 47.8 | 1.03x |
| host memory RDMA, bidirectional | 47.3 | 47.6 | 1.01x |
| GPU memory RDMA, **unidirectional** | 47.5 | 49.4 | 1.04x |
| **GPU memory RDMA, bidirectional** | **27.2** | **48.7** | **1.79x** |
| NCCL 2-node SendRecv | 26.6 | 49.7 | 1.87x |
| NCCL 2-node AllGather (node aggregate) | 218 | 383 | 1.76x |

**The defect is specific and narrow.** Host memory is fine in both directions.
GPU memory is fine *unidirectionally*. It collapses only when GPU-memory RDMA
runs in **both directions simultaneously** — which is exactly what every NCCL
ring collective does. `27.2 x 8 rails = 218 GB/s`, which is precisely AICR's
measured AllGather, so this microbenchmark gap *is* the collective gap.

**Target: ~48-49 GB/s/dir on GPU bidirectional, which lifts NCCL 2-node from
26.6 to ~49 GB/s SendRecv and AllGather from 218 to ~380 GB/s.**

## 2. It is not a hardware limit

A published interpretation (`aicr_benchmarks_resubmit.pdf`, Section IV B) reads
AICR's 26.6 GB/s as "a silicon-level wall that no NCCL tuning can overcome",
derived from a claimed ~53.5 GB/s DMA budget *shared* between transmit and
receive.

Engaging reaches **48.7 GB/s per direction simultaneously — 97.3 GB/s total —
on the same GPU and NIC generation.** PCIe Gen5 x16 is full duplex (~63 GB/s
*each* way); there is no shared budget to halve. The wall does not exist on
healthy hardware, so AICR's number is a defect to fix, not a ceiling to accept.

## 3. Already ruled out — do not spend time here

| Hypothesis | Status | Evidence |
|---|---|---|
| `EnablePCIERelaxedOrderingMode` (NVreg) | **Exonerated** | Value is `0` on **both** clusters. Engaging reaches full bidirectional rate with the same value. It is the vendor default and discriminates nothing. |
| PCIe relaxed ordering generally | **Not the lever** | Toggling it changes nothing on either cluster: AICR 32.0 vs 31.5, Engaging 48.7 vs 48.7 (identical to 2 dp). |
| IOMMU | **Already correct on AICR** | AICR boots `amd_iommu=off iommu=off`. (This *was* Engaging's problem historically — it booted `iommu=pt intel_iommu=on` and measured 12.7 GB/s SendRecv; disabling IOMMU is what took Engaging to 49.7.) |
| ACS | **Already disabled on AICR** | AICR boots `pci=noacs`; Engaging does **not** have this flag and is nonetheless faster. Worth *verifying* it took effect (section 5), but it is not the missing setting. |
| `nvidia_peermem` | Loaded on both | Not the difference. |
| GPU PCIe endpoint health | Fine on both | Part A `cudaMemcpy` full duplex: AICR 98.3 GB/s total, Engaging 101.5. Both healthy — the defect is in the P2P/GDRDMA path, not the GPU link. |
| NCCL version / tuning | Not the cause | Cannot halve a raw point-to-point RDMA transfer; the gap reproduces in `ib_write_bw` with no NCCL involved. |

## 4. Prime suspect: the PCIe switch

After the eliminations above, **one structural difference remains**:

| | AICR | Engaging |
|---|---|---|
| PCIe path GPU -> NIC | **Broadcom PEX890xx Gen5 switch** -> AMD Turin GPP Bridge | **Mellanox MT2910** (ConnectX-family) bridge chain |

A third-party PCIe switch between GPU and NIC is a well-known place for
peer-to-peer performance to be lost, and the failure signature fits it closely:
unidirectional traffic is fine, but **concurrent bidirectional** P2P exhausts
switch resources — posted/non-posted credits, replay buffers, or per-port
queueing — and throughput halves. That is a switch behaviour, not a GPU or NIC
behaviour, and it is consistent with everything measured.

## 5. Diagnostic checklist (needs root on AICR)

Run these on an AICR B200 node. Note that **unprivileged `lspci -vv` silently
omits the PCIe Express capability**, so `ACSCtl` and `RlxdOrd` cannot be read
without root — their absence in an unprivileged dump means "not visible", *not*
"clean".

```bash
# 0. GPU <-> NIC affinity. Confirm which NIC is PIX/PXB to each GPU.
nvidia-smi topo -m

# 1. Is ACS actually disabled on every bridge in the path (incl. the PEX switch)?
lspci -vvv | grep -i -B12 acsctl | grep -E "^[0-9a-f]|ACSCtl"

# 2. Relaxed ordering / no-snoop enable bits, and MaxPayload / MaxReadReq,
#    on the GPU, the NIC, and every bridge between them.
lspci -vvv -s <gpu_bdf>    | grep -E "DevCtl|MaxPayload|MaxReadReq|RlxdOrd|NoSnoop"
lspci -vvv -s <nic_bdf>    | grep -E "DevCtl|MaxPayload|MaxReadReq|RlxdOrd|NoSnoop"
lspci -vvv -s <bridge_bdf> | grep -E "DevCtl|MaxPayload|MaxReadReq|RlxdOrd|NoSnoop"

# 3. Broadcom PEX890xx firmware version and P2P configuration.
#    (Broadcom PEX tooling, e.g. PlxCm / pex_util, or the vendor's switch utility)

# 4. ConnectX firmware and PCI ordering setting.
mlxconfig -d <nic_bdf> q | grep -Ei "PCI_WR_ORDERING|ADVANCED_PCI|MAX_ACC_OUT_READ"
mlxfwmanager --query        # AICR is on 28.41.1000; Engaging on 28.49.1120
```

### Rail-affinity trap — verify before trusting any measurement

We hit this on Engaging and it cost a full round of wrong conclusions. Using a
NIC that is **NODE distance** from the GPU instead of its **PXB** partner gave
**18.6 GB/s instead of 49.4 GB/s** — a 2.6x error that looks exactly like a
hardware defect:

| Rail (Engaging, GPU at `0000:1b:00.0`) | GPU RDMA | Affinity |
|---|---:|---|
| `mlx5_4` (`0000:18:00.0`) | **49.4 GB/s** | PXB, same bridge |
| `mlx5_7` (`0000:40:00.0`) | 18.6 GB/s | NODE, cross-CPU |
| `mlx5_8` (`0000:4f:00.0`) | 18.6 GB/s | NODE, cross-CPU |

Adding queue pairs does **not** compensate (18.5 / 19.2 / 19.4 / 19.4 / 16.5
GB/s at q = 1 / 2 / 4 / 8 / 16).

AICR's unidirectional C3 of 47.5 GB/s indicates its test *was* correctly paired,
so this is probably not AICR's issue — but confirm with `nvidia-smi topo -m`
before drawing conclusions, and make sure NCCL is pairing rail-optimally in
production too (`NCCL_DEBUG=INFO` prints `GPU Direct RDMA Enabled for GPU/... -
NET/...` per rank; each GPU should map to its own affine NIC).

## 6. Candidate fixes, in order

1. **Update / reconfigure the Broadcom PEX890xx.** Check for firmware addressing
   P2P or bidirectional credit handling, and review its P2P routing and
   ingress/egress credit allocation. Highest prior, being the only structural
   difference left.
2. **Verify ACS is genuinely off on the PEX switch ports**, not merely requested
   on the kernel cmdline. `pci=noacs` does not always take effect on every
   downstream port of a third-party switch.
3. **Align MaxPayloadSize and MaxReadRequest** across GPU, switch, and NIC. A
   small MaxReadReq on the switch path throttles NIC-reads-from-GPU specifically,
   which is the direction that collapses.
4. **Bring ConnectX firmware to parity** — AICR 28.41.1000 vs Engaging 28.49.1120
   — and compare `mlxconfig` output between clusters, especially
   `PCI_WR_ORDERING` and `MAX_ACC_OUT_READ`.
5. **Isolate the switch if the topology allows.** If any GPU/NIC pair on an AICR
   node reaches the NIC without traversing the PEX890xx, measure that pair. If it
   reaches ~48 GB/s bidirectional while switch-traversing pairs sit at 27, the
   switch is confirmed outright.

## 7. How to verify a fix

Single decisive command — no NCCL, ~30 seconds, on the **PXB-affinity** rail:

```bash
# server node
ib_write_bw -d <pxb_rail> -s 8388608 -n 2000 -F --report_gbits --use_cuda=0 -b
# client node
ib_write_bw -d <pxb_rail> -s 8388608 -n 2000 -F --report_gbits --use_cuda=0 -b <server>
```

perftest reports bidirectional rows as the **sum of both directions**, so divide
the Gb/s figure by 16 for GB/s per direction.

| Result | Meaning |
|---|---|
| ~**780 Gb/s** (48-49 GB/s/dir) | Fixed — matches Engaging |
| ~435 Gb/s (27 GB/s/dir) | Unchanged |

Then confirm end to end with NCCL: 2-node SendRecv should move from 26.6 to
~49 GB/s, and AllGather from 218 to ~380 GB/s.

The full counter-test used to produce these numbers is `run-engaging-check.sh` in
this directory — it is read-only, needs no root, and takes ~5 minutes. Run it on
AICR with `NIC_FORCE=<pxb_rail>` for a like-for-like comparison.

---

## Appendix: reference data

| File | Contents |
|---|---|
| `RESULTS.md` | Full Engaging counter-test results and verdicts |
| `eng-gdr-19884807.out` | Raw counter-test output, correct rail (`mlx5_4`) |
| `eng-gdr-19881928.out` | Earlier run on a NODE-distance rail — retained as an example of the affinity trap |
| `eng-rail-19884230.out` | Per-rail sweep showing the PXB/NODE split |
| `eng-qp-19883270.out` | Queue-pair sweep (negative result) |
| `../out-nccl-2node/summary.md` | Engaging NCCL 2-node, all collectives, vs hardware ceiling |
| `../notes-aicr.md` | Why the paper's Section IV B model is wrong |
| `aicr-reference/` | AICR's own raw dumps, for config diffing |

*Engaging measurements: node5501 + node5502, 2026-08-07. AICR figures from
`aicr-reference/` and the paper.*
