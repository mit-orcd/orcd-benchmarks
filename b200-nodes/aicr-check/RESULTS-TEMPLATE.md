# Engaging GDRDMA counter-test — results

Fill this in from `eng-gdr-<jobid>.out` and send it back together with the raw file.
Convert everything to **GB/s per direction**: `Gb/s ÷ 8` for GB/s, then `÷ 2` again for any
**bidirectional** row (perftest reports those as the sum of both directions).

**Run details**

| | |
|---|---|
| Cluster / partition | |
| Nodes | |
| Rail used | |
| Job ID | |
| Date | |
| perftest version | |
| CUDA available (Part A ran?) | |

---

## Part A — PCIe full duplex, no IB involved

| | GB/s |
|---|---:|
| H2D alone | |
| D2H alone | |
| **Concurrent, each way** | |
| **Concurrent, total** | |

AICR reference: 57.6 / 57.3 / 49.1 each way / **98.3 total**.

Control test — both clusters are expected to pass. If Engaging is *below* ~90 GB/s total, say so;
that would be unexpected and changes the picture.

---

## Part C — RDMA matrix

| Test | Description | Raw (Gb/s) | **GB/s per direction** | AICR |
|---|---|---:|---:|---:|
| C1 | host unidirectional | | | 46.3 |
| C2 | host bidirectional | | | 47.3 |
| C3 | GPU unidirectional | | | 47.5 |
| **C4** | **GPU bidirectional, RO on** | | | **27.2** |
| C5 | GPU bidirectional, RO **off** | | | 31.5 |
| C6 | host bidirectional, RO off | | | 44.0 |

---

## Verdicts

**1. Is Engaging healthy?** (from C4)

- [ ] C4 ≈ 45–50 GB/s/dir → **healthy**; AICR's 27.2 is a genuine cluster defect
- [ ] C4 ≈ 27–32 GB/s/dir → **Engaging shows it too** — overturns the diagnosis, report prominently
- [ ] neither → record the value and say so

C4 value: ______ GB/s/dir

**2. Is relaxed ordering the lever?** (C5 vs C4)

- [ ] C5 collapses (~27) while C4 healthy (~47) → **RO is the lever**; AICR's bug is "RO never
      reaches the wire" → chase NIC `DevCtl.RlxdOrd`, ConnectX `PCI_WR_ORDERING`, PCIe switch
- [ ] C5 ≈ C4, both healthy → **RO is not the lever**; the difference is elsewhere → Part B diff
- [ ] contradictory → report verbatim

**3. The NVIDIA driver parameter** (Part B1)

`EnablePCIERelaxedOrderingMode` on Engaging: ______

- [ ] `0` **and** C4 healthy → parameter **exonerated permanently** (it is the vendor default)
- [ ] `1` → a real inter-cluster difference; it becomes the leading candidate again

---

## Part B — configuration differences from AICR

Diff Part B against `aicr-reference/gdr-root-b0031-317105`. Note anything that differs:

| Item | Engaging | AICR | Significant? |
|---|---|---|---|
| `/proc/driver/nvidia/params` relax lines | | `EnablePCIERelaxedOrderingMode: 0` | |
| PCIe bridge chain above GPU | | Broadcom PEX890xx Gen5 switch → AMD Turin GPP Bridge | |
| kernel cmdline (iommu / acs) | | `amd_iommu=off iommu=off pci=noacs` | |
| GPU link width / speed | | Gen5 x16 of x16 | |
| ConnectX firmware | | 28.41.1000 | |
| `nvidia_peermem` | | loaded | |
| BAR1 size | | 256 GB | |
| rail rates | | 8 × 400 Gb/s + 4 × 100 Gb/s | |

---

## Anything that failed, was skipped, or looked odd

Do not quietly drop a test. A contradictory result is more valuable than a confirming one.

---

## One-paragraph conclusion
