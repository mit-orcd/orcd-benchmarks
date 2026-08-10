# Engaging counter-test for the AICR inter-node GDRDMA defect

**If you are an agent picking this up cold: read this whole file first. It is the complete brief —
goal, how to run, what the numbers mean, and exactly what to report back. You need no prior
context and no root. Nothing here changes any system setting; every step is read-only.**

One SLURM job, 2 nodes × 1 GPU, ~5 minutes of runtime.

---

## 1. Goal

Decide **one question**:

> On healthy B200 hardware, does GPU-memory RDMA sustain full line rate in *both directions at
> once* — and if so, is PCIe relaxed ordering what makes the difference?

### Why it matters

On the MGHPCC **AICR** B200 cluster, GPU-memory RDMA collapses when both directions run
simultaneously, while host memory on the same rail does not:

| Path (per direction) | AICR measured |
|---|---:|
| host memory, unidirectional | 46.3 GB/s |
| host memory, **bidirectional** | 47.3 GB/s |
| GPU memory, unidirectional | 47.5 GB/s |
| **GPU memory, bidirectional** | **27.2 GB/s** ← the defect |

`27.2 GB/s × 8 rails = 218 GB/s`, which is exactly AICR's NCCL AllGather. So this microbenchmark
gap *is* the collective gap.

A published paper (`aicr_benchmarks_submitted.pdf`, Section IV B) interprets AICR's 26.6 GB/s
SendRecv as a hardware limit — "a silicon-level wall that no NCCL tuning can overcome" — derived
from a claimed ~53.5 GB/s DMA budget *shared* between transmit and receive. Separate measurement
on AICR has already refuted that model on AICR's own hardware (a plain `cudaMemcpy` full-duplex
test reaches 98.3 GB/s total, 49.1 GB/s each way at once). Engaging's NCCL AllGather implies
~47.9 GB/s per rail, i.e. healthy.

What is **still unknown** is *which setting* causes AICR's collapse. One hypothesis —
`EnablePCIERelaxedOrderingMode: 0` in the NVIDIA driver — was already tested on AICR and
**refuted**: toggling relaxed ordering there changes nothing (32.0 vs 31.5 GB/s/dir), and that
parameter is the vendor default anyway. This job runs the same tests on healthy hardware, which
is the cleanest remaining way to isolate the difference.

---

## 2. Run it

### Find the partition and nodes

```bash
sinfo -o "%.18P %.6a %.10l %.6D %.6t %N"        # list partitions
sinfo -N -o "%N %P %G" | grep -i b200           # which partition holds B200 nodes
```

The B200 nodes are named `node5500`, `node5501`, `node5502` (at least those three are known
good). Pick whichever partition contains them.

### Submit

```bash
cd engaging-check
sbatch -p <b200-partition> run-engaging-check.sh
```

Optional:

```bash
sbatch -p <part> -w node5500,node5501 run-engaging-check.sh   # pin specific nodes
NIC_FORCE=mlx5_4 sbatch -p <part> run-engaging-check.sh       # pin a specific rail
```

Watch it: `squeue -u $USER`. Output appears in `eng-gdr-<jobid>.out`.

**The job does the arithmetic and the verdict for you.** The last two blocks of the output are a
`CONVERTED RESULTS` table (already in GB/s per direction, with AICR values alongside) and an
`AUTOMATIC VERDICT` section that states which conclusion follows. You should still sanity-check
it against §5 rather than trusting it blindly — but you do not need to convert Gb/s by hand, and
the verdict thresholds (healthy ≥ 42, collapsed < 38 GB/s/dir) are visible in the script.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Failed status 12` / `transport retry exceeded` | Server and client are on rails in different IB subnets. Re-run with `NIC_FORCE=<rail>` pinning one 400 Gb/s rail both sides. Find candidates with `ibstat \| grep -E "^CA \|Rate:"`. |
| Part A says `nvcc NOT FOUND` | Harmless — Parts B and C still answer the question. To fix, `module avail cuda` and load one, then resubmit. |
| `ib_write_bw: command not found` | Load the OFED/perftest module (`module avail perftest` or `module avail ofed`), or point to a full path in the script. |
| Job pends a long time | Try the devel/debug partition, or drop to `-N 2 --gpus-per-node=1` on a less loaded partition. It only needs 1 GPU per node. |
| `FATAL: no 400 Gb/s rail found` | `ibstat -l` and `ibstat <dev> \| grep Rate` to see what exists; pass the right one via `NIC_FORCE`. |

**Do not** change any system setting, module parameter, or firmware value. This is a
read-only measurement. If something looks misconfigured, report it — do not fix it.

---

## 3. Reading the numbers

perftest reports **Gb/s**. Two conversions, both easy to get wrong:

1. **Gb/s ÷ 8 = GB/s**
2. **Bidirectional rows are the SUM of both directions** → divide by 2 again

Worked example: `435.89 Gb/s` bidirectional = 54.5 GB/s total = **27.2 GB/s per direction**.

Also note: **perftest requests MR-level relaxed ordering by DEFAULT.** The flag
`--disable_pcie_relaxed` turns it *off*. So C4 is "RO on" and C5 is "RO off" — not the reverse.

---

## 4. Expected results

| Test | What it is | AICR measured | Expected on Engaging |
|---|---|---:|---:|
| **A** | PCIe full duplex, `cudaMemcpy`, no IB | 49.1 each way, **98.3 total** | ~90–100 GB/s total (control; both clusters should pass) |
| **C1** | host unidirectional | 46.3 | ~46–48 GB/s |
| **C2** | host bidirectional | 47.3 | ~45–48 GB/s/dir |
| **C3** | GPU unidirectional | 47.5 | ~46–48 GB/s |
| **C4** | **GPU bidirectional, RO on** | **27.2** | **~47 GB/s/dir if healthy** |
| **C5** | GPU bidirectional, RO **off** | 31.5 | ? — this is the fingerprint |
| **C6** | host bidirectional, RO off | 44.0 | ~44–48 GB/s/dir |

**C4 is the headline number.** Everything else is context for it.

---

## 5. Decision table — what to conclude

Work through these in order.

### First: is Engaging healthy?

| C4 (GPU bidir, RO on) | Conclusion |
|---|---|
| **~45–50 GB/s/dir** | **Engaging is healthy.** AICR's 27.2 is a genuine cluster defect, now confirmed by direct contrast on identical hardware. Proceed to the next table. |
| **~27–32 GB/s/dir** | **Engaging shows it too.** This would mean it is *not* an AICR misconfiguration — it may be a B200/ConnectX platform property. This overturns the whole diagnosis and puts the paper's model back in play. Unlikely, but report it prominently and stop; the follow-up analysis changes completely. |
| anything else | Report the number and note it fits neither case. |

### Then: is relaxed ordering the lever?

Only if C4 was healthy. Compare C5 against C4.

| C5 vs C4 | Conclusion |
|---|---|
| **C5 collapses (~27) while C4 healthy (~47)** | **Relaxed ordering IS the lever**, and it works on Engaging but not on AICR. AICR's bug is therefore "RO never reaches the wire" — the search narrows to the NIC's `DevCtl.RlxdOrd` enable bit, ConnectX firmware `PCI_WR_ORDERING`, or the PCIe switch stripping the attribute. **This is the most informative outcome.** |
| **C5 ≈ C4, both healthy (~47)** | RO is **not** the lever on either cluster. AICR's difference lies elsewhere — switch config, BIOS, or firmware. Part B config diff becomes the primary lead. |
| C5 healthy but C4 degraded | Contradictory; report verbatim, do not rationalise. |

### Independently: the driver parameter

Look at **Part B1** (`/proc/driver/nvidia/params`, relaxed-ordering lines).

| B1 on Engaging | Conclusion |
|---|---|
| `EnablePCIERelaxedOrderingMode: 0` **and** C4 healthy | That parameter is **exonerated permanently** — it is the vendor default and discriminates nothing. An earlier AICR analysis wrongly named it as the root cause on the strength of this value alone; this closes it. |
| `EnablePCIERelaxedOrderingMode: 1` | Now it *is* a real difference between the clusters, and becomes the leading candidate again. Flag this clearly. |

---

## 6. What to report back

Send the raw file **and** a short written conclusion. Both.

1. **`eng-gdr-<jobid>.out`** — the complete raw output, unedited.

2. **A summary containing:**
   - A filled-in results table: Part A total, and C1–C6 **converted to GB/s per direction**
     (do the arithmetic; do not just paste Gb/s).
   - The **verdict on C4**: is Engaging healthy? Which row of the first decision table applies.
   - The **C5 vs C4 fingerprint** result and which conclusion follows.
   - The **B1 driver parameter value**, and whether it exonerates the NVreg hypothesis.
   - **Any config differences from AICR** worth chasing — see §7 for how to diff.
   - Anything that **failed, was skipped, or looked anomalous.** Do not quietly drop a test.

3. **Do not overstate.** If a test did not run, say so. If a number is ambiguous, say so. A
   result that contradicts the expected outcome is more valuable than one that confirms it —
   report it plainly rather than explaining it away.

---

## 7. Comparing against AICR

`aicr-reference/` holds the raw AICR output for side-by-side reading:

| File | Contents |
|---|---|
| `gdr-root-b0031-317105` | Part A equivalent + full config dump (node b0031) |
| `diag-b0029.txt` | config dump (node b0029) |
| `gdr-ab-300708` | host vs GPU, uni vs bidir (b0029+b0030) |
| `gdr-ro-317188` | the relaxed-ordering toggle matrix |

The most informative diff is **Part B against `gdr-root-b0031-317105`** — two B200 systems with a
known, reproducible 1.75× behavioural difference, so a config diff has a real chance of naming
the setting outright. Pay particular attention to:

- the relaxed-ordering lines in `/proc/driver/nvidia/params`
- the PCIe bridge chain above the GPU (AICR: **Broadcom PEX890xx Gen5 switch** → **AMD Turin GPP
  Bridge**) — a different switch vendor or a direct root-complex attachment on Engaging would
  itself be a strong lead
- kernel cmdline: AICR has `amd_iommu=off iommu=off pci=noacs`
- link widths and speeds (AICR: all Gen5 x16)
- ConnectX firmware version (AICR: 28.41.1000)

**AICR platform reference:** AMD Turin GPP root complex → Broadcom PEX890xx Gen5 switch → B200 /
ConnectX-7 (fw 28.41.1000), 8 × NDR400 compute rails, `amd_iommu=off iommu=off pci=noacs`, all
links Gen5 x16, `nvidia_peermem` loaded, BAR1 256 GB.

---

## 8. What the job actually does

| Part | Test | Needs |
|---|---|---|
| **A** | PCIe full duplex via `cudaMemcpyAsync` — no IB, no NIC, no RDMA | `nvcc` (auto-detected; skipped if absent) |
| **B** | Node config: driver params, cmdline, IOMMU, peermem, PCIe link widths, bridge chain, rail rates | nothing |
| **C** | RDMA matrix: host vs GPU × uni vs bidir × relaxed ordering on/off | `ib_write_bw` |

### Traps already handled in the script

- Server and client are pinned to the **same rail** — different rails can sit on different IB
  subnets and fail with `Failed status 12`.
- The rail is chosen by **PIX affinity** to the allocated GPU and filtered to 400 Gb/s, not
  hardcoded. An earlier AICR script hardcoded `mlx5_0`, which was a cross-host-bridge path and
  measured the wrong thing entirely.
- CUDA is auto-detected; Part A degrades gracefully.
- Unprivileged `lspci -vv` **silently omits** the PCIe Express capability, so `ACSCtl` and
  `RlxdOrd` cannot be read without root — absence of those lines means "not visible", **not**
  "clean". Part B uses the world-readable sysfs link attributes instead.
