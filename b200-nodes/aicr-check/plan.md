# Plan — running the AICR counter-test on Engaging

Prepared 2026-08-07, before submitting. Everything below was verified against the
brief (`README.md`), the template (`RESULTS-TEMPLATE.md`), the job script
(`run-engaging-check.sh`), and the live cluster state.

## What the test does

One SLURM job, 2 nodes x 1 GPU, ~5 minutes:

| Part | Test | Needs |
|---|---|---|
| **A** | PCIe full duplex via `cudaMemcpyAsync` — no IB, no NIC, no RDMA (control) | `nvcc` |
| **B** | Node config: driver params, cmdline, IOMMU, peermem, PCIe link widths, bridge chain, rail rates | nothing |
| **C** | RDMA matrix C1-C6: host vs GPU x uni vs bidir x relaxed ordering on/off | `ib_write_bw` |

## The question being decided

Is **C4** (GPU bidirectional, relaxed ordering ON) healthy on Engaging
(~47 GB/s/dir) versus AICR's **27.2 GB/s/dir**?

- If C4 is healthy → AICR's collapse is a genuine cluster defect, confirmed by
  direct contrast on identical hardware.
- If C4 collapses here too → it is *not* an AICR misconfiguration, and the
  paper's hardware-limit model goes back in play. Report prominently and stop.

Then **C5 vs C4** is the fingerprint: if C5 (RO off) collapses while C4 (RO on)
is healthy, relaxed ordering *is* the lever and AICR's bug is "RO never reaches
the wire" — narrowing the search to the NIC `DevCtl.RlxdOrd` enable bit,
ConnectX `PCI_WR_ORDERING`, or the PCIe switch stripping the attribute.

Independently, **Part B1** (`EnablePCIERelaxedOrderingMode`) either exonerates
the NVreg hypothesis (value `0` *and* C4 healthy → it is the vendor default and
discriminates nothing) or revives it (value `1`).

## Prerequisites verified

- CUDA modules present (`cuda/12.9.1`, `cuda/13.0.1`, `cuda/13.1.0`) → Part A
  should run rather than be skipped.
- `ib_write_bw` is not on the login node but is present on the compute nodes
  (confirmed by the earlier perftest work recorded in `../notes.md`).
- The script does its own unit conversion and prints a `CONVERTED RESULTS` table
  and an `AUTOMATIC VERDICT` (thresholds: healthy >= 42, collapsed < 38
  GB/s/dir). We sanity-check that against README section 5 rather than trusting
  it blindly.

## Node selection — one deviation

`node5500` is **reserved for another user (`rres_joohye`) until 2026-08-12**, so
the run uses **node5501 + node5502**. Both are idle, both are B200, and both are
already characterised by our earlier NCCL work. README section 2 names
node5500/5501/5502 as the known-good set, so this stays in scope.

## Steps

1. `sbatch -p mit_testing -w node5501,node5502 run-engaging-check.sh`
   (submitted from this directory so `eng-gdr-<jobid>.out` lands here).
2. If the Part C tests fail with `Failed status 12` / `transport retry exceeded`,
   resubmit with `NIC_FORCE=mlx5_4` — our known-good NDR rail from `../notes.md`.
3. When the job completes, `analyze-engaging-check.py` runs automatically (chained
   as a SLURM dependency, so it survives logout) and writes `RESULTS.md`:
   converted GB/s-per-direction table, the three verdicts, and a Part B diff
   against `aicr-reference/gdr-root-b0031-317105`.
4. Report the raw `eng-gdr-<jobid>.out` together with the written conclusion.

## Stated prior, so it cannot masquerade as a finding

We already expect Engaging to look healthy: our 2-node NCCL sendrecv reaches
49.7 GB/s (~99% of the 50 GB/s NDR line rate), and AllGather implies ~47.9 GB/s
per rail. That is a **prior, not a result**. If C4 comes back collapsed, it gets
reported prominently as overturning the diagnosis — not explained away. Likewise
anything that fails or is skipped is reported, not quietly dropped.

## Constraints honoured

Read-only measurement. No system setting, module parameter, or firmware value is
changed on either cluster. Anything that looks misconfigured is reported, not
fixed.
