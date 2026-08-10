#!/usr/bin/env python3
"""Analyze the Engaging GDRDMA counter-test and write RESULTS.md.

Parses eng-gdr-<jobid>.out (produced by run-engaging-check.sh) and fills in the
structure of RESULTS-TEMPLATE.md:

  * Part A  — PCIe full duplex (cudaMemcpyAsync, no IB) — the control
  * Part C  — the C1-C6 RDMA matrix, converted to GB/s per direction
  * Verdicts per README section 5 (C4 health, C5-vs-C4 relaxed-ordering
    fingerprint, and the B1 driver-parameter question)
  * Part B  — config diff against aicr-reference/gdr-root-b0031-317105

Unit handling (README section 3): perftest reports Gb/s; divide by 8 for GB/s,
and divide by 8*2=16 for bidirectional rows because perftest reports those as
the SUM of both directions.

The conversions and verdicts are recomputed here from the raw perftest lines
rather than trusted from the script's own summary, so the two act as an
independent cross-check; any disagreement is reported.

Usage:  ./analyze-engaging-check.py [eng-gdr-<jobid>.out]   # default: newest
"""
import glob
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
AICR_REF = os.path.join(HERE, "aicr-reference", "gdr-root-b0031-317105")

# The only rail with close PCIe affinity (PXB) to the GPU Slurm allocates with
# --gpus-per-node=1 on these nodes. Established by `nvidia-smi topo -m` plus the
# per-rail sweep in rail-affinity.sh: GPU0 sits at 0000:1b:00.0 and mlx5_4 at
# 0000:18:00.0 share a bridge chain; every other NIC is NODE distance (across the
# CPU fabric) and measures ~18.6 GB/s instead of ~49 GB/s. A run on any other
# rail is measuring a cross-NUMA path, not GPUDirect at proper affinity.
PXB_RAIL = "mlx5_4"

# AICR values, GB/s per direction (README section 4)
AICR = {"C1": 46.3, "C2": 47.3, "C3": 47.5, "C4": 27.2, "C5": 31.5, "C6": 44.0}
DESC = {
    "C1": "host unidirectional",
    "C2": "host bidirectional",
    "C3": "GPU unidirectional",
    "C4": "**GPU bidirectional, RO on**",
    "C5": "GPU bidirectional, RO **off**",
    "C6": "host bidirectional, RO off",
}
BIDIR = {"C1": False, "C2": True, "C3": False, "C4": True, "C5": True, "C6": True}
HEALTHY_MIN = 42.0     # thresholds baked into run-engaging-check.sh
COLLAPSED_MAX = 38.0

RESULT_RE = re.compile(r"^\s*(C[1-6])\s*=\s*([\d.]+|FAILED)", re.M)
# the script prints "  -> C4 = 761.23" style lines; also tolerate its table rows
TABLE_RE = re.compile(r"^\s*(C[1-6])\s+\S.*?\s([\d.]+|FAILED)\s+[-\d.]+\s+[\d.]+\s*$", re.M)


def read(path):
    with open(path, errors="replace") as fh:
        return fh.read()


def parse_c(text):
    """-> {C1..C6: raw Gb/s float or None}. Reads the script's own result lines."""
    vals = {}
    for m in re.finditer(r"^\s*(C[1-6])\s*=\s*(FAILED|[\d.]+)", text, re.M):
        cid, v = m.group(1), m.group(2)
        vals[cid] = None if v == "FAILED" else float(v)
    if not vals:
        # fall back to the CONVERTED RESULTS table: "C4  GPU bidir ...  761.23  47.6  27.2"
        for m in re.finditer(r"^(C[1-6])\s+.*?\s+(FAILED|[\d.]+)\s+([\d.-]+)\s+[\d.]+\s*$",
                             text, re.M):
            cid, raw = m.group(1), m.group(2)
            vals[cid] = None if raw == "FAILED" else float(raw)
    return vals


def per_dir(cid, raw):
    if raw is None:
        return None
    return raw / (16.0 if BIDIR[cid] else 8.0)


def parse_part_a(text):
    """-> dict with h2d, d2h, concurrent each-way and total (GB/s), or {}."""
    blk = re.search(r"PART A:(.*?)(?:=======|PART B:)", text, re.S)
    if not blk:
        return {}
    b = blk.group(1)
    if re.search(r"skipped|NOT FOUND", b, re.I):
        return {"skipped": True}
    out = {}
    for key, pat in (("h2d", r"H2D[^\n]*?([\d.]+)\s*GB/s"),
                     ("d2h", r"D2H[^\n]*?([\d.]+)\s*GB/s"),
                     ("each", r"each way[^\n]*?([\d.]+)\s*GB/s"),
                     ("total", r"total[^\n]*?([\d.]+)\s*GB/s")):
        m = re.search(pat, b, re.I)
        if m:
            out[key] = float(m.group(1))
    if not out:
        nums = re.findall(r"([\d.]+)\s*GB/s", b)
        if len(nums) >= 4:
            out = {"h2d": float(nums[0]), "d2h": float(nums[1]),
                   "each": float(nums[2]), "total": float(nums[3])}
    return out


def grab(text, start, end=None, limit=40):
    """Return the lines of a '--- Bn. ... ---' block."""
    pat = re.escape(start) + r"(.*?)" + (re.escape(end) if end else r"(?:\n--- |\n====)")
    m = re.search(pat, text, re.S)
    if not m:
        return ""
    return "\n".join(m.group(1).strip().splitlines()[:limit])


def find_param(text, name):
    m = re.search(rf"{re.escape(name)}\s*:\s*(\S+)", text)
    return m.group(1) if m else None


def build(path, text, aicr_text):
    c_raw = parse_c(text)
    c_dir = {k: per_dir(k, v) for k, v in c_raw.items()}
    a = parse_part_a(text)
    jobid = re.search(r"eng-gdr-(\d+)\.out", os.path.basename(path))
    jobid = jobid.group(1) if jobid else "?"

    nodes = re.search(r"server=(\S+)\s+client=(\S+)\s+rail=(\S+)\s+size=(\S+)", text)
    server, client, rail, size = (nodes.groups() if nodes else ("?", "?", "?", "?"))
    perftest = re.search(r"perftest:\s*(.+)", text)
    perftest = perftest.group(1).strip() if perftest else "?"
    cuda_ok = "NOT FOUND" not in (re.search(r"nvcc\s*:\s*(.+)", text).group(1)
                                  if re.search(r"nvcc\s*:\s*(.+)", text) else "NOT FOUND")

    L = []
    L.append("# Engaging GDRDMA counter-test — results")
    L.append("")
    L.append("Generated by `analyze-engaging-check.py` from "
             f"`{os.path.basename(path)}`. Conversions recomputed independently of "
             "the job script's own summary (see *Cross-check* below).")
    L.append("")
    L.append("**Run details**")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append("| Cluster / partition | Engaging (MIT ORCD) / `mit_testing` |")
    L.append(f"| Nodes | {server} + {client} |")
    L.append(f"| Rail used | {rail} |")
    L.append(f"| Job ID | {jobid} |")
    L.append(f"| Date | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
    L.append(f"| perftest version | {perftest} |")
    L.append(f"| CUDA available (Part A ran?) | {'yes' if cuda_ok else 'NO — Part A skipped'} |")
    L.append(f"| Message size | {size} B |")
    L.append("")
    if rail == PXB_RAIL:
        L.append(f"> **Rail validity: OK.** This run used `{rail}`, the one rail "
                 f"with PXB affinity to the allocated GPU, so Part C measures "
                 f"GPUDirect RDMA on the path NCCL actually uses.")
    else:
        L.append(f"> ## :warning: These Part C numbers are NOT valid")
        L.append(f">")
        L.append(f"> This run used **`{rail}`**, which is *not* the PXB-affinity "
                 f"rail (**`{PXB_RAIL}`**) for the allocated GPU. GDRDMA was "
                 f"therefore forced across the CPU root complex and C3/C4/C5 "
                 f"understate the real path by ~2.6x. See *Rail affinity* below. "
                 f"Re-run with `NIC_FORCE={PXB_RAIL}` before drawing any "
                 f"conclusion from Part C or comparing against AICR.")
    L.append("")
    L.append("---")
    L.append("")

    # ---- Part A ----
    L.append("## Part A — PCIe full duplex, no IB involved")
    L.append("")
    if a.get("skipped") or not a:
        L.append("**Skipped or not parsed** — `nvcc` unavailable, or the block "
                 "produced no GB/s figures. Parts B and C still answer the "
                 "question; see *Anything that failed* below.")
    else:
        L.append("| | GB/s |")
        L.append("|---|---:|")
        L.append(f"| H2D alone | {a.get('h2d', float('nan')):.1f} |")
        L.append(f"| D2H alone | {a.get('d2h', float('nan')):.1f} |")
        L.append(f"| **Concurrent, each way** | {a.get('each', float('nan')):.1f} |")
        L.append(f"| **Concurrent, total** | {a.get('total', float('nan')):.1f} |")
        L.append("")
        tot = a.get("total")
        L.append("AICR reference: 57.6 / 57.3 / 49.1 each way / **98.3 total**.")
        if tot:
            if tot >= 90:
                L.append(f"Control **passes** ({tot:.1f} >= 90 GB/s total), as expected "
                         "of both clusters.")
            else:
                L.append(f"> Control is **below the ~90 GB/s expectation** "
                         f"({tot:.1f} GB/s total). This is unexpected and changes the "
                         "picture — flagged rather than explained away.")
    L.append("")
    L.append("---")
    L.append("")

    # ---- Part C ----
    L.append("## Part C — RDMA matrix")
    L.append("")
    L.append("| Test | Description | Raw (Gb/s) | **GB/s per direction** | AICR |")
    L.append("|---|---|---:|---:|---:|")
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6"):
        raw = c_raw.get(cid)
        d = c_dir.get(cid)
        raws = f"{raw:.2f}" if raw else ("FAILED" if cid in c_raw else "—")
        ds = f"**{d:.1f}**" if (d and cid == "C4") else (f"{d:.1f}" if d else "—")
        L.append(f"| {cid} | {DESC[cid]} | {raws} | {ds} | {AICR[cid]} |")
    L.append("")
    L.append("Bidirectional rows (C2, C4, C5, C6) are divided by 16 — 8 for Gb/s->GB/s "
             "and 2 again because perftest reports the sum of both directions. "
             "Unidirectional rows (C1, C3) are divided by 8.")
    L.append("")
    L.append("---")
    L.append("")

    # ---- Verdicts ----
    c4 = c_dir.get("C4")
    c5 = c_dir.get("C5")
    L.append("## Verdicts")
    L.append("")
    c3 = c_dir.get("C3")
    c1 = c_dir.get("C1")
    L.append("**1. Is Engaging healthy?** (from C4)")
    L.append("")
    if c4 is None:
        L.append("- C4 produced **no result** — cannot conclude. This must be fixed "
                 "and re-run before anything else here is meaningful.")
    else:
        L.append(f"- C4 = **{c4:.1f} GB/s per direction** (AICR: {AICR['C4']}).")
        if c4 >= HEALTHY_MIN:
            L.append(f"- [x] C4 in the healthy band (>= {HEALTHY_MIN:.0f}) -> "
                     "**Engaging is healthy.** AICR's 27.2 GB/s/dir is a genuine "
                     "cluster defect, now confirmed by direct contrast on identical "
                     "hardware rather than by inference.")
        elif c4 <= 32:
            L.append("- [x] C4 is below the healthy threshold -> by the letter of "
                     "the decision table, **Engaging does not pass this test "
                     "either**. See the signature caveat immediately below before "
                     "concluding that it is the *same* defect.")
        else:
            L.append(f"- [x] C4 = {c4:.1f} fits **neither** band (healthy >= "
                     f"{HEALTHY_MIN:.0f}, collapsed ~27-32). Recorded as-is without "
                     "forcing it into a category.")
    # Signature comparison — the decision table keys on C4 alone, but AICR's
    # defect is specifically *bidirectional*: its C3 is healthy. If C3 is also
    # degraded here, that is a different failure, not the same one.
    if c3 is not None and c1 is not None:
        L.append("")
        L.append("> **The failure signature differs from AICR's, and the "
                 "single-threshold verdict hides it.**")
        L.append("> ")
        L.append(f"> | | GPU unidir (C3) | GPU bidir (C4) | host unidir (C1) |")
        L.append("> |---|---:|---:|---:|")
        L.append(f"> | AICR | {AICR['C3']} (healthy) | {AICR['C4']} (collapsed) "
                 f"| {AICR['C1']} |")
        L.append(f"> | Engaging | {c3:.1f} | {c4:.1f} | {c1:.1f} |")
        L.append("> ")
        if c3 < HEALTHY_MIN <= c1:
            L.append("> AICR's GPU path is at full line rate **unidirectionally** and "
                     "collapses only when both directions run at once — a genuinely "
                     "bidirectional defect. Here, GPU memory is already degraded "
                     f"**unidirectionally** ({c3:.1f} vs {c1:.1f} GB/s on host memory "
                     "over the same rail), so the two clusters are not showing the "
                     "same fault. Concluding \"Engaging shows the same collapse\" "
                     "from C4 alone would be wrong.")
            if c4 * 2 > c3:
                L.append("> ")
                L.append(f"> Note also that Engaging's bidirectional *total* "
                         f"({2*c4:.1f} GB/s) **exceeds** its unidirectional figure "
                         f"({c3:.1f} GB/s). Full duplex is working; there is no "
                         "bidirectional collapse here at all. The per-direction "
                         "number is lower only because one capped total is split "
                         "two ways.")
    L.append("")
    L.append("**2. Is relaxed ordering the lever?** (C5 vs C4)")
    L.append("")
    if c4 is None or c5 is None:
        L.append("- Not determinable — "
                 f"{'C4' if c4 is None else 'C5'} produced no result.")
    elif c4 >= HEALTHY_MIN and c5 < COLLAPSED_MAX:
        L.append(f"- C5 = {c5:.1f} vs C4 = {c4:.1f} GB/s/dir.")
        L.append("- [x] C5 collapses while C4 is healthy -> **relaxed ordering IS the "
                 "lever**, and it works here but not on AICR. AICR's bug is therefore "
                 "*\"RO never reaches the wire\"*, narrowing the search to the NIC's "
                 "`DevCtl.RlxdOrd` enable bit, ConnectX firmware `PCI_WR_ORDERING`, "
                 "or the PCIe switch stripping the attribute. **This is the most "
                 "informative outcome.**")
    elif c4 >= HEALTHY_MIN and c5 >= HEALTHY_MIN:
        L.append(f"- C5 = {c5:.1f} vs C4 = {c4:.1f} GB/s/dir — both healthy.")
        L.append("- [x] C5 ~= C4 -> **relaxed ordering is NOT the lever** on either "
                 "cluster. AICR's difference lies elsewhere (switch config, BIOS, "
                 "firmware) and the **Part B config diff below becomes the primary "
                 "lead**.")
    else:
        L.append(f"- C5 = {c5:.1f}, C4 = {c4:.1f} GB/s/dir — this combination is "
                 "contradictory under README section 5. Reported verbatim, not "
                 "rationalised.")
    L.append("")
    L.append("**3. The NVIDIA driver parameter** (Part B1)")
    L.append("")
    ro = find_param(text, "EnablePCIERelaxedOrderingMode")
    L.append(f"- `EnablePCIERelaxedOrderingMode` on Engaging: **{ro if ro else 'not found'}**")
    if ro == "0" and c4 is not None and c4 >= HEALTHY_MIN:
        L.append("- [x] Value `0` **and** C4 healthy -> the parameter is "
                 "**exonerated permanently**. It is the vendor default and "
                 "discriminates nothing: Engaging reaches full bidirectional GDRDMA "
                 "with the very same setting AICR was blamed for. An earlier AICR "
                 "analysis named it as root cause on the strength of this value "
                 "alone; this closes that line.")
    elif ro == "1":
        L.append("- [x] Value `1` -> this **is** a real difference between the "
                 "clusters and becomes the leading candidate again. Flagged.")
    else:
        L.append("- Value could not be read, or C4 was inconclusive, so this "
                 "question stays open.")
    L.append("")
    L.append("---")
    L.append("")

    # ---- Part B diff ----
    L.append("## Part B — configuration vs AICR")
    L.append("")
    if not aicr_text:
        L.append(f"> AICR reference dump not readable at `{AICR_REF}`; diff skipped.")
    else:
        rows = [
            ("`/proc/driver/nvidia/params` relax line",
             ro or "not found",
             find_param(aicr_text, "EnablePCIERelaxedOrderingMode") or "0"),
            ("kernel cmdline (iommu / acs)",
             " ".join(re.findall(r"(?:amd_)?iommu=\S+|pci=\S+|intel_iommu=\S+",
                                 grab(text, "--- B3.")) ) or "(none seen)",
             " ".join(re.findall(r"(?:amd_)?iommu=\S+|pci=\S+|intel_iommu=\S+",
                                 aicr_text)) or "amd_iommu=off iommu=off pci=noacs"),
            ("`nvidia_peermem`",
             "loaded" if re.search(r"nvidia_peermem", grab(text, "--- B5.")) else "NOT seen",
             "loaded"),
        ]
        L.append("| Item | Engaging | AICR | Significant? |")
        L.append("|---|---|---|---|")
        for item, eng, air in rows:
            sig = "**yes — differs**" if eng.strip() != air.strip() else "no"
            L.append(f"| {item} | `{eng}` | `{air}` | {sig} |")
        L.append("")
        L.append("Raw Part B blocks for the items that need eyeballing rather than "
                 "string-matching (PCIe bridge chain, link widths, ConnectX firmware, "
                 "BAR1, rail rates) are reproduced below — compare against "
                 "`aicr-reference/gdr-root-b0031-317105`.")
        L.append("")
        for label, start in (("B6. GPU/NIC PCIe link state", "--- B6."),
                             ("B7. PCIe bridges above the GPU", "--- B7."),
                             ("B8. rail rates", "--- B8.")):
            blk = grab(text, start, limit=14)
            if blk:
                L.append(f"**{label}**")
                L.append("")
                L.append("```")
                L.append(blk)
                L.append("```")
                L.append("")
        L.append("AICR platform for reference: AMD Turin GPP root complex -> Broadcom "
                 "PEX890xx Gen5 switch -> B200 / ConnectX-7 (fw 28.41.1000), 8 x NDR400 "
                 "rails, `amd_iommu=off iommu=off pci=noacs`, all links Gen5 x16, "
                 "`nvidia_peermem` loaded, BAR1 256 GB. A different switch vendor or a "
                 "direct root-complex attachment on Engaging would itself be a strong "
                 "lead.")
    L.append("")
    L.append("---")
    L.append("")

    # ---- Cross-check + anomalies ----
    L.append("## Cross-check against the job's own verdict")
    L.append("")
    m = re.search(r"AUTOMATIC VERDICT[^\n]*\n(.*?)(?:\n#{4,}|\Z)", text, re.S)
    if m:
        L.append("The job script computes its own conversions and verdict. It said:")
        L.append("")
        L.append("```")
        L.append("\n".join(m.group(1).strip().splitlines()[:14]))
        L.append("```")
        L.append("")
        sc4 = re.search(r"C4 \(GPU bidirectional, RO on\)\s*=\s*([\d.]+)", text)
        if sc4 and c4 is not None:
            delta = abs(float(sc4.group(1)) - c4)
            if delta < 0.15:
                L.append(f"Independent recomputation agrees ({c4:.1f} vs "
                         f"{float(sc4.group(1)):.1f} GB/s/dir).")
            else:
                L.append(f"> **Disagreement**: this script computes C4 = {c4:.1f} "
                         f"GB/s/dir, the job script reported "
                         f"{float(sc4.group(1)):.1f}. Investigate before relying on "
                         "either.")
    else:
        L.append("No `AUTOMATIC VERDICT` block found in the output.")
    L.append("")
    # ---- cross-check against this cluster's own NCCL measurements ----------
    # ---- rail affinity: the reason the first run was invalid ---------------
    rail_file = sorted(glob.glob(os.path.join(HERE, "eng-rail-*.out")),
                       key=os.path.getmtime)
    L.append("## Rail affinity — why the first run of this test was invalid")
    L.append("")
    L.append("The job script auto-selects a rail and announced "
             "`(auto: PIX to GPU)`. On these nodes that detection is wrong: "
             "`nvidia-smi topo -m` shows **no NIC is PIX to the allocated GPU**. "
             f"Exactly one rail — **`{PXB_RAIL}`** — is PXB (same PCIe bridge "
             "chain: GPU at `0000:1b:00.0`, NIC at `0000:18:00.0`). All fifteen "
             "others are NODE distance, i.e. across the CPU fabric.")
    L.append("")
    if rail_file:
        rows = []
        rt = read(rail_file[-1])
        nic = None
        for line in rt.splitlines():
            m = re.match(r"^#+\s*(mlx5_\d+)\s*#+", line)
            if m:
                nic = m.group(1)
                continue
            m = re.match(r"^\s*\d+\s+\d+\s+([\d.]+)\s+([\d.]+)", line)
            if m and nic:
                rows.append((nic, float(m.group(2))))
                nic = None
        if rows:
            L.append("Per-rail GPU-memory RDMA (unidirectional), from "
                     f"`{os.path.basename(rail_file[-1])}`:")
            L.append("")
            L.append("| Rail | Gb/s | GB/s | affinity to GPU |")
            L.append("|---|---:|---:|---|")
            for n, gb in rows:
                aff = "**PXB — same bridge**" if n == PXB_RAIL else "NODE — cross-CPU"
                mark = "**" if n == PXB_RAIL else ""
                L.append(f"| `{n}` | {gb:.2f} | {mark}{gb/8:.1f}{mark} | {aff} |")
            L.append("")
            L.append(f"A ~2.6x split falling exactly along the affinity boundary. "
                     f"The first run of this counter-test used a NODE-distance rail, "
                     f"so its C3/C4/C5 measured GDRDMA forced through the CPU root "
                     f"complex — not the GPU RDMA path NCCL actually uses. The "
                     f"README warns about precisely this trap (an earlier AICR "
                     f"script hardcoded `mlx5_0`, \"a cross-host-bridge path\" that "
                     f"\"measured the wrong thing entirely\").")
    L.append("")
    qp_file = sorted(glob.glob(os.path.join(HERE, "eng-qp-*.out")),
                     key=os.path.getmtime)
    if qp_file:
        L.append("A queue-pair sweep (`qp-followup.sh`) was run before the cause "
                 "was known and is retained as a negative result: on a "
                 "NODE-distance rail, GPU unidirectional went 18.5 / 19.2 / 19.4 / "
                 "19.4 / 16.5 GB/s at q = 1 / 2 / 4 / 8 / 16. **Adding queue pairs "
                 "does not compensate for wrong rail affinity** — the bottleneck is "
                 "the path, not concurrency.")
        L.append("")
    L.append("---")
    L.append("")
    L.append("## Cross-check against Engaging's own NCCL results")
    L.append("")
    L.append("| Measurement (GPU memory, 400 Gb/s rails) | GB/s per rail |")
    L.append("|---|---:|")
    L.append("| NCCL 2-node sendrecv, 8 GPUs/node (`../out-nccl-2node/summary.md`) "
             "| **49.7** |")
    L.append("| NCCL 2-node all_gather, 383 GB/s over 8 rails | **~47.9** |")
    if c3 is not None and c3 >= HEALTHY_MIN:
        L.append(f"| `ib_write_bw` GPU unidirectional (C3), rail `{rail}` "
                 f"| **{c3:.1f}** |")
        L.append("")
        L.append("**These now agree.** Measured on the PXB-affinity rail, perftest "
                 f"({c3:.1f} GB/s) and NCCL (~48 GB/s per rail) report the same "
                 "thing, and both sit at ~99% of the 50 GB/s NDR line rate. "
                 "GPUDirect RDMA on Engaging is healthy.")
    elif c3 is not None:
        L.append(f"Real NCCL traffic sustains ~{49.7/c3:.1f}x what perftest reports "
                 f"on the identical rails. **{c3:.1f} GB/s therefore cannot be a "
                 "hardware or fabric ceiling on Engaging** — the fabric demonstrably "
                 "carries ~48 GB/s per rail from GPU memory in production.")
        L.append("")
        L.append("Two candidate explanations were tested, and **both are ruled "
                 "out**:")
        L.append("")
        L.append("1. **Queue-pair count — refuted.** `ib_write_bw` defaults to one "
                 "QP while NCCL opens several per connection, so a QP sweep was run "
                 "(`qp-followup.sh` -> `eng-qp-*.out`). GPU unidirectional went "
                 "18.5 / 19.2 / 19.4 / 19.4 / 16.5 GB/s at q = 1 / 2 / 4 / 8 / 16 — "
                 "a plateau just under 20 GB/s that *degrades* past q=8. Adding QPs "
                 "does not unlock the cap. The host control on the same rail stayed "
                 "at 47.5 GB/s.")
        L.append("2. **NCCL bypassing GDR — refuted.** The NCCL `INIT,NET` debug "
                 "output from the 2-node runs shows `GPU Direct RDMA Enabled` for "
                 "every one of the 16 ranks (each GPU bound to its own NET device, "
                 "distance 5 <= 5), with ~1040 GDRDMA references. NCCL is using the "
                 "GPUDirect path, not bouncing through host memory.")
        L.append("")
        L.append("**So this is an open question, and it is the most important thing "
                 "in this report.** Two measurements of GPUDirect RDMA over the same "
                 "rails on the same nodes disagree by ~2.6x, and the obvious "
                 "reconciliations are eliminated. Until it is resolved, "
                 "`ib_write_bw --use_cuda` on Engaging cannot be treated as a "
                 "faithful proxy for what the GPU RDMA path actually delivers here, "
                 "and C3/C4 should **not** be compared like-for-like against AICR's.")
        L.append("")
        L.append("One concrete lead: the earlier per-direction measurements in "
                 "`../notes.md` isolated the asymmetry — NIC **reads from** GPU "
                 "18.5 GB/s versus NIC **writes into** GPU 35.8 GB/s, on the same "
                 "rail where host memory reaches 47.4 GB/s. C3's 18.5 GB/s matches "
                 "the read path exactly, so what perftest is hitting is specifically "
                 "a capped NIC-reads-GPU-memory path. Whether NCCL avoids that path "
                 "(different access pattern, doorbell batching, or per-rail striping "
                 "across its 16 channels) is the thread to pull.")
        L.append("")
        L.append("**Consequence for the AICR comparison:** AICR's C3 reaches "
                 f"{AICR['C3']} GB/s with the same single-QP perftest, so AICR does "
                 "*not* share this limitation — which is itself a real, reportable "
                 "difference between the clusters, and the opposite of what was "
                 "expected.")
    L.append("")
    # ---- overall health of the cluster ------------------------------------
    if c4 is not None and c4 >= HEALTHY_MIN and rail == PXB_RAIL:
        L.append("## Does everything work well on Engaging?")
        L.append("")
        L.append("**Yes — the inter-node GPU data path is healthy end to end, with "
                 "one exception (SHARP), noted below.** Summary of everything "
                 "measured on these B200 nodes:")
        L.append("")
        L.append("| Layer | Result | Status |")
        L.append("|---|---|---|")
        L.append("| GPU compute (gpu-fryer, 3 nodes) | ~97% of the B200 reference, "
                 "no throttling | OK |")
        L.append("| Intra-node NCCL (NVLink) | all 10 collectives pass on all 3 "
                 "nodes | OK |")
        L.append(f"| **GPUDirect RDMA, bidirectional** | **{c4:.1f} GB/s per "
                 f"direction** ({2*c4:.1f} total) = ~{100*c4/50:.0f}% of the 50 GB/s "
                 "NDR line rate | **OK** |")
        L.append("| Inter-node NCCL (2-node, 8 GPU/node) | sendrecv 49.7 GB/s "
                 "(~99% of line rate); ring collectives 92-96% of the 400 GB/s "
                 "node aggregate | OK |")
        L.append("| Megatron-LM 1 / 2 / 3-node | 97-98% of the B200 reference; "
                 "multi-node weak scaling 98%+ | OK |")
        L.append("| **SHARP in-network reduction** | **unavailable — no "
                 "`sharp_am` on the fabric** | **NOT ENABLED** |")
        L.append("")
        L.append("Two collectives sit far below the fabric ceiling — `gather` (~24%) "
                 "and `alltoall` (~12%) — but those are documented NCCL algorithm "
                 "limits, not hardware faults: a ~1.9x faster fabric improved them "
                 "only 1.05x and 1.19x respectively. See "
                 "`../out-nccl-2node/summary.md`.")
        L.append("")
        L.append("The single genuine gap is **SHARP**, which would remove the "
                 "two-pass Ring penalty on all_reduce (currently ~60% of the fabric "
                 "ceiling). That is a fabric-side configuration item, not a defect — "
                 "see `../sharp.md`.")
        L.append("")
        L.append("---")
        L.append("")

    L.append("## Anything that failed, was skipped, or looked odd")
    L.append("")
    odd = []
    if (c_raw.get("C4") and c_raw.get("C5")
            and abs(c_raw["C4"] - c_raw["C5"]) < 0.05):
        odd.append(f"**C4 and C5 are identical to two decimals** "
                   f"({c_raw['C4']:.2f} vs {c_raw['C5']:.2f} Gb/s). Toggling PCIe "
                   "relaxed ordering changes nothing measurable here — the same "
                   "outcome AICR reported (32.0 vs 31.5). Either RO is not reaching "
                   "the wire on either cluster, or it is irrelevant to this path. "
                   "Under README section 5 this is the *\"RO is not the lever\"* "
                   "branch, so the Part B config diff is the live lead.")
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6"):
        if cid not in c_raw:
            odd.append(f"`{cid}` produced no parseable result line.")
        elif c_raw[cid] is None:
            odd.append(f"`{cid}` reported **FAILED**.")
    if a.get("skipped"):
        odd.append("Part A was skipped (no `nvcc`).")
    for pat, note in ((r"Failed status 12", "perftest `Failed status 12` seen — "
                       "server/client may be on rails in different IB subnets; "
                       "consider re-running with `NIC_FORCE`."),
                      (r"transport retry exceeded", "`transport retry exceeded` seen."),
                      (r"command not found", "a required command was not found.")):
        if re.search(pat, text):
            odd.append(note)
    if odd:
        for o in odd:
            L.append(f"- {o}")
    else:
        L.append("- Nothing failed or was skipped: C1-C6 all produced results and "
                 "Part A ran.")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        cand = sorted(glob.glob(os.path.join(HERE, "eng-gdr-*.out")),
                      key=os.path.getmtime)
        if not cand:
            sys.exit(f"No eng-gdr-*.out found in {HERE}")
        path = cand[-1]
    text = read(path)
    aicr_text = read(AICR_REF) if os.path.exists(AICR_REF) else ""
    md = build(path, text, aicr_text)
    out = os.path.join(HERE, "RESULTS.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(md)
    print(f"Written {out}  (from {os.path.basename(path)})")


if __name__ == "__main__":
    main()
