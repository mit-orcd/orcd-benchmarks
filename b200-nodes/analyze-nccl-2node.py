#!/usr/bin/env python3
"""Analyze nccl-tests sendrecv_perf 2-node output(s) and write a Markdown summary.

Scans every *.out in out-nccl-2node/, keeps the newest parseable run per node-pair
(crashed runs with no data table are skipped), and writes out-nccl-2node/summary.md:
a run overview, a bus-bandwidth-vs-message-size table (out-of-place + in-place),
and a comparison against the MIT aicr-benchmarks B200 2-node reference (Table 2 of
results_b200.md) — the same reference file used by the 1-node analysis.

Inter-node sendrecv is bounded by the GPU's GDRDMA bidirectional per-pair ceiling
(~26.7 GB/s on B200), not NVLink. The *_perf binaries print one row per message
size; `busbw` (GB/s) is the figure of merit (for sendrecv it equals algbw).

Usage:
    ./analyze-nccl-2node.py [file ...]   # default: newest parseable *.out per pair
"""
import glob
import os
import re
import sys
from datetime import datetime

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out-nccl-2node")
REFERENCE_FILE = (
    "/home/shaohao/data022/aicr-benchmarks/Benchmark_WG/nccl-tests/results_b200.md"
)

CONFIG_RE = re.compile(
    r"nThread\s+(\d+)\s+nGpus\s+(\d+)\s+minBytes\s+(\d+)\s+maxBytes\s+(\d+).*"
    r"warmup iters:\s*(\d+)\s+iters:\s*(\d+)"
)
DEVICE_RE = re.compile(
    r"Rank\s+(\d+).*on\s+(\S+)\s+device\s+(\d+)\s+\[[^\]]+\]\s+(.+?)\s*$"
)
DATA_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(-?\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s*$"
)
AVG_RE = re.compile(r"Avg bus bandwidth\s*:\s*([\d.]+)")
OOB_RE = re.compile(r"Out of bounds values\s*:\s*(\d+)\s*(\w+)")
PROG_RE = re.compile(r"%+\s*(\w+)\s*%+")
# Table 2 sendrecv row: | sendrecv | 26.6 | 26.6 | 26.7 (per-pair bidir) | **~100%** | ... |
REF_ROW_RE = re.compile(
    r"^\|\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\**~?([\d.]+)[^|]*\|"
    r"\s*\**~?([\d.]+)\s*%"
)


def fmt_size(n):
    for div, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if n >= div and n % div == 0:
            return f"{n // div} {unit}"
    for div, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def parse_reference(path, collective):
    """2-node B200 reference row for `collective` from Table 2 of results_b200.md."""
    try:
        in_table2 = False
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("## Table 2"):
                    in_table2 = True
                    continue
                if in_table2 and s.startswith("## ") and not s.startswith("## Table 2"):
                    break
                if not in_table2:
                    continue
                m = REF_ROW_RE.match(s)
                if m and m.group(1).lower() == collective.lower():
                    return {
                        "algbw": float(m.group(2)),
                        "busbw": float(m.group(3)),
                        "gdrdma_max": float(m.group(4)),
                        "pct": float(m.group(5)),
                    }
    except FileNotFoundError:
        return None
    return None


def parse_file(path):
    cfg = None
    devices, rows = [], []
    avg = oob = program = None
    with open(path) as fh:
        for line in fh:
            m = PROG_RE.search(line)
            if m and m.group(1).endswith("_perf"):
                program = m.group(1)
            m = CONFIG_RE.search(line)
            if m:
                cfg = {
                    "nThread": int(m.group(1)), "nGpus": int(m.group(2)),
                    "minBytes": int(m.group(3)), "maxBytes": int(m.group(4)),
                    "warmup": int(m.group(5)), "iters": int(m.group(6)),
                }
                continue
            m = DEVICE_RE.search(line)
            if m:
                devices.append({"rank": int(m.group(1)), "node": m.group(2),
                                "name": m.group(4)})
                continue
            m = DATA_RE.match(line)
            if m:
                rows.append({
                    "size": int(m.group(1)),
                    "oop_time": float(m.group(6)), "oop_busbw": float(m.group(8)),
                    "oop_wrong": m.group(9),
                    "ip_time": float(m.group(10)), "ip_busbw": float(m.group(12)),
                    "ip_wrong": m.group(13),
                })
                continue
            m = AVG_RE.search(line)
            if m:
                avg = float(m.group(1))
            m = OOB_RE.search(line)
            if m:
                oob = (int(m.group(1)), m.group(2))
    if not rows:
        return None
    node_pair = "+".join(sorted({d["node"] for d in devices})) or os.path.basename(path)
    gpu_name = devices[0]["name"] if devices else "?"
    big = max(rows, key=lambda r: r["size"])
    converged = max(big["oop_busbw"], big["ip_busbw"])
    peak = max(max(r["oop_busbw"], r["ip_busbw"]) for r in rows)
    peak_size = max(rows, key=lambda r: max(r["oop_busbw"], r["ip_busbw"]))["size"]
    ok = all(r["oop_wrong"] in ("0", "N/A") and r["ip_wrong"] in ("0", "N/A")
             for r in rows) and (oob is None or oob[0] == 0)
    return {
        "path": path, "pair": node_pair, "gpu_name": gpu_name,
        "ngpu_per_rank": cfg["nGpus"] if cfg else 1, "nranks": len(devices),
        "program": program or "sendrecv_perf", "cfg": cfg, "rows": rows, "avg": avg,
        "converged": converged, "converged_size": big["size"],
        "peak": peak, "peak_size": peak_size, "ok": ok,
    }


def build(runs, reference):
    L = []
    collective = runs[0]["program"]
    cfg = runs[0]["cfg"]
    gpu = runs[0]["gpu_name"]
    ceiling = reference["gdrdma_max"] if reference else 26.7

    L.append(f"# nccl-tests ({collective.replace('_perf','')}) 2-node summary")
    L.append("")
    L.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- Collective: {collective}  (inter-node, InfiniBand + GPUDirect RDMA)")
    L.append(f"- Runs: {', '.join(r['pair'] for r in runs)} "
             f"(2 nodes x 1 {gpu}, one GPU per node)")
    if cfg:
        L.append(f"- Config: {cfg['nThread']} thread, "
                 f"{fmt_size(cfg['minBytes'])}-{fmt_size(cfg['maxBytes'])}, "
                 f"{cfg['warmup']} warmup + {cfg['iters']} iters")
    if reference:
        L.append(f"- Reference (MIT aicr-benchmarks, `results_b200.md` Table 2, "
                 f"b0029+b0030): sendrecv busbw **{reference['busbw']:.1f} GB/s** = "
                 f"{reference['pct']:.0f}% of the {ceiling:.1f} GB/s GDRDMA bidir "
                 f"per-pair ceiling")
    L.append("")

    # Run overview
    L.append("## Overview")
    L.append("")
    hdr = ("| Run | Avg busbw (GB/s) | Peak busbw (GB/s) | Converged (GB/s) "
           "| % of GDRDMA ceiling")
    sep = "|-----|-----------------:|------------------:|-----------------:|----------------:"
    if reference:
        hdr += " | % of reference"
        sep += "|-------------:"
    hdr += " | Correctness |"
    sep += "|---|"
    L.append(hdr)
    L.append(sep)
    for r in runs:
        row = (f"| {r['pair']} | {r['avg']:.1f} | {r['peak']:.1f} | {r['converged']:.1f} "
               f"| {100*r['converged']/ceiling:.0f}%")
        if reference:
            row += f" | {100*r['converged']/reference['busbw']:.0f}%"
        row += f" | {'PASS' if r['ok'] else 'FAIL'} |"
        L.append(row)
    if reference:
        L.append(f"| **reference (b0029+b0030)** | — | {reference['busbw']:.1f} | "
                 f"{reference['busbw']:.1f} | {reference['pct']:.0f}% | 100% | — |")
    L.append("")
    L.append("Converged = busbw at the largest message size "
             f"({fmt_size(runs[0]['converged_size'])}), best of out-of-place / in-place "
             "(matches the reference's methodology). The GDRDMA bidir per-pair ceiling "
             f"(~{ceiling:.1f} GB/s) is the B200 hardware limit for a single cross-node "
             "GPU pair: one PCIe Gen5 x16 DMA engine shared between simultaneous TX+RX.")
    L.append("")

    # Bandwidth vs size
    L.append("## Bus bandwidth vs message size (GB/s)")
    L.append("")
    for r in runs:
        if len(runs) > 1:
            L.append(f"### {r['pair']}")
            L.append("")
        L.append("| Message size | OOP time | OOP busbw | IP time | IP busbw |")
        L.append("|-------------:|---------:|----------:|--------:|---------:|")
        for row in r["rows"]:
            ot = (f"{row['oop_time']/1000:.2f} ms" if row["oop_time"] >= 1000
                  else f"{row['oop_time']:.1f} us")
            it = (f"{row['ip_time']/1000:.2f} ms" if row["ip_time"] >= 1000
                  else f"{row['ip_time']:.1f} us")
            L.append(f"| {fmt_size(row['size'])} | {ot} | {row['oop_busbw']:.1f} "
                     f"| {it} | {row['ip_busbw']:.1f} |")
        L.append("")
    L.append("OOP = out-of-place, IP = in-place. Bandwidth rises with message size and "
             "saturates once the per-pair GDRDMA DMA budget is the binding constraint.")
    L.append("")

    # Network fabric (from ibstat on node5500 / node5502, 2026-07-13)
    L.append("## Network fabric")
    L.append("")
    L.append("`ibstat` on **both node5500 and node5502** — the inter-node data path is "
             "**NDR (400 Gb/s), not HDR**:")
    L.append("")
    L.append("| NICs | Rate | Role |")
    L.append("|------|------|------|")
    L.append("| mlx5_4, 7, 8, 9, 10, 13, 14, 15 | **400 Gb/s (4X NDR)** | "
             "8 GPU compute rails (active) |")
    L.append("| mlx5_0, 1, 2, 3 | 100 Gb/s (2X HDR / HDR100) | secondary (storage/mgmt), active |")
    L.append("| mlx5_5, 6, 11, 12 | down (SDR/QDR placeholder) | unused |")
    L.append("")
    L.append("The sendrecv run bound to `mlx5_4` (NDR 400 Gb/s) on both nodes, so the "
             "~12.7 GB/s ceiling is **not** a network-rate limit — a single 400 Gb/s NDR "
             "link carries ~50 GB/s per direction, well above what was achieved. The "
             "HDR100 NICs (mlx5_0-3) are not on the NCCL data path.")
    L.append("")

    # Comparison to reference
    if reference:
        L.append("## Comparison to B200 reference (sendrecv, 2-node)")
        L.append("")
        L.append("| Metric | This run | Reference (b0029+b0030) |")
        L.append("|--------|---------:|------------------------:|")
        r = runs[0]
        L.append(f"| Converged busbw (GB/s) | {r['converged']:.1f} | {reference['busbw']:.1f} |")
        L.append(f"| % of GDRDMA ceiling ({ceiling:.1f} GB/s) | "
                 f"{100*r['converged']/ceiling:.0f}% | {reference['pct']:.0f}% |")
        L.append(f"| This run / reference | "
                 f"{100*r['converged']/reference['busbw']:.0f}% | 100% |")
        L.append("")
        gap = 100 * r["converged"] / reference["busbw"]
        if gap < 85:
            L.append(f"> The measured {r['converged']:.1f} GB/s is ~{gap:.0f}% of the "
                     f"reference and ~{100*r['converged']/ceiling:.0f}% of the "
                     f"{ceiling:.1f} GB/s per-pair hardware ceiling — a clean factor of "
                     "~2 short. Diagnostics on node5500/node5502 rule out the usual "
                     "suspects: the GPU-facing NICs are **NDR** (mlx5_4 Active, Rate "
                     "400 Gb/s — not HDR), `nvidia_peermem` is loaded on both nodes, the "
                     "GPU runs **PCIe Gen5 x16**, and NCCL uses a GDRDMA path with good "
                     "GPU-NIC affinity (PXB, same PCIe switch). Installing "
                     "`nvidia_peermem` on node5500 did **not** change the result "
                     "(still 12.7 GB/s across reruns), so GDR/peermem is not the "
                     "bottleneck. The leading remaining hypothesis is that a single "
                     "cross-node GPU pair over one NIC/QP does not saturate the "
                     f"~{ceiling:.1f} GB/s bidirectional PCIe-DMA budget the reference "
                     "figure assumes. Next steps: (1) re-test at **8 GPUs/node** so all "
                     "8 NICs are active and measure the aggregate fabric; (2) try "
                     "`NCCL_IB_QPS_PER_CONNECTION=4` and more channels to add "
                     "concurrency on the single pair.")
            L.append("")
    return "\n".join(L) + "\n"


def collect(args):
    files = args if args else glob.glob(os.path.join(OUT_DIR, "*.out"))
    parsed = {}
    for f in sorted(files, key=os.path.getmtime):  # newest wins per pair
        r = parse_file(f)
        if r:
            parsed[r["pair"]] = r
    return [parsed[k] for k in sorted(parsed)]


def main():
    runs = collect(sys.argv[1:])
    if not runs:
        sys.exit(f"No parseable nccl-tests 2-node results in {OUT_DIR}")
    reference = parse_reference(REFERENCE_FILE, runs[0]["program"].replace("_perf", ""))
    md = build(runs, reference)
    summary = os.path.join(OUT_DIR, "summary.md")
    with open(summary, "w") as fh:
        fh.write(md)
    print(md)
    print(f"Written to {summary}  ({len(runs)} run(s))")


if __name__ == "__main__":
    main()
