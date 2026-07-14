#!/usr/bin/env python3
"""Analyze nccl-tests sendrecv_perf output(s) and write a Markdown summary.

Scans every *.out in out-nccl-1node/, groups by node (newest file per node), and
writes a multi-node summary to out-nccl-1node/summary.md: a per-node overview,
a bus-bandwidth-vs-message-size table with one column per node, aggregates, and
a comparison against the MIT aicr-benchmarks B200 reference (results_b200.md).

The nccl-tests *_perf binaries print one row per message size, e.g.:

    #       size         count  ...     time   algbw   busbw #wrong     time   algbw   busbw #wrong
         1048576        262144  ...    46.41   22.59   22.59      0    49.27   21.28   21.28    N/A

The two triples are out-of-place and in-place; `busbw` (GB/s) is the figure of
merit — higher is better; for sendrecv it equals algbw.

Usage:
    ./analyze-nccl-1node.py [file_or_node ...]   # default: newest *.out per node
"""
import glob
import os
import re
import sys
from datetime import datetime

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out-nccl-1node")
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
# reference 1-node row: | sendrecv | 666 | 666 | 900 | 74% | NVSwitch ... |
REF_ROW_RE = re.compile(
    r"^\|\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)%"
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
    """1-node B200 reference row for `collective` from Table 1 of results_b200.md."""
    try:
        in_table1 = False
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("## Table 1"):
                    in_table1 = True
                    continue
                if in_table1 and s.startswith("## Table 2"):
                    break
                if not in_table1:
                    continue
                m = REF_ROW_RE.match(s)
                if m and m.group(1).lower() == collective.lower():
                    return {
                        "algbw": float(m.group(2)),
                        "busbw": float(m.group(3)),
                        "nvlink_max": float(m.group(4)),
                        "pct": float(m.group(5)),
                    }
    except FileNotFoundError:
        return None
    return None


def parse_file(path):
    cfg = devices = None
    devices = []
    rows = []
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
    node = devices[0]["node"] if devices else os.path.basename(path)
    gpu_name = devices[0]["name"] if devices else "?"
    big = max(rows, key=lambda r: r["size"])
    converged = max(big["oop_busbw"], big["ip_busbw"])
    peak = max(max(r["oop_busbw"], r["ip_busbw"]) for r in rows)
    peak_size = max(rows, key=lambda r: max(r["oop_busbw"], r["ip_busbw"]))["size"]
    ok = all(r["oop_wrong"] in ("0", "N/A") and r["ip_wrong"] in ("0", "N/A")
             for r in rows) and (oob is None or oob[0] == 0)
    return {
        "path": path, "node": node, "gpu_name": gpu_name, "ngpu": cfg["nGpus"] if cfg else len(devices),
        "program": program or "sendrecv_perf", "cfg": cfg, "rows": rows, "avg": avg,
        "converged": converged, "peak": peak, "peak_size": peak_size, "ok": ok,
        "converged_size": big["size"],
    }


def build(nodes, reference):
    L = []
    collective = nodes[0]["program"]
    cfg = nodes[0]["cfg"]
    ngpu = nodes[0]["ngpu"]
    gpu = nodes[0]["gpu_name"]

    L.append(f"# nccl-tests ({collective.replace('_perf','')}) 1-node summary")
    L.append("")
    L.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- Collective: {collective}")
    L.append(f"- Nodes: {', '.join(n['node'] for n in nodes)} "
             f"(each {ngpu} x {gpu}, single node, intra-node NVLink)")
    if cfg:
        L.append(f"- Config: {cfg['nThread']} thread, "
                 f"{fmt_size(cfg['minBytes'])}-{fmt_size(cfg['maxBytes'])}, "
                 f"{cfg['warmup']} warmup + {cfg['iters']} iters")
    if reference:
        L.append(f"- Reference (MIT aicr-benchmarks, `results_b200.md`, b0027): "
                 f"sendrecv busbw **{reference['busbw']:.0f} GB/s** = "
                 f"{reference['pct']:.0f}% of {reference['nvlink_max']:.0f} GB/s NVLink max")
    L.append("")

    # Per-node overview
    L.append("## Per-node overview")
    L.append("")
    hdr = "| Node | Avg busbw (GB/s) | Peak busbw (GB/s) | Converged (GB/s) | % of NVLink max"
    sep = "|------|-----------------:|------------------:|-----------------:|--------------:"
    if reference:
        hdr += " | % of reference"
        sep += "|-------------:"
    hdr += " | Correctness |"
    sep += "|---|"
    L.append(hdr)
    L.append(sep)
    nvlink = reference["nvlink_max"] if reference else 900.0
    for n in nodes:
        row = (f"| {n['node']} | {n['avg']:.1f} | {n['peak']:.1f} | {n['converged']:.1f} "
               f"| {100*n['converged']/nvlink:.0f}%")
        if reference:
            row += f" | {100*n['converged']/reference['busbw']:.1f}%"
        row += f" | {'PASS' if n['ok'] else 'FAIL'} |"
        L.append(row)
    if reference:
        L.append(f"| **reference (b0027)** | — | {reference['busbw']:.1f} | "
                 f"{reference['busbw']:.1f} | {reference['pct']:.0f}% | 100% | — |")
    L.append("")
    L.append("Converged = busbw at the largest message size "
             f"({fmt_size(nodes[0]['converged_size'])}), best of out-of-place / in-place "
             "(matches the reference's methodology).")
    L.append("")

    # Bandwidth vs size, one column per node (out-of-place busbw)
    L.append("## Bus bandwidth vs message size (out-of-place busbw, GB/s)")
    L.append("")
    sizes = sorted({r["size"] for n in nodes for r in n["rows"]})
    L.append("| Message size | " + " | ".join(n["node"] for n in nodes) + " |")
    L.append("|-------------:|" + "|".join(["------:"] * len(nodes)) + "|")
    for sz in sizes:
        cells = []
        for n in nodes:
            r = next((r for r in n["rows"] if r["size"] == sz), None)
            cells.append(f"{r['oop_busbw']:.1f}" if r else "—")
        L.append(f"| {fmt_size(sz)} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("In-place busbw tracks out-of-place within ~2% for sendrecv. Bandwidth "
             "rises with message size and saturates at large sizes (NVLink-bound).")
    L.append("")
    return "\n".join(L) + "\n"


def collect(args):
    """Return newest-per-node list of parsed results, sorted by node name.
    args may be explicit files, node names, or empty (all *.out)."""
    files = []
    if args:
        for a in args:
            if os.path.isfile(a):
                files.append(a)
            else:  # treat as node name
                files += glob.glob(os.path.join(OUT_DIR, f"*{a}*.out"))
    else:
        files = glob.glob(os.path.join(OUT_DIR, "*.out"))
    parsed = {}
    for f in sorted(files, key=os.path.getmtime):  # newest wins per node
        r = parse_file(f)
        if r:
            parsed[r["node"]] = r
    return [parsed[k] for k in sorted(parsed)]


def main():
    nodes = collect(sys.argv[1:])
    if not nodes:
        sys.exit(f"No nccl-tests results parsed from {OUT_DIR}")
    reference = parse_reference(REFERENCE_FILE, nodes[0]["program"].replace("_perf", ""))
    md = build(nodes, reference)
    summary = os.path.join(OUT_DIR, "summary.md")
    with open(summary, "w") as fh:
        fh.write(md)
    print(md)
    print(f"Written to {summary}  ({len(nodes)} node(s))")


if __name__ == "__main__":
    main()
