#!/usr/bin/env python3
"""Analyze gpu-fryer output(s) and write a summary of converged TFLOP/s.

Scans every *.out in out-gpu-fryer/, groups by node (newest file per node), and
writes a multi-node summary to out-gpu-fryer/summary.md: a per-node mean-throughput
overview (FP32 / BF16 / FP8), per-GPU detail per node, and a comparison against the
MIT aicr-benchmarks B200 reference (gpu-fryer/summary.md).

gpu-fryer prints, at the end of each precision run, one final per-GPU line:

    GPU #0: 757961 Gflops/s (min: 741070.84, max: 781752.77, dev: 757960.97)
             Throttling HW: false, Thermal SW: false, Thermal HW: false

The leading number is the converged (sustained-average) throughput.

Usage:
    ./analyze-gpu-fryer.py [file_or_node ...]   # default: newest *.out per node
"""
import glob
import os
import re
import sys
from datetime import datetime

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out-gpu-fryer")
REFERENCE_FILE = (
    "/home/shaohao/data022/aicr-benchmarks/Benchmark_WG/gpu-fryer/summary.md"
)

NODE_RE = re.compile(r"Node\s*=\s*(\S+)")
SECTION_RE = re.compile(r"=+\s*Run with\s+(\w+)\s*=+", re.IGNORECASE)
GPU_RE = re.compile(r"GPU #(\d+):\s+([\d.]+)\s+Gflops/s")
THROTTLE_RE = re.compile(
    r"Throttling HW:\s*(\w+),\s*Thermal SW:\s*(\w+),\s*Thermal HW:\s*(\w+)"
)
# reference per-GPU row: | #0 | 779.7 / 796.0 | 1,512.3 / 1,516.2 | 4,138.7 / 4,156.2 | 65C |
REF_ROW_RE = re.compile(
    r"^\|\s*#(\d+)\s*\|\s*\*?\*?([\d,.]+)\s*/\s*[\d,.]+\*?\*?\s*\|"
    r"\s*\*?\*?([\d,.]+)\s*/\s*[\d,.]+\*?\*?\s*\|"
    r"\s*\*?\*?([\d,.]+)\s*/\s*[\d,.]+\*?\*?\s*\|"
)


def parse_reference(path):
    """Return {precision: mean_tflops} for the B200 node in the reference file
    (mean over the 8 per-GPU 'mean' values under '### B200')."""
    data = {"FP32": {}, "BF16": {}, "FP8": {}}
    in_b200 = False
    try:
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("### B200"):
                    in_b200 = True
                    continue
                if in_b200 and s.startswith("###") and not s.startswith("### B200"):
                    break
                if not in_b200:
                    continue
                m = REF_ROW_RE.match(s)
                if m:
                    g = int(m.group(1))
                    data["FP32"][g] = float(m.group(2).replace(",", ""))
                    data["BF16"][g] = float(m.group(3).replace(",", ""))
                    data["FP8"][g] = float(m.group(4).replace(",", ""))
    except FileNotFoundError:
        return None
    if not data["FP32"]:
        return None
    return {p: sum(d.values()) / len(d) for p, d in data.items()}


def parse_file(path):
    node = None
    order, data = [], {}
    throttled = False
    cur = None
    with open(path) as fh:
        for line in fh:
            m = NODE_RE.search(line)
            if m and node is None:
                node = m.group(1)
            m = SECTION_RE.search(line)
            if m:
                cur = m.group(1).upper()
                order.append(cur)
                data[cur] = {}
                continue
            if cur is None:
                continue
            g = GPU_RE.search(line)
            if g:
                data[cur][int(g.group(1))] = float(g.group(2)) / 1000.0  # -> TFLOP/s
                continue
            t = THROTTLE_RE.search(line)
            if t and any(v.lower() == "true" for v in t.groups()):
                throttled = True
    order = [p for p in order if data.get(p)]
    if not order:
        return None
    return {
        "path": path, "node": node or os.path.basename(path),
        "order": order, "data": data, "throttled": throttled,
    }


def mean(vals):
    return sum(vals) / len(vals)


def build(nodes, reference):
    L = []
    # union of precisions in first-seen order
    order = []
    for n in nodes:
        for p in n["order"]:
            if p not in order:
                order.append(p)

    L.append("# gpu-fryer summary")
    L.append("")
    L.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- Nodes: {', '.join(n['node'] for n in nodes)} (8 x NVIDIA B200 each)")
    L.append(f"- Precisions: {', '.join(order)}")
    if reference:
        ref_str = ", ".join(f"{p} {reference[p]:.0f}" for p in order if p in reference)
        L.append(f"- Reference (MIT aicr-benchmarks, `gpu-fryer/summary.md`, b0025): "
                 f"per-GPU mean TFLOP/s — {ref_str}")
    L.append("")

    # Per-node mean overview
    L.append("## Per-node mean converged throughput (TFLOP/s)")
    L.append("")
    L.append("| Node | " + " | ".join(order) + " | Health |")
    L.append("|------|" + "|".join(["------:"] * len(order)) + "|---|")
    for n in nodes:
        cells = [f"{mean(list(n['data'][p].values())):.0f}" if p in n["data"] else "—"
                 for p in order]
        health = "THROTTLING" if n["throttled"] else "ok"
        L.append(f"| {n['node']} | " + " | ".join(cells) + f" | {health} |")
    if reference:
        refc = [f"{reference[p]:.0f}" if p in reference else "—" for p in order]
        L.append("| **reference (b0025)** | " + " | ".join(f"**{c}**" for c in refc) + " | — |")
        # % of reference per node
        L.append("")
        L.append("### % of B200 reference (mean)")
        L.append("")
        L.append("| Node | " + " | ".join(order) + " |")
        L.append("|------|" + "|".join(["------:"] * len(order)) + "|")
        for n in nodes:
            cells = []
            for p in order:
                if p in n["data"] and p in reference:
                    cells.append(f"{100*mean(list(n['data'][p].values()))/reference[p]:.0f}%")
                else:
                    cells.append("—")
            L.append(f"| {n['node']} | " + " | ".join(cells) + " |")
    L.append("")

    # Per-GPU detail per node
    L.append("## Per-GPU converged throughput (TFLOP/s)")
    L.append("")
    for n in nodes:
        gpus = sorted({g for p in n["order"] for g in n["data"][p]})
        L.append(f"### {n['node']}")
        L.append("")
        L.append("| GPU | " + " | ".join(n["order"]) + " |")
        L.append("|-----|" + "|".join(["------:"] * len(n["order"])) + "|")
        for g in gpus:
            cells = [f"{n['data'][p].get(g, float('nan')):.1f}" for p in n["order"]]
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        mins = [min(n["data"][p].values()) for p in n["order"]]
        means = [mean(list(n["data"][p].values())) for p in n["order"]]
        maxs = [max(n["data"][p].values()) for p in n["order"]]
        L.append("| **min** | " + " | ".join(f"**{v:.1f}**" for v in mins) + " |")
        L.append("| **mean** | " + " | ".join(f"**{v:.1f}**" for v in means) + " |")
        L.append("| **max** | " + " | ".join(f"**{v:.1f}**" for v in maxs) + " |")
        L.append("")

    L.append("Converged = the final sustained-average throughput gpu-fryer reports per "
             "GPU at the end of each precision run. Higher is better; large spread across "
             "GPUs or any throttling flag indicates a problem.")
    L.append("")
    return "\n".join(L) + "\n"


def collect(args):
    files = []
    if args:
        for a in args:
            if os.path.isfile(a):
                files.append(a)
            else:
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
        sys.exit(f"No gpu-fryer results parsed from {OUT_DIR}")
    reference = parse_reference(REFERENCE_FILE)
    md = build(nodes, reference)
    summary = os.path.join(OUT_DIR, "summary.md")
    with open(summary, "w") as fh:
        fh.write(md)
    print(md)
    print(f"Written to {summary}  ({len(nodes)} node(s))")


if __name__ == "__main__":
    main()
