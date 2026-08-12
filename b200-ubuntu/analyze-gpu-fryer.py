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
# Rocky 8 results (node5500-5502), pulled in automatically as the reference set
ROCKY_DIR = "/orcd/data/orcd/022/benchmarks/b200-nodes/out-gpu-fryer"
REFERENCE_FILE = (
    "/home/shaohao/data022/aicr-benchmarks/Benchmark_WG/gpu-fryer/summary.md"
)

NODE_RE = re.compile(r"Node\s*=\s*(\S+)")
SECTION_RE = re.compile(r"=+\s*Run with\s+(\w+)\s*=+", re.IGNORECASE)
GPU_RE = re.compile(r"GPU #(\d+):\s+([\d.]+)\s+Gflops/s")
# nvidia-smi -L line: "GPU 0: NVIDIA B200 (UUID: GPU-...)"
GPU_MODEL_RE = re.compile(r"GPU\s+\d+:\s+NVIDIA\s+([A-Za-z0-9 ]+?)\s*\(UUID")
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
    gpu_model = None
    order, data = [], {}
    throttled = False
    cur = None
    with open(path) as fh:
        for line in fh:
            m = NODE_RE.search(line)
            if m and node is None:
                node = m.group(1)
            if gpu_model is None and cur is None:
                gm = GPU_MODEL_RE.search(line)
                if gm:
                    gpu_model = gm.group(1).strip()
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
        "gpu": gpu_model or "unknown", "order": order, "data": data,
        "throttled": throttled,
    }


def mean(vals):
    return sum(vals) / len(vals)


# node5500-5502 run Rocky 8, node5700-5701 run Ubuntu 24.04. The detailed
# sections (speed-up plot, per-GPU tables) cover the Ubuntu nodes only; the
# Rocky 8 nodes stay in the per-node overview as a cross-OS reference.
def node_os(name):
    if re.match(r"node55\d\d$", name):
        return "Rocky 8"
    if re.match(r"node57\d\d$", name):
        return "Ubuntu 24.04"
    return "unknown"


def is_ubuntu(node):
    return node_os(node["node"]) == "Ubuntu 24.04"


def pct_diff(ubuntu, rocky):
    """Signed % difference of an Ubuntu value vs the Rocky 8 mean.
    '+' means the Ubuntu node is faster, '-' means slower."""
    return f"{(ubuntu - rocky) / rocky * 100:+.1f}%"


# node curve colors, in the order nodes are encountered
NODE_COLORS = ["#1f6feb", "#2ea043", "#8957e5", "#d29922"]


def speedup_curve(node, prec):
    """Cumulative aggregate over GPUs 0..N-1, normalised by GPU 0 -> [(N, speedup)]."""
    vals = [node["data"][prec][g] for g in sorted(node["data"][prec])]
    if not vals:
        return []
    base, run, pts = vals[0], 0.0, []
    for i, v in enumerate(vals, start=1):
        run += v
        pts.append((i, run / base))
    return pts


def svg_speedup(nodes, path):
    """Speed-up vs number of GPUs for ONE node, one curve per precision (SVG).

    gpu-fryer stresses all 8 GPUs concurrently and reports one converged figure per
    GPU; it does not run separate 1/2/.../8-GPU jobs. The curve is therefore derived
    from that single run as the cumulative aggregate over GPUs 0..N-1, normalised by
    GPU 0. It is linear by construction — its value is that any *departure* from the
    ideal line marks a slow or throttling GPU.

    Returns {"node", "curves": {prec: pts}, "others": {node: {prec: speedup_at_max}}}
    so the summary can state how closely the remaining nodes agree.
    """
    W, H = 760, 470
    ml, mr, mt, mb = 70, 160, 46, 56
    pw, ph = W - ml - mr, H - mt - mb
    if not nodes:
        return None
    main = nodes[0]                      # representative node (first, sorted by name)
    curves = {p: speedup_curve(main, p) for p in main["order"]
              if main["data"].get(p)}
    curves = {p: pts for p, pts in curves.items() if pts}
    if not curves:
        return None
    names = sorted(curves)               # precision names, used for the legend
    xmax = max(p[0] for pts in curves.values() for p in pts)
    ymax = max([p[1] for pts in curves.values() for p in pts] + [xmax])
    ymax = float(int(ymax) + 1)

    def X(g):
        return ml + (g - 1) / max(xmax - 1, 1) * pw

    def Y(v):
        return mt + ph - (v / ymax) * ph

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'font-family="sans-serif" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{ml}" y="24" font-size="16" font-weight="bold">'
         f'gpu-fryer: speed-up vs number of GPUs — {main["node"]} '
         f'({main["gpu"]})</text>']
    v = 0
    while v <= ymax:
        y = Y(v)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                 f'stroke="#e0e0e0"/>')
        s.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">'
                 f'{v}</text>')
        v += 1
    for g in range(1, int(xmax) + 1):
        x = X(g)
        s.append(f'<line x1="{x:.1f}" y1="{mt+ph}" x2="{x:.1f}" y2="{mt+ph+5}" '
                 f'stroke="#555"/>')
        s.append(f'<text x="{x:.1f}" y="{mt+ph+20}" text-anchor="middle" '
                 f'fill="#555">{g}</text>')
    s.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#555"/>')
    s.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#555"/>')
    s.append(f'<text x="{ml+pw/2}" y="{H-14}" text-anchor="middle">'
             f'number of GPUs</text>')
    s.append(f'<text x="18" y="{mt+ph/2}" text-anchor="middle" '
             f'transform="rotate(-90 18 {mt+ph/2})">speed-up (x single GPU)</text>')
    # ideal linear
    s.append(f'<line x1="{X(1):.1f}" y1="{Y(1):.1f}" x2="{X(xmax):.1f}" '
             f'y2="{Y(xmax):.1f}" stroke="#bbb" stroke-width="2" '
             f'stroke-dasharray="6 5"/>')
    for i, name in enumerate(names):
        c = NODE_COLORS[i % len(NODE_COLORS)]
        pts = curves[name]
        poly = " ".join(f"{X(g):.1f},{Y(v):.1f}" for g, v in pts)
        s.append(f'<polyline points="{poly}" fill="none" stroke="{c}" '
                 f'stroke-width="2.5"/>')
        for g, v in pts:
            s.append(f'<circle cx="{X(g):.1f}" cy="{Y(v):.1f}" r="4" fill="{c}"/>')
    lx, ly = ml + pw + 16, mt + 6
    items = [(NODE_COLORS[i % len(NODE_COLORS)], n, "line")
             for i, n in enumerate(names)]
    items.append(("#bbb", "ideal linear", "dash"))
    for i, (c, lbl, kind) in enumerate(items):
        yy = ly + i * 22
        if kind == "line":
            s.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+22}" y2="{yy}" '
                     f'stroke="{c}" stroke-width="2.5"/>')
            s.append(f'<circle cx="{lx+11}" cy="{yy}" r="4" fill="{c}"/>')
        else:
            s.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+22}" y2="{yy}" '
                     f'stroke="{c}" stroke-width="2" stroke-dasharray="6 5"/>')
        s.append(f'<text x="{lx+28}" y="{yy+4}" fill="#333">{lbl}</text>')
    s.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(s))

    # how closely do the remaining nodes track the plotted one? compare the
    # end-point speed-up (all GPUs) per precision.
    others = {}
    for n in nodes[1:]:
        per_prec = {}
        for p in names:
            if n["data"].get(p):
                pts = speedup_curve(n, p)
                if pts:
                    per_prec[p] = pts[-1][1]
        if per_prec:
            others[n["node"]] = per_prec
    endpoints = {p: curves[p][-1][1] for p in names}
    max_dev = max((abs(v - endpoints[p])
                   for d in others.values() for p, v in d.items()), default=0.0)
    return {"node": main["node"], "curves": curves, "names": names,
            "endpoints": endpoints, "others": others, "max_dev": max_dev,
            "xmax": xmax}


def build(nodes, reference, speedup=None, svg_name="gpu-fryer-speedup.svg",
          detail_nodes=None):
    """detail_nodes: nodes to show the per-GPU tables for (default: all)."""
    L = []
    if detail_nodes is None:
        detail_nodes = nodes
    detail_names = {n["node"] for n in detail_nodes}
    others = [n for n in nodes if n["node"] not in detail_names]

    # union of precisions in first-seen order
    order = []
    for n in nodes:
        for p in n["order"]:
            if p not in order:
                order.append(p)

    def is_b200(n):
        return "B200" in n["gpu"].upper()

    L.append("# gpu-fryer summary")
    L.append("")
    L.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("- Nodes: " + ", ".join(
        f"{n['node']} (8 x {n['gpu']}, {node_os(n['node'])})" for n in nodes))
    L.append(f"- Precisions: {', '.join(order)}")
    L.append("- **node5500-5502 run Rocky 8; node5700-5701 run Ubuntu 24.04.** "
             "The speed-up plot and the per-GPU tables below cover the Ubuntu "
             "nodes; the Rocky 8 nodes appear in the per-node overview only, "
             "since their results are very close (see below). Full Rocky 8 "
             "detail: `../b200-nodes/out-gpu-fryer/summary.md`.")
    if reference:
        ref_str = ", ".join(f"{p} {reference[p]:.0f}" for p in order if p in reference)
        L.append(f"- Reference (MIT aicr-benchmarks, `gpu-fryer/summary.md`, b0025, "
                 f"**B200**): per-GPU mean TFLOP/s — {ref_str}")
        if not all(is_b200(n) for n in nodes):
            L.append("- Note: the reference is a B200 node; the `% of B200 reference` "
                     "comparison is only meaningful for B200 nodes and is shown as `—` "
                     "for other GPU types.")
    L.append("")

    # Per-node mean overview
    L.append("## Per-node mean converged throughput (TFLOP/s)")
    L.append("")
    # Rocky 8 mean per precision, used as the baseline for the signed diffs
    rocky = [n for n in nodes if node_os(n["node"]) == "Rocky 8"]
    rocky_mean = {}
    for p in order:
        vals = [mean(list(n["data"][p].values())) for n in rocky if p in n["data"]]
        if vals:
            rocky_mean[p] = mean(vals)

    dhdr = " | ".join(f"{p} vs Rocky 8" for p in order) if rocky_mean else ""
    L.append("| Node | OS | GPU | " + " | ".join(order)
             + (f" | {dhdr}" if rocky_mean else "") + " | Health |")
    L.append("|------|----|-----|" + "|".join(["------:"] * len(order)) + "|"
             + ("|".join(["------:"] * len(order)) + "|" if rocky_mean else "")
             + "---|")
    for n in nodes:
        cells = [f"{mean(list(n['data'][p].values())):.0f}" if p in n["data"] else "—"
                 for p in order]
        health = "THROTTLING" if n["throttled"] else "ok"
        row = (f"| {n['node']} | {node_os(n['node'])} | {n['gpu']} | "
               + " | ".join(cells))
        if rocky_mean:
            # only the Ubuntu nodes are compared; Rocky nodes are the baseline
            if is_ubuntu(n):
                dcells = [pct_diff(mean(list(n["data"][p].values())), rocky_mean[p])
                          if p in n["data"] and p in rocky_mean else "—"
                          for p in order]
            else:
                dcells = ["baseline"] * len(order)
            row += " | " + " | ".join(dcells)
        L.append(row + f" | {health} |")
    if reference:
        refc = [f"{reference[p]:.0f}" if p in reference else "—" for p in order]
        L.append("| **reference (b0025)** | **B200** | "
                 + " | ".join(f"**{c}**" for c in refc) + " | — |")
        # % of reference per node (B200 nodes only)
        b200_nodes = [n for n in nodes if is_b200(n)]
        if b200_nodes:
            L.append("")
            L.append("### % of B200 reference (mean, B200 nodes only)")
            L.append("")
            L.append("| Node | " + " | ".join(order) + " |")
            L.append("|------|" + "|".join(["------:"] * len(order)) + "|")
            for n in b200_nodes:
                cells = []
                for p in order:
                    if p in n["data"] and p in reference:
                        cells.append(f"{100*mean(list(n['data'][p].values()))/reference[p]:.0f}%")
                    else:
                        cells.append("—")
                L.append(f"| {n['node']} | " + " | ".join(cells) + " |")
    L.append("")

    # Speed-up figure: one node, one curve per precision
    if speedup:
        su_node = speedup["node"]
        precs = speedup["names"]
        L.append("## Speed-up vs number of GPUs")
        L.append("")
        L.append(f"![Speed-up vs number of GPUs]({svg_name})")
        L.append("")
        L.append(f"Shown for **{su_node}**, one curve per precision.")
        L.append("")
        L.append("| #GPUs | " + " | ".join(precs) + " | ideal |")
        L.append("|------:|" + "|".join(["------:"] * len(precs)) + "|------:|")
        npts = max(len(speedup["curves"][p]) for p in precs)
        for i in range(npts):
            cells = []
            for p in precs:
                pts = speedup["curves"][p]
                cells.append(f"{pts[i][1]:.2f}" if i < len(pts) else "—")
            L.append(f"| {i+1} | " + " | ".join(cells) + f" | {i+1}.00 |")
        L.append("")
        if speedup["others"]:
            other_names = ", ".join(sorted(speedup["others"]))
            ends = ", ".join(f"{p} {speedup['endpoints'][p]:.2f}x" for p in precs)
            dev = speedup["max_dev"]
            same = "essentially identical" if dev <= 0.1 else "close"
            L.append(f"The other nodes ({other_names}) are {same} and are omitted "
                     f"from the plot to keep it readable: at {speedup['xmax']} GPUs "
                     f"{su_node} reaches {ends}, and no other node differs by more "
                     f"than **{dev:.2f}x** on any precision.")
            L.append("")
        L.append("> **How to read this.** gpu-fryer stresses all 8 GPUs "
                 "*concurrently* and reports one converged figure per GPU — it does "
                 "not run separate 1, 2, ... 8-GPU jobs. The curve above is therefore "
                 "**derived** from that single run: speed-up(N) = (sum of GPUs 0..N-1) "
                 "/ GPU 0. It is linear by construction and is **not** a measured "
                 "scaling study; what it shows is per-GPU *uniformity* — a curve that "
                 "tracks the dashed ideal line means every GPU sustains the same "
                 "throughput, while a curve bending below it marks a slow or "
                 "throttling GPU. For real scaling behaviour see the Megatron-LM "
                 "weak-scaling results in `output-megatron/summary.md`.")
        L.append("")

    # Per-GPU detail — Ubuntu nodes only
    L.append("## Per-GPU converged throughput (TFLOP/s)")
    L.append("")
    if others:
        L.append("Ubuntu nodes (" + ", ".join(n["node"] for n in detail_nodes)
                 + "). Per-GPU detail for the Rocky 8 nodes is omitted — "
                 "they are very close, as quantified after the tables.")
        L.append("")
    for n in detail_nodes:
        gpus = sorted({g for p in n["order"] for g in n["data"][p]})
        L.append(f"### {n['node']} (8 x {n['gpu']})")
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
        if rocky_mean:
            dcells = [pct_diff(mean(list(n["data"][p].values())), rocky_mean[p])
                      if p in rocky_mean else "—" for p in n["order"]]
            L.append("| **vs Rocky 8** | " + " | ".join(f"**{c}**" for c in dcells)
                     + " |")
        L.append("")

    # how closely the omitted (Rocky 8) nodes track the Ubuntu ones
    if others and detail_nodes:
        worst, worst_p, worst_node = 0.0, None, None
        for p in order:
            base = [mean(list(n["data"][p].values())) for n in detail_nodes
                    if p in n["data"]]
            if not base:
                continue
            ref = mean(base)
            for o in others:
                if p not in o["data"]:
                    continue
                d = abs(mean(list(o["data"][p].values())) - ref) / ref * 100
                if d > worst:
                    worst, worst_p, worst_node = d, p, o["node"]
        if worst_p:
            names = ", ".join(o["node"] for o in others)
            L.append(f"**The Rocky 8 nodes ({names}) are very close.** Their per-node "
                     f"mean throughput sits within **{worst:.1f}%** of the Ubuntu "
                     f"node(s) on every precision (largest gap: {worst_node}, "
                     f"{worst_p}). Per-GPU tables and the speed-up plot for those "
                     f"nodes are in `../b200-nodes/out-gpu-fryer/summary.md`.")
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
        # every Ubuntu run in this dir, plus the Rocky 8 runs as a reference.
        # Globbing both (rather than naming files on the command line) is what
        # keeps a newly-added node — e.g. node5701 — from being left out.
        files = glob.glob(os.path.join(OUT_DIR, "*.out"))
        files += glob.glob(os.path.join(ROCKY_DIR, "*.out"))
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
    svg_name = "gpu-fryer-speedup.svg"
    # plot + per-GPU tables: Ubuntu nodes only (node5700/5701); the Rocky 8
    # nodes stay in the overview table as a cross-OS reference
    ubuntu = [n for n in nodes if is_ubuntu(n)] or nodes
    speedup = svg_speedup(ubuntu, os.path.join(OUT_DIR, svg_name))
    md = build(nodes, reference, speedup, svg_name, detail_nodes=ubuntu)
    summary = os.path.join(OUT_DIR, "summary.md")
    with open(summary, "w") as fh:
        fh.write(md)
    print(md)
    print(f"Written to {summary}  ({len(nodes)} node(s))")


if __name__ == "__main__":
    main()
