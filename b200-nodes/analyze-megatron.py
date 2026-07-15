#!/usr/bin/env python3
"""Analyze the Megatron-LM 1-node GPU sweeps and write a Markdown summary + SVG.

Parses the per-GPU-count job outputs in output/ (megatron-1node-<node>-g<N>.<jobid>),
extracts the last-iteration throughput (TFLOP/s/GPU) and iteration time — the same
metric the reference uses — and writes output/summary.md with:

  1. an apples-to-apple comparison against the B200 reference in
     ~/data022/aicr-benchmarks/Benchmark_WG/megatron-lm/output/summary.md, at the
     GPU counts the reference measured (1, 2, 4, 8), and
  2. a scaling table (per-GPU + aggregate TFLOP/s, weak-scaling efficiency) for
     every GPU count, plus a scaling figure (output/megatron-scaling.svg).

Every node found in output/ gets its own column in the comparison table, its own
scaling table, and its own curve in the figure.

Config is apples-to-apple: ~7B GPT (36L/4096H/ffn14336/32heads/seq2048), micro-batch
4, GBS = 128 x total_GPUs, bf16, 100 iters, no activation recompute.

Usage:  ./analyze-megatron.py [output_dir]     # default: ./output
"""
import glob
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output")
REF_SUMMARY = os.path.expanduser(
    "~/data022/aicr-benchmarks/Benchmark_WG/megatron-lm/output/summary.md"
)
REF_COUNTS = (1, 2, 4, 8)   # GPU counts the reference measured (1-node)

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*?throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)"
)
FNAME_RE = re.compile(r"megatron-1node-(\S+?)-g(\d+)\.(\d+)$")
# reference B200 group table row: | 1 | 1 | b0004 | 128 | 996.1 | ...
REF_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*[^|]+\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|"
)


def parse_output(path):
    """Return dict for one run: gpus, tflops (last iter), iter_ms, last_iter,
    total_iters, ok. ok = reached the final iteration with a throughput value."""
    m = FNAME_RE.search(os.path.basename(path))
    gpus = int(m.group(2)) if m else None
    node = m.group(1) if m else "?"
    last = None
    total = None
    with open(path, errors="replace") as fh:
        for line in fh:
            mm = ITER_RE.search(line)
            if mm:
                it, tot, ims, tf = mm.groups()
                last = {"iter": int(it), "iter_ms": float(ims), "tflops": float(tf)}
                total = int(tot)
    if last is None:
        return {"gpus": gpus, "node": node, "ok": False, "tflops": None,
                "iter_ms": None, "last_iter": 0, "total_iters": total, "path": path}
    return {"gpus": gpus, "node": node, "ok": (last["iter"] == total),
            "tflops": last["tflops"], "iter_ms": last["iter_ms"],
            "last_iter": last["iter"], "total_iters": total, "path": path}


def parse_reference(path):
    """Return {gpus: tflops} for the 1-node B200 reference (max if duplicates)."""
    ref = {}
    in_b200 = False
    try:
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("## Group") and "B200" in s:
                    in_b200 = True
                    continue
                if in_b200 and s.startswith("## Group"):
                    break
                if not in_b200:
                    continue
                m = REF_ROW_RE.match(s)
                if m:
                    nodes, gpn, gbs, tf = m.groups()
                    if int(nodes) == 1:
                        g = int(gpn)
                        ref[g] = max(ref.get(g, 0.0), float(tf))
    except FileNotFoundError:
        return {}
    return ref


# node curve colors, in the order nodes are encountered
NODE_COLORS = ["#1f6feb", "#2ea043", "#8957e5", "#d29922"]


# ---------- SVG scaling figure (no external deps) ----------
def svg_scaling(by_node, ref, path):
    """Aggregate TFLOP/s vs #GPUs: one curve per node, ideal-linear, reference."""
    W, H = 760, 470
    ml, mr, mt, mb = 78, 150, 46, 56          # margins (right margin holds legend)
    pw, ph = W - ml - mr, H - mt - mb
    nodes = sorted(by_node)
    agg_by_node = {n: {r["gpus"]: r["tflops"] * r["gpus"] for r in by_node[n]}
                   for n in nodes}
    # ideal line anchors on the best 1-GPU per-GPU rate across nodes
    ideal1 = max([r["tflops"] for n in nodes for r in by_node[n] if r["gpus"] == 1]
                 or [0])
    xmin, xmax = 1, 8
    ymax = max([v for n in nodes for v in agg_by_node[n].values()] + [ideal1 * 8] +
               [ref[g] * g for g in ref if g <= 8] + [1])
    # round ymax up to a nice step
    step = 1000
    ymax = step * (int(ymax // step) + 1)

    def X(g):
        return ml + (g - xmin) / (xmax - xmin) * pw

    def Y(v):
        return mt + ph - (v / ymax) * ph

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'font-family="sans-serif" font-size="13">')
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{ml}" y="24" font-size="16" font-weight="bold">'
             f'Megatron-LM ~7B GPT: aggregate throughput scaling (B200, single node)'
             f'</text>')
    # y gridlines + labels
    v = 0
    while v <= ymax:
        y = Y(v)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                 f'stroke="#e0e0e0"/>')
        s.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">'
                 f'{v}</text>')
        v += step
    # x ticks
    for g in range(1, 9):
        x = X(g)
        s.append(f'<line x1="{x:.1f}" y1="{mt+ph}" x2="{x:.1f}" y2="{mt+ph+5}" '
                 f'stroke="#555"/>')
        s.append(f'<text x="{x:.1f}" y="{mt+ph+20}" text-anchor="middle" '
                 f'fill="#555">{g}</text>')
    # axes
    s.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#555"/>')
    s.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#555"/>')
    s.append(f'<text x="{ml+pw/2}" y="{H-14}" text-anchor="middle">'
             f'number of GPUs</text>')
    s.append(f'<text x="20" y="{mt+ph/2}" text-anchor="middle" '
             f'transform="rotate(-90 20 {mt+ph/2})">aggregate TFLOP/s</text>')
    # ideal linear line
    s.append(f'<line x1="{X(1):.1f}" y1="{Y(ideal1):.1f}" x2="{X(8):.1f}" '
             f'y2="{Y(ideal1*8):.1f}" stroke="#bbb" stroke-width="2" '
             f'stroke-dasharray="6 5"/>')
    # measured polyline, one per node
    for i, n in enumerate(nodes):
        c = NODE_COLORS[i % len(NODE_COLORS)]
        agg = agg_by_node[n]
        pts = " ".join(f"{X(g):.1f},{Y(agg[g]):.1f}" for g in sorted(agg))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" '
                 f'stroke-width="2.5"/>')
        for g in sorted(agg):
            s.append(f'<circle cx="{X(g):.1f}" cy="{Y(agg[g]):.1f}" r="4.5" '
                     f'fill="{c}"/>')
    # reference points
    for g in sorted(ref):
        if g <= 8:
            s.append(f'<rect x="{X(g)-4:.1f}" y="{Y(ref[g]*g)-4:.1f}" width="8" '
                     f'height="8" fill="#e3742f"/>')
    # legend
    lx, ly = ml + pw + 16, mt + 6
    items = [(NODE_COLORS[i % len(NODE_COLORS)], n, "line")
             for i, n in enumerate(nodes)]
    items += [("#bbb", "ideal linear", "dash"),
              ("#e3742f", "reference B200", "sq")]
    for i, (c, lbl, kind) in enumerate(items):
        yy = ly + i * 22
        if kind == "line":
            s.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+22}" y2="{yy}" '
                     f'stroke="{c}" stroke-width="2.5"/>')
            s.append(f'<circle cx="{lx+11}" cy="{yy}" r="4" fill="{c}"/>')
        elif kind == "dash":
            s.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+22}" y2="{yy}" '
                     f'stroke="{c}" stroke-width="2" stroke-dasharray="6 5"/>')
        else:
            s.append(f'<rect x="{lx+7}" y="{yy-4}" width="8" height="8" fill="{c}"/>')
        s.append(f'<text x="{lx+28}" y="{yy+4}" fill="#333">{lbl}</text>')
    s.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(s))


def build(by_node_all, ref, svg_name):
    nodes = sorted(by_node_all)
    L = []
    L.append("# Megatron-LM 1-node GPU sweep — B200")
    L.append("")
    L.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- Nodes: {', '.join(nodes)} (single node each, data-parallel, "
             f"TP=1, PP=1)")
    L.append("- Model: ~7B GPT — 36 layers, hidden 4096, FFN 14336, 32 heads, "
             "seq 2048, bf16")
    L.append("- Per run: micro-batch 4, global batch = 128 x total_GPUs, 100 iters, "
             "no activation recompute")
    L.append("- Metric: last-iteration throughput (TFLOP/s/GPU), same as the reference")
    L.append("- Reference: MIT aicr-benchmarks `megatron-lm/output/summary.md`, "
             "B200 1-node group")
    L.append("")

    # 1. Apples-to-apple comparison at the reference's GPU counts, node by node
    L.append("## Apples-to-apple vs B200 reference")
    L.append("")
    hdr = "| #GPUs | GBS | reference TFLOP/s/GPU |"
    sep = "|------:|----:|----------------------:|"
    for n in nodes:
        hdr += f" {n} TFLOP/s/GPU | {n} / ref |"
        sep += "-----------------:|-----------:|"
    L.append(hdr)
    L.append(sep)
    ok_by_node = {n: {r["gpus"]: r for r in by_node_all[n]
                      if r["ok"] and r["tflops"] is not None} for n in nodes}
    for g in REF_COUNTS:
        rf = ref.get(g)
        row = f"| {g} | {128*g} | {rf:.1f} |" if rf else f"| {g} | {128*g} | — |"
        for n in nodes:
            r = ok_by_node[n].get(g)
            row += f" {r['tflops']:.1f} |" if r else " — |"
            row += f" {100*r['tflops']/rf:.1f}% |" if (r and rf) else " — |"
        L.append(row)
    L.append("")
    L.append("Reference values are the best B200 1-node result per GPU count from "
             "`summary.md` (last-iteration TFLOP/s/GPU).")
    L.append("")

    # 2. Scaling table across all measured GPU counts, one per node
    L.append("## Scaling (1 -> 8 GPUs, single node)")
    L.append("")
    for n in nodes:
        runs = sorted(by_node_all[n], key=lambda r: (r["gpus"] or 0))
        base = next((r["tflops"] for r in runs
                     if r["gpus"] == 1 and r["ok"] and r["tflops"]), None)
        L.append(f"### {n}")
        L.append("")
        L.append("| #GPUs | GBS | per-GPU TFLOP/s | aggregate TFLOP/s | iter (ms) | "
                 "weak-scaling eff. | status |")
        L.append("|------:|----:|----------------:|------------------:|----------:|"
                 "------------------:|--------|")
        for r in runs:
            g = r["gpus"]
            if r["ok"] and r["tflops"] is not None:
                agg = r["tflops"] * g
                eff = f"{100*r['tflops']/base:.1f}%" if base else "—"
                L.append(f"| {g} | {128*g} | {r['tflops']:.1f} | {agg:.0f} | "
                         f"{r['iter_ms']:.0f} | {eff} | ok |")
            else:
                st = f"incomplete (iter {r['last_iter']}/{r['total_iters']})" \
                     if r["last_iter"] else "no data / failed"
                L.append(f"| {g} | {128*g} | — | — | — | — | {st} |")
        L.append("")
    L.append("Aggregate = per-GPU x #GPUs. Weak-scaling efficiency = per-GPU(N) / "
             "per-GPU(1) on that node. Per-GPU work is held constant (GBS scales "
             "with #GPUs).")
    L.append("")

    # 3. Figure
    L.append("## Scaling figure")
    L.append("")
    L.append(f"![Aggregate TFLOP/s vs number of GPUs]({svg_name})")
    L.append("")
    L.append("Aggregate throughput vs #GPUs: one curve per node, ideal linear scaling "
             "from the best 1-GPU point (dashed), and the B200 reference (orange).")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "megatron-1node-*-g*")))
    files = [f for f in files if FNAME_RE.search(os.path.basename(f))]
    if not files:
        sys.exit(f"No megatron 1-node output files in {OUT_DIR}")
    # newest job per (node, GPU count)
    seen = {}
    for f in sorted(files, key=os.path.getmtime):
        r = parse_output(f)
        if r["gpus"] is not None:
            seen[(r["node"], r["gpus"])] = r
    by_node_all = {}
    for (node, _), r in seen.items():
        by_node_all.setdefault(node, []).append(r)
    for node in by_node_all:
        by_node_all[node].sort(key=lambda r: r["gpus"])
    ref = parse_reference(REF_SUMMARY)

    svg_name = "megatron-scaling.svg"
    by_node_ok = {n: [r for r in rs if r["ok"] and r["tflops"] is not None]
                  for n, rs in by_node_all.items()}
    by_node_ok = {n: rs for n, rs in by_node_ok.items() if rs}
    if by_node_ok:
        svg_scaling(by_node_ok, ref, os.path.join(OUT_DIR, svg_name))

    md = build(by_node_all, ref, svg_name)
    summary = os.path.join(OUT_DIR, "summary.md")
    with open(summary, "w") as fh:
        fh.write(md)
    print(md)
    n_ok = sum(len(rs) for rs in by_node_ok.values())
    n_all = sum(len(rs) for rs in by_node_all.values())
    print(f"Written {summary} and {svg_name} ({n_ok}/{n_all} runs ok, "
          f"{len(by_node_all)} node(s))")


if __name__ == "__main__":
    main()
