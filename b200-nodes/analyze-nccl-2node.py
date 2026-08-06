#!/usr/bin/env python3
"""Analyze nccl-tests 2-node output(s) and write a Markdown summary.

Handles multi-collective runs: a single output file may contain several
collectives, each delimited by a `%%%%% <binary> %%%%%` marker (as emitted by
job-nccl-2node.sh). Each collective segment is parsed independently.

Scans every *.out in out-nccl-2node/, keeps the newest parseable run per
node-pair, and writes out-nccl-2node/summary.md:
  1. a per-collective comparison table against the MIT aicr-benchmarks B200
     2-node reference (Table 2 of results_b200.md), and
  2. a per-collective bus-bandwidth-vs-message-size table.

`busbw` (GB/s) is the figure of merit. Reference collectives are measured at
16 GPUs (8/node) except sendrecv, whose reference figure is a per-pair bidir
measurement; when this run uses a different GPU count for sendrecv that is
flagged in the summary rather than compared blindly.

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
# Table 2 rows, e.g.:
#   | sendrecv | 26.6 | 26.6 | 26.7 (per-pair bidir) | **~100%** | ... |
#   | allreduce | 163 | 163 | 214 | 76% | ... |
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


def parse_reference(path):
    """Return {collective: {algbw, busbw, gdrdma_max, pct}} for every Table 2 row."""
    ref = {}
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
                if m:
                    ref[m.group(1).lower()] = {
                        "algbw": float(m.group(2)),
                        "busbw": float(m.group(3)),
                        "gdrdma_max": float(m.group(4)),
                        "pct": float(m.group(5)),
                    }
    except FileNotFoundError:
        return {}
    return ref


def _finish_segment(seg):
    """Turn an accumulated segment dict into a result, or None if it has no data."""
    rows = seg["rows"]
    if not rows:
        return None
    big = max(rows, key=lambda r: r["size"])
    converged = max(big["oop_busbw"], big["ip_busbw"])
    peak = max(max(r["oop_busbw"], r["ip_busbw"]) for r in rows)
    peak_size = max(rows, key=lambda r: max(r["oop_busbw"], r["ip_busbw"]))["size"]
    ok = all(r["oop_wrong"] in ("0", "N/A") and r["ip_wrong"] in ("0", "N/A")
             for r in rows) and (seg["oob"] is None or seg["oob"][0] == 0)
    # GPUs in this collective's communicator = distinct ranks in its device list;
    # fall back to cfg nGpus if the run printed no per-rank lines.
    ngpus = len({d["rank"] for d in seg["devices"]})
    if ngpus == 0:
        ngpus = seg["cfg"]["nGpus"] if seg["cfg"] else 1
    return {
        "program": seg["program"], "cfg": seg["cfg"], "rows": rows,
        "avg": seg["avg"], "converged": converged, "converged_size": big["size"],
        "peak": peak, "peak_size": peak_size, "ok": ok, "ngpus": ngpus,
    }


def parse_file(path):
    """Parse one output file into {pair, gpu_name, collectives: [segment,...]}."""
    all_devices = []
    segments = []
    seg = None

    def new_seg(program):
        return {"program": program, "cfg": None, "rows": [], "avg": None,
                "oob": None, "devices": []}

    with open(path) as fh:
        for line in fh:
            m = PROG_RE.search(line)
            if m and m.group(1).endswith("_perf"):
                if seg is not None:
                    segments.append(seg)
                seg = new_seg(m.group(1))
                continue
            if seg is None:
                # data before the first program marker (single-collective legacy file)
                seg = new_seg(None)
            m = DEVICE_RE.search(line)
            if m:
                dev = {"rank": int(m.group(1)), "node": m.group(2), "name": m.group(4)}
                seg["devices"].append(dev)
                all_devices.append(dev)
                continue
            m = CONFIG_RE.search(line)
            if m:
                seg["cfg"] = {
                    "nThread": int(m.group(1)), "nGpus": int(m.group(2)),
                    "minBytes": int(m.group(3)), "maxBytes": int(m.group(4)),
                    "warmup": int(m.group(5)), "iters": int(m.group(6)),
                }
                continue
            m = DATA_RE.match(line)
            if m:
                seg["rows"].append({
                    "size": int(m.group(1)),
                    "oop_time": float(m.group(6)), "oop_busbw": float(m.group(8)),
                    "oop_wrong": m.group(9),
                    "ip_time": float(m.group(10)), "ip_busbw": float(m.group(12)),
                    "ip_wrong": m.group(13),
                })
                continue
            m = AVG_RE.search(line)
            if m:
                seg["avg"] = float(m.group(1))
            m = OOB_RE.search(line)
            if m:
                seg["oob"] = (int(m.group(1)), m.group(2))
    if seg is not None:
        segments.append(seg)

    collectives = [c for c in (_finish_segment(s) for s in segments) if c]
    if not collectives:
        return None
    node_pair = ("+".join(sorted({d["node"] for d in all_devices}))
                 or os.path.basename(path))
    gpu_name = all_devices[0]["name"] if all_devices else "?"
    return {"path": path, "pair": node_pair, "gpu_name": gpu_name,
            "nnodes": len({d["node"] for d in all_devices}),
            "collectives": collectives}


def coll_name(program):
    # binary stem sans "_perf" == the reference Table-2 row name
    # (sendrecv, all_reduce, all_gather, reduce_scatter, alltoall, ...)
    return (program or "?").replace("_perf", "")


def build(runs, reference):
    L = []
    run = runs[0]
    gpu = run["gpu_name"]
    # representative config (first collective) for the header
    seg0 = run["collectives"][0]
    cfg = seg0["cfg"]
    nnodes = run["nnodes"] or 2
    tot = seg0["ngpus"]
    gpn = tot // nnodes if nnodes else tot

    L.append("# nccl-tests 2-node summary (multi-collective)")
    L.append("")
    L.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- Runs: {', '.join(r['pair'] for r in runs)}")
    L.append(f"- GPUs: {gpn}/node x {nnodes} nodes = {tot} x {gpu} "
             f"(inter-node, InfiniBand + GPUDirect RDMA)")
    if cfg:
        L.append(f"- Config: {fmt_size(cfg['minBytes'])}-{fmt_size(cfg['maxBytes'])}, "
                 f"{cfg['warmup']} warmup + {cfg['iters']} iters")
    L.append("- Reference: MIT aicr-benchmarks `results_b200.md` Table 2 "
             "(b0029+b0030, 16x B200 / NDR IB). busbw is the figure of merit.")
    L.append("")

    # 1. Per-collective comparison vs reference
    L.append("## Per-collective busbw vs B200 reference")
    L.append("")
    multi = len(runs) > 1
    pair_col = "Node pair | " if multi else ""
    pair_sep = "-----------|" if multi else ""
    L.append(f"| {pair_col}Collective | GPUs | converged busbw (GB/s) "
             "| peak busbw (GB/s) | reference busbw (GB/s) | ours / ref "
             "| correctness |")
    L.append(f"|{pair_sep}------------|-----:|-----------------------:"
             "|------------------:|-----------------------:|-----------:"
             "|:-----------:|")
    caveats = []
    for r in runs:
        for seg in r["collectives"]:
            name = coll_name(seg["program"])
            g = seg["ngpus"]
            rf = reference.get(name)
            refbw = f"{rf['busbw']:.1f}" if rf else "—"
            # sendrecv reference is a per-pair (2-GPU) measurement; only compare
            # like-for-like, otherwise flag it.
            comparable = rf and not (name == "sendrecv" and g != 2)
            ratio = f"{100*seg['converged']/rf['busbw']:.0f}%" if comparable else "—"
            if rf and name == "sendrecv" and g != 2:
                caveats.append(
                    f"sendrecv here uses {g} GPUs (ring), but the reference "
                    f"{rf['busbw']:.1f} GB/s is a per-pair (2-GPU) bidir figure, so the "
                    "two are not directly comparable and `ours / ref` is left blank.")
            pair_cell = f"{r['pair']} | " if multi else ""
            L.append(f"| {pair_cell}{name} | {g} | {seg['converged']:.1f} "
                     f"| {seg['peak']:.1f} | {refbw} | {ratio} "
                     f"| {'PASS' if seg['ok'] else 'FAIL'} |")
    L.append("")
    L.append("Converged = busbw at the largest message size, best of out-of-place / "
             "in-place (matches the reference methodology).")
    if caveats:
        L.append("")
        for c in dict.fromkeys(caveats):   # de-dup, preserve order
            L.append(f"> Note: {c}")
    L.append("")

    # 2. Per-collective bandwidth vs message size
    L.append("## Bus bandwidth vs message size (GB/s)")
    L.append("")
    for r in runs:
        multi_pair = len(runs) > 1
        for seg in r["collectives"]:
            title = coll_name(seg["program"])
            if multi_pair:
                title = f"{r['pair']} — {title}"
            L.append(f"### {title}")
            L.append("")
            L.append("| Message size | OOP time | OOP busbw | IP time | IP busbw |")
            L.append("|-------------:|---------:|----------:|--------:|---------:|")
            for row in seg["rows"]:
                ot = (f"{row['oop_time']/1000:.2f} ms" if row["oop_time"] >= 1000
                      else f"{row['oop_time']:.1f} us")
                it = (f"{row['ip_time']/1000:.2f} ms" if row["ip_time"] >= 1000
                      else f"{row['ip_time']:.1f} us")
                L.append(f"| {fmt_size(row['size'])} | {ot} | {row['oop_busbw']:.1f} "
                         f"| {it} | {row['ip_busbw']:.1f} |")
            L.append("")
    L.append("OOP = out-of-place, IP = in-place.")
    L.append("")

    # 3. Network fabric (static: measured on the B200 nodes 2026-07-13)
    L.append("## Network fabric")
    L.append("")
    L.append("The inter-node data path on the B200 nodes is **NDR (400 Gb/s)**:")
    L.append("")
    L.append("| NICs | Rate | Role |")
    L.append("|------|------|------|")
    L.append("| mlx5_4, 7, 8, 9, 10, 13, 14, 15 | **400 Gb/s (4X NDR)** | "
             "8 GPU compute rails (active) |")
    L.append("| mlx5_0, 1, 2, 3 | 100 Gb/s (HDR100) | secondary (storage/mgmt) |")
    L.append("| mlx5_5, 6, 11, 12 | down | unused |")
    L.append("")
    L.append("`nvidia_peermem` is loaded on both nodes, enabling GPUDirect RDMA so the "
             "NIC DMAs directly to/from GPU HBM over InfiniBand.")
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
    reference = parse_reference(REFERENCE_FILE)
    md = build(runs, reference)
    summary = os.path.join(OUT_DIR, "summary.md")
    with open(summary, "w") as fh:
        fh.write(md)
    print(md)
    ncol = sum(len(r["collectives"]) for r in runs)
    print(f"Written to {summary}  ({len(runs)} run(s), {ncol} collective segment(s))")


if __name__ == "__main__":
    main()
