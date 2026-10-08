#!/usr/bin/env python3
"""Summarize run_sendrecv_check.sh into results/ubuntu/sendrecv-check.md.
Usage: analyze_sendrecv_check.py <logs/node6100/rccl/sendrecv_check_TS>"""
import glob, os, re, sys

RAIL_GBS = 50.0  # one 400 Gb/s rail
out = sys.argv[1].rstrip("/")
root = "/orcd/data/orcd/022/benchmarks/amd-ubuntu"
dst = os.path.join(root, "results/ubuntu/sendrecv-check.md")

def ib(tag):
    f = os.path.join(out, f"ib_{tag}.log")
    try:
        lines = open(f, errors="replace").read().splitlines()
    except OSError:
        return None
    for i, l in enumerate(lines):
        if "#bytes" in l and i + 1 < len(lines):
            p = lines[i + 1].split()
            if len(p) >= 4:
                try:
                    return float(p[3])
                except ValueError:
                    pass
    return None

def sendrecv(d):
    logs = glob.glob(os.path.join(d or "", "sendrecv_ppn*_np*.log"))
    if not logs:
        return None
    rows = [l.split() for l in open(logs[0], errors="replace")
            if re.match(r"^ *\d+ +\d+ +[a-z]", l)]
    if not rows:
        return None
    last = rows[-1]                       # 16 GiB: out-of-place busbw [7], in-place [11]
    best = max(max(float(r[7]), float(r[11])) for r in rows)
    return float(last[11]), best

def channels(d):
    try:
        for l in open(os.path.join(d, "rccl_summary.txt")):
            if l.startswith("channels"):
                return "GDRDMA" if "GDRDMA" in l else "no GDRDMA"
    except (OSError, TypeError):
        pass
    return "-"

L = ["# 2-node sendrecv check (node6100 → node6101)", "",
     f"Raw logs: `{os.path.relpath(out, root)}`. Script: `rccl-tests/run_sendrecv_check.sh`.",
     "Question: why is 2-node sendrecv only 57% of one rail (28.6 of 50 GB/s) when the other "
     "collectives reach 92–95% of 400 GB/s?", "",
     "## A. Raw RDMA on one rail (ib_write_bw, host memory, 1 MiB, 10 s)", "",
     "| path | queue pairs | Gb/s | % of 400 |", "|---|---:|---:|---:|"]
for tag, path, q in [("same_q1", "ionic_0 → ionic_0 (same rail)", 1),
                     ("same_q2", "ionic_0 → ionic_0 (same rail)", 2),
                     ("same_q4", "ionic_0 → ionic_0 (same rail)", 4),
                     ("cross_q1", "ionic_7 → ionic_0 (cross rail)", 1),
                     ("cross_q4", "ionic_7 → ionic_0 (cross rail)", 4)]:
    v = ib(tag)
    L.append(f"| {path} | {q} | {v:,.0f} | {v / 4:.0f}% |" if v else f"| {path} | {q} | failed | – |")

L += ["", "## B. RCCL sendrecv with different settings (16 GiB, busbw)", "",
      "| run | ranks per node | setting | busbw at 16 GiB (GB/s) | best (GB/s) | % of one rail | transport |",
      "|---|---:|---|---:|---:|---:|---|"]
base = None
try:
    runs = [l.rstrip("\n").split("|") for l in open(os.path.join(out, "rccl_runs.txt"))]
except OSError:
    runs = []
for tag, ppn, env, d in runs:
    r = sendrecv(d)
    if not r:
        L.append(f"| {tag} | {ppn} | `{env or 'default'}` | failed | – | – | – |"); continue
    if tag == "default":
        base = r[0]
    L.append(f"| {tag} | {ppn} | `{env or 'default'}` | {r[0]:.1f} | {r[1]:.1f} | "
             f"{100 * r[0] / RAIL_GBS:.0f}% | {channels(d)} |")
L += ["", "How to read it:",
      "- **A, 1 queue pair well below 400 Gb/s** → one RDMA connection cannot fill a rail; "
      "RCCL settings that add queue pairs or channels (B: qps*, nch*) should then help.",
      "- **A, cross rail slower than same rail** → the 16-rank sendrecv pair GPU7@node6100 → GPU0@node6101 "
      "goes from rail 7 to rail 0 and leaves its rail; the `pxn` run (route through the GPU on the matching rail, "
      "as NCCL does on B200) should then help.",
      "- **B, `default_ppn1`** (GPU0 ↔ GPU0, same rail) vs `default` (16 ranks, includes the cross-rail pair) "
      "separates the two causes.", ""]
if base:
    L.append(f"Baseline `default` run: {base:.1f} GB/s ({100 * base / RAIL_GBS:.0f}% of one rail).")
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "w").write("\n".join(L) + "\n")
print("\n".join(L))
