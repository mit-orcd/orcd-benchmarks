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

# ---- hardware ceiling for THIS cluster (measured from ibstat on node5500-5502) --
# Each B200 has one NDR rail: 400 Gb/s = 50 GB/s per direction. PCIe Gen5 x16 is
# full-duplex (~63 GB/s each way), so the NIC — not the PCIe DMA path — is the
# binding constraint in both directions. 8 rails per node.
NIC_GBPS = 50.0                       # per GPU-NIC pair, per direction
NICS_PER_NODE = 8
AGG_GBPS = NIC_GBPS * NICS_PER_NODE   # 400 GB/s per node, per direction
# sendrecv's busbw is a single pair's rate; every other collective here is
# ring/symmetric or root-anchored and drives all 8 rails concurrently.
PER_PAIR_COLLECTIVES = {"sendrecv"}


def hw_ceiling(collective):
    """(ceiling GB/s, basis) for this cluster's NDR fabric."""
    if collective in PER_PAIR_COLLECTIVES:
        return NIC_GBPS, "per-pair"
    return AGG_GBPS, "node aggregate"

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


# One Rocky 8 node pair carried as a reference column, so the Ubuntu results
# can be read against the older nodes. The three Rocky pairs agree closely
# (see ../b200-nodes/out-nccl-2node/summary.md for the full set).
ROCKY_DIR = "/orcd/data/orcd/022/benchmarks/b200-nodes/out-nccl-2node"
ROCKY_PAIR = "5500-5502"


def parse_rocky_ref():
    """Newest Rocky 8 run for ROCKY_PAIR, same parser. None if absent."""
    files = sorted(glob.glob(os.path.join(ROCKY_DIR, f"*{ROCKY_PAIR}*.out")),
                   key=os.path.getmtime)
    return parse_file(files[-1]) if files else None


def pct_diff(ubuntu, rocky):
    """Signed % difference vs the Rocky 8 reference.
    '+' means the Ubuntu pair is faster, '-' means slower."""
    if not rocky:
        return "—"
    return f"{(ubuntu - rocky) / rocky * 100:+.1f}%"


def why_section(L, rocky):
    """Static narrative explaining the Ubuntu vs Rocky 8 differences.
    Numbers cited come from the tables above and a repeat scatter run."""
    if not rocky:
        return
    L.append("## 2. Why three collectives differ between Ubuntu and Rocky 8")
    L.append("")
    L.append("**The dominant pattern is a latency advantage on the Ubuntu "
             "nodes that decays with message size.** At 1 MiB the Ubuntu pair "
             "leads on every collective (alltoall +203%, scatter +277%, "
             "reduce +121%, all_reduce +109%, broadcast +103%), and the gap "
             "shrinks monotonically as messages grow. That is the signature "
             "of a lower fixed per-transfer cost, not of more bandwidth.")
    L.append("")
    L.append("**Collectives that reach the fabric ceiling converge.** reduce, "
             "broadcast, all_gather, reduce_scatter and sendrecv all run at "
             "89-98% of `HW max` at 16 GiB, and there they land within +-2% of "
             "Rocky 8. Once the wire is the constraint, the OS and driver "
             "stack cannot help.")
    L.append("")
    L.append("**all_reduce and alltoall are not special cases** — they are "
             "simply the only collectives that never reach the ceiling (67% "
             "and 14% of `HW max`). all_reduce is bound by its ring/tree "
             "schedule; alltoall is 15 separate peer transfers per GPU and is "
             "latency-bound throughout. With headroom left, the per-transfer "
             "advantage still shows at 16 GiB: **+15.5%** and **+11.1%**.")
    L.append("")
    L.append("**scatter is the one real large-message regression.** Ubuntu "
             "leads it at small sizes (+277% at 1 MiB, +23% at 256 MiB), then "
             "**plateaus at ~290 GB/s** from 1 GiB onward while Rocky 8 climbs "
             "to 327. scatter is root-anchored — one GPU feeds all 15 peers, 8 "
             "of them remote — so it is bound by that root node\'s outbound "
             "aggregate, and the Ubuntu pair hits a lower ceiling there. The "
             "plateau reproduces exactly (289.9 GB/s in the sweep, 290.1 GB/s "
             "on a repeat run), so it is not run-to-run noise on this side.")
    L.append("")
    L.append("Two caveats on that last point. The **mechanism is not "
             "established** — confirming it needs `NCCL_DEBUG=INFO` "
             "channel/protocol inspection on both clusters. And the Rocky 8 "
             "figure rests on a **single run** whose curve jumps oddly from "
             "236 GB/s at 1 GiB to 318 at 4 GiB, so part of the gap may be "
             "variance in that measurement.")
    L.append("")
    L.append("Note the contrast with **gather**, root-anchored like scatter: "
             "both clusters plateau at exactly 92.9 GB/s (0.0% difference). "
             "Where a structural limit binds, the two are identical — which is "
             "what makes scatter\'s asymmetry worth a closer look rather than "
             "dismissing it as OS noise.")
    L.append("")
    L.append("**Suspected cause of the latency advantage.** See "
             "*Why small messages favour the Ubuntu nodes* in section 4 for the "
             "systematic version. In short: the cost is paid **per network "
             "operation**, not per byte. `sendrecv` and `gather` — the two "
             "collectives NCCL does not split across its 8 channels — are "
             "identical on both clusters at every size, which rules out a bulk "
             "GPUDirect or bandwidth cap as the explanation. The remaining "
             "candidates are IOTLB pressure from many small descriptors under the "
             "IOMMU the Rocky 8 nodes boot with, and per-operation CPU cost in "
             "NCCL\'s proxy thread. The builds (CUDA 13.1 vs 12.9) and drivers "
             "(590.48.01 vs 570.211.01) also differ, so this data alone cannot "
             "single one out.")
    L.append("")



def small_message_section(L, main_run, rocky):
    """Systematic analysis of the small-message advantage, computed from the runs.

    Compares per-operation *time* (not bandwidth) at the smallest message sizes,
    and uses every available Rocky 8 pair as a control.
    """
    if not rocky:
        return
    MiB = 1024 ** 2
    ub = {coll_name(s["program"]): s for s in main_run["collectives"]}
    rk = {coll_name(s["program"]): s for s in rocky["collectives"]}

    def t_at(seg, size):
        r = next((x for x in seg["rows"] if x["size"] == size), None)
        return r["oop_time"] if r else None

    names = [n for n in ("sendrecv", "gather", "broadcast", "all_reduce", "reduce",
                         "reduce_scatter", "all_gather", "scatter", "alltoall")
             if n in ub and n in rk]
    if not names:
        return

    L.append("### Why small messages favour the Ubuntu nodes")
    L.append("")
    L.append("At 1 MiB the bandwidth columns above are really a *latency* "
             "measurement: the transfer is too short for bandwidth to matter, so "
             "busbw is dominated by fixed per-operation cost. Comparing the raw "
             "**times** is therefore the cleaner view.")
    L.append("")
    L.append("| Collective | Ubuntu 1 MiB (us) | Rocky 8 1 MiB (us) | Rocky / Ubuntu |")
    L.append("|------------|------------------:|-------------------:|---------------:|")
    for n in names:
        u, r = t_at(ub[n], MiB), t_at(rk[n], MiB)
        if u and r:
            L.append(f"| {n} | {u:.1f} | {r:.1f} | **{r/u:.2f}x** |")
    L.append("")
    L.append("**The advantage is not uniform — and the exceptions are the "
             "finding.** `sendrecv` and `gather` show essentially no difference "
             "(~1.0-1.1x) at 1 MiB *and at every larger size*, while every other "
             "collective costs 1.9-3.0x more time on Rocky 8. Any explanation has "
             "to account for that split.")
    L.append("")

    # decay of the gap with message size, averaged over the affected collectives
    affected = [n for n in names if n not in ("sendrecv", "gather")]
    sizes = sorted({r["size"] for n in affected for r in ub[n]["rows"]})
    L.append("| Message size | mean Rocky/Ubuntu time (affected collectives) |")
    L.append("|-------------:|----------------------------------------------:|")
    for sz in sizes:
        ratios = []
        for n in affected:
            u, r = t_at(ub[n], sz), t_at(rk[n], sz)
            if u and r:
                ratios.append(r / u)
        if ratios:
            L.append(f"| {fmt_size(sz)} | {sum(ratios)/len(ratios):.2f}x |")
    L.append("")
    L.append("The gap decays steadily with size, which is the signature of a "
             "**fixed per-operation cost** rather than a per-byte (bandwidth) "
             "one: a constant overhead is a large fraction of a 1 MiB transfer "
             "and a negligible one of a 16 GiB transfer.")
    L.append("")
    L.append("**What the data rules out.**")
    L.append("")
    L.append("- *NCCL version or topology* — both clusters run NCCL 2.29.2 with "
             "the same 16-rank, 8-GPU-per-node layout. Only the CUDA flavour "
             "differs (12.9 here vs 13.1 there, forced by the r570 driver).")
    L.append("- *A single bad node pair* — all three Rocky 8 pairs "
             "(5500+5501, 5500+5502, 5501+5502) show the same slow small-message "
             "times, so this is systematic, not one degraded node.")
    L.append("- *A bulk GPUDirect/bandwidth cap* — this is the important one. "
             "`sendrecv` moves its 1 MiB as one contiguous chunk per pair over "
             "the same NIC and GPU-memory path, and it is **identical** on both "
             "clusters (52 us vs 50 us) and at line rate at large sizes. A "
             "degraded bulk GDR path would slow `sendrecv` too. It does not.")
    L.append("")
    L.append("**What remains.** The collectives that differ are exactly those "
             "that split their payload across NCCL's 8 parallel channels and run "
             "multiple phases: a 1 MiB all_gather becomes 8 chunks of 128 KiB "
             "plus cross-phase synchronisation, where a 1 MiB sendrecv is one "
             "chunk. So the extra cost on Rocky 8 is paid **per network "
             "operation and per synchronisation**, not per byte — consistent with "
             "IOTLB pressure from many small scattered descriptors under the "
             "IOMMU those nodes boot with, and/or with per-operation CPU cost in "
             "NCCL's proxy thread, which posts each RDMA operation. `gather` fits "
             "the same rule from the other side: it is a fan-in to a single root "
             "that NCCL does not spread across channels, and it shows no gap.")
    L.append("")
    L.append("Distinguishing IOTLB pressure from proxy-thread CPU cost needs a "
             "test this data cannot provide: an `ib_write_bw` sweep at small "
             "message sizes (many small ops vs one large op) between two Rocky "
             "nodes and between node5700/node5701, or a boot with `iommu=off` on "
             "one Rocky node. Both are outside what these benchmark runs measure.")
    L.append("")


def build(runs, reference, rocky=None):
    L = []
    run = runs[0]
    # collective -> converged busbw for the Rocky 8 reference pair
    rbw = {coll_name(s["program"]): s["converged"]
           for s in rocky["collectives"]} if rocky else {}
    rlabel = f"Rocky 8 ({rocky['pair']})" if rocky else None
    gpu = run["gpu_name"]
    # representative config (first collective) for the header
    seg0 = run["collectives"][0]
    cfg = seg0["cfg"]
    nnodes = run["nnodes"] or 2
    tot = seg0["ngpus"]
    gpn = tot // nnodes if nnodes else tot

    L.append("# nccl-tests 2-node summary — Ubuntu B200 nodes")
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
    if rocky:
        L.append(f"- **node5700+node5701 run Ubuntu 24.04; node5500-5502 run "
                 f"Rocky 8.** One Rocky 8 node pair (**{rocky['pair']}**, newest "
                 f"run) is carried in the tables below as a reference, with a "
                 f"signed difference column: **+** means the Ubuntu pair is "
                 f"faster, **-** slower. Full Rocky 8 set: "
                 f"`../b200-nodes/out-nccl-2node/summary.md`.")
    L.append("")

    # 1. Per-collective comparison vs reference
    L.append("## 1. Results — bandwidth for every collective")
    L.append("")
    L.append("A *collective* is one communication pattern that all 16 GPUs take "
             "part in together (e.g. `all_reduce` sums a buffer across every GPU; "
             "`broadcast` sends one GPU's buffer to all). Each row is one such "
             "pattern, measured at 1 MiB-16 GiB; the figure of merit is **busbw** "
             "(bus bandwidth, GB/s) at the largest message size.")
    L.append("")
    # One representative node pair keeps the table readable; the others are
    # summarised as a spread underneath.
    main_run = runs[0]
    other_runs = runs[1:]
    L.append(f"Representative node pair: **{main_run['pair']}**.")
    L.append("")
    rcols = f" {rlabel} (GB/s) | vs Rocky 8 |" if rocky else ""
    rseps = "---------------:|-----------:|" if rocky else ""
    # the aicr reference columns are only useful when that file was found
    fcols = " reference busbw (GB/s) | ours / ref |" if reference else ""
    fseps = "-----------------------:|-----------:|" if reference else ""
    L.append("| Collective | GPUs | Ubuntu busbw (GB/s) |" + rcols + fcols
             + " HW max (GB/s) | % of HW max | correctness |")
    L.append("|------------|-----:|-----------------------:|" + rseps + fseps
             + "--------------:|--------------:|:-----------:|")
    caveats = []
    for r in [main_run]:
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
            hwmax, _basis = hw_ceiling(name)
            hw_pct = f"{100*seg['converged']/hwmax:.0f}%"
            rcell = ""
            if rocky:
                rv = rbw.get(name)
                rcell = (f" {rv:.1f} | {pct_diff(seg['converged'], rv)} |"
                         if rv else " — | — |")
            fcell = f" {refbw} | {ratio} |" if reference else ""
            L.append(f"| {name} | {g} | {seg['converged']:.1f} |" + rcell + fcell
                     + f" {hwmax:.0f} | {hw_pct} "
                     f"| {'PASS' if seg['ok'] else 'FAIL'} |")
    L.append("")
    L.append("Converged = busbw at the largest message size, best of out-of-place / "
             "in-place (matches the reference methodology).")

    # How closely do the remaining pairs track the representative one?
    if other_runs:
        base = {coll_name(s["program"]): s["converged"]
                for s in main_run["collectives"]}
        worst_pct, worst_what = 0.0, None
        for r in other_runs:
            for seg in r["collectives"]:
                nm = coll_name(seg["program"])
                b = base.get(nm)
                if b:
                    d = 100 * abs(seg["converged"] - b) / b
                    if d > worst_pct:
                        worst_pct, worst_what = d, f"{nm} on {r['pair']}"
        names_o = ", ".join(r["pair"] for r in other_runs)
        agree = "essentially identical" if worst_pct <= 5 else "very close"
        L.append("")
        L.append(f"The other node pair(s) — {names_o} — give {agree} results and "
                 f"are omitted here to keep the table readable: across every "
                 f"collective the largest deviation from {main_run['pair']} is "
                 f"**{worst_pct:.1f}%** ({worst_what}). No pair stands out as slow, "
                 f"so the fabric behaves the same whichever two of the three nodes "
                 f"are used. Per-pair message-size detail for all pairs is in the "
                 f"next section.")
    L.append("")
    L.append(f"`HW max` is the hardware ceiling of **this** cluster's fabric, not a "
             f"figure taken from any paper. Each B200 owns one NDR rail at 400 Gb/s "
             f"= **{NIC_GBPS:.0f} GB/s per direction**, and each node has "
             f"**{NICS_PER_NODE} rails** (mlx5_4/7/8/9/10/13/14/15, confirmed by "
             f"`ibstat`), so:")
    L.append("")
    L.append(f"- **sendrecv** — busbw is one pair's rate => ceiling "
             f"**{NIC_GBPS:.0f} GB/s**.")
    L.append(f"- **all other collectives** — ring/symmetric or root-anchored, "
             f"driving all {NICS_PER_NODE} rails concurrently => ceiling "
             f"{NICS_PER_NODE} x {NIC_GBPS:.0f} = **{AGG_GBPS:.0f} GB/s** per node "
             f"per direction.")
    L.append("")
    L.append("The NIC is the binding constraint in both directions because PCIe "
             "Gen5 x16 is full-duplex (~63 GB/s *each* way), comfortably above the "
             f"{NIC_GBPS:.0f} GB/s rail. A collective well below its ceiling is "
             "limited by the NCCL algorithm, not by this hardware.")
    L.append("")

    # ---- Interpretation: is each percentage expected? ----------------------
    why_section(L, rocky)

    L.append("## 3. How close each collective gets to the hardware limit")
    L.append("")
    L.append(f"Dividing each result by one rail's line rate ({NIC_GBPS:.0f} GB/s) "
             f"gives the most useful view: **how many of the {NICS_PER_NODE} rails "
             f"the collective actually engages**.")
    L.append("")
    L.append(f"| Collective | ours / HW max | effective rails (of {NICS_PER_NODE}) "
             f"| verdict |")
    L.append("|------------|--------------:|----------------:|---------|")
    VERDICT = {
        "sendrecv": "at line rate",
        "reduce_scatter": "at fabric limit",
        "reduce": "at fabric limit",
        "broadcast": "at fabric limit",
        "all_gather": "at fabric limit",
        "scatter": "expected (root-anchored)",
        "all_reduce": "expected Ring two-pass penalty",
        "gather": "NCCL algorithm limit",
        "alltoall": "NCCL algorithm limit",
    }
    rows_i = []
    for seg in main_run["collectives"]:
        nm = coll_name(seg["program"])
        hwmax, _b = hw_ceiling(nm)
        rows_i.append((nm, 100 * seg["converged"] / hwmax,
                       seg["converged"] / NIC_GBPS))
    for nm, pct, rails in sorted(rows_i, key=lambda x: -x[1]):
        note = "(per pair)" if nm in PER_PAIR_COLLECTIVES else ""
        L.append(f"| {nm} | {pct:.0f}% | {rails:.2f} {note} "
                 f"| {VERDICT.get(nm, '—')} |")
    L.append("")
    # Computed from this run's own numbers, so the prose cannot drift from the
    # table above (the Rocky-8 version of this script hard-coded them).
    def _pct(nm):
        seg = next((s for s in main_run["collectives"]
                    if coll_name(s["program"]) == nm), None)
        if not seg:
            return None
        return 100 * seg["converged"] / hw_ceiling(nm)[0]

    at_limit = [(nm, _pct(nm)) for nm in
                ("reduce_scatter", "reduce", "all_gather", "broadcast")]
    at_limit = [(nm, v) for nm, v in at_limit if v is not None]
    sr, ar = _pct("sendrecv"), _pct("all_reduce")
    a2a, gat, sca = _pct("alltoall"), _pct("gather"), _pct("scatter")

    if sr and at_limit:
        lo = min(v for _, v in at_limit)
        hi = max(v for _, v in at_limit)
        L.append(f"**At the fabric limit ({lo:.0f}-{hi:.0f}%).** `sendrecv` is the "
                 f"cleanest validation in the table: each GPU saturates its own "
                 f"rail, so {sr:.0f}% of {NIC_GBPS:.0f} GB/s means nothing is left "
                 f"on the table — it is the single number that certifies the "
                 f"fabric is healthy. The ring collectives "
                 f"({', '.join('`%s`' % nm for nm, _ in at_limit)}) sit at "
                 f"{lo*NICS_PER_NODE/100:.1f}-{hi*NICS_PER_NODE/100:.1f} effective "
                 f"rails because NCCL runs {NICS_PER_NODE} parallel ring channels, "
                 f"each crossing the node boundary on a different rail; the "
                 f"missing few percent is ring fill/drain and protocol overhead, "
                 f"which cannot be recovered.")
        L.append("")
    if ar and sca:
        L.append(f"**Expected shortfalls.** `all_reduce` at {ar:.0f}% is the Ring "
                 f"two-pass penalty: it runs reduce_scatter then all_gather, and "
                 f"the busbw formula already divides out the doubled traffic "
                 f"(factor 2(N-1)/N), so a perfectly pipelined all_reduce would "
                 f"score the *same* as all_gather. It does not, because the ring "
                 f"fills and drains twice and pays the phase-transition latency — "
                 f"a fixed cost that does not shrink as bandwidth grows. "
                 f"`scatter` at {sca:.0f}% is root-anchored and unidirectional, "
                 f"limited by the root GPU's own outbound capacity.")
        L.append("")
    if a2a and gat:
        L.append(f"**Algorithm-limited, and the numbers say so precisely.** "
                 f"`alltoall` at {a2a:.0f}% engages roughly "
                 f"**{a2a*NICS_PER_NODE/100:.1f} of the {NICS_PER_NODE} rails** — a "
                 f"literal quantification of NCCL's N^2 point-to-point transfers "
                 f"not being pipelined across NICs. `gather` at {gat:.0f}% is about "
                 f"{gat*NICS_PER_NODE/100:.1f} rails, the same story for fan-in to "
                 f"a single root. Neither is a fabric problem: a faster network "
                 f"barely helps a collective that does not use it.")
        L.append("")
    L.append(f"> **Caveat on the denominators.** The {AGG_GBPS:.0f} GB/s ceiling is "
             "exact for the ring collectives, whose traffic streams around a ring "
             "bottlenecked by its inter-node links. It is an *approximation* for "
             "the root-anchored and all-to-all patterns, where only a fraction of "
             "traffic crosses the node boundary (for alltoall, 8 of each GPU's 15 "
             "peers are remote; the rest go over NVLink). A per-collective ceiling "
             "would shift those percentages — most likely lowering `scatter`'s "
             "apparent figure. It does not change any conclusion: gather and "
             "alltoall are 4-8x below any reasonable ceiling and are algorithm-"
             "bound under every accounting.")
    if caveats:
        L.append("")
        for c in dict.fromkeys(caveats):   # de-dup, preserve order
            L.append(f"> Note: {c}")
    L.append("")

    # 2. Per-collective bandwidth vs message size
    L.append("## 4. Bandwidth vs message size (GB/s)")
    L.append("")
    for r in runs:
        multi_pair = len(runs) > 1
        for seg in r["collectives"]:
            title = coll_name(seg["program"])
            if multi_pair:
                title = f"{r['pair']} — {title}"
            L.append(f"### {title}")
            L.append("")
            # Rocky 8 out-of-place busbw for the same collective, by size
            rrows = {}
            if rocky:
                rseg = next((s for s in rocky["collectives"]
                             if coll_name(s["program"]) == coll_name(seg["program"])),
                            None)
                if rseg:
                    rrows = {r["size"]: r["oop_busbw"] for r in rseg["rows"]}
            rhdr = f" {rlabel} OOP busbw | vs Rocky 8 |" if rrows else ""
            rsep = "---------------:|-----------:|" if rrows else ""
            L.append("| Message size | OOP time | OOP busbw | IP time | IP busbw |"
                     + rhdr)
            L.append("|-------------:|---------:|----------:|--------:|---------:|"
                     + rsep)
            for row in seg["rows"]:
                ot = (f"{row['oop_time']/1000:.2f} ms" if row["oop_time"] >= 1000
                      else f"{row['oop_time']:.1f} us")
                it = (f"{row['ip_time']/1000:.2f} ms" if row["ip_time"] >= 1000
                      else f"{row['ip_time']:.1f} us")
                line = (f"| {fmt_size(row['size'])} | {ot} | {row['oop_busbw']:.1f} "
                        f"| {it} | {row['ip_busbw']:.1f} |")
                if rrows:
                    rv = rrows.get(row["size"])
                    line += (f" {rv:.1f} | {pct_diff(row['oop_busbw'], rv)} |"
                             if rv else " — | — |")
                L.append(line)
            L.append("")
    L.append("OOP = out-of-place, IP = in-place.")
    L.append("")

    small_message_section(L, main_run, rocky)

    # 3. Network fabric (static: measured on the B200 nodes 2026-07-13)
    L.append("## 5. Network fabric")
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
    """Newest result *per collective* per node pair.

    Merging per collective (rather than keeping only the newest file) means a
    targeted re-run of a single collective updates just that collective instead
    of wiping the rest of the sweep from the summary.
    """
    files = args if args else glob.glob(os.path.join(OUT_DIR, "*.out"))
    parsed = {}
    for f in sorted(files, key=os.path.getmtime):   # oldest first, newest wins
        r = parse_file(f)
        if not r:
            continue
        prev = parsed.get(r["pair"])
        if prev is None:
            parsed[r["pair"]] = r
            continue
        by_name = {coll_name(s["program"]): s for s in prev["collectives"]}
        by_name.update({coll_name(s["program"]): s for s in r["collectives"]})
        prev["collectives"] = sorted(by_name.values(),
                                     key=lambda s: coll_name(s["program"]))
    return [parsed[k] for k in sorted(parsed)]


def main():
    runs = collect(sys.argv[1:])
    if not runs:
        sys.exit(f"No parseable nccl-tests 2-node results in {OUT_DIR}")
    reference = parse_reference(REFERENCE_FILE)
    rocky = parse_rocky_ref()      # one Rocky 8 pair, as a reference column
    md = build(runs, reference, rocky)
    summary = os.path.join(OUT_DIR, "summary.md")
    with open(summary, "w") as fh:
        fh.write(md)
    print(md)
    ncol = sum(len(r["collectives"]) for r in runs)
    print(f"Written to {summary}  ({len(runs)} run(s), {ncol} collective segment(s))")


if __name__ == "__main__":
    main()
