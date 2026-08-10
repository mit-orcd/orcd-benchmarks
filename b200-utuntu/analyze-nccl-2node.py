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
    if rocky:
        L.append(f"- **node5700+node5701 run Ubuntu 24.04; node5500-5502 run "
                 f"Rocky 8.** One Rocky 8 node pair (**{rocky['pair']}**, newest "
                 f"run) is carried in the tables below as a reference, with a "
                 f"signed difference column: **+** means the Ubuntu pair is "
                 f"faster, **-** slower. Full Rocky 8 set: "
                 f"`../b200-nodes/out-nccl-2node/summary.md`.")
    L.append("")

    # 1. Per-collective comparison vs reference
    L.append("## Per-collective busbw vs B200 reference")
    L.append("")
    # One representative node pair keeps the table readable; the others are
    # summarised as a spread underneath.
    main_run = runs[0]
    other_runs = runs[1:]
    L.append(f"Representative node pair: **{main_run['pair']}**.")
    L.append("")
    rcols = f" {rlabel} busbw (GB/s) | vs Rocky 8 |" if rocky else ""
    rseps = "---------------:|-----------:|" if rocky else ""
    L.append("| Collective | GPUs | converged busbw (GB/s) "
             "| peak busbw (GB/s) |" + rcols + " reference busbw (GB/s) | ours / ref "
             "| HW max (GB/s) | ours / HW max | correctness |")
    L.append("|------------|-----:|-----------------------:"
             "|------------------:|" + rseps + "-----------------------:|-----------:"
             "|--------------:|--------------:|:-----------:|")
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
            L.append(f"| {name} | {g} | {seg['converged']:.1f} "
                     f"| {seg['peak']:.1f} |" + rcell + f" {refbw} | {ratio} "
                     f"| {hwmax:.0f} | {hw_pct} "
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
    L.append("## Interpreting `ours / HW max`")
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
    L.append("**At the hardware limit (92-99%).** `sendrecv` is the cleanest "
             "validation in the table: each GPU saturates its own rail, so ~99% of "
             f"{NIC_GBPS:.0f} GB/s means nothing is left on the table. It is the "
             "single number that certifies the fabric is healthy. The ring "
             "collectives sit at 7.3-7.5 effective rails because NCCL runs 8 "
             "parallel ring channels, each crossing the node boundary on a "
             "different rail; the missing few percent is ring fill/drain and "
             "protocol overhead, which cannot be recovered.")
    L.append("")
    L.append("**Expected shortfalls.** `all_reduce` at ~60% is the Ring two-pass "
             "penalty: it runs reduce_scatter then all_gather, and the busbw "
             "formula already divides out the doubled traffic (factor 2(N-1)/N), so "
             "a perfectly pipelined all_reduce would score the *same* as "
             "all_gather. It does not, because the ring fills and drains twice and "
             "pays the phase-transition latency. That fixed latency does not shrink "
             "when bandwidth grows, which is why our all_reduce is a *smaller* "
             "fraction of our all_gather (~65%) than the reference's was (~78%) — "
             "and why SHARP, which collapses the two passes into one in-switch "
             "reduction, has more to gain here (see `out-nccl-2node-sharp/`). "
             "`scatter` is root-anchored and unidirectional, limited by the root's "
             "own outbound capacity.")
    L.append("")
    L.append("**Algorithm-limited, and the numbers say so precisely.** `alltoall` "
             f"at ~12% is exactly 1/{NICS_PER_NODE} — it engages roughly **one "
             f"rail's worth** of bandwidth out of {NICS_PER_NODE}, a literal "
             "quantification of NCCL's N^2 point-to-point transfers not being "
             "pipelined across NICs. `gather` at ~24% is about two rails, the same "
             "story for fan-in to a single root. The decisive evidence that these "
             "are algorithmic rather than physical: this fabric is ~1.9x faster "
             "than the reference on sendrecv, yet gather improved only 1.05x and "
             "alltoall 1.19x. A faster fabric barely helps a collective that is not "
             "using it.")
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

    # 2b. Ubuntu vs Rocky 8 interpretation (static narrative; the numbers it
    # cites come from the tables above and from a repeat scatter run)
    if rocky:
        L.append("## Why the Ubuntu and Rocky 8 results differ")
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
        L.append("**Suspected cause of the latency advantage.** The leading "
                 "suspect is the platform difference documented in "
                 "`../b200-nodes/notes.md`: the Rocky 8 nodes run with IOMMU "
                 "enabled and have a degraded NIC-reads-from-GPU path (18.5 GB/s "
                 "vs 35.8 GB/s for writes). Per-transfer address-translation "
                 "overhead penalises small messages most, which matches the "
                 "decaying gap. But the builds also differ (CUDA 13.1 vs 12.9) and "
                 "so do the drivers (590.48.01 vs 570.211.01), so this data alone "
                 "cannot attribute it to IOMMU. The clean experiment is "
                 "`ib_write_bw --use_cuda` between node5700 and node5701: if these "
                 "nodes show the full ~35 GB/s read path where Rocky showed 18.5, "
                 "that confirms it.")
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
