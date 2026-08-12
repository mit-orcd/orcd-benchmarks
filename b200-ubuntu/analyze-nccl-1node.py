#!/usr/bin/env python3
"""Analyze nccl-tests 1-node output(s) and write a Markdown summary.

Each output file may contain several collectives, one per section marked
"%%%%% <program>_perf %%%%%" (as produced by run-nccl-1node.sh). This script
scans every *.out in out-nccl-1node/, groups by node (newest file per node),
splits each file into per-collective sections, and writes a summary to
out-nccl-1node/summary.md:

  - a per-node / per-collective converged-busbw comparison against the MIT
    aicr-benchmarks B200 reference (results_b200.md, Table 1), and
  - a bus-bandwidth-vs-message-size table per collective, one column per node.

The nccl-tests *_perf binaries print one row per message size, e.g.:

    #       size         count  ...     time   algbw   busbw #wrong     time   algbw   busbw #wrong
         1048576        262144  ...    46.41   22.59   22.59      0    49.27   21.28   21.28    N/A

The two triples are out-of-place and in-place; `busbw` (GB/s) is the figure of
merit — higher is better.

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
# reference row: | sendrecv | 666 | 666 | 900 | 74% | ... |  (numbers may be **bold**)
REF_ROW_RE = re.compile(
    r"^\|\s*(\w+)\s*\|\s*\**([\d.]+)\**\s*\|\s*\**([\d.]+)\**\s*\|"
    r"\s*\**([\d.]+)\**\s*\|\s*\**~?([\d.]+)%"
)

# preferred display order (matches reference Table 1)
COLL_ORDER = ["sendrecv", "reduce", "broadcast", "gather", "scatter",
              "reduce_scatter", "all_gather", "all_reduce", "alltoall", "hypercube"]


def fmt_size(n):
    for div, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if n >= div and n % div == 0:
            return f"{n // div} {unit}"
    for div, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def coll_name(program):
    """sendrecv_perf -> sendrecv, all_reduce_perf -> all_reduce."""
    return program[:-5] if program.endswith("_perf") else program


def parse_reference(path):
    """Return {collective: {algbw, busbw, nvlink_max, pct}} from Table 1."""
    ref = {}
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
                if m:
                    ref[m.group(1).lower()] = {
                        "algbw": float(m.group(2)),
                        "busbw": float(m.group(3)),
                        "nvlink_max": float(m.group(4)),
                        "pct": float(m.group(5)),
                    }
    except FileNotFoundError:
        return {}
    return ref


def summarize_rows(rows, oob):
    big = max(rows, key=lambda r: r["size"])
    converged = max(big["oop_busbw"], big["ip_busbw"])
    peak = max(max(r["oop_busbw"], r["ip_busbw"]) for r in rows)
    peak_size = max(rows, key=lambda r: max(r["oop_busbw"], r["ip_busbw"]))["size"]
    ok = all(r["oop_wrong"] in ("0", "N/A") and r["ip_wrong"] in ("0", "N/A")
             for r in rows) and (oob is None or oob[0] == 0)
    return converged, big["size"], peak, peak_size, ok


def parse_file(path):
    """Parse a multi-collective 1-node output file.

    Returns {"node", "gpu_name", "ngpu", "cfg", "collectives": [ ... ]} where
    each collective is {program, coll, rows, avg, converged, converged_size,
    peak, peak_size, ok}. Sections with a program marker but no data rows are
    recorded with status FAILED (rows == []).
    """
    node = gpu_name = None
    cfg = None
    ngpu = None
    order = []           # program names in file order
    sect = {}            # program -> {"rows": [...], "avg", "oob"}
    cur = None

    def ensure(prog):
        if prog not in sect:
            sect[prog] = {"rows": [], "avg": None, "oob": None}
            order.append(prog)
        return prog

    with open(path) as fh:
        for line in fh:
            m = PROG_RE.search(line)
            if m and m.group(1).endswith("_perf"):
                cur = ensure(m.group(1))
                continue
            m = CONFIG_RE.search(line)
            if m:
                cfg = {
                    "nThread": int(m.group(1)), "nGpus": int(m.group(2)),
                    "minBytes": int(m.group(3)), "maxBytes": int(m.group(4)),
                    "warmup": int(m.group(5)), "iters": int(m.group(6)),
                }
                ngpu = cfg["nGpus"]
                continue
            m = DEVICE_RE.search(line)
            if m:
                if node is None:
                    node, gpu_name = m.group(2), m.group(4)
                continue
            m = DATA_RE.match(line)
            if m and cur is not None:
                sect[cur]["rows"].append({
                    "size": int(m.group(1)),
                    "oop_time": float(m.group(6)), "oop_busbw": float(m.group(8)),
                    "oop_wrong": m.group(9),
                    "ip_time": float(m.group(10)), "ip_busbw": float(m.group(12)),
                    "ip_wrong": m.group(13),
                })
                continue
            m = AVG_RE.search(line)
            if m and cur is not None:
                sect[cur]["avg"] = float(m.group(1))
                continue
            m = OOB_RE.search(line)
            if m and cur is not None:
                sect[cur]["oob"] = (int(m.group(1)), m.group(2))

    if not order:
        return None

    collectives = []
    for prog in order:
        s = sect[prog]
        entry = {"program": prog, "coll": coll_name(prog),
                 "rows": s["rows"], "avg": s["avg"]}
        if s["rows"]:
            (entry["converged"], entry["converged_size"], entry["peak"],
             entry["peak_size"], entry["ok"]) = summarize_rows(s["rows"], s["oob"])
        else:
            entry.update(converged=None, converged_size=None, peak=None,
                         peak_size=None, ok=False)
        collectives.append(entry)

    return {"path": path, "node": node or os.path.basename(path),
            "gpu_name": gpu_name or "?", "ngpu": ngpu or (cfg["nGpus"] if cfg else 8),
            "cfg": cfg, "collectives": collectives}


def coll_sort_key(name):
    return (COLL_ORDER.index(name), name) if name in COLL_ORDER else (len(COLL_ORDER), name)


# One Rocky 8 node (node5500-5502 all run Rocky 8) carried as a reference
# column, so the Ubuntu results can be read against the older nodes without
# repeating all three — the three Rocky nodes agree to within a few percent
# (see ../b200-nodes/out-nccl-1node/summary.md for the full set).
ROCKY_DIR = "/orcd/data/orcd/022/benchmarks/b200-nodes/out-nccl-1node"
ROCKY_NODE = "node5502"


def parse_rocky_ref():
    """Newest run of ROCKY_NODE, parsed with the same parser. None if absent."""
    files = sorted(glob.glob(os.path.join(ROCKY_DIR, f"*{ROCKY_NODE}*.out")),
                   key=os.path.getmtime)
    if not files:
        return None
    r = parse_file(files[-1])
    if r:
        r["src"] = files[-1]
    return r


def pct_diff(ubuntu, rocky):
    """Signed % difference of an Ubuntu value vs the Rocky 8 reference.
    '+' means the Ubuntu node is faster, '-' means slower."""
    if not rocky:
        return "—"
    return f"{(ubuntu - rocky) / rocky * 100:+.1f}%"


def build(nodes, reference, rocky=None):
    L = []
    # collective -> entry for the Rocky 8 reference node
    ridx = {c["coll"]: c for c in rocky["collectives"]} if rocky else {}
    rlabel = f"{rocky['node']} (Rocky 8)" if rocky else None
    cfg = nodes[0]["cfg"]
    ngpu = nodes[0]["ngpu"]
    gpu = nodes[0]["gpu_name"]

    # union of collectives across nodes, in reference order
    coll_names = sorted({c["coll"] for n in nodes for c in n["collectives"]},
                        key=coll_sort_key)
    # index: node -> coll -> entry
    idx = {n["node"]: {c["coll"]: c for c in n["collectives"]} for n in nodes}

    L.append("# nccl-tests 1-node collective summary")
    L.append("")
    L.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- Nodes: {', '.join(n['node'] for n in nodes)} "
             f"(each {ngpu} x {gpu}, single node, intra-node NVLink)")
    if cfg:
        L.append(f"- Config: {cfg['nThread']} thread, "
                 f"{fmt_size(cfg['minBytes'])}-{fmt_size(cfg['maxBytes'])}, "
                 f"{cfg['warmup']} warmup + {cfg['iters']} iters")
    L.append(f"- Collectives: {', '.join(coll_names)}")
    if rocky:
        L.append(f"- **node5700/node5701 run Ubuntu 24.04; node5500-5502 run "
                 f"Rocky 8.** One Rocky 8 node (**{rocky['node']}**, newest run) "
                 f"is carried below as a reference column — the three Rocky "
                 f"nodes agree to within a few percent, so one stands in for "
                 f"them all. Full Rocky 8 set: "
                 f"`../b200-nodes/out-nccl-1node/summary.md`.")
    if reference:
        L.append("- Reference: MIT aicr-benchmarks `results_b200.md`, Table 1 "
                 "(b0027, 8x B200, NVLink 5.0 / NVSwitch), busbw at 900 GB/s NVLink max")
    L.append("")
    L.append("Converged busbw = busbw at the largest message size, best of "
             "out-of-place / in-place (matches the reference methodology). "
             "busbw (bus bandwidth) is the figure of merit.")
    L.append("")

    # Main comparison table: rows = collectives, per-node converged + % of ref
    L.append("## Converged bus bandwidth by collective (GB/s)")
    L.append("")
    hdr = "| Collective |"
    sep = "|---|"
    for n in nodes:
        hdr += f" {n['node']} |"
        sep += "---:|"
    if rocky:
        hdr += f" {rlabel} |"
        sep += "---:|"
        for n in nodes:
            hdr += f" {n['node']} vs Rocky 8 |"
            sep += "---:|"
    if reference:
        hdr += " Reference (b0027) |"
        sep += "---:|"
        for n in nodes:
            hdr += f" {n['node']} % of ref |"
            sep += "---:|"
    hdr += " Correctness |"
    sep += "---|"
    L.append(hdr)
    L.append(sep)
    for cn in coll_names:
        row = f"| {cn} |"
        oks = []
        for n in nodes:
            e = idx[n["node"]].get(cn)
            if e and e["converged"] is not None:
                row += f" {e['converged']:.1f} |"
                oks.append(e["ok"])
            elif e:
                row += " FAILED |"
                oks.append(False)
            else:
                row += " — |"
        if rocky:
            re_ = ridx.get(cn)
            rval = re_["converged"] if re_ else None
            if rval is not None:
                row += f" {rval:.1f} |"
            elif re_:
                row += " FAILED |"
            else:
                row += " — |"
            # signed difference of each Ubuntu node against that reference
            for n in nodes:
                e = idx[n["node"]].get(cn)
                if e and e["converged"] is not None and rval:
                    row += f" {pct_diff(e['converged'], rval)} |"
                else:
                    row += " — |"
        ref = reference.get(cn) if reference else None
        if reference:
            row += f" {ref['busbw']:.0f} |" if ref else " — |"
            for n in nodes:
                e = idx[n["node"]].get(cn)
                if e and e["converged"] is not None and ref:
                    row += f" {100*e['converged']/ref['busbw']:.0f}% |"
                else:
                    row += " — |"
        if all(oks):
            row += " PASS |"
        elif any(o for o in oks):
            row += " mixed |"
        else:
            row += " FAIL |"
        L.append(row)
    L.append("")

    # Per-collective busbw vs message size (out-of-place), one column per node
    L.append("## Bus bandwidth vs message size (out-of-place busbw, GB/s)")
    L.append("")
    for cn in coll_names:
        entries = [(n["node"], idx[n["node"]].get(cn)) for n in nodes]
        if rocky:
            entries.append((rlabel, ridx.get(cn)))
        entries = [(nm, e) for nm, e in entries if e and e["rows"]]
        if not entries:
            L.append(f"### {cn}")
            L.append("")
            L.append("_No data (run failed or produced no rows)._")
            L.append("")
            continue
        L.append(f"### {cn}")
        L.append("")
        sizes = sorted({r["size"] for _, e in entries for r in e["rows"]})
        # the Rocky 8 reference is the last entry when present; the Ubuntu mean
        # is compared against it per message size
        rk_entry = entries[-1][1] if (rocky and entries[-1][0] == rlabel) else None
        ub_entries = entries[:-1] if rk_entry else entries
        head = "| Message size | " + " | ".join(nm for nm, _ in entries) + " |"
        seps = "|-------------:|" + "|".join(["------:"] * len(entries)) + "|"
        if rk_entry and ub_entries:
            head += " Ubuntu vs Rocky 8 |"
            seps += "------:|"
        L.append(head)
        L.append(seps)

        def busbw_at(e, sz):
            r = next((r for r in e["rows"] if r["size"] == sz), None)
            return r["oop_busbw"] if r else None

        for sz in sizes:
            cells = []
            for _, e in entries:
                v = busbw_at(e, sz)
                cells.append(f"{v:.1f}" if v is not None else "—")
            line = f"| {fmt_size(sz)} | " + " | ".join(cells) + " |"
            if rk_entry and ub_entries:
                rv = busbw_at(rk_entry, sz)
                uvs = [busbw_at(e, sz) for _, e in ub_entries]
                uvs = [v for v in uvs if v is not None]
                line += (f" {pct_diff(sum(uvs)/len(uvs), rv)} |"
                         if uvs and rv else " — |")
            L.append(line)
        L.append("")
    return "\n".join(L) + "\n"


def collect(args):
    """Return newest-per-node list of parsed results, sorted by node name."""
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
    reference = parse_reference(REFERENCE_FILE)
    rocky = parse_rocky_ref()          # one Rocky 8 node, as a reference column
    md = build(nodes, reference, rocky)
    summary = os.path.join(OUT_DIR, "summary.md")
    with open(summary, "w") as fh:
        fh.write(md)
    print(md)
    print(f"Written to {summary}  ({len(nodes)} node(s))")


if __name__ == "__main__":
    main()
