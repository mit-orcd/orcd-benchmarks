#!/usr/bin/env python3
"""Parse the nccl-tests output written by run-nccl.sh into markdown tables.

Reads out-1node/ and out-2node/, and prints (a) a peak bus-bandwidth table per
collective, (b) the message-size sweep for the headline collectives.
Used by ./make-summary.sh to build summary.md.
"""
import os, re, sys, glob, json

COLL_ORDER = ["sendrecv_perf", "all_reduce_perf", "all_gather_perf",
              "reduce_scatter_perf", "reduce_perf", "broadcast_perf",
              "alltoall_perf", "gather_perf", "scatter_perf", "hypercube_perf"]
PRETTY = {"sendrecv_perf": "SendRecv", "all_reduce_perf": "AllReduce",
          "all_gather_perf": "AllGather", "reduce_scatter_perf": "ReduceScatter",
          "reduce_perf": "Reduce", "broadcast_perf": "Broadcast",
          "alltoall_perf": "AllToAll", "gather_perf": "Gather",
          "scatter_perf": "Scatter", "hypercube_perf": "Hypercube"}


def human(nbytes):
    n = int(nbytes)
    for unit, div in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if n >= div:
            v = n / div
            return f"{v:g}{unit}"
    return str(n)


def parse(path):
    """-> dict(label=..., collectives={name: [(size, busbw_oop, busbw_ip), ...]})"""
    info = {"file": os.path.basename(path), "label": None, "gpus": None,
            "collectives": {}}
    cur = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.match(r"^nodes?\s+=\s+(\S+)", line)
            if m and info["label"] is None:
                info["label"] = m.group(1)
                continue
            m = re.match(r"^num_gpu_per_task\s+=\s+(\d+)", line)
            if m:
                info["gpus"] = int(m.group(1))
                continue
            m = re.match(r"^%+\s+(\S+_perf)\s+%+", line)
            if m:
                cur = m.group(1)
                info["collectives"].setdefault(cur, [])
                continue
            if cur is None or line.startswith("#") or not line.strip():
                continue
            f = line.split()
            if len(f) < 9 or not f[0].isdigit():
                continue
            try:
                size = int(f[0])
                busbw_oop = float(f[-6])
                busbw_ip = float(f[-2])
            except ValueError:
                continue
            info["collectives"][cur].append((size, busbw_oop, busbw_ip))
    return info


def peak(rows):
    """bus bandwidth at the largest message size (best of in/out of place)."""
    if not rows:
        return None
    size, oop, ip = max(rows, key=lambda r: r[0])
    return max(oop, ip)


def load(dirname):
    runs = []
    for path in sorted(glob.glob(os.path.join(dirname, "*"))):
        if os.path.isfile(path):
            info = parse(path)
            if info["collectives"]:
                runs.append(info)
    return runs


def table_peak(runs, title):
    if not runs:
        return f"_no results in {title}_\n"
    cols = [r["label"] or r["file"] for r in runs]
    out = ["| Collective | " + " | ".join(cols) + " | mean |",
           "|---|" + "---|" * (len(cols) + 1)]
    for coll in COLL_ORDER:
        vals = [peak(r["collectives"].get(coll, [])) for r in runs]
        if all(v is None for v in vals):
            continue
        cells = [f"{v:,.1f}" if v is not None else "—" for v in vals]
        got = [v for v in vals if v is not None]
        mean = f"**{sum(got)/len(got):,.1f}**" if got else "—"
        out.append(f"| {PRETTY.get(coll, coll)} | " + " | ".join(cells) + f" | {mean} |")
    return "\n".join(out) + "\n"


def table_sweep(runs, coll, title):
    """message-size sweep of one collective, averaged over the runs."""
    if not runs:
        return f"_no results in {title}_\n"
    sizes = {}
    for r in runs:
        for size, oop, ip in r["collectives"].get(coll, []):
            sizes.setdefault(size, []).append(max(oop, ip))
    if not sizes:
        return f"_{PRETTY.get(coll, coll)} not present in {title}_\n"
    out = ["| Message size | busbw min | busbw mean | busbw max |",
           "|---|---|---|---|"]
    for size in sorted(sizes):
        v = sizes[size]
        out.append(f"| {human(size)} | {min(v):,.1f} | {sum(v)/len(v):,.1f} | {max(v):,.1f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    one = load("out-1node")
    two = load("out-2node")
    if mode == "json":
        print(json.dumps({"one": one, "two": two}, indent=1, default=str))
        sys.exit(0)
    print("### 1-node peak busbw (GB/s)\n")
    print(table_peak(one, "out-1node"))
    print("\n### 2-node peak busbw (GB/s)\n")
    print(table_peak(two, "out-2node"))
    for coll in ("sendrecv_perf", "all_reduce_perf"):
        print(f"\n### 1-node {PRETTY[coll]} sweep\n")
        print(table_sweep(one, coll, "out-1node"))
        print(f"\n### 2-node {PRETTY[coll]} sweep\n")
        print(table_sweep(two, coll, "out-2node"))
