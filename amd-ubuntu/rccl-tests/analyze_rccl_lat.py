#!/usr/bin/env python3
"""Latency tables from run_rccl_lat_2node.sh.

rccl_runs.txt lines: tag|hosts|PPN|EXTRA_ENV|log dir. Each dir has <coll>_ppn<P>_np<N>.log from
rccl-tests. A data row is: size count type redop root  time algbw busbw #wrong  time algbw busbw
#wrong (out-of-place, then in-place); time is in microseconds per call. We report the
out-of-place time, as the bandwidth tables do.
"""
import argparse, glob, os, re, datetime

ROW = re.compile(r"^\s*\d+\s+\d+\s+[a-z]")
SIZES = [8, 64, 1024, 8192, 65536, 1048576, 4194304]
COLLS = ["sendrecv", "all_reduce", "all_gather", "reduce_scatter", "broadcast", "alltoall"]


def human(n):
    for u, s in ((1 << 20, "M"), (1 << 10, "K")):
        if n >= u and n % u == 0:
            return f"{n // u}{s}"
    return str(n)


def parse(log):
    t = {}
    for line in open(log, errors="replace"):
        if ROW.match(line):
            f = line.split()
            try:
                t[int(f[0])] = float(f[5])
            except ValueError:
                pass
    return t


def label(tag, hosts, ppn, env):
    n = len(hosts.split())
    s = f"{n} node{'s' if n > 1 else ''}, {ppn} GPU/node"
    return s + (f", {env}" if env else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--title", default="RCCL latency, node6100 + node6101")
    ap.add_argument("--tool", default="rccl-tests")
    ap.add_argument("--intra", default="XGMI", help="intra-node link name for the 1-node note")
    a = ap.parse_args()
    runs = []  # (label, {coll: {size: us}})
    for line in open(a.runs):
        tag, hosts, ppn, env, d = line.rstrip("\n").split("|")
        data = {}
        for log in glob.glob(os.path.join(d, "*_ppn*_np*.log")):
            coll = os.path.basename(log).split("_ppn")[0]
            data[coll] = parse(log)
        runs.append((label(tag, hosts, ppn, env), data))

    out = [f"# {a.title}", "",
           f"{a.tool}, 1000 iterations + 100 warm-up per size, no data check, NUMA-bound ranks, "
           f"{datetime.date.today()}. Time per call in µs (out-of-place). "
           f"\"1 node\" columns are {a.intra} only (no network) for reference; the difference to "
           "the 2-node columns is what the network adds.", "",
           f"Runs: `{os.path.dirname(os.path.abspath(a.runs))}`", ""]
    for coll in COLLS:
        cols = [(l, d[coll]) for l, d in runs if d.get(coll)]
        if not cols:
            continue
        out += [f"## {coll}", "", "| bytes | " + " | ".join(l for l, _ in cols) + " |",
                "|---:|" + "---:|" * len(cols)]
        for s in SIZES:
            vals = [f"{t[s]:.1f}" if s in t else "–" for _, t in cols]
            out.append(f"| {human(s)} | " + " | ".join(vals) + " |")
        out.append("")
    open(a.out, "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
