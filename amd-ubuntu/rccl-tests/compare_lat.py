#!/usr/bin/env python3
"""Latency write-ups from run_rccl_lat_2node.sh (MI355X) and b200-nodes/job-nccl-lat-2node.sh.

  compare_lat.py ubuntu            -> results/ubuntu/latency.md (our nodes: ROCm 7.2.4 vs 7.14)
  compare_lat.py b200 <b200 dir>   -> section 5 + one Summary row in results/mi355x-vs-b200.md
                                      (MI355X ROCm 7.14 vs B200; never 7.2.4 vs B200)
Inputs are found by default as the newest finished run dirs under logs/node6100/.
"""
import glob, os, re, sys, statistics, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_rccl_lat import parse, human  # noqa: E402

BENCH = "/orcd/data/orcd/022/benchmarks/amd-ubuntu"
LOGS = f"{BENCH}/logs/node6100"
COLLS = ["sendrecv", "all_reduce", "all_gather", "reduce_scatter", "broadcast", "alltoall"]
SIZES = [8, 1024, 65536, 1048576]
MAIN = f"{BENCH}/results/mi355x-vs-b200.md"
BEGIN, END = "<!-- LATENCY-BEGIN (auto: rccl-tests/compare_lat.py) -->", "<!-- LATENCY-END -->"


def newest_done(pattern):
    for d in sorted(glob.glob(pattern), reverse=True):
        st = os.path.join(d, "STATE.txt")
        if os.path.exists(st) and "latency run DONE" in open(st).read():
            return d
    return None


def load_runs(d):
    """{(setting, layout): {coll: {size: us}}}; layout = 2n_ppn1 / 2n_ppn8 / 1n_ppn8."""
    out = {}
    for line in open(os.path.join(d, "rccl_runs.txt")):
        tag, _, _, _, rd = line.rstrip("\n").split("|")
        m = re.match(r"(?:(\w+?)_)?(\dn_ppn\d)$", tag)
        setting, layout = (m.group(1) or "default"), m.group(2)
        out[(setting, layout)] = {os.path.basename(f).split("_ppn")[0]: parse(f)
                                  for f in glob.glob(os.path.join(rd, "*_ppn*_np*.log"))}
    return out


def ib_lat(logdir):
    """{test: {rail: {size: median_us}}} from perftest logs <test>_<rail>.log."""
    res = {}
    for f in glob.glob(os.path.join(logdir, "*_*.log")):
        t, rail = os.path.basename(f)[:-4].split("_", 1)
        if t not in ("write", "send"):
            continue
        rows = {}
        for line in open(f, errors="replace"):
            p = line.split()
            if len(p) >= 9 and p[0].isdigit() and p[1].isdigit():
                rows[int(p[0])] = float(p[4])  # t_typical
        res.setdefault(t, {})[rail] = rows
    return res


def rail_median(lat, t, s):
    v = [r[s] for r in lat.get(t, {}).values() if s in r]
    return statistics.median(v) if v else None


def g(runs, setting, layout, coll, s):
    return runs.get((setting, layout), {}).get(coll, {}).get(s)


f1 = lambda x: "–" if x is None else f"{x:.1f}"
f2 = lambda x: "–" if x is None else f"{x:.2f}"


def ratio(slow_ref, x):  # speed ratio = ref time / x time (>1: x faster)
    return None if not (slow_ref and x) else slow_ref / x


def span(v):
    lo, hi = min(v), max(v)
    return f"{lo:.2f}x" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f}–{hi:.2f}x"


def rx(r, red_below=None):
    if r is None:
        return "–"
    s = f"{r:.2f}x"
    return f"$\\color{{red}}{{\\textbf{{{s}}}}}$" if red_below and r < red_below else s


# ---------------------------------------------------------------- ubuntu/latency.md
def ubuntu():
    d724 = newest_done(f"{LOGS}/rccl/lat_2node_*")
    d714 = newest_done(f"{LOGS}/rocm7.14/rccl/lat_2node_*")
    ibd = sorted(glob.glob(f"{LOGS}/net/ib_lat_*"))
    a, b = (load_runs(d) if d else {} for d in (d724, d714))
    lat = ib_lat(ibd[-1]) if ibd else {}
    L = ["# Network latency, node6100 ↔ node6101", "",
         f"Updated {datetime.date.today()} by `rccl-tests/compare_lat.py ubuntu`. Scripts: "
         "`net/run_ib_lat.sh` (host RDMA, perftest), `rccl-tests/run_rccl_lat_2node.sh` (RCCL, rccl-tests 8 B–4 MB, "
         "1000 iterations + 100 warm-up, no data check, one rank per GPU, NUMA-bound).", "",
         "Compared: host ROCm 7.2.4 (RCCL 2.27.7) with ROCm 7.14 (RCCL 2.30.4, default IB-CAST transport and "
         "`NCCL_NET=IB`). Time per call in µs, lower is better."
         " **ROCm 7.14 is faster than 7.2.4 in almost every test (1.04–1.58x lower latency); the one exception is all_reduce at 1 MB on 16 ranks (0.89x). On 7.14 the default transport is faster than `NCCL_NET=IB` in most tests.**", ""]
    if lat:
        L += ["## 1. Raw network: RDMA latency, one rail at a time (no GPU, no RCCL)", "",
              f"perftest one-way latency (median, µs), host memory. Logs: `{ibd[-1][len(BENCH)+1:]}`.", "",
              "| test | bytes | median of 8 rails | fastest rail | slowest rail |", "|---|---:|---:|---:|---:|"]
        for t in ("write", "send"):
            for s in (2, 8, 64, 4096, 65536):
                v = [r[s] for r in lat.get(t, {}).values() if s in r]
                if v:
                    L.append(f"| ib_{t}_lat | {s} | {statistics.median(v):.2f} | {min(v):.2f} | {max(v):.2f} |")
        L.append("")
    if a or b:
        L += ["## 2. RCCL, 2 nodes", "",
              "Columns: 7.2.4 | 7.14 default | 7.14 `NCCL_NET=IB` | 1 node × 8 GPUs on 7.14 (XGMI only, no network). "
              "The ratio is 7.2.4 time ÷ 7.14 time (>1: 7.14 faster).", ""]
        for layout, title in (("2n_ppn1", "2 nodes × 1 GPU (GPU → NIC → wire → NIC → GPU)"),
                              ("2n_ppn8", "2 nodes × 8 GPUs (16 ranks, as in the bandwidth tables)")):
            L += [f"### {title}", "", "| collective | bytes | 7.2.4 | 7.14 default | 7.14 NCCL_NET=IB | 7.14 best vs 7.2.4 | 1 node, 7.14 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
            for c in COLLS:
                for s in SIZES:
                    x = g(a, "default", layout, c, s)
                    y, z = g(b, "default", layout, c, s), g(b, "netib", layout, c, s)
                    if x is None and y is None:
                        continue
                    best = min([v for v in (y, z) if v] or [None]) if (y or z) else None
                    one = g(b, "default", "1n_ppn8", c, s) if layout == "2n_ppn8" else None
                    L.append(f"| {c} | {human(s)} | {f1(x)} | {f1(y)} | {f1(z)} | {rx(ratio(x, best))} | {f1(one)} |")
            L.append("")
        L += [f"Runs: 7.2.4 `{(d724 or 'not finished')[len(BENCH)+1:]}`, 7.14 `{(d714 or 'not finished')[len(BENCH)+1:]}`.", ""]
    out = f"{BENCH}/results/ubuntu/latency.md"
    open(out, "w").write("\n".join(L))
    print("wrote", out)


# ---------------------------------------------------------------- vs-b200 section 5
def b200(bdir):
    d714 = newest_done(f"{LOGS}/rocm7.14/rccl/lat_2node_*")
    ibd = sorted(glob.glob(f"{LOGS}/net/ib_lat_*"))
    if not (d714 and os.path.exists(os.path.join(bdir, "rccl_runs.txt"))):
        sys.exit(f"missing input: 7.14={d714} b200={bdir}")
    m, n = load_runs(d714), load_runs(bdir)
    la, lb = (ib_lat(ibd[-1]) if ibd else {}), ib_lat(bdir)

    def best(layout, c, s):  # MI355X 7.14: the faster of default and NCCL_NET=IB
        v = [x for x in (g(m, "default", layout, c, s), g(m, "netib", layout, c, s)) if x]
        return min(v) if v else None

    # headline: 8-byte speed ratios across 2 nodes x 8 GPUs
    r8 = {c: ratio(g(n, "default", "2n_ppn8", c, 8), best("2n_ppn8", c, 8)) for c in COLLS}
    r8 = {c: r for c, r in r8.items() if r}
    faster = [c for c, r in r8.items() if r >= 1.03]
    slower = [c for c, r in r8.items() if r <= 0.97]
    if slower and not faster:
        verdict = f"B200 has the lower latency for small messages (MI355X {span(r8.values())} the speed at 8 bytes)."
    elif faster and not slower:
        verdict = f"MI355X has the lower latency for small messages ({span(r8.values())} the speed of B200 at 8 bytes)."
    else:
        verdict = (f"Mixed: at 8 bytes MI355X is faster in {', '.join(faster) or 'none'} and slower in "
                   f"{', '.join(slower) or 'none'} ({span(r8.values())}).") if r8 else "No data."

    L = [BEGIN, "## 5. Latency across 2 nodes (small messages)", "",
         "**Compared:** MI355X ROCm 7.14 (RCCL, AMD Pollara network) with B200 (NCCL, NVIDIA InfiniBand), same tests on both: "
         "perftest RDMA latency on one rail (network only: Pollara vs InfiniBand), "
         "then rccl-tests / nccl-tests from 8 B to 4 MB with 1000 iterations, one rank per GPU, NUMA-bound. "
         f"Time per call in µs, lower is better; ratio = B200 time ÷ MI355X time (>1: MI355X faster). **{verdict}**", "",
         "MI355X column = the faster of RCCL's default and `NCCL_NET=IB` (both are in `ubuntu/latency.md`).", ""]
    def winner(rs, what, ref="InfiniBand"):  # highlighted one-line verdict above a table
        rs = [r for r in rs if r]
        if not rs:
            return []
        if max(rs) < 0.97:
            w = f"**InfiniBand is better.** {what} is {span(rs)} the speed of {ref} (lower latency on InfiniBand in every row)."
        elif min(rs) > 1.03:
            w = f"**Pollara is better.** {what} is {span(rs)} the speed of {ref} (lower latency on Pollara in every row)."
        else:
            w = f"**Mixed.** {what} is {span(rs)} the speed of {ref}."
        return [f"> {w}", ""]

    if la and lb:
        rows, rs = [], []
        for t in ("write", "send"):
            for s in (8, 4096, 65536):
                x, y = rail_median(la, t, s), rail_median(lb, t, s)
                rs.append(ratio(y, x))
                rows.append(f"| ib_{t}_lat | {human(s)} | {f2(x)} | {f2(y)} | {rx(ratio(y, x))} |")
        L += ["### Raw network, no GPU: Pollara vs InfiniBand", "",
              *winner(rs, "Pollara"),
              "perftest one-way latency, median of the 8 rails, µs:", "",
              "| test | bytes | AMD Pollara (RoCEv2) | NVIDIA InfiniBand (ConnectX-7) | Pollara / InfiniBand speed |", "|---|---:|---:|---:|---:|",
              *rows, ""]
    rows, rs = [], []
    for c in ("sendrecv", "all_reduce"):
        for s in SIZES:
            x, y = best("2n_ppn1", c, s), g(n, "default", "2n_ppn1", c, s)
            rs.append(ratio(y, x))
            rows.append(f"| {c} | {human(s)} | {f1(x)} | {f1(y)} | {rx(ratio(y, x))} |")
    L += ["### 2 nodes × 1 GPU: Pollara vs InfiniBand", "",
          *winner(rs, "MI355X over Pollara", "B200 over InfiniBand"),
          "One GPU talking to one GPU over the network, so the network sets the time:", "",
          "| collective | bytes | MI355X (Pollara) | B200 (InfiniBand) | MI355X / B200 speed |", "|---|---:|---:|---:|---:|",
          *rows]
    L += ["", "### 2 nodes × 8 GPUs: MI355X vs B200", "", "16 ranks. \"Network adds\" = 2-node time − 1-node time at 8 bytes "
          "(1 node = XGMI on MI355X, NVLink on B200; the added time crosses Pollara on MI355X, InfiniBand on B200):", "",
          "| collective | MI355X 8 B | B200 8 B | MI355X / B200 speed | MI355X 1 MB | B200 1 MB | MI355X / B200 speed | network adds, Pollara | network adds, InfiniBand |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in COLLS:
        x8, y8 = best("2n_ppn8", c, 8), g(n, "default", "2n_ppn8", c, 8)
        xm, ym = best("2n_ppn8", c, 1 << 20), g(n, "default", "2n_ppn8", c, 1 << 20)
        x1, y1 = best("1n_ppn8", c, 8), g(n, "default", "1n_ppn8", c, 8)
        ax = None if not (x8 and x1) else x8 - x1
        ay = None if not (y8 and y1) else y8 - y1
        L.append(f"| {c} | {f1(x8)} | {f1(y8)} | {rx(ratio(y8, x8))} | {f1(xm)} | {f1(ym)} | {rx(ratio(ym, xm))} | {f1(ax)} | {f1(ay)} |")
    L += ["", f"Sources: `{d714[len(BENCH)+1:]}`, `{os.path.relpath(bdir, os.path.dirname(MAIN))}`. "
          f"Updated {datetime.date.today()}.", END, ""]
    sec = "\n".join(L)

    txt = open(MAIN).read()
    if BEGIN in txt:
        txt = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", lambda _: sec, txt, flags=re.S)
    else:  # put it before "## 6." (section 5)
        i = txt.index("\n## 6. ")
        txt = txt[:i].rstrip("\n") + "\n\n" + sec + txt[i:]
    if r8:
        row = (f"| Latency across 2 nodes, 8 bytes (§5) | {span(r8.values())} "
               "(speed; MI355X ROCm 7.14) | ✅ same tests |")
        lines = txt.split("\n")
        lines = [l for l in lines if not l.startswith("| Latency across 2 nodes")]
        k = next(i for i, l in enumerate(lines) if l.startswith("| Collectives across 2 nodes"))
        lines.insert(k + 1, row)
        txt = "\n".join(lines)
    open(MAIN, "w").write(txt)
    print("updated", MAIN, "|", verdict)


if __name__ == "__main__":
    {"ubuntu": lambda: ubuntu(), "b200": lambda: b200(sys.argv[2])}[sys.argv[1]]()
