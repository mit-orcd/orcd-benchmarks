#!/usr/bin/env python3
"""Latency write-ups from run_rccl_lat_2node.sh (MI355X) and b200-nodes/job-nccl-lat-2node.sh.

  compare_lat.py ubuntu            -> results/ubuntu/latency.md (our nodes: ROCm 7.2.4 vs 7.14)
  compare_lat.py b200 <b200 dir>   -> §2.3 of results/mi355x-vs-b200-interconnect.md + one Summary row in results/mi355x-vs-b200.md
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
SIZES8 = [8, 256, 65536, 1048576]  # 2 nodes x 8 GPUs table; 256 B = smallest real size for all 6 at 16 ranks
MAIN = f"{BENCH}/results/mi355x-vs-b200.md"               # Summary row
IC = f"{BENCH}/results/mi355x-vs-b200-interconnect.md"    # §2.3 latency section
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


# ---------------------------------------------------------------- vs-b200 §2.3 (interconnect file)
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

    # Summary-table row: 8-byte speed ratios across 2 nodes x 8 GPUs
    r8 = {c: ratio(g(n, "default", "2n_ppn8", c, 8), best("2n_ppn8", c, 8)) for c in COLLS}
    r8 = {c: r for c, r in r8.items() if r}

    MAINC = ("sendrecv", "all_reduce")  # the tests run in both layouts
    unit = lambda s: {8: "8 B", 1024: "1 KB", 65536: "64 KB", 1 << 20: "1 MB"}.get(s, f"{s} B")

    def who(r):
        return None if r is None else "MI355X" if r >= 1.03 else "B200" if r <= 0.97 else "equal"

    def verdict(layout):  # one highlighted line: who is faster, per collective
        parts, wins = [], set()
        for c in MAINC:
            rs = {s: ratio(g(n, "default", layout, c, s), best(layout, c, s)) for s in SIZES}
            rs = {s: r for s, r in rs.items() if r}
            if not rs:
                continue
            by = {}
            for s, r in rs.items():
                by.setdefault(who(r), []).append(s)
            wins |= set(by)
            if len(by) == 1:
                w = next(iter(by))
                parts.append(f"{c}: **{w} faster** at every size ({span(rs.values())})" if w != "equal"
                             else f"{c}: equal ({span(rs.values())})")
            else:
                parts.append(f"{c}: " + ", ".join(
                    f"**{w} faster** at {', '.join(unit(s) for s in ss)} ({span([rs[s] for s in ss])})" if w != "equal"
                    else f"equal at {', '.join(unit(s) for s in ss)}" for w, ss in by.items()))
        head = ("B200 (InfiniBand) is faster" if wins == {"B200"} else
                "MI355X (Pollara) is faster" if wins == {"MI355X"} else "Mixed")
        return f"> **{head}.** " + "; ".join(parts) + ".", head

    def netpart(c, s):  # network part of a 2 x 8 GPU call = 2-node time - 1-node time (same size)
        x2, x1 = best("2n_ppn8", c, s), best("1n_ppn8", c, s)
        y2, y1 = g(n, "default", "2n_ppn8", c, s), g(n, "default", "1n_ppn8", c, s)
        return (x2 - x1 if x2 and x1 else None), (y2 - y1 if y2 and y1 else None)

    def table(layout):
        rows = ["| collective | bytes | MI355X (Pollara) | B200 (InfiniBand) | MI355X / B200 speed |", "|---|---:|---:|---:|---:|"]
        for c in MAINC:
            for s in SIZES:
                x, y = best(layout, c, s), g(n, "default", layout, c, s)
                rows.append(f"| {c} | {human(s)} | {f1(x)} | {f1(y)} | {rx(ratio(y, x))} |")
        return rows

    def net_table():  # 2 x 8 GPUs: network part only (+ whole-call speed for reference)
        rows = ["| collective | bytes | network part, Pollara | network part, InfiniBand | Pollara / InfiniBand speed | whole call, MI355X / B200 speed |",
                "|---|---:|---:|---:|---:|---:|"]
        for c in MAINC:
            for s in SIZES:
                nx, ny = netpart(c, s)
                r = ratio(ny, nx) if nx and ny and nx > 0 and ny > 0 else None
                rows.append(f"| {c} | {human(s)} | {f1(nx)} | {f1(ny)} | {rx(r)} | "
                            f"{rx(ratio(g(n, 'default', '2n_ppn8', c, s), best('2n_ppn8', c, s)))} |")
        return rows

    def net_verdict():
        rs = {(c, s): ratio(ny, nx) for c in MAINC for s in SIZES for nx, ny in [netpart(c, s)]
              if nx and ny and nx > 0 and ny > 0}
        if not rs:
            return "> No data.", "No data"
        ib = [r for r in rs.values() if r <= 0.97]
        oth = {k: r for k, r in rs.items() if r > 0.97}
        if not oth:
            return (f"> **InfiniBand is faster.** In the network part, Pollara is {span(ib)} the speed of InfiniBand "
                    "in every row.", "InfiniBand is faster")
        if not ib:
            return f"> **Pollara is faster or equal.** Pollara is {span(rs.values())} the speed of InfiniBand.", "Pollara is faster"
        exc = ", ".join(f"{c} {unit(s)} ({r:.2f}x)" for (c, s), r in oth.items())
        return (f"> **InfiniBand is faster** in {len(ib)} of {len(rs)} rows: Pollara is {span(ib)} the speed of InfiniBand "
                f"in the network part. Exception: {exc}.", "InfiniBand is faster, mostly")

    v1, h1 = verdict("2n_ppn1")
    v8, h8 = verdict("2n_ppn8")
    nv, hn = net_verdict()
    L = [BEGIN, "### 2.3 Measured latency (small messages)", "",
         "MI355X ROCm 7.14 (RCCL, AMD Pollara RoCEv2) vs B200 (NCCL, NVIDIA InfiniBand). Time per call in µs, lower is better; "
         "speed = B200 time ÷ MI355X time (>1: MI355X faster). MI355X = the faster of RCCL's default and `NCCL_NET=IB` "
         "(both in `ubuntu/latency.md`).", ""]
    if la and lb:
        rows, rs = [], []
        for tt in ("write", "send"):
            for s in (8, 4096, 65536):
                x, y = rail_median(la, tt, s), rail_median(lb, tt, s)
                rs.append(ratio(y, x))
                rows.append(f"| ib_{tt}_lat | {human(s)} | {f2(x)} | {f2(y)} | {rx(ratio(y, x))} |")
        rs = [r for r in rs if r]
        w = ("InfiniBand is faster" if rs and max(rs) < 0.97 else "Pollara is faster" if rs and min(rs) > 1.03 else "Mixed")
        L += ["#### Raw network, no GPU: Pollara vs InfiniBand", "",
              f"> **{w}.** Pollara is {span(rs)} the speed of InfiniBand (perftest one-way latency, median of the 8 rails).", "",
              "| test | bytes | AMD Pollara (RoCEv2) | NVIDIA InfiniBand (ConnectX-7) | Pollara / InfiniBand speed |", "|---|---:|---:|---:|---:|",
              *rows, ""]
    L += ["#### Main result: 2 nodes × 1 GPU and 2 nodes × 8 GPUs, same tests", "",
          "**2 nodes × 1 GPU** (one GPU per node, so only the network is between them):", "", v1, "", *table("2n_ppn1"), "",
          "**2 nodes × 8 GPUs, network part only** (network part = 2-node time − 1-node time, same test and size; "
          "this removes the step among the 8 GPUs inside each node):", "",
          nv, "", *net_table(), ""]
    # Collectives run only with 8 GPUs per node: speed only, brief
    ext = [c for c in COLLS if c not in MAINC]
    rows = []
    for c in ext:
        rows.append(f"| {c} | " + " | ".join(rx(ratio(g(n, "default", "2n_ppn8", c, s), best("2n_ppn8", c, s)))
                                             for s in (256, 65536, 1 << 20)) + " |")
    L += ["**Only in 2 nodes × 8 GPUs** (other collectives; speed only, >1: MI355X faster):", "",
          "| collective | 256 B | 64 KB | 1 MB |", "|---|---:|---:|---:|", *rows, "",
          f"Sources: `{d714[len(BENCH)+1:]}`, `{os.path.relpath(bdir, os.path.dirname(MAIN))}`. "
          f"Updated {datetime.date.today()}.", END, ""]
    verdict_txt = f"2 nodes x 1 GPU: {h1}; 2 nodes x 8 GPUs: {h8}"
    sec = "\n".join(L)

    ic = open(IC).read()
    if BEGIN in ic:
        ic = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", lambda _: sec, ic, flags=re.S)
    else:  # last section of the interconnect file
        ic = ic.rstrip("\n") + "\n\n" + sec
    open(IC, "w").write(ic)
    txt = open(MAIN).read()
    if r8:
        def rng(layout):
            v = [ratio(g(n, "default", layout, c, s), best(layout, c, s)) for c in MAINC for s in SIZES]
            return span([r for r in v if r])
        row = (f"| Latency across 2 nodes, sendrecv + all_reduce, 8 B–1 MB (§3) | 1 GPU/node: {h1} ({rng('2n_ppn1')}); "
               f"8 GPUs/node: {h8} ({rng('2n_ppn8')}) (speed; MI355X ROCm 7.14) | ✅ same tests |")
        lines = txt.split("\n")
        lines = [l for l in lines if not l.startswith("| Latency across 2 nodes")]
        k = next(i for i, l in enumerate(lines) if l.startswith("| Collectives across 2 nodes"))
        lines.insert(k + 1, row)
        txt = "\n".join(lines)
    open(MAIN, "w").write(txt)
    print("updated", IC, "§2.3 and the Summary row of", MAIN, "|", verdict_txt)


if __name__ == "__main__":
    {"ubuntu": lambda: ubuntu(), "b200": lambda: b200(sys.argv[2])}[sys.argv[1]]()
