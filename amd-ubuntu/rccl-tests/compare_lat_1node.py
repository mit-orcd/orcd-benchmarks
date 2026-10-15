#!/usr/bin/env python3
"""Latency inside one node (8 GPUs): MI355X (RCCL, ROCm 7.14) vs B200 (NCCL).

Inputs (all finished runs found, so it can run after either side finishes):
  MI355X: amd-ubuntu/logs/node610[01]/rocm7.14/rccl/lat_1node_*  (rccl-tests/run_rccl_lat_1node.sh)
  B200  : b200-nodes/out-nccl-lat-1node/<job>                     (b200-nodes/job-nccl-lat-1node.sh)
Outputs:
  results/latency-1node-mi355x-vs-b200.md   every node, setting and collective
  results/mi355x-vs-b200-interconnect.md    §1.3 (between the INTRA-LAT markers)
  results/mi355x-vs-b200.md                 one Summary row
"""
import glob, os, re, statistics, datetime

from analyze_rccl_lat import parse, human

BENCH = "/orcd/data/orcd/022/benchmarks"
AMD = f"{BENCH}/amd-ubuntu"
MAIN = f"{AMD}/results/mi355x-vs-b200.md"               # Summary row
IC = f"{AMD}/results/mi355x-vs-b200-interconnect.md"    # §1.3
DETAIL = f"{AMD}/results/latency-1node-mi355x-vs-b200.md"
AMD_GLOB = f"{AMD}/logs/node610[01]/rocm7.14/rccl/lat_1node_*"
B200_GLOB = f"{BENCH}/b200-nodes/out-nccl-lat-1node/[0-9]*"
BEGIN, END = "<!-- INTRA-LAT-BEGIN (auto: rccl-tests/compare_lat_1node.py) -->", "<!-- INTRA-LAT-END -->"
COLLS = ["sendrecv", "all_reduce", "all_gather", "reduce_scatter", "broadcast", "alltoall"]
MAIN_SIZES = [8, 1024, 65536, 1 << 20]
ALL_SIZES = [8, 64, 256, 1024, 8192, 65536, 1 << 20, 4 << 20]
FIXED = {"ll": "NCCL_PROTO=LL", "ringll": "NCCL_ALGO=Ring NCCL_PROTO=LL",
         "treell": "NCCL_ALGO=Tree NCCL_PROTO=LL", "nvls0": "NCCL_NVLS_ENABLE=0"}
unit = lambda s: {8: "8 B", 1024: "1 KB", 65536: "64 KB", 1 << 20: "1 MB"}.get(s, f"{human(s)} B")


def done(d):
    st = os.path.join(d, "STATE.txt")
    return os.path.exists(st) and "latency run DONE" in open(st).read()


def load(d):
    """[(node, setting, {coll: {size: us}})] from one run dir's rccl_runs.txt."""
    out = []
    for line in open(os.path.join(d, "rccl_runs.txt")):
        tag, host, _, _, rd = line.rstrip("\n").split("|")
        setting = tag[: -len("_1n_ppn8")]
        data = {os.path.basename(f).split("_ppn")[0]: parse(f) for f in glob.glob(os.path.join(rd, "*_ppn*_np*.log"))}
        out.append((host.split(".")[0], setting, data))
    return out


def gather(pattern):
    runs, dirs = [], []
    for d in sorted(glob.glob(pattern)):
        if done(d):
            runs += load(d)
            dirs.append(d)
    return runs, dirs


def default_vals(runs, c, s):
    return [r[c][s] for _, st, r in runs if re.fullmatch(r"r\d", st) and s in r.get(c, {})]


def med(v):
    return statistics.median(v) if v else None


def best_fixed(runs, c, s):
    v = [(r[c][s], st) for _, st, r in runs if st in FIXED and s in r.get(c, {})]
    return min(v) if v else None


f1 = lambda x: "–" if x is None else f"{x:.1f}"
ratio = lambda ref, x: None if not (ref and x) else ref / x
rx = lambda r: "–" if r is None else f"{r:.2f}x"


def span(v):
    v = [x for x in v if x]
    if not v:
        return "–"
    lo, hi = min(v), max(v)
    return f"{lo:.2f}x" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f}–{hi:.2f}x"


def main():
    m, mdirs = gather(AMD_GLOB)
    b, bdirs = gather(B200_GLOB)
    today = datetime.date.today()
    mnodes = sorted({n for n, _, _ in m}) or ["pending"]
    bnodes = sorted({n for n, _, _ in b}) or ["pending"]
    rel = lambda d: os.path.relpath(d, os.path.dirname(MAIN))

    # ---- §1.3 of the interconnect file -----------------------------------------------------
    L = [BEGIN, "### 1.3 Measured latency (8 GPUs, small messages)", "",
         "rccl-tests (MI355X, ROCm 7.14) / nccl-tests (B200), 8 B–4 MB, 1000 iterations + 100 warm-up, one rank per GPU, "
         "NUMA-bound, each run started only after all GPUs read 0 % use. Time per call in µs, lower is better; "
         "speed = B200 time ÷ MI355X time (>1: MI355X faster). Default = median of 3 default runs per node "
         f"(MI355X: {', '.join(mnodes)}; B200: {', '.join(bnodes)}).", ""]
    if not (m and b):
        L += [f"> **Pending.** MI355X runs finished: {len(mdirs)}; B200 runs finished: {len(bdirs)}. "
              "This section fills in when both sides have at least one finished run.", ""]
    else:
        wins = {}
        for c in COLLS:
            for s in MAIN_SIZES:
                r = ratio(med(default_vals(b, c, s)), med(default_vals(m, c, s)))
                if r:
                    wins.setdefault(c, []).append(r)
        parts, who_all, wins_by = [], set(), {}
        for c, rs in wins.items():
            w = "MI355X" if min(rs) > 1.03 else "B200" if max(rs) < 0.97 else "mixed"
            who_all.add(w)
            wins_by.setdefault(w, []).append(c)
            parts.append(f"{c}: **{w + ' faster' if w != 'mixed' else 'mixed'}** ({span(rs)})")
        head = ("MI355X (XGMI) is faster" if who_all == {"MI355X"} else
                "B200 (NVLink) is faster" if who_all == {"B200"} else "Mixed")
        L += [f"> **{head}** with default settings, 8 B–1 MB. " + "; ".join(parts) + ".", "",
              "| collective | bytes | MI355X (XGMI) | B200 (NVLink) | MI355X / B200 speed | best fixed setting, MI355X | best fixed setting, B200 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for c in COLLS:
            for s in MAIN_SIZES:
                x, y = med(default_vals(m, c, s)), med(default_vals(b, c, s))
                if x is None and y is None:
                    continue
                bx, by = best_fixed(m, c, s), best_fixed(b, c, s)
                fx = f"{bx[0]:.1f} ({FIXED[bx[1]]})" if bx else "–"
                fy = f"{by[0]:.1f} ({FIXED[by[1]]})" if by else "–"
                L.append(f"| {c} | {unit(s)} | {f1(x)} | {f1(y)} | {rx(ratio(y, x))} | {fx} | {fy} |")
        # repeatability of the default runs, all_reduce + sendrecv
        L += ["", "**Run-to-run spread of the default runs** (min–max over all runs and nodes, µs):", "",
              "| collective | bytes | MI355X | B200 |", "|---|---:|---:|---:|"]
        for c in ("sendrecv", "all_reduce"):
            for s in (8, 1024):
                vm, vb = default_vals(m, c, s), default_vals(b, c, s)
                fm = f"{min(vm):.1f}–{max(vm):.1f}" if vm else "–"
                fb = f"{min(vb):.1f}–{max(vb):.1f}" if vb else "–"
                L.append(f"| {c} | {unit(s)} | {fm} | {fb} |")
        L += ["", f"All nodes, settings and sizes: [`latency-1node-mi355x-vs-b200.md`](latency-1node-mi355x-vs-b200.md)."]
    missing = [n for n in ("node6100", "node6101") if n not in mnodes]
    if missing and m:
        L += ["", f"Still to come: the MI355X run on {', '.join(missing)} (waiting for the GPUs); "
              "this section updates itself when it finishes."]
    L += ["", "Sources: " + ", ".join(f"`{rel(d)}`" for d in mdirs + bdirs) + f". Updated {today}.", END, ""]
    sec = "\n".join(L)

    txt = open(IC).read()
    if BEGIN in txt:
        txt = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", lambda _: sec, txt, flags=re.S)
    else:  # after 3.1, before "## 4."
        i = txt.index("\n## 2. Inter-node")
        txt = txt[:i].rstrip("\n") + "\n\n" + sec + txt[i:]
    tmp = IC + ".tmp-intra"
    open(tmp, "w").write(txt)
    os.replace(tmp, IC)
    if m and b:  # one Summary row in the main file, after the 1-node bandwidth row
        desc = "; ".join(f"{'mixed' if w == 'mixed' else w + ' faster'} in {', '.join(cs)} "
                         f"({span([r for c in cs for r in wins[c]])})" for w, cs in wins_by.items())
        row = f"| Latency inside one node, 8 B–1 MB (§3) | {desc} (default settings, MI355X ROCm 7.14) | ✅ same tests |"
        lines = [l for l in open(MAIN).read().split("\n") if not l.startswith("| Latency inside one node")]
        k = next(i for i, l in enumerate(lines) if l.startswith("| Collectives inside one node"))
        lines.insert(k + 1, row)
        tmp = MAIN + ".tmp-intra"
        open(tmp, "w").write("\n".join(lines))
        os.replace(tmp, MAIN)

    # ---- detail file ---------------------------------------------------------------------
    D = ["# Latency inside one node: MI355X (RCCL) vs B200 (NCCL)", "",
         "Every run of `amd-ubuntu/rccl-tests/run_rccl_lat_1node.sh` (MI355X, ROCm 7.14) and "
         "`b200-nodes/job-nccl-lat-1node.sh` (B200). 8 GPUs, one rank per GPU, 1000 iterations + 100 warm-up. "
         f"Time per call in µs (out-of-place), lower is better. Updated {today}.", ""]
    for c in COLLS:
        cols = [(f"MI355X {n} {st}", r.get(c, {})) for n, st, r in m] + [(f"B200 {n} {st}", r.get(c, {})) for n, st, r in b]
        cols = [(l, t) for l, t in cols if t]
        if not cols:
            continue
        D += [f"## {c}", "", "| bytes | " + " | ".join(l for l, _ in cols) + " |", "|---:|" + "---:|" * len(cols)]
        for s in ALL_SIZES:
            D.append(f"| {human(s)} | " + " | ".join(f1(t.get(s)) for _, t in cols) + " |")
        D.append("")
    D += ["Settings: r1/r2/r3 = default; " + "; ".join(f"{k} = `{v}`" for k, v in FIXED.items()) + " (nvls0: B200 only).", "",
          "Sources: " + ", ".join(f"`{os.path.relpath(d, os.path.dirname(DETAIL))}`" for d in mdirs + bdirs) + "."]
    open(DETAIL, "w").write("\n".join(D) + "\n")
    print(f"updated {IC} §1.3, {MAIN} Summary row and {DETAIL}: MI355X runs {len(mdirs)}, B200 runs {len(bdirs)}")


if __name__ == "__main__":
    main()
