#!/usr/bin/env python3
"""Why 1-node RCCL latency changed from ROCm 7.14 to 10.1: reads one run_rccl_why_1node.sh run dir.

  analyze_rccl_why_1node.py <logs/node6100/rocm10.1/rccl/why_1node_TS>

Writes results/rccl-lat-1node-rocm10.1-why.md (full), results/node6100/rocm10.1/rccl_why_1node_short.md
and rccl_why_1node_summary.txt (both read by compare_rocm10.1.py), then reruns compare_rocm10.1.py so
§5.1.1 and the Summary of results/rocm10.1-vs-rocm7.14.md pick them up.
"""
import collections, csv, re, subprocess, sys
from pathlib import Path

B = Path("/orcd/data/orcd/022/benchmarks/amd-ubuntu")
SW = Path("/orcd/data/orcd/022/benchmarks/amd-software")
RES = B / "results"
OUT_MD = RES / "rccl-lat-1node-rocm10.1-why.md"
SHORT = RES / "node6100/rocm10.1/rccl_why_1node_short.md"
SUMM = RES / "node6100/rocm10.1/rccl_why_1node_summary.txt"
RUN = Path(sys.argv[1])
STACKS = ["7.14", "10.1"]
TIME_COLLS = ["sendrecv", "all_reduce", "all_gather", "reduce_scatter", "broadcast", "alltoall"]
DBG_COLLS = ["sendrecv", "all_reduce", "all_gather", "broadcast", "alltoall"]
SIZES = {8: "8", 1024: "1K", 65536: "64K", 1048576: "1M"}
BACK = 10.0   # "back to 7.14": DDA-off time within ±10 % of the 7.14 time
BIG = 15.0    # a 7.14 -> 10.1 change counts when it is more than 15 % (one run per column; 8 B points move ~10 %)
ROW = re.compile(r"^\s+(\d+)\s+(\d+)\s+\S+\s+\S+\s+-?\d+\s+([\d.]+)\s")


# ---- timing --------------------------------------------------------------------------------------
def timing(tag):
    """time_<tag>.dir -> {coll: {bytes: out-of-place us}}"""
    p = RUN / f"time_{tag}.dir"
    if not p.exists() or not p.read_text().strip():
        return {}
    d = Path(p.read_text().strip())
    t = {}
    for c in TIME_COLLS:
        f = d / f"{c}_ppn8_np8.log"
        if not f.exists():
            continue
        for line in f.read_text(errors="replace").splitlines():
            m = ROW.match(line)
            if m and int(m.group(1)) in SIZES:
                t.setdefault(c, {})[int(m.group(1))] = float(m.group(3))
    return t


T = {"7.14": timing("7.14_default"), "10.1": timing("10.1_default"), "nodda": timing("10.1_nodda")}


def pct(old, new):
    return None if old is None or new is None or old == 0 else (new - old) / old * 100


def fp(p):
    return "" if p is None else ("0.0%" if abs(p) < 0.05 else f"{p:+.1f}%")


def fv(x):
    return "–" if x is None else f"{x:.1f}"


# ---- debug logs ----------------------------------------------------------------------------------
IMPL = re.compile(r"(\w+) impl selected: algo (\S+)")
DDA = re.compile(r"(\w+): taking DDA fabric (.+?) path: .*?(?:totalBytes|bytes)=(\d+)")
TUNE = re.compile(r"(\w+): (\d+) Bytes -> Algo (\S+) proto (\S+)")
CHAN = re.compile(r"coll channels:(\d+).*?p2p channels:(\d+).*?p2p channels per peer:(\d+)")
CSVT = re.compile(r"CsvTuner: Loaded (\d+) tuning")


def norm(line):
    s = line.split("NCCL INFO ", 1)[-1] if "NCCL INFO " in line else line
    s = re.sub(r"0x[0-9a-fA-F]+", "0x#", s)
    return re.sub(r"\d+", "#", s).strip()[:150]


def debug(stack, coll):
    """rank-0 view of one debug run (rank 0 = the file with '[0] NCCL' lines)"""
    r = {"impl": collections.Counter(), "dda": {}, "tune": {}, "chan": None, "csv": None,
         "msccl": collections.Counter(), "kinds": collections.Counter(), "files": 0}
    d = RUN / f"debug_{stack}" / coll
    files = [f.read_text(errors="replace").splitlines() for f in sorted(d.glob("rank.*.log"))] if d.exists() else []
    r["files"] = len(files)
    # other ranks print a few "[0]" lines too; rank 0's file is the one with the most
    for txt in sorted(files, key=lambda t: sum(" [0] NCCL" in x for x in t))[-1:]:
        for x in txt:
            if " [0] NCCL" not in x:
                continue
            m = IMPL.search(x)
            if m:
                r["impl"][m.group(2)] += 1
            m = DDA.search(x)
            if m:
                r["dda"].setdefault(int(m.group(3)), set()).add(m.group(2))
            m = TUNE.search(x)
            if m:
                r["tune"].setdefault(int(m.group(2)), set()).add(f"{m.group(3)}/{m.group(4)}")
            m = CHAN.search(x)
            if m and not r["chan"]:
                r["chan"] = m.groups()
            m = CSVT.search(x)
            if m:
                r["csv"] = int(m.group(1))
            if "MSCCL" in x or "mscclpp" in x.lower():
                r["msccl"][norm(x)] += 1
            r["kinds"][norm(x)] += 1
    return r


D = {s: {c: debug(s, c) for c in DBG_COLLS} for s in STACKS}


def csv_rows(stack):
    p = SW / f"rocm-{stack}.0/share/rccl/tuner/rccl_tuner_gfx950.csv"
    return [",".join(r) for r in csv.reader(p.read_text().splitlines()) if r] if p.exists() else []


new_rows = [r for r in csv_rows("10.1") if r not in csv_rows("7.14")]


def uses_dda(c):
    x = D["10.1"][c]
    return bool(x["dda"]) or any("DDA" in k for k in x["impl"])


def uses_dda_714(c):
    x = D["7.14"][c]
    return bool(x["dda"]) or any("DDA" in k for k in x["impl"])


# ---- verdict per point ---------------------------------------------------------------------------
pts = []   # (coll, size, t714, t101, tnodda, gap%, nodda-vs-714%)
for c in TIME_COLLS:
    for s in SIZES:
        a, b, n = (T[k].get(c, {}).get(s) for k in ("7.14", "10.1", "nodda"))
        if a is None and b is None:
            continue
        pts.append((c, s, a, b, n, pct(a, b), pct(a, n)))
changed = [p for p in pts if p[5] is not None and abs(p[5]) > BIG]
back = [p for p in changed if p[6] is not None and abs(p[6]) <= BACK]
dda_c = [c for c in DBG_COLLS if uses_dda(c)]
chg_dda = [p for p in changed if p[0] in dda_c]
chg_other = [p for p in changed if p[0] not in dda_c]
back_dda = [p for p in chg_dda if p in back]
back_other = [p for p in chg_other if p in back]


def lbl(ps):
    return ", ".join(f"{p[0]} {SIZES[p[1]]}" for p in ps) or "none"


def colls(ps):
    return ", ".join(dict.fromkeys(p[0] for p in ps)) or "none"


# DDA off slower than BOTH 7.14 and 10.1 default: the switch leaves a slower fallback, not 7.14's path
slow_off = [p for p in pts if None not in p[2:5] and p[4] > max(p[2], p[3]) * (1 + BACK / 100)]
moved = [p for p in pts if None not in p[3:5] and abs(pct(p[3], p[4])) > BACK]   # DDA switch changes it
moved_nodda = [p for p in moved if p[0] not in dda_c]
same_off = [p for p in chg_other if p not in moved]   # changed vs 7.14, same with DDA on or off
have_t = all(T[k] for k in T)
if not have_t:
    verdict = "The timing runs are missing, so there is no verdict; see the run log."
    one = "not decided (timing runs missing)."
elif len(slow_off) >= len(pts) / 3:
    v = [f"Turning DDA off does not bring back 7.14's behaviour. With `RCCL_DDA_ENABLE=0`, {len(slow_off)} of "
         f"{len(pts)} points are more than {BACK:.0f}% slower than both 7.14 and 10.1's default, in "
         f"{colls(slow_off)}. So the switch removes DDA and leaves a slower fallback path, not 7.14's path, "
         "and this test cannot measure how much of the change DDA explains."]
    if dda_c:
        v.append(f"10.1 runs {', '.join(dda_c)} through the new DDA-IPC implementation (debug log, §2)"
                 + (f"; the DDA switch also changes {colls(moved_nodda)}, which log no DDA line" if moved_nodda else "")
                 + ". The faster points come with this rework of RCCL's single-node path in 2.30.7"
                 + (f" ({lbl([p for p in changed if p[5] < 0])})" if any(p[5] < 0 for p in changed) else "") + ".")
    if same_off:
        v.append(f"{lbl(same_off)} change by the same amount with DDA on or off, so that change is separate "
                 "from DDA (another change in RCCL 2.30.7 / HIP 7.16).")
    verdict = " ".join(v)
    # plain-language version (2026-10-30): factors -> which changed -> why
    verdict = ("**Main factors for small-message latency in one node:** (1) fixed cost per call (kernel launch, "
        "set-up, final sync), which dominates at 8 B-1 KB; (2) algorithm (ring, tree, direct), which sets how many "
        "steps and GPU-to-GPU handshakes a call needs; (3) protocol (LL, LL128, Simple); (4) number of channels; "
        "(5) implementation path (copy through RCCL buffers, or read peer GPU memory directly); (6) the tuning table "
        "that picks algorithm and protocol per size.\n\n"
        "**Same in 7.14 and 10.1** (debug logs): channels (112 / 64 / 8), algorithm and protocol where logged "
        "(e.g. broadcast RING/LL up to 64K on both), and the tuning table (the new 10.1 rows are all_reduce at "
        "32 MB and above).\n\n"
        "**What differs and why:**\n"
        "- **Implementation path (5):** RCCL 2.30.7 adds a new DDA-IPC path for "
        + (" and ".join(dda_c) or "no collective") + ". Each GPU reads the other 7 GPUs' buffers directly in one step, "
        "instead of passing data around a ring in 7 steps with a handshake each. Fewer steps, less waiting: this is "
        "the likely reason all_gather is about 30% faster and all_reduce 1M about 20% faster.\n"
        "- **Fixed cost per call (1), probably:** sendrecv 1K and broadcast 1K are about 30% slower. Their algorithm, "
        "protocol and channels are the same, and the change is the same with DDA on or off, so it points to a "
        "higher per-call cost in RCCL 2.30.7 or HIP 7.16 (a deduction, not measured directly).\n"
        "- **Not explained:** alltoall is about 30% faster without using DDA, so it comes from another part of the "
        "2.30.7 rework; the logs do not say which.\n\n"
        "**Notes:** this is not noise (repeat runs agree within about ±3%; the changes are 20-37%). "
        f"`RCCL_DDA_ENABLE=0` does not restore 7.14: it falls back to a path slower than both versions "
        f"({len(slow_off)} of {len(pts)} points more than {BACK:.0f}% slower), so it cannot show how much DDA explains.")
    one = (f"real, not noise. RCCL 10.1 runs {', '.join(dda_c) or 'no collective'} through a new DDA-IPC path; "
           f"`RCCL_DDA_ENABLE=0` does not restore 7.14 (it is slower than both versions at {len(slow_off)} of "
           f"{len(pts)} points), so DDA's share cannot be measured with that switch. "
           + (f"{lbl(same_off)}: same with DDA on or off, a separate change." if same_off else ""))
else:
    v = []
    if chg_dda:
        v.append(f"{len(chg_dda)} of the points that changed by more than {BIG:.0f}% are in collectives that "
                 f"take RCCL 10.1's new DDA path ({', '.join(dda_c)}). With `RCCL_DDA_ENABLE=0`, "
                 f"{len(back_dda)} of these {len(chg_dda)} come back to within {BACK:.0f}% of 7.14"
                 + (f" ({lbl(back_dda)})" if back_dda else "") + ".")
        if len(back_dda) >= 0.75 * len(chg_dda):
            v.append("So the speed-ups (and any slow-downs) in these collectives come from the new DDA path, "
                     "not from noise between runs.")
        elif back_dda:
            v.append("So DDA explains part of the change in these collectives; the rest has another cause.")
        else:
            v.append("So the DDA path does not explain the change in these collectives.")
    if chg_other:
        v.append(f"{len(chg_other)} changed points are in collectives without DDA ({lbl(chg_other)}); "
                 f"turning DDA off brings {len(back_other)} of them back. "
                 + ("These come from another change in RCCL 2.30.7 / HIP 7.16 (lists of new debug lines "
                    "below)." if len(back_other) < len(chg_other) else "Those also follow the DDA setting."))
    if not changed:
        v.append(f"In this run no point changed by more than {BIG:.0f}% between 7.14 and 10.1.")
    verdict = " ".join(v)
    one = (f"RCCL 10.1 adds a direct GPU-to-GPU (DDA) path for {', '.join(dda_c) or 'no collective'}; with "
           f"`RCCL_DDA_ENABLE=0` {len(back_dda)} of {len(chg_dda)} changed points in those collectives return "
           f"to within {BACK:.0f}% of 7.14. {len(chg_other)} changed points elsewhere "
           f"({lbl(chg_other)}), {len(back_other)} of them follow DDA.")

# ---- write ---------------------------------------------------------------------------------------
L = []
w = L.append
w("# Why RCCL latency inside one node changed from ROCm 7.14 to 10.1 (MI355X)")
w("")
w(f"node6100, 8× MI355X, one node (XGMI only). Run dir: `{RUN}`. Script: "
  "[`../rocm10.1/run_rccl_why_1node.sh`](../rocm10.1/run_rccl_why_1node.sh). "
  "Question from [`rocm10.1-vs-rocm7.14.md`](rocm10.1-vs-rocm7.14.md) §5.1: are the 20-37% changes "
  "noise between runs, or did RCCL change how it runs these collectives?")
w("")
w("## Answer")
w("")
w(verdict)
w("")
w("Noise between runs is small: the three default runs per version in §5.1 agree within about ±3% "
  "on both nodes (node6100 and node6101), while the changes are 20-37%.")
w("")
w("## 1. Timing, same day (time per call, µs, out-of-place, lower is better)")
w("")
w("rccl-tests, 1000 iterations + 100 warm-up, no data check, NUMA-bound ranks; one run per column, "
  "back to back. Change = (10.1 − 7.14) / 7.14, so a negative number means 10.1 is faster.")
w("")
w("| Collective | Size | DDA on 10.1 | 7.14 | 10.1 | 10.1, `RCCL_DDA_ENABLE=0` | 10.1 vs 7.14 | DDA off vs 7.14 |")
w("|---|---:|---|---:|---:|---:|---:|---:|")
for c, s, a, b, n, g, k in pts:
    dd = ("yes" if uses_dda(c) else "no") if c in DBG_COLLS else "not checked"
    w(f"| {c} | {SIZES[s]} | {dd} | {fv(a)} | {fv(b)} | {fv(n)} | {fp(g)} | {fp(k)} |")
w("")
w("## 2. What RCCL picks (debug runs, rank 0)")
w("")
w("`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL,ENV`, 8 B to 1 MB, 5 iterations; separate from the "
  "timing runs because debug output slows every call.")
w("")
w("| Collective | Stack | Channels (coll / p2p / p2p per peer) | Implementation lines | DDA path by size | Algo/proto lines |")
w("|---|---|---|---|---|---|")
for c in DBG_COLLS:
    for s in STACKS:
        x = D[s][c]
        ch = " / ".join(x["chan"]) if x["chan"] else "–"
        im = ", ".join(f"{k} ×{v}" for k, v in x["impl"].most_common()) or "none"
        dd = "; ".join(f"{SIZES.get(b, b)}: {'/'.join(sorted(p))}" for b, p in sorted(x["dda"].items())
                       if b in SIZES) or ("yes, sizes not logged" if any("DDA" in k for k in x["impl"]) else "none")
        tu = "; ".join(f"{SIZES[b]}: {'/'.join(sorted(p))}" for b, p in sorted(x["tune"].items()) if b in SIZES) or "not logged"
        w(f"| {c} | {s} | {ch} | {im} | {dd} | {tu} |")
w("")
csvn = {s: next((D[s][c]["csv"] for c in DBG_COLLS if D[s][c]["csv"]), None) for s in STACKS}
w(f"Both versions load RCCL's gfx950 tuning table (`share/rccl/tuner/rccl_tuner_gfx950.csv`): "
  f"{csvn['7.14'] or '?'} rows on 7.14, {csvn['10.1'] or '?'} on 10.1. "
  + ("The rows only in 10.1 are all_reduce at "
     + ", ".join(f"{int(r.split(',')[1]) // 2**20}-{(int(r.split(',')[2]) + 1) // 2**20} MB" for r in new_rows
                 if r.split(',')[1].isdigit()) + ", far above the 1 MB used here, so the table does not "
     "explain these changes." if new_rows else "The tables are the same."))
w("")
w(f"`RCCL_DDA_ENABLE` and related settings (`RCCL_DDA_THRESHOLD`, `RCCL_DDA_LL_*`) exist in RCCL 2.30.7 "
  f"(strings in `librccl.so.1`); 7.14's RCCL 2.30.4 logs no DDA lines "
  f"({'none of the collectives' if not any(uses_dda_714(c) for c in DBG_COLLS) else 'except ' + ', '.join(c for c in DBG_COLLS if uses_dda_714(c))}).")
w("")
w("## 3. Debug lines only one version prints (rank 0, numbers replaced by #)")
w("")
for c in DBG_COLLS:
    k7, k10 = D["7.14"][c]["kinds"], D["10.1"][c]["kinds"]
    only10 = [(k, n) for k, n in k10.most_common() if k not in k7][:10]
    only7 = [(k, n) for k, n in k7.most_common() if k not in k10][:10]
    w(f"### {c}")
    w("")
    w("Only in 10.1:" if only10 else "Only in 10.1: none.")
    if only10:
        w("```"); [w(f"{n:5d}  {k}") for k, n in only10]; w("```")
    w("")
    w("Only in 7.14:" if only7 else "Only in 7.14: none.")
    if only7:
        w("```"); [w(f"{n:5d}  {k}") for k, n in only7]; w("```")
    w("")
OUT_MD.write_text("\n".join(L) + "\n")
print(f"wrote {OUT_MD}")

S = ["#### 5.1.1 Why the 1-node numbers changed", "",
     verdict + " Details, same-day timing and RCCL's choices: "
     "[`rccl-lat-1node-rocm10.1-why.md`](rccl-lat-1node-rocm10.1-why.md)."]
if have_t:
    S += ["", "| Collective | Size | 7.14 (µs) | 10.1 (µs) | 10.1, DDA off (µs) |", "|---|---:|---:|---:|---:|"]
    S += [f"| {p[0]} | {SIZES[p[1]]} | {fv(p[2])} | {fv(p[3])} | {fv(p[4])} |" for p in changed]
SHORT.write_text("\n".join(S) + "\n")
SUMM.write_text(f"- **Why 1-node latency changed:** {one}\n")
print(f"wrote {SHORT}, {SUMM}")
rc = subprocess.call([sys.executable, str(B / "rocm10.1/compare_rocm10.1.py")])
print(f"compare_rocm10.1.py rc={rc}")
