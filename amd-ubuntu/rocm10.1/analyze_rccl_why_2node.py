#!/usr/bin/env python3
"""Why 2-node all_reduce latency differs between ROCm 7.14 and 10.1: reads one run_rccl_why_2node.sh run dir.

  analyze_rccl_why_2node.py <logs/node6100/rocm10.1/rccl/why_2node_TS>

Writes results/rccl-lat-2node-rocm10.1-why.md (full), results/node6100/rocm10.1/rccl_why_2node_short.md
and rccl_why_2node_summary.txt (both read by compare_rocm10.1.py), then reruns compare_rocm10.1.py so
§5.2.1 and the Summary of results/rocm10.1-vs-rocm7.14.md pick them up.
"""
import re, subprocess, sys
from pathlib import Path

B = Path("/orcd/data/orcd/022/benchmarks/amd-ubuntu")
RES = B / "results"
OUT_MD = RES / "rccl-lat-2node-rocm10.1-why.md"
SHORT = RES / "node6100/rocm10.1/rccl_why_2node_short.md"
SUMM = RES / "node6100/rocm10.1/rccl_why_2node_summary.txt"
RUN = Path(sys.argv[1])
STACKS = ["7.14", "10.1"]
SETS = [("default", "default"), ("ll128", "`NCCL_PROTO=LL128`"), ("treell", "`NCCL_ALGO=Tree NCCL_PROTO=LL`"),
        ("ringll", "`NCCL_ALGO=Ring NCCL_PROTO=LL`"), ("nodda", "`RCCL_DDA_ENABLE=0`")]
SIZES = {8: "8", 1024: "1K", 8192: "8K", 65536: "64K", 262144: "256K", 1048576: "1M"}
SAME = 10.0   # within ±10 % counts as the same (one run per cell)
ROW = re.compile(r"^\s+(\d+)\s+(\d+)\s+\S+\s+\S+\s+-?\d+\s+([\d.]+)\s+\S+\s+\S+\s+(\S+)")
TUNE = re.compile(r"AllReduce: (\d+) Bytes -> Algo (\S+) proto (\S+)")


def runs():
    d = {}
    p = RUN / "runs.txt"
    for line in p.read_text().splitlines() if p.exists() else []:
        f = line.split("|")
        if len(f) >= 4 and f[3]:
            d[(f[0], f[1])] = Path(f[3]) / "all_reduce_ppn8_np16.log"
    return d


R = runs()


def rows(stack, tag):
    """-> {bytes: (us, wrong)}"""
    f = R.get((stack, tag))
    out = {}
    if f and f.exists():
        for line in f.read_text(errors="replace").splitlines():
            m = ROW.match(line)
            if m:
                w = m.group(4)
                out[int(m.group(1))] = (float(m.group(3)), int(w) if w.isdigit() else None)
    return out


def picks(stack):
    """rank 0 (node6100, local [0]) algo/proto per size from the debug run"""
    f = R.get((stack, "debug"))
    out = {}
    if f and f.exists():
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("node6100:") and " [0] NCCL" in line:
                m = TUNE.search(line)
                if m:
                    out.setdefault(int(m.group(1)), set()).add(f"{m.group(2)}/{m.group(3)}")
    return out


LAT = {(s, t): rows(s, f"{t}_lat") for s in STACKS for t, _ in SETS}
CHK = {(s, t): rows(s, f"{t}_check") for s in STACKS for t, _ in SETS}
PK = {s: picks(s) for s in STACKS}


def wrong(s, t):
    """sizes with wrong results, or None when the check run is missing"""
    c = CHK[(s, t)]
    return None if not c else [b for b, (_, w) in sorted(c.items()) if w]


def t(s, tag, b):
    return LAT[(s, tag)].get(b, (None, None))[0]


def pct(a, b):
    return None if a is None or b is None or a == 0 else (b - a) / a * 100


def fp(p):
    return "" if p is None else ("0.0%" if abs(p) < 0.05 else f"{p:+.1f}%")


def fv(x):
    return "–" if x is None else (f"{x:,.0f}" if x >= 100 else f"{x:.1f}")


def hs(b):
    return SIZES.get(b) or (f"{b // 1024}K" if b % 1024 == 0 and b < 2**20 else f"{b // 2**20}M" if b % 2**20 == 0 else str(b))


def fw(s, tg):
    w = wrong(s, tg)
    return "not run" if w is None else ("none" if not w else ", ".join(hs(b) for b in w))


have = bool(LAT[("7.14", "default")]) and bool(LAT[("10.1", "default")])

# ---- findings, from the data ------------------------------------------------------------------
F = []
same_pick = all(PK["7.14"].get(b) == PK["10.1"].get(b) for b in SIZES) and PK["7.14"] and PK["10.1"]
if PK["7.14"] and PK["10.1"]:
    p7 = "; ".join(f"{SIZES[b]}: {'/'.join(sorted(PK['7.14'][b]))}" for b in SIZES if b in PK["7.14"])
    if same_pick:
        F.append(f"Both versions pick the same algorithm and protocol by default ({p7}).")
    else:
        p10 = "; ".join(f"{SIZES[b]}: {'/'.join(sorted(PK['10.1'][b]))}" for b in SIZES if b in PK["10.1"])
        F.append(f"The default picks differ: 7.14 {p7}; 10.1 {p10}.")
w7, w10 = wrong("7.14", "default"), wrong("10.1", "default")
if w7 is not None and w10 is not None:
    F.append(f"Default setting, data check: wrong results on 7.14 at {', '.join(hs(b) for b in w7) or 'no size'}, "
             f"on 10.1 at {', '.join(hs(b) for b in w10) or 'no size'}.")
tw7, tw10 = wrong("7.14", "treell"), wrong("10.1", "treell")
small = [b for b in SIZES if b <= 262144]
slow_tree = [b for b in small if None not in (t("7.14", "treell", b), t("10.1", "treell", b))
             and t("10.1", "treell", b) > 3 * t("7.14", "treell", b)]
if tw7 is not None and tw10 is not None:
    s = (f"Tree/LL forced on both: wrong results on 7.14 at {', '.join(hs(b) for b in tw7) or 'no size'}, "
         f"on 10.1 at {', '.join(hs(b) for b in tw10) or 'no size'}")
    if slow_tree:
        s += (f"; 10.1 is more than 3× slower at {', '.join(SIZES[b] for b in slow_tree)} "
              f"(e.g. {SIZES[slow_tree[0]]}: {fv(t('7.14', 'treell', slow_tree[0]))} vs "
              f"{fv(t('10.1', 'treell', slow_tree[0]))} µs)")
    F.append(s + ".")
    if (tw10 or slow_tree) and not tw7:
        F.append("So the 2-node default problem on 10.1 is RCCL 2.30.7's Tree/LL path across nodes: 7.14 runs "
                 "the same algorithm and protocol correctly" + (" and fast" if slow_tree else "") +
                 ". This is a regression in RCCL, not a setting in this benchmark.")
ll = [(b, pct(t("7.14", "ll128", b), t("10.1", "ll128", b))) for b in SIZES]
ll = [(b, p) for b, p in ll if p is not None]
if ll:
    big = [(b, p) for b, p in ll if abs(p) > SAME]
    F.append("Same setting on both (`NCCL_PROTO=LL128`), 10.1 vs 7.14: "
             + "; ".join(f"{SIZES[b]} {fp(p)}" for b, p in ll) + ". "
             + ("Within ±%.0f%% at every size, so with LL128 on both the versions are about the same: the §5.2 "
                "gap at small sizes comes from comparing 10.1's LL128 with 7.14's default, not from a slower 10.1."
                % SAME if not big else
                f"{len(big)} sizes differ by more than {SAME:.0f}% even on the same setting."))
    d7 = [(b, pct(t("7.14", "default", b), t("7.14", "ll128", b))) for b in SIZES]
    F.append("On 7.14 itself, LL128 vs its default: " + "; ".join(f"{SIZES[b]} {fp(p)}" for b, p in d7 if p is not None)
             + " (positive = LL128 slower).")
nd = wrong("10.1", "nodda")
if nd is not None:
    F.append(f"10.1 with `RCCL_DDA_ENABLE=0`: wrong results at {', '.join(hs(b) for b in nd) or 'no size'}; "
             f"8 B {fv(t('10.1', 'nodda', 8))} µs vs {fv(t('10.1', 'default', 8))} µs default"
             + (", so DDA is not behind the 2-node problem." if (nd and w10) or
                (t('10.1', 'nodda', 8) and t('10.1', 'default', 8) and abs(pct(t('10.1', 'default', 8), t('10.1', 'nodda', 8))) <= SAME)
                else "."))
if not have:
    F = ["The latency runs are missing, so there is no verdict; see the run log."]

# ---- write --------------------------------------------------------------------------------------
L = []
w = L.append
w("# Why 2-node all_reduce latency differs between ROCm 7.14 and 10.1 (MI355X)")
w("")
w(f"node6100 + node6101, 2 nodes × 8 MI355X, 8 RoCE rails (RCCL NET/IB). Run dir: `{RUN}`. Script: "
  "[`../rocm10.1/run_rccl_why_2node.sh`](../rocm10.1/run_rccl_why_2node.sh). Question from "
  "[`rocm10.1-vs-rocm7.14.md`](rocm10.1-vs-rocm7.14.md) §5.2: there, 7.14 runs RCCL's default and 10.1 runs "
  "`NCCL_PROTO=LL128` (10.1's default gives wrong results), so the all_reduce rows compare different settings. "
  "Here both versions run the same settings, back to back.")
w("")
w("## Answer")
w("")
for x in F:
    w(f"- {x}")
w("")
w("## 1. Latency (time per call, µs, out-of-place, lower is better)")
w("")
w("rccl-tests all_reduce, 1000 iterations + 100 warm-up, no data check, NUMA-bound ranks; one run per column. "
  "Change = (10.1 − 7.14) / 7.14, so a negative number means 10.1 is faster.")
w("")
w("| Size | " + " | ".join(f"{lb}, 7.14 | {lb}, 10.1 | Change" for tg, lb in SETS if tg != "nodda") + " | `RCCL_DDA_ENABLE=0`, 10.1 |")
w("|---:|" + "---:|---:|---:|" * (len(SETS) - 1) + "---:|")
for b in SIZES:
    c = []
    for tg, _ in SETS:
        if tg == "nodda":
            continue
        a, n = t("7.14", tg, b), t("10.1", tg, b)
        c += [fv(a), fv(n), fp(pct(a, n))]
    w(f"| {SIZES[b]} | " + " | ".join(c) + f" | {fv(t('10.1', 'nodda', b))} |")
w("")
w("## 2. Correctness (data check on, 20 iterations, 8 B to 1 MB)")
w("")
w("| Setting | 7.14: sizes with wrong results | 10.1: sizes with wrong results |")
w("|---|---|---|")
for tg, lb in SETS:
    w(f"| {lb} | {fw('7.14', tg) if tg != 'nodda' else 'not run'} | {fw('10.1', tg)} |")
w("")
w("## 3. What RCCL picks by default (debug run, rank 0)")
w("")
w("`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL`, 2 iterations.")
w("")
w("| Size | 7.14 | 10.1 |")
w("|---:|---|---|")
for b in SIZES:
    w(f"| {SIZES[b]} | {'/'.join(sorted(PK['7.14'].get(b, []))) or '–'} | {'/'.join(sorted(PK['10.1'].get(b, []))) or '–'} |")
w("")
OUT_MD.write_text("\n".join(L) + "\n")
print(f"wrote {OUT_MD}")

S = ["#### 5.2.1 Why 2-node all_reduce differs", ""] + [f"- {x}" for x in F] + [
     "", "Same-day runs with the same settings on both versions, correctness checks and RCCL's choices: "
     "[`rccl-lat-2node-rocm10.1-why.md`](rccl-lat-2node-rocm10.1-why.md)."]
SHORT.write_text("\n".join(S) + "\n")
SUMM.write_text("- **Why 2-node all_reduce differs:** " + (F[0] if len(F) == 1 else " ".join(
    x for x in F if x.startswith(("So the", "Same setting")))) + "\n")
print(f"wrote {SHORT}, {SUMM}")
rc = subprocess.call([sys.executable, str(B / "rocm10.1/compare_rocm10.1.py")])
print(f"compare_rocm10.1.py rc={rc}")
