#!/usr/bin/env python3
"""ROCm 10.1.0 (+ hipBLASLt 1.4.1, RCCL 2.30.7) vs the current ROCm 7.14 results, MI355X node6100.

Reads results/node6100/rocm7.14/ (current, not changed) and results/node6100/rocm10.1/ (new),
writes results/rocm10.1-vs-rocm7.14.md and results/node6100/rocm10.1/rochpl.csv.
No existing md file is changed. A test with no new result is shown as "not run".
"""
import csv, re, statistics, sys
from pathlib import Path

B = Path("/orcd/data/orcd/022/benchmarks/amd-ubuntu")
OLD = B / "results/node6100/rocm7.14"
NEW = B / "results/node6100/rocm10.1"
OUT = B / "results/rocm10.1-vs-rocm7.14.md"
NA = "not run"
SAME = 3.0   # |change| <= 3 % counts as "about the same"
changes = {}  # test -> list of (label, pct)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(old, new, lower_better=False):
    if old is None or new is None or old == 0:
        return None
    p = (new - old) / old * 100
    return -p if lower_better else p


def fpct(p):
    if p is None:
        return ""
    return "0.0%" if abs(p) < 0.05 else f"{p:+.1f}%"


def fv(v, d=1):
    return NA if v is None else f"{v:,.{d}f}"


def note(test, label, p):
    if p is not None:
        changes.setdefault(test, []).append((label, p))


def read_csv(p):
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def versions():
    d, sec = {}, None
    p = NEW / "versions.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.match(r"\[(.+)\]", line)
            if m:
                sec = m.group(1); d[sec] = {}; continue
            if "=" in line and sec:
                k, v = line.split("=", 1); d[sec][k.strip()] = v.strip()
    return d


L = []
w = L.append
v = versions()
o, n, h = v.get("rocm7.14", {}), v.get("rocm10.1", {}), v.get("host", {})
w("# ROCm 10.1 vs ROCm 7.14 on MI355X")
w("")
w("ROCm 10.1.0 (released 2026-10-05) against the current ROCm 7.14 results, on the same node "
  "(node6100, 8× MI355X; node6101 joins for the 2-node tests). Same scripts and settings; "
  "only the ROCm user-space stack differs. Higher is better unless the table says otherwise. "
  f"Change = (10.1 − 7.14) / 7.14; within ±{SAME:.0f}% counts as about the same.")
w("")
w("Plan: [`../plan-new-version.txt`](../plan-new-version.txt). Scripts: [`../rocm10.1/`](../rocm10.1/).")
w("")
w("## Versions")
w("")
w("| | Current | New |")
w("|---|---|---|")
w(f"| ROCm | 7.14.0 | 10.1.0 |")
for k, name in [("hip", "HIP"), ("hipblaslt", "hipBLASLt"), ("rccl", "RCCL"), ("rocblas", "rocBLAS"), ("clang", "AMD clang")]:
    a, b = o.get(k, "?"), n.get(k, "?")
    if k == "clang":
        a, b = (re.sub(r"\s*\(.*", "", x).replace("AMD clang version ", "") for x in (a, b))
    w(f"| {name} | {a} | {b}{' (same version)' if a == b and a not in ('?', '') else ''} |")
w(f"| amdgpu driver (host, not changed) | {h.get('driver', '?')} | {h.get('driver', '?')} |")
w("| RVS | 1.7.8, commit c82e4aeb | same commit, built against 10.1 |")
w(f"| rocHPL / Open MPI / UCX | rocHPL {h.get('rochpl', '?')}, Open MPI 5.0.8, UCX 1.18.0 | same versions, rebuilt against 10.1 |")
w("")
if o.get("hipblaslt") and o.get("hipblaslt") == n.get("hipblaslt"):
    w(f"hipBLASLt reports the same version ({n['hipblaslt']}) in both stacks: ROCm 7.14 already shipped "
      "hipBLASLt 1.4.1. Its kernels and the rest of the stack (HIP, compiler, rocBLAS) still changed, "
      "so the GEMM results can still differ.")
    w("")

# ---- what is affected ----------------------------------------------------------------------
w("## What the update affects")
w("")
w("| Result | Uses | Affected? | Rerun here |")
w("|---|---|---|---|")
w("| RVS GEMM, all precisions (§1, §3 of `mi355x-vs-b200.md`) | RVS `gst` → hipBLASLt | Yes | Yes |")
w("| hipBLASLt MX GEMM correctness (MXFP8, MXFP4) | hipBLASLt | Yes | Yes (both stacks) |")
w("| rocHPL FP64 (§2, `rochpl-vs-rvs.md`) | rocBLAS + HIP + Open MPI/UCX | Yes | Yes |")
w("| RCCL bandwidth and latency, 1 and 2 nodes (§4, interconnect and latency files) | RCCL | Yes | Yes |")
w("| Network alone (`ib_write_bw`, `ib_lat`) | perftest, host memory, no ROCm | No | No |")
w("| Training, Primus / Megatron-LM (§5) | container with its own ROCm | No | No |")
w("| Inference, Kimi-K3 (§6, Kimi files) | containers with their own ROCm | No | No |")
w("")

# ---- 1. MX correctness ----------------------------------------------------------------------
w("## 1. hipBLASLt MX GEMM correctness (MXFP8, MXFP6, MXFP4)")
w("")
mx = read_csv(NEW / "mx_check.csv")
if not mx:
    w(f"_{NA}._"); w("")
else:
    w("`hipblaslt-bench -v` (result checked against a CPU reference), AMD's own bench from each "
      "release's gfx950 test tarball, 1 GPU. Scale mode 3 = MX block scaling (one E8M0 scale per 32 elements). "
      "Check at 2048×2048×2048 (PASS = error within the bench's own tolerance); speed at 8192×8192×8192.")
    w("")
    w("The MX fixes in the 10.1 release notes are in Composable Kernel: most name gfx1250, and one fixes "
      "MX GEMMs where B is packed more densely than A (such as FP8 × FP4). The FP8 × FP4 and FP4 × FP8 "
      "rows test that case on gfx950.")
    w("")
    w("| Case | A × B | 7.14 check (error ≤ limit) | 7.14 TFLOPS | 10.1 check | 10.1 TFLOPS | Change |")
    w("|---|---|---|---:|---|---:|---:|")
    cases = {}
    for r in mx:
        cases.setdefault(r["case"], {})[r["version"]] = r
    for c, d in cases.items():
        a, b = d.get("7.14"), d.get("10.1")
        ref = a or b
        ta = fnum(a["gflops"]) / 1000 if a and fnum(a["gflops"]) else None
        tb = fnum(b["gflops"]) / 1000 if b and fnum(b["gflops"]) else None
        def chk(r):
            if not r:
                return NA
            e, t = fnum(r.get("error")), fnum(r.get("rtol"))
            return r["status"] + (f" ({e:.2g} ≤ {t:g})" if r["status"] == "PASS" and e is not None and t is not None
                                  else (f" ({e:.2g}, limit {t:g})" if e is not None and t is not None else ""))
        sa, sb = chk(a), chk(b)
        p = pct(ta, tb); note("hipBLASLt MX GEMM", c, p)
        w(f"| {c} | {ref['a_type']} × {ref['b_type']} | {sa} | {fv(ta)} | {sb} | {fv(tb)} | {fpct(p)} |")
    w("")

# ---- 2. RVS ------------------------------------------------------------------------------------
w("## 2. RVS GEMM (hipBLASLt), TFLOPS")
w("")
ro = {(r["gpus"], r["precision"]): r for r in read_csv(OLD / "rvs_tflops.csv")}
rn = {(r["gpus"], r["precision"]): r for r in read_csv(NEW / "rvs_tflops.csv")}
precs = [p for p in ["fp4", "fp6", "bf6", "fp8", "bf8", "fp16", "bf16", "fp32", "fp64"] if ("8", p) in ro]
if not rn:
    w(f"_{NA}._"); w("")
else:
    w("Same RVS `gst` settings as §1/§3 of `mi355x-vs-b200.md` (level-5 matrix sizes, 30 s per test, "
      "peak of the 3 s samples). FP4, FP6 and FP8 are the MX formats.")
    w("")
    w("| Precision | 8 GPUs, 7.14 | 8 GPUs, 10.1 | Change | 1 GPU, 7.14 | 1 GPU, 10.1 | Change |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for p in precs:
        a8, b8 = fnum(ro.get(("8", p), {}).get("aggregate_tflops")), fnum(rn.get(("8", p), {}).get("aggregate_tflops"))
        a1, b1 = fnum(ro.get(("1", p), {}).get("aggregate_tflops")), fnum(rn.get(("1", p), {}).get("aggregate_tflops"))
        p8, p1 = pct(a8, b8), pct(a1, b1)
        note("RVS GEMM", f"{p} 8 GPUs", p8)
        w(f"| {p.upper()} | {fv(a8)} | {fv(b8)} | {fpct(p8)} | {fv(a1)} | {fv(b1)} | {fpct(p1)} |")
    allp = [pct(fnum(ro[k]["aggregate_tflops"]), fnum(rn[k]["aggregate_tflops"])) for k in ro if k in rn]
    allp = [x for x in allp if x is not None]
    if allp:
        w("")
        w(f"All {len(allp)} points (1–8 GPUs × precisions): median change {statistics.median(allp):+.1f}%, "
          f"range {min(allp):+.1f}% to {max(allp):+.1f}%.")
    w("")

# ---- 3. rocHPL ---------------------------------------------------------------------------------
w("## 3. rocHPL FP64 Linpack, TFLOPS")
w("")
ho = {r["gpus"]: r for r in read_csv(OLD / "rochpl.csv")}
hn = {}
rd = NEW / "rochpl_run_dir.txt"
if rd.exists():
    run = Path(rd.read_text().strip())
    s = run / "summary.txt"
    if s.exists():
        for line in s.read_text().splitlines():
            f = line.split()
            if len(f) >= 10:
                G, P, Q, N, NB, t, tf, ok, rc, frac = f[:10]
                hn[G] = dict(gpus=G, P=P, Q=Q, N=N, NB=NB, time_s=t, hpl_tflops=tf, passed=ok, rc=rc, frac=frac, run=run.name)
        with open(NEW / "rochpl.csv", "w", newline="") as fo:
            cw = csv.writer(fo)
            cw.writerow(["run", "gpus", "nodes", "P", "Q", "N", "NB", "time_s", "hpl_tflops", "hpl_per_gpu", "mem_frac", "passed"])
            for G, r in hn.items():
                tf = fnum(r["hpl_tflops"]) or 0
                cw.writerow([r["run"], G, 2 if int(G) > 8 else 1, r["P"], r["Q"], r["N"], r["NB"], r["time_s"],
                             r["hpl_tflops"], f"{tf / int(G):.2f}", r["frac"], r["passed"]])
if not hn:
    w(f"_{NA}._"); w("")
else:
    w("Same rocHPL commit, N, NB (512) and process grids as the ROCm 7.14 run "
      "(`rochpl_20261020_204342`, reported in `rochpl-vs-rvs.md`). Each run must pass HPL's residual check.")
    w("")
    w("| GPUs | P×Q | N (7.14 / 10.1) | 7.14 | 10.1 | Change | Passed (7.14 / 10.1) |")
    w("|---:|---|---|---:|---:|---:|---|")
    for G in sorted(set(ho) | set(hn), key=int):
        a, b = ho.get(G), hn.get(G)
        ta, tb = fnum(a["hpl_tflops"]) if a else None, fnum(b["hpl_tflops"]) if b else None
        p = pct(ta, tb); note("rocHPL", f"{G} GPUs", p)
        pq = f"{(a or b)['P']}×{(a or b)['Q']}"
        nn = f"{a['N'] if a else '–'} / {b['N'] if b else '–'}"
        ok = f"{'yes' if a and a['passed'] == '1' else ('no' if a else '–')} / {'yes' if b and b['passed'] == '1' else ('no' if b else NA)}"
        w(f"| {G} | {pq} | {nn} | {fv(ta)} | {fv(tb)} | {fpct(p)} | {ok} |")
    w("")

# ---- 4. RCCL bandwidth ----------------------------------------------------------------------------
w("## 4. RCCL bandwidth (bus bandwidth, GB/s)")
w("")
w("### 4.1 One node, 8 GPUs")
w("")


def first(rows, key):
    d = {}
    for r in rows:
        d.setdefault(key(r), r)
    return d


co = first([r for r in read_csv(OLD / "rccl.csv") if r["config"] == "default" and r["gpus"] == "8"], lambda r: r["collective"])
cn = first([r for r in read_csv(NEW / "rccl.csv") if r["config"] == "default" and r["gpus"] == "8"], lambda r: r["collective"])
if not cn:
    w(f"_{NA}._"); w("")
else:
    w("rccl-tests, default RCCL settings, peak bus bandwidth over message sizes up to 8 GB.")
    w("")
    w("| Collective | 7.14 | 10.1 | Change |")
    w("|---|---:|---:|---:|")
    for c in sorted(co):
        a, b = fnum(co[c]["peak_busbw_GBps"]), fnum(cn.get(c, {}).get("peak_busbw_GBps"))
        p = pct(a, b); note("RCCL bandwidth, 1 node", c, p)
        w(f"| {c} | {fv(a)} | {fv(b)} | {fpct(p)} |")
    w("")

w("### 4.2 Two nodes, 16 GPUs")
w("")
to = first([r for r in read_csv(OLD / "rccl_2node.csv") if r["ppn"] == "8"], lambda r: r["collective"])
tn = first([r for r in read_csv(NEW / "rccl_2node.csv") if r["ppn"] == "8"], lambda r: r["collective"])
if not tn:
    w(f"_{NA}._"); w("")
else:
    w("rccl-tests over the 8 RoCE rails (RCCL NET/IB), 8 GPUs per node, peak bus bandwidth and bus bandwidth at 1 GB.")
    w("")
    w("| Collective | Peak, 7.14 | Peak, 10.1 | Change | 1 GB, 7.14 | 1 GB, 10.1 | Change |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for c in sorted(to):
        a, b = fnum(to[c]["peak_busbw"]), fnum(tn.get(c, {}).get("peak_busbw"))
        a1, b1 = fnum(to[c]["busbw_1G"]), fnum(tn.get(c, {}).get("busbw_1G"))
        p = pct(a, b); note("RCCL bandwidth, 2 nodes", c, p)
        w(f"| {c} | {fv(a)} | {fv(b)} | {fpct(p)} | {fv(a1)} | {fv(b1)} | {fpct(pct(a1, b1))} |")
    w("")

# ---- 5. RCCL latency --------------------------------------------------------------------------------


def parse_lat(p):
    """md from analyze_rccl_lat.py -> {collective: {column: {bytes: us}}}"""
    d, coll, cols = {}, None, None
    if not p.exists():
        return d
    for line in p.read_text().splitlines():
        m = re.match(r"## (\S+)", line)
        if m:
            coll, cols = m.group(1), None; d[coll] = {}; continue
        if coll and line.startswith("| bytes"):
            cols = [c.strip() for c in line.strip("|").split("|")][1:]; continue
        if coll and cols and line.startswith("|") and not line.startswith("|---"):
            f = [c.strip() for c in line.strip("|").split("|")]
            for c, x in zip(cols, f[1:]):
                d[coll].setdefault(c, {})[f[0]] = fnum(x)
    return d


SIZES = ["8", "1K", "64K", "1M"]


def lat_table(old, new, pick, title, test, pick_new=None):
    """pick(columns) -> list of columns to take the median of; pick_new(collective, columns) for 10.1"""
    w(title)
    w("")
    if not new:
        w(f"_{NA}._"); w("")
        return
    w("| Collective | Size | 7.14 (µs) | 10.1 (µs) | Change (lower time = positive) |")
    w("|---|---:|---:|---:|---:|")
    for c in old:
        if c not in new:
            continue
        co_ = pick(old[c]); cn_ = pick_new(c, new[c]) if pick_new else pick(new[c])
        for s in SIZES:
            a = [old[c][k].get(s) for k in co_ if old[c][k].get(s) is not None]
            b = [new[c][k].get(s) for k in cn_ if new[c][k].get(s) is not None]
            if not a:
                continue
            ma = statistics.median(a); mb = statistics.median(b) if b else None
            p = pct(ma, mb, lower_better=True); note(test, f"{c} {s}", p)
            w(f"| {c} | {s} | {ma:.1f} | {fv(mb)} | {fpct(p)} |")
    w("")


w("## 5. RCCL latency (time per call, µs, lower is better)")
w("")
w("rccl-tests, 1000 iterations + 100 warm-up, no data check. In the Change column a positive number "
  "means 10.1 is faster.")
w("")
lo, ln = parse_lat(OLD / "rccl_lat_1node.md"), parse_lat(NEW / "rccl_lat_1node.md")
lat_table(lo, ln, lambda cols: [k for k in cols if "default" in k],
          "### 5.1 One node, 8 GPUs (median of the three default runs)", "RCCL latency, 1 node")
WHY = NEW / "rccl_why_1node_short.md"   # from analyze_rccl_why_1node.py
if WHY.exists():
    w(WHY.read_text().rstrip()); w("")
lo2, ln2 = parse_lat(OLD / "rccl_lat.md"), parse_lat(NEW / "rccl_lat.md")
FIXC = "2 nodes, 8 GPU/node, NCCL_PROTO=LL128"
fixed = any(FIXC in ln2.get(c, {}) for c in ln2)
if fixed:
    w("On ROCm 10.1, RCCL 2.30.7's default all_reduce across 2 nodes x 8 GPUs gives wrong results "
      "(rccl-tests data check) and takes 1.4-2.2 ms per call from 8 B to 256 KB. `NCCL_PROTO=LL128` gives "
      "correct results at every size, so the 10.1 all_reduce rows in 5.2 use it; the other 10.1 rows "
      "and all 7.14 rows use RCCL's default (all settings: `node6100/rocm10.1/fix_check.md`, "
      "`node6100/rocm10.1/rccl_lat.md`).")
    w("")
lat_table(lo2, ln2, lambda cols: [k for k in cols if k == "2 nodes, 8 GPU/node"],
          "### 5.2 Two nodes, 8 GPUs per node (default setting; 10.1 all_reduce with `NCCL_PROTO=LL128`)"
          if fixed else "### 5.2 Two nodes, 8 GPUs per node (default setting)", "RCCL latency, 2 nodes",
          pick_new=(lambda c, cols: [FIXC] if c == "all_reduce" and FIXC in cols
                    else [k for k in cols if k == "2 nodes, 8 GPU/node"]) if fixed else None)
lat_table(lo2, ln2, lambda cols: [k for k in cols if k == "2 nodes, 1 GPU/node"],
          "### 5.3 Two nodes, 1 GPU per node (default setting)", "RCCL latency, 2 nodes, 1 GPU/node")

# ---- summary ---------------------------------------------------------------------------------------
S = []
for test, items in changes.items():
    ps = [p for _, p in items]
    if not ps:
        continue
    up = sum(p > SAME for p in ps); dn = sum(p < -SAME for p in ps)
    best = max(items, key=lambda x: x[1]); worst = min(items, key=lambda x: x[1])
    s = f"- **{test}:** median {fpct(statistics.median(ps))} over {len(ps)} points; {up} faster, {dn} slower, {len(ps) - up - dn} about the same"
    if best[1] > SAME:
        s += f". Largest gain: {best[0]} ({best[1]:+.1f}%)"
    if worst[1] < -SAME:
        s += f". Largest loss: {worst[0]} ({worst[1]:+.1f}%)"
    S.append(s + ".")
    if test == "RCCL latency, 1 node" and (NEW / "rccl_why_1node_summary.txt").exists():
        S.append((NEW / "rccl_why_1node_summary.txt").read_text().rstrip())
if mx:
    bad = {(r["version"], r["case"]) for r in mx if r["status"] != "PASS"}
    if bad:
        S.append("- **MX correctness:** not passing: " + ", ".join(f"{c} on {ver}" for ver, c in sorted(bad)) + ".")
    else:
        S.append("- **MX correctness:** every MX case passed on both versions.")
summary = ["## Summary", ""] + (S if S else ["_No new results yet._"]) + [
    f"- **Not affected, not rerun:** network-only tests, Primus training and Kimi-K3 inference.", ""]
# summary goes right after the intro
i = L.index("## Versions")
L[i:i] = summary
OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT}")
