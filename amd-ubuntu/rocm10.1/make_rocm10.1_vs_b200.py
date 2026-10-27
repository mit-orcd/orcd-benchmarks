#!/usr/bin/env python3
"""Build results/mi355x-rocm10.1-vs-b200.md and results/mi355x-rocm10.1-vs-b200-interconnect.md.

MI355X = ROCm 10.1 (hipBLASLt 1.4.1, RCCL 2.30.7) results on node6100/node6101; B200/B300 numbers are the
same measurements as in the B200 result files (constants below). Same sections and points as the MI355X vs
B200 write-up; §5 (training), §6 (Kimi-K3) and the apple-to-apple notes do not depend on the host ROCm and
are taken from mi355x-vs-b200.md (MI355X BF16 numbers inside them are replaced by the ROCm 10.1 ones).
2-node items show "pending" until run_2node_fix_rocm10.1.sh has finished.
Then runs compare_lat_1node_rocm10.1.py (§1.3 + Summary row) and, when the 2-node runs are done,
compare_lat_rocm10.1.py (§2.3 + Summary row). §2 (HPL) is kept from the existing file when it is filled
(b200-nodes/hpl/analyze-hpl-b200.py fills it when the B200 HPL runs finish).
"""
import csv, glob, json, os, re, statistics, subprocess, sys
from pathlib import Path

B = Path("/orcd/data/orcd/022/benchmarks")
A = B / "amd-ubuntu"
R = A / "results"
N0, N1 = R / "node6100/rocm10.1", R / "node6101/rocm10.1"
OLD = R / "mi355x-vs-b200.md"            # template for §5, §6 and the apple-to-apple table
OLD_IC = R / "mi355x-vs-b200-interconnect.md"   # template for the on-paper sections and Appendix A
MAIN = R / "mi355x-rocm10.1-vs-b200.md"
IC = R / "mi355x-rocm10.1-vs-b200-interconnect.md"
PY = sys.executable
B200_LAT2 = B / "b200-nodes/out-nccl-lat/25690497"
RED = lambda s: f"$\\color{{red}}{{\\textsf{{{s}}}}}$"
REDB = lambda s: f"$\\color{{red}}{{\\textbf{{{s}}}}}$"

# ---- constants: datasheet peaks and B200/B300 measurements (TFLOP/s per GPU, GB/s) ----------------
PAPER_MI = dict(fp4=10066, fp6=10066, fp8=5033, bf16=2516, fp16=2516, fp64=78.6, fp32=157.3)
PAPER_B2 = dict(fp4=9000, fp8=4500, bf16=2250, fp16=2250, fp64=37, tf32=1100)
PAPER_B3 = dict(fp4=13500, fp8=4500, bf16=2250, fp16=2250, fp64=1.25, tf32=1100)
FRYER_B2 = dict(bf16=1450, fp8=4018, tf32=751)
FRYER_B3 = dict(bf16=1521, fp8=4228, tf32=797)
CUBLAS_B2 = dict(fp4=5708, fp8=2524, bf16=1396, fp16=1322, fp64=36, tf32=720)
CUBLAS_B3 = dict(fp4=6877, fp8=2524, bf16=1437, fp16=1333, fp64=1.10, tf32=756)
BW1_B2 = dict(all_reduce=839.4, all_gather=681.6, reduce_scatter=691.6, broadcast=682.1, reduce=687.5,
              gather=718.0, scatter=731.2, alltoall=658.1, sendrecv=668.6)
BW2_B2 = dict(all_reduce=233.1, all_gather=375.4, reduce_scatter=382.4, broadcast=359.1, alltoall=48.9, sendrecv=48.8)
CEIL2 = dict(all_reduce=400, all_gather=400, reduce_scatter=400, broadcast=400, alltoall=93.75, sendrecv=50)


def rows(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def c0(x):
    return "—" if x is None else f"{x:,.0f}"


def pc(x, peak):
    return f"{100 * x / peak:.0f}%"


def rt(x, y, d=2):
    return f"{x / y:.{d}f}x"


# ---- MI355X GEMM (RVS) ---------------------------------------------------------------------------
def rvs(n, gpus):
    out = {}
    for r in rows(n / "rvs_tflops.csv"):
        if r["gpus"] == str(gpus):
            out[r["precision"]] = fnum(r["aggregate_tflops"]) / gpus
    return out


r8 = {p: mean([rvs(N0, 8).get(p), rvs(N1, 8).get(p)]) for p in ["fp4", "fp6", "bf6", "fp8", "bf8", "fp16", "bf16", "fp32", "fp64"]}
r1_0, r1_1 = rvs(N0, 1).get("fp4"), rvs(N1, 1).get("fp4")
fp4_1 = mean([r1_0, r1_1])
mp = sorted(glob.glob(str(A / "logs/node6100/rocm10.1/rvs/fp4_multiproc_*/summary.txt")))
fp4_mp = fp4_mp_sum = fp4_mp_lo = fp4_mp_hi = None
if mp:
    for line in open(mp[-1]):
        m = re.match(r"multi101\s+GPUs=(\d+) sum=(\d+) mean/GPU=(\d+) \|(.*)", line)
        if m:
            fp4_mp, fp4_mp_sum = float(m.group(3)), float(m.group(2))
            per = [float(x.split(":")[1]) for x in m.group(4).split()]
            fp4_mp_lo, fp4_mp_hi = min(per), max(per)
need = [r8["bf16"], r8["fp8"], r8["fp4"], fp4_1, fp4_mp]
if any(x is None for x in need):
    sys.exit(f"missing RVS inputs: {need}")
bf16, fp8, fp16, fp64, fp32, fp6, fp4_8 = (r8[k] for k in ("bf16", "fp8", "fp16", "fp64", "fp32", "fp6", "fp4"))

# ---- RCCL 1 node, 8 GPUs (mean of both nodes) -----------------------------------------------------
def bw1(n):
    d = {}
    for r in rows(n / "rccl.csv"):
        if r["config"] == "default" and r["gpus"] == "8":
            d.setdefault(r["collective"], fnum(r["busbw_at_max_GBps"]))
    return d


b0, b1 = bw1(N0), bw1(N1)
BW1 = {c: mean([b0.get(c), b1.get(c)]) for c in BW1_B2}
ONE_NODE_BOTH = bool(b1)

# ---- RCCL 2 nodes ------------------------------------------------------------------------------------
FIXD = None
if (N0 / "fix_2node_dir.txt").exists():
    FIXD = Path((N0 / "fix_2node_dir.txt").read_text().strip())
DONE2 = (A / "logs/node6100/rocm10.1/driver/DONE2").exists() and FIXD and (FIXD / "STATE.txt").exists() \
    and "2-node follow-up DONE" in (FIXD / "STATE.txt").read_text()
fc = json.load(open(N0 / "fix_choice.json")) if (N0 / "fix_choice.json").exists() and DONE2 else {}
FIX = fc.get("fix_env", "")


def busbw_max(logf):
    """busbw (out-of-place) at the largest size in a rccl-tests log."""
    best = None
    for line in open(logf, errors="replace"):
        p = line.split()
        if len(p) >= 13 and p[0].isdigit() and p[1].isdigit():
            v = fnum(p[7])
            if v is not None:
                best = (int(p[0]), v) if best is None or int(p[0]) >= best[0] else best
    return best[1] if best else None


BW2 = {}
for r in rows(N0 / "rccl_2node.csv"):
    if r["ppn"] == "8":
        BW2.setdefault(r["collective"], {})["default"] = fnum(r["busbw_max_size"])
if DONE2:
    for line in open(FIXD / "bw_runs.txt"):
        tag, env, rd = line.rstrip("\n").split("|")
        for f in glob.glob(f"{rd}/*_ppn8_np16.log"):
            c = os.path.basename(f).split("_ppn")[0]
            BW2.setdefault(c, {})[tag.replace("bw_", "")] = busbw_max(f)


def bw2_best(c):
    d = BW2.get(c, {})
    v = {k: x for k, x in d.items() if x}
    if not v:
        return None, None
    k = max(v, key=v.get)
    return v[k], k


SETTING = {"default": "defaults", "netib": "`NCCL_NET=IB`", "netib_ch4": "`NCCL_NET=IB` + `NCCL_NCHANNELS_PER_NET_PEER=4`",
           "ch4": "`NCCL_NCHANNELS_PER_NET_PEER=4`", "fix": f"`{FIX}`"}

# =====================================================================================================
# interconnect file
# =====================================================================================================
old_ic = OLD_IC.read_text()


def chunk(text, start, end):
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


paper1 = chunk(old_ic, "### 1.1 On paper", "### 1.2 Measured bandwidth")
paper2 = chunk(old_ic, "### 2.1 On paper", "### 2.2 Measured bandwidth")
appA = old_ic[old_ic.index("## Appendix A."):].replace("with ROCm 7.14's default", "with RCCL's default (also on ROCm 10.1)")
for t in (paper1, paper2, appA):
    assert "7.14" not in t and "7.2.4" not in t

# 1.2 table
t12 = ["| Collective | MI355X (RCCL) | MI355X on paper | MI355X % of paper | B200 (NCCL) | B200 on paper | B200 % of paper | MI355X / B200 |",
       "|---|---:|---:|---:|---:|---:|---:|---:|"]
r12 = {}
for c, y in BW1_B2.items():
    x = BW1.get(c)
    pk = 76.8 if c == "sendrecv" else 537.6
    if x:
        r12[c] = x / y
        t12.append(f"| {c} | {x:.1f} | {pk} | {pc(x, pk)} | {y} | 900 | {pc(y, 900)} | **{rt(x, y)}** |")
coll_r = [v for k, v in r12.items() if k != "sendrecv"]
lo1, hi1 = min(coll_r), max(coll_r)
sr1 = r12.get("sendrecv")
mi_pk = [100 * BW1[c] / (76.8 if c == "sendrecv" else 537.6) for c in BW1 if BW1[c]]
b2_pk = [100 * y / 900 for y in BW1_B2.values()]
nodes1 = "mean of node6100 and node6101" if ONE_NODE_BOTH else "node6100"

# 2.2 table + notes
if DONE2 and BW2:
    t22 = ["| Collective | MI355X (RCCL) | MI355X ceiling | MI355X % of ceiling | B200 (NCCL) | B200 ceiling | B200 % of ceiling | MI355X / B200 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    r22, used = {}, {}
    for c in ["all_reduce", "all_gather", "reduce_scatter", "broadcast", "alltoall", "sendrecv"]:
        x, k = bw2_best(c)
        if c == "all_reduce" and "default" in BW2.get(c, {}):
            x, k = BW2[c]["default"], "default"   # large messages: RCCL's default is correct and fastest
        y, ce = BW2_B2[c], CEIL2[c]
        if not x:
            continue
        r22[c], used[c] = x / y, k
        low_b = c in ("all_reduce", "alltoall")
        yb = RED(f"{y}") if low_b else f"{y}"
        yp = RED(pc(y, ce).replace("%", "\\\\%")) if low_b else pc(y, ce)
        rr = REDB(rt(x, y)) if low_b else rt(x, y)
        mark = "" if k == "default" else f" ({SETTING.get(k, k)})"
        t22.append(f"| {c}{mark} | {x:.1f} | {ce} | {pc(x, ce)} | {yb} | {ce} | {yp} | {rr} |")
    ring = [r22[c] for c in ("all_gather", "reduce_scatter", "broadcast") if c in r22]
    alt = {k: v for k, v in BW2.get("alltoall", {}).items() if v}
    srv = {k: v for k, v in BW2.get("sendrecv", {}).items() if v}
    defaults_line = "; ".join(
        [f"alltoall: " + ", ".join(f"{SETTING[k]} {v:.1f} GB/s ({pc(v, 93.75)})" for k, v in alt.items())] +
        [f"sendrecv: " + ", ".join(f"{SETTING[k]} {v:.1f} GB/s ({pc(v, 50)})" for k, v in srv.items())])
else:
    t22 = None
fixmd = (N0 / "fix_check.md").read_text() if DONE2 and (N0 / "fix_check.md").exists() else ""

IC_L = []
w = IC_L.append
w("# AMD MI355X vs NVIDIA B200: interconnect (RCCL vs NCCL)")
w("")
w("Every ratio is **MI355X / B200**: above 1 means MI355X is faster (higher bandwidth or lower latency). "
  "MI355X = RCCL 2.30.7 on ROCm 10.1, node6100 + node6101; B200 = NCCL on the B200 nodes.")
w("")
w("- [Quick summary](#quick-summary)")
w("- [1. Intra-node (8 GPUs in one node)](#1-intra-node-8-gpus-in-one-node): 1.1 on paper, 1.2 measured bandwidth, 1.3 measured latency")
w("- [2. Inter-node (2 nodes, 16 GPUs)](#2-inter-node-2-nodes-16-gpus): 2.1 on paper, 2.2 measured bandwidth, 2.3 measured latency")
w("- [Appendix A. Pollara: RoCEv2 vs Ultra Ethernet](#appendix-a-pollara-rocev2-vs-ultra-ethernet)")
w("")
w("## Quick summary")
w("")
w("Which is faster and by how much. Here N× faster is the faster side's speed over the slower side's; the sections below give the same results as MI355X / B200 ratios.")
w("")
w("<!-- QS-TABLE -->")
w("")
w("On paper: links inside one node 537.6 vs 900 GB/s per GPU (0.60x, §1.1); network 400 vs 400 GB/s per node (1.00x, §2.1); neither vendor publishes latency figures.")
w("")
w("## 1. Intra-node (8 GPUs in one node)")
w("")
w(paper1.rstrip())
w("")
w("### 1.2 Measured bandwidth (large messages)")
w("")
w(f"busbw (GB/s) at the largest message size, **8 GPUs in one node on both systems** (one rank per GPU, the GPUs talk over XGMI / NVLink only). "
  f"MI355X = RCCL on ROCm 10.1, {nodes1} (max 8 GB; alltoall 4 GB). B200 = mean of 10 nodes (max 16 GiB).")
w("")
IC_L += t12
w("")
w("On paper = vendor peak per GPU, one direction, GB/s:")
w("- **MI355X**: 7 Infinity Fabric (XGMI) links × 76.8 GB/s = **537.6 GB/s** per GPU; **sendrecv** uses one GPU pair = one link = **76.8 GB/s**.")
w("- **B200**: NVLink 5 through NVSwitch = **900 GB/s** per GPU, for every collective (a pair can use all of it, so also for sendrecv).")
w(f"- **Note on all_reduce (expected):** B200 reaches 93% of its link peak, above its ≈76% on the other collectives, because **NVLink SHARP** "
  "(the NVSwitch adds the data up inside the switch) can improve all_reduce on B200 nodes; NCCL uses it by default when available "
  "(not checked in the B200 logs). MI355X has no switch and no such in-network reduction.")
w("")
w("**Apple-to-apple: close, not exact.**")
w("- Same test design: one process driving 8 GPUs, 20 iterations after 5 warm-up, busbw at the largest size.")
w("- Largest message differs (8 GB vs 16 GiB). Both curves are flat at that point, so the effect is small.")
w(f"- The ratio reflects the hardware: MI355X connects the GPUs in a full mesh of XGMI links (537.6 GB/s per GPU in total, 76.8 GB/s per pair on paper). "
  f"B200 connects all GPUs through NVSwitch (900 GB/s per GPU, any pair can use all of it). That is why sendrecv, which uses one pair of GPUs, is {sr1:.2f}x.")
w("")
w("> [!NOTE]")
w(f"> Inside one node, NVIDIA NVLink with NVSwitch is faster: AMD Infinity Fabric reaches only {lo1:.2f}–{hi1:.2f}x of its collective bandwidth.")
w("")
w("<!-- INTRA-LAT-BEGIN (auto: rccl-tests/compare_lat_1node.py) -->")
w("<!-- INTRA-LAT-END -->")
w("")
w("<!-- INTRA-LAT-NOTES -->")
w("")
w("## 2. Inter-node (2 nodes, 16 GPUs)")
w("")
w(paper2.rstrip())
w("")
w("### 2.2 Measured bandwidth (large messages)")
w("")
if t22:
    w("**Compared: MI355X (RCCL 2.30.7, ROCm 10.1, node6100 + node6101) vs B200 (NCCL, mean of the 4 healthy node pairs), 2 nodes × 8 GPUs, one rank per GPU.** "
      "busbw (GB/s) at the largest message size (16 GiB; MI355X alltoall 4 GB). Both have 8 × 400G rails per node = 400 GB/s (§2.1). "
      "RCCL is AMD's version of NCCL and reads the same `NCCL_*` setting names, so the `NCCL_*` variables below are RCCL settings on MI355X. "
      "MI355X rows use RCCL's defaults unless a setting is named; the named setting is the fastest one tested for that collective.")
    w("")
    IC_L += t22
    w("")
    w("Ceiling = what the network can carry for that pattern, per node and direction. **400 GB/s** for the ring/tree collectives: up to 8 rings, one per rail, "
      "together use the whole network of the node. sendrecv and alltoall use no ring, so their ceiling is what one GPU can send over its own rail:")
    w("- **alltoall, 93.75 GB/s:** each GPU sends half of its data straight to the other node over its own rail (50 GB/s), so the call runs at up to 100 GB/s; "
      "in busbw units that is 100 × 15/16. It is an upper bound: protocol overhead makes the real ceiling about 5% lower.")
    w("- **sendrecv, 50 GB/s:** each GPU pair that crosses the nodes uses one rail, and the slowest pair sets the busbw.")
    w(f"- **All MI355X settings tested** (busbw at the largest size): {defaults_line}.")
    w("")
    if fixmd:
        w("**all_reduce, small and medium messages: RCCL's default is wrong across nodes.** With RCCL 2.30.7's defaults, all_reduce on 2 nodes × 8 GPUs "
          f"returns wrong data at {', '.join(f'{s // 1024} KB' for s in fc.get('wrong_sizes_default', []))} (rccl-tests data check) and takes 1.4–2.2 ms "
          "from 8 B to 256 KB (about 30–50 µs is normal here). Large messages are correct and fast, so the bandwidth above is not affected. "
          + (f"`{FIX}` gives correct results at every size and is used for all_reduce latency in §2.3. " if FIX else
             "None of the settings tested gives correct results at every size. ")
          + "All settings, time per call in µs (✗ = wrong results at that size), data check on, 100 iterations:")
        w("")
        w(fixmd.rstrip())
        w("")
        if fc.get("algo_debug"):
            w(f"RCCL's own log (`NCCL_DEBUG_SUBSYS=TUNING`) shows it picks {fc['algo_debug']} for these sizes by default.")
            w("")
    w("> [!NOTE]")
    w(f"> Between two nodes, AMD Pollara (Ethernet) performs about the same as NVIDIA InfiniBand: all_gather, reduce_scatter and broadcast are "
      f"{min(ring):.2f}–{max(ring):.2f}x, both near the full network. MI355X is faster in all_reduce ({r22['all_reduce']:.2f}x) and alltoall "
      f"({r22['alltoall']:.2f}x), because NCCL is low there; B200 all_reduce could improve with SHARP (data reduced inside the InfiniBand switch), "
      f"which was not used here. sendrecv is {r22['sendrecv']:.2f}x.")
    w("")
    w("**Apple-to-apple: nearly.** Same test, 16 ranks, same message range, same network speed. Differences: RoCEv2 (AMD Pollara) vs InfiniBand (NVIDIA), "
      "RCCL vs NCCL. reduce, gather and scatter were not run on MI355X across 2 nodes.")
    w("")
    w("**Results far from paper** (also listed in the Quick summary of [`mi355x-rocm10.1-vs-b200.md`](mi355x-rocm10.1-vs-b200.md)):")
    w("")
    w(f"- **B200 all_reduce reaches only 58% ({r22['all_reduce']:.2f}x vs 1.00x on paper):** other NCCL collectives reach 90–96% on the same network, "
      "so **the likely cause is NCCL's all_reduce algorithm, not the network** (the algorithm NCCL picked was not logged). **SHARP should help:** "
      "the switches add the data up, so each GPU sends it once instead of twice. It could not be tested: the fabric has no SHARP Aggregation Manager (`sharp_am`).")
    w(f"- **B200 alltoall reaches only 52% ({r22['alltoall']:.2f}x vs 1.00x on paper):** NCCL's alltoall sends a separate piece directly to each of the "
      "15 other GPUs and reaches only half of the ceiling. **Likely too few channels per network peer.** Not tested on B200; "
      "`NCCL_NCHANNELS_PER_NET_PEER=4` would be the first setting to try.")
    w("")
else:
    r22 = {}
    w("_Pending: the 2-node runs on ROCm 10.1 (bandwidth with all settings, all_reduce data check) are running._")
    w("")
w("<!-- LATENCY-BEGIN (auto: rccl-tests/compare_lat.py) -->")
if not DONE2:
    w("### 2.3 Measured latency (small messages)")
    w("")
    w("_Pending: the 2-node latency runs on ROCm 10.1 are running._")
w("<!-- LATENCY-END -->")
w("")
w(appA.rstrip())
w("")
IC.write_text("\n".join(IC_L) + "\n")

# =====================================================================================================
# main file
# =====================================================================================================
old = OLD.read_text()
sec5_6 = chunk(old, "## 5. Training", "## Quick summary: results far")
apple = old[old.index("## Apple-to-apple?"):]
assert "7.14" not in sec5_6 and "7.14" not in apple
# MI355X BF16/FP16 GEMM numbers inside §5 come from §1/§3: use the ROCm 10.1 ones
bf16_fr, bf16_cu, fp16_cu = bf16 / FRYER_B2["bf16"], bf16 / CUBLAS_B2["bf16"], fp16 / CUBLAS_B2["fp16"]
for a, b in [("(MI355X RVS 1,672, B200 cuBLASLt 1,396), training at 1 GPU reaches ≈70% on both (1,166 / 1,672 and 989 / 1,396)",
              f"(MI355X RVS {c0(bf16)}, B200 cuBLASLt 1,396), training at 1 GPU reaches ≈{100 * 1166 / bf16:.0f}% and ≈71% "
              f"(1,166 / {c0(bf16)} and 989 / 1,396)"),
             ("consistent with the BF16 GEMM (1.15x vs gpu-fryer, 1.20x vs cuBLASLt, §1 and §3)",
              f"consistent with the BF16 GEMM ({bf16_fr:.2f}x vs gpu-fryer, {bf16_cu:.2f}x vs cuBLASLt, §1 and §3)"),
             ("(FP16 GEMM gives the same 1.19x, §3)", f"(FP16 GEMM gives {fp16_cu:.2f}x, §3)")]:
    assert a in sec5_6, a
    sec5_6 = sec5_6.replace(a, b)

hpl_block = "<!-- HPL-B200-START -->\nB200 HPL runs are pending; this section is filled in automatically when they finish.\n<!-- HPL-B200-END -->"
hpl_row = "pending"
if MAIN.exists():
    m = re.search(r"<!-- HPL-B200-START -->.*?<!-- HPL-B200-END -->", MAIN.read_text(), re.S)
    if m and "pending" not in m.group(0):
        hpl_block = m.group(0)
    m2 = re.search(r"^\| FP64 HPL \(Linpack\) \(§2\) \| ([^|]+) \|", MAIN.read_text(), re.M)
    if m2:
        hpl_row = m2.group(1).strip()

ratio_fp4_1, ratio_fp4_8, ratio_fp4_mp = fp4_1 / CUBLAS_B2["fp4"], fp4_8 / CUBLAS_B2["fp4"], fp4_mp / CUBLAS_B2["fp4"]
mi_fp4_pct = sorted([100 * fp4_1 / PAPER_MI["fp4"], 100 * fp4_mp / PAPER_MI["fp4"]])
fp4r = sorted([ratio_fp4_1, ratio_fp4_mp])
fp6_of_fp4 = fp6 / fp4_mp


def pr(lo, hi, d=2, suf="x"):
    a, b = f"{lo:.{d}f}", f"{hi:.{d}f}"
    return f"{a}{suf}" if a == b else f"{a}–{b}{suf}"


faster_b3 = [p.upper() for p, x in (("bf16", bf16 / FRYER_B3["bf16"]), ("fp8", fp8 / FRYER_B3["fp8"])) if x > 1]
slower_b3 = [p.upper() for p, x in (("bf16", bf16 / FRYER_B3["bf16"]), ("fp8", fp8 / FRYER_B3["fp8"])) if x <= 1]

# §4 quick-summary table (shared by both files), built from the tables above and the latency blocks
L = []
w = L.append
w("# AMD MI355X vs NVIDIA B200")
w("")
w("Written 2026-10-27. Every ratio is **MI355X / B200**: above 1 means MI355X is higher. For latency (TPOT, time per output token) a ratio above 1 means MI355X is **slower**.")
w("")
w("$\\color{red}{\\textsf{Red}}$ = a result far from what the specs on paper predict; each is explained in the **Quick summary** at the end of the file. (Red text renders on GitHub; other viewers show the `$\\color{red}...$` code.)")
w("")
w("| | MI355X (this directory) | B200 |")
w("|---|---|---|")
w("| Nodes | node6100, node6101: 8 × MI355X (288 GB) each | node5500–5502, node5600–5802: 8 × B200 (192 GB) each |")
w("| Network | 8 × 400G AMD Pollara RoCEv2 per node (400 GB/s) | 8 × 400G NDR InfiniBand per node (400 GB/s) |")
w("| Software | ROCm 10.1 (user-space install; hipBLASLt 1.4.1, RCCL 2.30.7, rocBLAS 5.7.0) for RVS (ROCm Validation Suite), rocHPL and RCCL; apptainer containers (Primus, ATOM, vLLM) | CUDA 13.x, NGC containers, vLLM |")
w("| Sources | `results/node6100/rocm10.1/` and `results/node6101/rocm10.1/` (RVS, RCCL, rocHPL); `logs/node6100/rocm10.1/rvs/fp4_multiproc_*`; `results/ubuntu/{primus,kimi-recipe-old-vs-new}.md` | `../b200-nodes/out-gpu-fryer/`, `../cuBLASLt/summary.md`, `../b200-nodes/out-nccl-{1node,2node}/`, `../b200-nodes/output-megatron/`, `../b200-kimi/results/kimi-k3-{base,improve}-b200.md` |")
w("")
w("## Summary")
w("")
w("| Area | MI355X / B200 | Apple-to-apple? |")
w("|---|---|---|")
w(f"| BF16 GEMM | {pr(min(bf16_fr, bf16_cu), max(bf16_fr, bf16_cu))} | ❌ |")
w(f"| FP8 GEMM | {fp8 / FRYER_B2['fp8']:.2f}x (vs gpu-fryer) or {fp8 / CUBLAS_B2['fp8']:.2f}x (vs cuBLASLt) | ❌ |")
w(f"| FP4 GEMM | {ratio_fp4_1:.2f}x (1 GPU), {ratio_fp4_mp:.2f}x (8 GPUs, one process per GPU) | ❌ |")
w(f"| FP64 GEMM | {fp64 / CUBLAS_B2['fp64']:.2f}x | ❌ |")
w(f"| FP64 HPL (Linpack) (§2) | {hpl_row} | ✅ same benchmark, each vendor's own build |")
w(f"| Collectives inside one node, bandwidth (§4) | {lo1:.2f}–{hi1:.2f}x (sendrecv {sr1:.2f}x) | ⚠️ close |")
if r22:
    named = {c: used[c] for c in used if used[c] != "default"}
    note = "; ".join(f"{c} with {SETTING.get(k, k)}" for c, k in named.items())
    w(f"| Collectives across 2 nodes, bandwidth (§4) | {min(ring):.2f}–{max(ring):.2f}x; all_reduce {r22['all_reduce']:.2f}x; "
      f"alltoall {r22['alltoall']:.2f}x; sendrecv {r22['sendrecv']:.2f}x" + (f" (MI355X {note})" if note else "") + " | ✅ nearly |")
else:
    w("| Collectives across 2 nodes, bandwidth (§4) | pending | ✅ nearly |")
    w("| Latency across 2 nodes, sendrecv + all_reduce, 8 B–1 MB (§4) | pending | ✅ same tests |")
w("| Training, 7B, 1–8 GPUs | 1.12–1.19x | ❌ different model/settings |")
w("| Kimi-K3 (ATOM, vLLM recipe 1, vLLM recipe 2), per GPU | 1.05–2.43x (best recipe: 1.21–1.63x up to 64 users, 2.03–2.43x from 128) | ⚠️ 8 vs 16 GPUs |")
w("| Kimi-K3, one user's streaming speed (TPOT) | B200 1.27–1.94x faster up to 64 users (1.37x vs the best recipe at 1 user); same or slower from 128 users | ⚠️ 8 vs 16 GPUs |")
w("")
w("---")
w("")
w("## 1. RVS vs gpu-fryer (BF16, FP8, FP32/TF32 GEMM stress test, all 8 GPUs at once)")
w("")
w("TFLOP/s per GPU. MI355X = RVS `gst` on ROCm 10.1, N = 8, mean of node6100 and node6101 (aggregate / 8). B200 = gpu-fryer mean of node5500–5502 (the 7 newer nodes are within 1%).")
w("")
w("| Precision | MI355X (RVS) | MI355X on paper | MI355X % of paper | B200 (gpu-fryer) | B200 on paper | B200 % of paper | MI355X / B200, measured | MI355X / B200, on paper |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
w(f"| BF16 | {c0(bf16)} | 2,516 | {pc(bf16, 2516)} | 1,450 | 2,250 | 64% | **{rt(bf16, 1450)}** | 1.12x |")
w(f"| FP8 | {c0(fp8)} | 5,033 | {pc(fp8, 5033)} | 4,018 | 4,500 | 89% | **{rt(fp8, 4018)}** | 1.12x |")
w(f"| FP32 (MI355X) / TF32 (B200) | {c0(fp32)} | 157.3 | {pc(fp32, 157.3)} | 751 | 1,100 | 68% | {rt(fp32, 751)} *(not comparable: B200 uses TF32 tensor cores, MI355X runs true FP32)* | — |")
w("")
w("On paper = vendor datasheet peak per GPU, dense (no sparsity), TFLOP/s.")
w("")
w("**Apple-to-apple: no.**")
w("- Different tools: RVS `gst` (hipBLASLt) vs gpu-fryer (cuBLASLt).")
w("- RVS runs 30 s and reports the **peak** per-GPU sample. gpu-fryer runs 300 s and reports the **converged mean**. This favours RVS by a few percent.")
w("- Matrix shapes are not the same: RVS uses 8192×8192×16384, gpu-fryer uses its own fixed shape.")
w("- \"FP32\" means a different operation on each GPU (see the table). MI355X has no TF32.")
w("")
w("### 1.1 MI355X vs B300 (RVS vs gpu-fryer)")
w("")
w(f"**Compared:** the MI355X numbers above with B300 (node5900-c1, gpu-fryer, same method as B200; `../b300-nodes/COMPARISON-b200-vs-b300.md` §3). "
  f"**MI355X is faster in {', '.join(faster_b3) or 'neither'}, B300 is faster in {', '.join(slower_b3) or 'neither'}.**")
w("")
w("| Precision | MI355X (RVS) | MI355X on paper | MI355X % of paper | B300 (gpu-fryer) | B300 on paper | B300 % of paper | MI355X / B300, measured | MI355X / B300, on paper |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
w(f"| BF16 | {c0(bf16)} | 2,516 | {pc(bf16, 2516)} | 1,521 | 2,250 | 68% | **{rt(bf16, 1521)}** | 1.12x |")
w(f"| FP8 | {c0(fp8)} | 5,033 | {pc(fp8, 5033)} | 4,228 | 4,500 | 94% | **{rt(fp8, 4228)}** | 1.12x |")
w(f"| FP32 (MI355X) / TF32 (B300) | {c0(fp32)} | 157.3 | {pc(fp32, 157.3)} | 797 | 1,100 | 72% | {rt(fp32, 797)} *(not comparable)* | — |")
w("")
w("- B300 is only 4–6% faster than B200 here, so the ratios are close to those against B200. The apple-to-apple notes above apply.")
w("")
w("## 2. rocHPL vs NVIDIA HPL (FP64 Linpack, 1–8 GPUs and 2 nodes)")
w("")
w(hpl_block)
w("")
w("---")
w("")
w("## 3. RVS vs cuBLASLt (FP4, FP6, FP8, BF16, FP16, FP32/TF32, FP64 GEMM throughput, all 8 GPUs at once)")
w("")
w("TFLOP/s per GPU. MI355X = RVS `gst` N = 8 as above. B200 = `../cuBLASLt` **sustained** (60 s at the tuned shape), node5802-c1.")
w("")
H = "| Precision | MI355X (RVS) | MI355X on paper | MI355X % of paper | {b} | {b0} on paper | {b0} % of paper | MI355X / {b0}, measured | MI355X / {b0}, on paper |"


R56 = RED("56\\\\%")


def red_pct(x, pk):
    return RED(pc(x, pk).replace("%", "\\\\%"))


def gemm_table(cu, paper, name, fn):
    t = [H.format(b=f"{name} (cuBLASLt)", b0=name), "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pp4 = paper["fp4"]
    b4 = cu["fp4"]
    b4p = f"{100 * b4 / pp4:.0f}%"
    b4pc = RED(b4p.replace("%", "\\\\%")) if name == "B300" else b4p
    paper4 = f"{PAPER_MI['fp4'] / pp4:.2f}x"
    for lbl, x in ((f"FP4, 1 GPU", fp4_1), ("FP4, 8 GPUs, one RVS process for all 8 (standard run)", fp4_8),
                   ("FP4, 8 GPUs, one RVS process per GPU", fp4_mp)):
        xs = f"≈{x:,.0f}" if "per GPU" in lbl else c0(x)
        rr = x / b4
        rs = REDB(f"{rr:.2f}x") if (rr < 0.8 * PAPER_MI['fp4'] / pp4 or name == "B200") else f"{rr:.2f}x"
        t.append(f"| {lbl} | {xs} | 10,066 | {red_pct(x, 10066)} | {c0(b4)}{fn if lbl.endswith('1 GPU') else ''} | {pp4:,} | {b4pc} | {rs} | {paper4} |")
    b8 = cu["fp8"]
    t.append(f"| FP8 | {c0(fp8)} | 5,033 | {pc(fp8, 5033)} | {c0(b8)} | 4,500 | {R56} | {REDB(rt(fp8, b8))} | 1.12x |")
    t.append(f"| BF16 | {c0(bf16)} | 2,516 | {pc(bf16, 2516)} | {c0(cu['bf16'])} | 2,250 | {pc(cu['bf16'], 2250)} | **{rt(bf16, cu['bf16'])}** | 1.12x |")
    t.append(f"| FP16 | {c0(fp16)} | 2,516 | {pc(fp16, 2516)} | {c0(cu['fp16'])} | 2,250 | {pc(cu['fp16'], 2250)} | **{rt(fp16, cu['fp16'])}** | 1.12x |")
    f64r = fp64 / cu["fp64"]
    f64p = PAPER_MI["fp64"] / paper["fp64"]
    t.append(f"| FP64 | {fp64:.0f} | 78.6 | {pc(fp64, 78.6)} | {cu['fp64']:g} | {paper['fp64']:g} | {pc(cu['fp64'], paper['fp64'])} | "
             f"**{f64r:.2f}x** | {f64p:.2f}x |" if f64r < 10 else
             f"| FP64 | {fp64:.0f} | 78.6 | {pc(fp64, 78.6)} | {cu['fp64']:.2f} | {paper['fp64']:g} | {pc(cu['fp64'], paper['fp64'])} | "
             f"**{f64r:.0f}x** | {f64p:.0f}x |")
    t.append(f"| FP32 (MI355X) / TF32 ({name}) | {c0(fp32)} | 157.3 | {pc(fp32, 157.3)} | {c0(cu['tf32'])} | 1,100 | {pc(cu['tf32'], 1100)} | {rt(fp32, cu['tf32'])} *(not comparable)* | — |")
    t.append(f"| FP6 | {c0(fp6)} | 10,066 | {red_pct(fp6, 10066)} | — | — | — | — *(not run on {name})* | — |")
    return t


L += gemm_table(CUBLAS_B2, PAPER_B2, "B200", "¹")
w("")
w("On paper = vendor datasheet peak per GPU, dense (no sparsity), TFLOP/s. MI355X FP6 runs at the FP4 rate on paper.")
w("")
fp4_src = (f"mean of node6100 ({r1_0:,.0f}) and node6101 ({r1_1:,.0f})" if r1_1 else f"node6100 ({r1_0:,.0f})")
w(f"FP4 rows: MI355X 1 GPU = RVS N = 1, {fp4_src}; 8 GPUs = per GPU. "
  f"**FP4 needs one process per GPU:** with one RVS process for all 8 GPUs (the standard RVS run) the rate falls to ≈{fp4_8:,.0f} per GPU; "
  "the FP4 kernel runs at full speed, but the GPUs sit idle much of the time because one process cannot launch work fast enough "
  f"(profile: `vs-amd-cloud/rvs-fp4-recheck.md`). With 8 separate one-GPU processes at once, each GPU gives {fp4_mp_lo:,.0f}–{fp4_mp_hi:,.0f} "
  f"({fp4_mp_sum:,.0f} total), the same as 1 GPU. Real workloads run one process per GPU, so the last FP4 row is the one to use.")
w("")
w("¹ B200 FP4 was measured only with all 8 GPUs at once (one process per GPU; 5,647–5,824 per GPU), so the 1-GPU row uses the same B200 value.")
w("")
w("**Apple-to-apple: no.**")
w("- cuBLASLt is **tuned**: it tries 28 shapes and up to 8 algorithms per shape and keeps the fastest. RVS runs one fixed shape with the library's default choice. This favours B200.")
w(f"- The two B200 tools disagree on FP8: gpu-fryer gives 4,018 and cuBLASLt 2,524 (56% of datasheet). The FP8 ratio therefore depends on which B200 number is used "
  f"({fp8 / FRYER_B2['fp8']:.2f}x vs {fp8 / CUBLAS_B2['fp8']:.2f}x).")
w("- FP4 formats differ: MI355X uses MXFP4 (32-element blocks), B200 uses NVFP4 (16-element blocks). The format does not change the FP4 work done, so the expected speed ratio between the two is about 1.0x for compute-bound GEMMs; NVFP4 only stores about 6% more bytes per element.")
w("")
w("**Results far from paper** (in the order of the table at the end of the file):")
w("")
w(f"- **Why MI355X FP4 is far below paper ({ratio_fp4_mp:.2f}x measured with one process per GPU and {ratio_fp4_1:.2f}x on 1 GPU, vs 1.12x on paper):** "
  f"MI355X reaches only {mi_fp4_pct[0]:.0f}–{mi_fp4_pct[1]:.0f}% of its FP4 peak (B200: 63%). Its FP8 reaches {pc(fp8, 5033)} and BF16 {pc(bf16, 2516)} in the same test. "
  "**The likely cause is the ROCm software: the FP4 GEMM kernels in ROCm's hipBLASLt library are less mature than its FP8/BF16 kernels, not the hardware.** "
  "**It is not power or clocks:** in the fp4 recheck the GPUs ran at 1.6–2.1 GHz and 260–800 W during FP4, well below MI355X's 1,400 W limit (`vs-amd-cloud/rvs-fp4-recheck.md`).")
w("  - The test favours B200: cuBLASLt picks the best of many shapes and algorithms; RVS runs one fixed shape.")
w(f"- **Why MI355X FP6 is only {pc(fp6, 10066)} of paper:** on paper FP6 runs at the FP4 rate, but RVS measures it at {100 * fp6_of_fp4:.0f}% of FP4 "
  f"({c0(fp6)} vs ≈{fp4_mp:,.0f}). **The likely cause is the same as for FP4, the ROCm software: hipBLASLt's FP6 GEMM kernels are even less mature than its FP4 ones, "
  "not the hardware.** Evidence: fp6 and bf6 give the same number (not profiled).")
fp8chk = re.search(r"- \*\*Why B200 FP8 is only 56% of paper with cuBLASLt.*", old).group(0)
w(fp8chk)
w("")
w("> [!NOTE]")
w(f"> In my test, MI355X FP4 GEMM is only {ratio_fp4_mp:.2f}x of B200, against 1.12x on paper. The likely reason is that ROCm's FP4 GEMM kernels are not fully optimized yet. "
  f"In the other precisions MI355X is faster than B200, as expected; FP64 is {fp64 / CUBLAS_B2['fp64']:.2f}x faster.")
w("")
w("### 3.1 MI355X vs B300 (RVS vs cuBLASLt)")
w("")
w("**Compared:** the MI355X numbers of the table above with B300 (Blackwell Ultra, SXM6 AC, air-cooled, 1,100 W limit), `../cuBLASLt` **sustained**, node5900-c1, "
  "same cuBLASLt method as for B200 (`../cuBLASLt/summary.md`, `../b300-nodes/COMPARISON-b200-vs-b300.md` §4). **MI355X is faster in FP8, BF16, FP16 and FP64; B300 is faster in FP4.**")
w("")
L += gemm_table(CUBLAS_B3, PAPER_B3, "B300", "²")
w("")
w("² As for B200, B300 FP4 was measured only with all 8 GPUs at once (one process per GPU), so the 1-GPU row uses the same value.")
w("")
w(f"- **The MI355X/B300 FP4 ratio is closer to paper than the MI355X/B200 one** ({fp4_mp / CUBLAS_B3['fp4']:.2f}x vs 0.75x on paper; B200: {ratio_fp4_mp:.2f}x vs 1.12x), "
  "because both sides are below paper: MI355X for the ROCm FP4 kernels (see above), and **B300 because of its power limit:** at 1,100 W it runs FP4 at 1,099 MHz, "
  "54% of its full clock, and reaches 51% of paper (`../b300-nodes/COMPARISON-b200-vs-b300.md`, \"Why FP4 is 1.20x measured but 1.5x on paper\").")
w("- **B300 FP8 is 56% of paper,** the same as B200 and for the same reason (power limit, lowest clock of all precisions: 1,114 MHz). B300 FP8 equals B200 FP8 (2,524), so the FP8, BF16 and FP16 ratios are about the same as against B200.")
w(f"- **FP64:** B300 has almost no FP64 hardware (1.25 TFLOP/s on paper, 0.03x of B200), so MI355X is {fp64 / CUBLAS_B3['fp64']:.0f}x faster.")
w("- The apple-to-apple notes above apply here too (tuned cuBLASLt vs fixed-shape RVS; NVFP4 vs MXFP4).")
w("")
w("## 4. Interconnect — RCCL vs NCCL, one node and two nodes")
w("")
w("<!-- QS-TABLE -->")
w("")
w("On paper: links inside one node 537.6 vs 900 GB/s per GPU (0.60x); network 400 vs 400 GB/s per node (1.00x); neither vendor publishes latency figures.")
w("")
w("**Details:** [`mi355x-rocm10.1-vs-b200-interconnect.md`](mi355x-rocm10.1-vs-b200-interconnect.md) — 1. intra-node, 2. inter-node; each with on paper, measured bandwidth, measured latency.")
w("")
L.append(sec5_6.rstrip())
w("")
w("---")
w("")
w("## Quick summary: results far from what the specs on paper predict")
w("")
w("On paper, MI355X / B200 is 1.12x for FP4/FP8/BF16 compute, 2.12x for FP64, 0.60x for GPU-to-GPU links in one node (537.6 vs 900 GB/s), 1.00x for the network between nodes and 1.00x for memory bandwidth.")
w("")
w("**Results as expected from the specs on paper**, one per section (measured MI355X / B200 vs on paper; % of each GPU's own peak):")
w("")
w(f"- **1. RVS vs gpu-fryer:** BF16 {bf16_fr:.2f}x vs 1.12x on paper; both GPUs ≈65% of peak.")
w(f"- **2. RVS vs cuBLASLt:** BF16 {bf16_cu:.2f}x and FP16 {fp16_cu:.2f}x vs 1.12x; FP64 {fp64 / CUBLAS_B2['fp64']:.2f}x vs 2.12x (both ≈97–98% of peak).")
most = [v for k, v in r12.items() if k not in ("sendrecv", "all_reduce", "reduce")]
w(f"- **3. Interconnect, one node (bandwidth):** Most collectives {min(most):.2f}–{max(most):.2f}x vs 0.60x link bandwidth on paper; both GPUs "
  f"{min(mi_pk + b2_pk):.0f}–{max(mi_pk + b2_pk):.0f}% of their link peak. all_reduce {r12['all_reduce']:.2f}x and reduce {r12['reduce']:.2f}x are also expected: "
  "B200 is helped by NVLink SHARP (in-switch reduction).")
if r22:
    w(f"- **3. Interconnect, two nodes (bandwidth):** all_gather, reduce_scatter, broadcast {min(ring):.2f}–{max(ring):.2f}x vs 1.00x network; "
      f"sendrecv {r22['sendrecv']:.2f}x" + (f" with {SETTING[used['sendrecv']]} on MI355X." if used.get('sendrecv', 'default') != 'default' else "."))
else:
    w("- **3. Interconnect, two nodes (bandwidth):** pending.")
w("<!-- QS-LAT-BULLET -->")
w("- **4. Training:** 7B models 1.12–1.19x vs 1.12x BF16; both use 43–46% of BF16 peak.")
w("- **5. Kimi-K3:** Model fits in one MI355X node but needs two B200 nodes, as 288 vs 192 GB per GPU predicts; per-GPU throughput 1.05–2.43x across the three MI355X recipes.")
w("")
w("**Results far from the specs on paper:**")
w("")
w("| Result | Which GPU is off | Measured | On paper | Likely reason (one sentence) |")
w("|---|---|---|---|---|")
w(f"| **FP4 GEMM** (§3) | MI355X (low) | MI355X {mi_fp4_pct[0]:.0f}–{mi_fp4_pct[1]:.0f}% of its peak (1 GPU and 8 GPUs), B200 63%; ratio **{pr(*fp4r)}** | 1.12x | "
  f"ROCm software: the FP4 kernels in ROCm's hipBLASLt are less mature than its FP8/BF16 kernels ({pc(fp8, 5033)}/{pc(bf16, 2516)} of peak), and cuBLASLt was tuned while RVS was not. |")
w(f"| **FP6 GEMM** (§3) | MI355X (low) | MI355X {pc(fp6, 10066)} of its peak | — (not on B200) | Same ROCm software cause as FP4: the FP6 kernels are even less developed. |")
w(f"| **FP8 GEMM on B200 with cuBLASLt** (§3) | B200 (low) | B200 56% of peak (gpu-fryer: 89%); ratio **{fp8 / CUBLAS_B2['fp8']:.2f}x** | 1.12x | Power limit: all precisions run at the ≈1,000 W cap, "
  "and FP8 uses the most power per cycle, so it gets the lowest clock (1,131 MHz); the kernel itself is ≈97% efficient per clock. gpu-fryer's 89% is inflated by its constant test data "
  "(every FP8 value clamped to 448), as the 1-GPU check in §3 confirms. |")
if r22:
    w(f"| **all_reduce across 2 nodes** (§4) | B200 (low) | MI355X {pc(BW2['all_reduce']['default'], 400)}, B200 58% of 400 GB/s; ratio **{r22['all_reduce']:.2f}x** | 1.00x | "
      "Likely NCCL's all_reduce algorithm, not the network (other NCCL collectives reach 90–96%). SHARP should help but is not enabled on the fabric (no `sharp_am`). |")
    a_x = BW2["alltoall"]
    a_def = a_x.get("default")
    w(f"| **alltoall across 2 nodes** (§4) | B200 (low) | MI355X {pc(bw2_best('alltoall')[0], 93.75)}"
      + (f" (with {SETTING[used['alltoall']]}; default {pc(a_def, 93.75)})" if used['alltoall'] != 'default' and a_def else "")
      + f", B200 52% of the 93.75 GB/s alltoall ceiling; ratio **{r22['alltoall']:.2f}x** | 1.00x | NCCL's alltoall (many point-to-point transfers, no ring) is known to be slow; "
      "likely too few channels per network peer (not tested on B200). |")
    if fc and not fc.get("default_ok", True):
        w(f"| **all_reduce across 2 nodes, 8 B–256 KB** (§4) | MI355X (wrong and slow) | RCCL 2.30.7's default: wrong results at "
          f"{', '.join(f'{s // 1024} KB' for s in fc.get('wrong_sizes_default', []))}, 1.4–2.2 ms per call up to 256 KB (normal: about 30–50 µs) | correct, tens of µs | "
          + (f"RCCL software: its default path for this case is broken; `{FIX}` gives correct results and normal times. |" if FIX else
             "RCCL software: its default path for this case is broken; no setting tested fixes it. |"))
else:
    w("| **all_reduce across 2 nodes** (§4) | pending | pending | 1.00x | pending |")
    w("| **alltoall across 2 nodes** (§4) | pending | pending | 1.00x | pending |")
w("| **Kimi-K3 single-user speed** (§6) | MI355X (slow) | B200 1.27–1.94x faster per user (TPOT) up to 64 users, depending on the MI355X recipe; same or slower from 128 users | 1.00x (same 8 TB/s memory bandwidth) | Not measured; most likely software (per-step kernel and launch overhead in vLLM on ROCm), since memory bandwidth on paper is the same. |")
w("")
L.append(apple.rstrip())
MAIN.write_text("\n".join(L) + "\n")

# ---- latency blocks + Summary rows (compare scripts write into MAIN and IC) ---------------------------
H_ = A / "rocm10.1"
subprocess.run([PY, str(H_ / "compare_lat_1node_rocm10.1.py")], check=False)
if DONE2:
    subprocess.run([PY, str(H_ / "compare_lat_rocm10.1.py"), str(B200_LAT2)], check=False)


# ---- quick-summary table for §4 / interconnect: built from the tables now in IC ------------------------
def lat_rows(text, begin, end):
    """[(coll, bytes, mi, b2, speed)] from the first table in a marker block."""
    m = re.search(re.escape(begin) + r"(.*?)" + re.escape(end), text, re.S)
    out = []
    if not m:
        return out
    for line in m.group(1).split("\n"):
        p = [x.strip() for x in line.strip("|").split("|")]
        if len(p) >= 5 and p[0] in ("sendrecv", "all_reduce", "all_gather", "reduce_scatter", "broadcast", "alltoall"):
            sp = re.search(r"([\d.]+)x", p[4])
            if sp:
                out.append((p[0], p[1], sp.group(1) and float(sp.group(1))))
    return out


ic = IC.read_text()
intra = lat_rows(ic, "<!-- INTRA-LAT-BEGIN", "<!-- INTRA-LAT-END -->")
bycoll = {}
for c, s, r in intra:
    bycoll.setdefault(c, []).append((s, r))
mi_win = [c for c, v in bycoll.items() if all(r >= 1 for _, r in v)]
mi_all = [r for c in mi_win for _, r in bycoll[c]] or [float("nan")]
b2_parts = []
for c, v in bycoll.items():
    lose = [(s, r) for s, r in v if r < 1]
    if lose:
        inv = [1 / r for _, r in lose]
        b2_parts.append(f"{c} at {', '.join(s for s, _ in lose)} ({pr(min(inv), max(inv), 1, '×')})")
intra_txt = (f"• MI355X {pr(min(mi_all), max(mi_all), 1, '×')} faster in {', '.join(mi_win)}" if mi_win else "") + \
            (f"<br>• B200 faster only in {'; '.join(b2_parts)}" if b2_parts else "")
winner_intra = "**MI355X** (Infinity Fabric)" if len(mi_win) >= 3 else "**Mixed**"

if r22:
    eq_txt = f"• Within {max(abs(1 - x) for x in ring) * 100:.0f}%<br>• sendrecv {r22['sendrecv']:.2f}x" + \
             (f" with {SETTING[used['sendrecv']]}" if used.get("sendrecv", "default") != "default" else "") + \
             f"<br>• MI355X {r22['all_reduce']:.1f}× faster in all_reduce and {r22['alltoall']:.1f}× in alltoall, where NCCL is low"
    inter_bw = f"| **Inter-node bandwidth** (2 nodes, large messages) | **Equal** (Pollara vs InfiniBand) | {eq_txt} |"
else:
    inter_bw = "| **Inter-node bandwidth** (2 nodes, large messages) | pending | pending |"
if DONE2 and "Main result" in ic:
    blk = re.search(r"<!-- LATENCY-BEGIN.*?<!-- LATENCY-END -->", ic, re.S).group(0)
    raw = [float(x) for x in re.findall(r"\| ib_\w+_lat \| \S+ \| [\d.]+ \| [\d.]+ \| ([\d.]+)x \|", blk)]
    one = re.search(r"\*\*2 nodes × 1 GPU\*\*.*?\n\n(> .*?)\n", blk, re.S)
    p1 = [float(x) for x in re.findall(r"\| (?:sendrecv|all_reduce) \| \S+ \| [\d.–]+ \| [\d.–]+ \| (?:\$\\color\{red\}\{\\textbf\{)?([\d.]+)x", blk.split("**2 nodes × 8 GPUs, network part only**")[0])]
    netblk = blk.split("network part only")[-1].split("**Only in")[0]
    whole8 = re.findall(r"\| (sendrecv|all_reduce) \| (\S+) \| [-\d.–]+ \| [-\d.–]+ \| [^|\n]+ \| (?:\$\\color\{red\}\{\\textbf\{)?([\d.]+)x", netblk)
    ar8 = [float(r) for c, s, r in whole8 if c == "all_reduce"]
    sr8 = [float(r) for c, s, r in whole8 if c == "sendrecv"]
    parts = []
    if raw:
        parts.append(f"• Network alone: InfiniBand {1 / max(raw):.1f}–{1 / min(raw):.0f}× faster")
    if p1:
        parts.append(f"• 1 GPU per node: " + (f"B200 {1 / max(p1):.1f}–{1 / min(p1):.1f}× faster" if max(p1) < 1 else f"MI355X / B200 {min(p1):.2f}–{max(p1):.2f}x"))
    if ar8 or sr8:
        parts.append("• 8 GPUs per node: " + "; ".join(
            f"{c} MI355X / B200 {min(v):.2f}–{max(v):.2f}x" for c, v in (("all_reduce", ar8), ("sendrecv", sr8)) if v))
    inter_lat = f"| **Inter-node latency** (2 nodes, small messages) | **B200** (InfiniBand) | {'<br>'.join(parts)} |"
    qs_lat = ("- **3. Interconnect, latency:** no latency on the spec sheets. Inside one node MI355X is faster for small messages "
              f"({pr(min(mi_all), max(mi_all))} in {', '.join(mi_win)}). Across 2 nodes point-to-point (sendrecv) is slower on MI355X "
              f"({pr(min(sr8), max(sr8)) if sr8 else '–'}; RoCE latency 5.2 vs 1.3 µs on InfiniBand).")
else:
    inter_lat = "| **Inter-node latency** (2 nodes, small messages) | pending | pending |"
    qs_lat = ("- **3. Interconnect, latency:** no latency on the spec sheets. Inside one node MI355X is faster for small messages "
              f"({pr(min(mi_all), max(mi_all))} in {', '.join(mi_win)}). Across 2 nodes: pending.")
qs = "\n".join(["| | Faster | By how much |", "|---|---|---|",
                f"| **Intra-node bandwidth** (8 GPUs, large messages) | **B200** (NVLink 5 + NVSwitch) | • B200 {1 / hi1:.1f}–{1 / lo1:.1f}× faster on collectives<br>• B200 {1 / sr1:.0f}× faster for one GPU pair (sendrecv) |",
                f"| **Intra-node latency** (8 GPUs, 8 B–1 MB) | {winner_intra} | {intra_txt} |",
                inter_bw, inter_lat])
qs_ic = qs.replace("(8 GPUs, large messages) |", "(8 GPUs, large messages), §1.2 |", 1) \
          .replace("(8 GPUs, 8 B–1 MB) |", "(8 GPUs, 8 B–1 MB), §1.3 |", 1) \
          .replace("(2 nodes, large messages) |", "(2 nodes, large messages), §2.2 |", 1) \
          .replace("(2 nodes, small messages) |", "(2 nodes, small messages), §2.3 |", 1)
IC.write_text(ic.replace("<!-- QS-TABLE -->", qs_ic).replace("<!-- INTRA-LAT-NOTES -->\n\n", ""))
mt = MAIN.read_text().replace("<!-- QS-TABLE -->", qs).replace("<!-- QS-LAT-BULLET -->", qs_lat)
MAIN.write_text(mt)
print(f"wrote {MAIN} and {IC}; 2-node done: {bool(DONE2)}; fix: {FIX or '-'}")
