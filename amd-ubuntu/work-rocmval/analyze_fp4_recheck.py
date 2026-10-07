#!/usr/bin/env python3
"""Report for fp4_recheck_and_profile.sh -> results/vs-amd-cloud/rvs-fp4-recheck.md

    analyze_fp4_recheck.py <out_dir> -o <md>

Stage 1: RVS (ROCm Validation Suite) gst fp4 per-GPU TFLOPS by stack/arm over repeats, with mean
sclk / power (rocm-smi) and rvs thread CPU (pidstat). Stage 2 (if run): rocprofv3 kernel traces
per GPU (main kernel, median duration, busy fraction = kernel time / wall span) and perf top.
"""
import argparse, collections, csv, re, statistics as st
from pathlib import Path

FLOP = 2 * 8192 * 8192 * 16384          # one gst fp4 GEMM (matrix_size_a/b/c of the sweep conf)
ARMS = [("n1", "1 GPU"), ("n2", "2 GPUs, one process"), ("n4", "4 GPUs, one process"),
        ("n8", "8 GPUs, one process (standard)"), ("m8", "8 GPUs, 8 processes")]
STACKS = [("7.14", "ROCm 7.14"), ("host", "ROCm 7.2.4")]


def f(x, nd=0):
    return "—" if x is None else f"{x:,.{nd}f}"


def smi(p: Path):
    """mean sclk (MHz) and power (W) over all GPUs and samples with sclk > 500 (under load)."""
    if not p.exists():
        return None, None
    t = p.read_text(errors="replace")
    clk = [float(x) for x in re.findall(r"sclk clock level:\s*\S+:\s*\((\d+)Mhz\)", t)]
    pw = [float(x) for x in re.findall(r"Power \(W\):\s*([\d.]+)", t)]
    clk = [c for c in clk if c > 500]
    pw = [w for w in pw if w > 200]
    return (st.mean(clk) if clk else None), (st.mean(pw) if pw else None)


def pidstat(p: Path):
    """(sum of rvs thread %CPU averaged over samples, busiest thread mean %CPU)"""
    if not p.exists():
        return None, None
    per_t, tot = collections.defaultdict(list), collections.defaultdict(float)
    for l in p.read_text(errors="replace").splitlines():
        s = l.split()
        # Time UID TGID TID %usr %system %guest %wait %CPU CPU Command ; thread rows have TGID '-'
        if len(s) >= 11 and s[2] == "-" and s[3].isdigit():
            try:
                v = float(s[8])
            except ValueError:
                continue
            per_t[s[3]].append(v); tot[s[0]] += v
    if not per_t:
        return None, None
    return st.mean(tot.values()), max(st.mean(v) for v in per_t.values())


def traces(d: Path):
    rows = []
    for c in d.rglob("*kernel_trace.csv"):
        with open(c) as fh:
            rows += list(csv.DictReader(fh))
    by = collections.defaultdict(list)
    for r in rows:
        try:
            by[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
        except (KeyError, ValueError):
            continue
    out = []
    for a, ks in sorted(by.items()):
        ks.sort()
        tot = collections.Counter()
        for s, e, n in ks:
            tot[n] += e - s
        main = tot.most_common(1)[0][0]
        md = [e - s for s, e, n in ks if n == main]
        # steady state: drop the first 20% of the run (warm-up, rotating buffers)
        t0, t1 = ks[0][0], ks[-1][1]
        cut = t0 + 0.2 * (t1 - t0)
        ss = [(s, e) for s, e, n in ks if s >= cut]
        busy = sum(e - s for s, e in ss) / max(1, (ss[-1][1] - ss[0][0])) if ss else None
        gaps = [ss[i + 1][0] - ss[i][1] for i in range(len(ss) - 1)]
        dmed = st.median(md) / 1e6 if md else None
        out.append({"agent": a, "kernel": main[:90], "n": len(md), "dur_ms": dmed,
                    "tflops_kernel": FLOP / (dmed / 1e3) / 1e12 if dmed else None,
                    "busy": busy, "gap_us": st.median(gaps) / 1e3 if gaps else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("-o", required=True)
    a = ap.parse_args()
    O = Path(a.out)
    L = ["# RVS (ROCm Validation Suite) fp4 recheck: ROCm 7.14 vs 7.2.4, 1 vs 8 GPUs", "",
         f"Run: `{O}` (script `work-rocmval/fp4_recheck_and_profile.sh`). gst fp4, the sweep's own "
         "configs (8192×8192×16384, 30 s), peak per-GPU sample like the sweep.", ""]
    state = (O / "STATE.txt").read_text() if (O / "STATE.txt").exists() else ""
    m = re.search(r"VERDICT (.*)", state)
    L += [f"**Verdict: {m.group(1) if m else 'not reached'}**", ""]

    # stage 1
    res = collections.defaultdict(list)
    for l in (O / "stage1.txt").read_text().splitlines()[1:]:
        p = l.split()
        if len(p) == 8 and p[3] != "0":
            res[(p[0], p[1])].append([float(x) for x in p[3:]])
    L += ["## 1. Recheck: per-GPU fp4 TFLOPS (mean over repeats)", "",
          "| arm | ROCm 7.14 per GPU | 7.14 vs its 1 GPU | 7.14 slowest–fastest GPU | ROCm 7.2.4 per GPU | 7.2.4 vs its 1 GPU | 7.2.4 slowest–fastest GPU | 7.14 / 7.2.4 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    mean = {}
    for k, v in res.items():
        mean[k] = (st.mean(r[2] for r in v), min(r[3] for r in v), max(r[4] for r in v), len(v))
    for arm, lab in ARMS:
        c = []
        for s, _ in STACKS:
            x, b = mean.get((s, arm)), mean.get((s, "n1"))
            c += [f(x[0]) if x else "—", f"{x[0]/b[0]:.2f}x" if x and b else "—",
                  f"{f(x[1])}–{f(x[2])}" if x else "—"]
        x, y = mean.get(("7.14", arm)), mean.get(("host", arm))
        L.append(f"| {lab} | " + " | ".join(c) + f" | {x[0]/y[0]:.2f}x |" if x and y else f"| {lab} | " + " | ".join(c) + " | — |")
    reps = max((v[3] for v in mean.values()), default=0)
    L += ["", f"{reps} repeats per cell. Per-run numbers: `stage1.txt`.", ""]

    L += ["### Clocks, power and host CPU during the runs (first repeat)", "",
          "| arm | stack | mean sclk MHz | mean power W/GPU | rvs CPU % (all threads) | busiest rvs thread % |",
          "|---|---|---:|---:|---:|---:|"]
    for arm, lab in ARMS:
        for s, sl in STACKS:
            d = O / "stage1" / s / arm / "r1"
            if not d.exists():
                continue
            c, w = smi(d / "smi.txt")
            tc, bt = pidstat(d / "pidstat.txt")
            L.append(f"| {lab} | {sl} | {f(c)} | {f(w)} | {f(tc)} | {f(bt)} |")
    L.append("")

    # stage 2
    s2 = O / "stage2"
    if s2.exists():
        L += ["## 2. Profile: GEMM kernels per GPU (rocprofv3 kernel trace, 10 s runs)", "",
              "Kernel TFLOPS = one GEMM's FLOPs / its median duration (what the GPU does while the kernel runs). "
              "Busy = kernel time / wall time after the first 20% of the run; gap = median idle time between kernels. "
              "Slower kernels point at the device (clocks, a different kernel); same kernels with low busy / large gaps point at the host side.", ""]
        summ = {}
        for name, lab in (("n1_714", "7.14, 1 GPU"), ("n8_714", "7.14, 8 GPUs one process"),
                          ("m8_714", "7.14, 8 processes"), ("n8_724", "7.2.4, 8 GPUs one process")):
            t = traces(s2 / name)
            if not t:
                L += [f"### {lab}", "", "*No kernel trace.*", ""]; continue
            summ[name] = t
            L += [f"### {lab}", "", "| agent | main kernel | count | median ms | kernel TFLOPS | busy | median gap µs |",
                  "|---|---|---:|---:|---:|---:|---:|"]
            for r in t:
                L.append(f"| {r['agent']} | `{r['kernel']}` | {r['n']} | {f(r['dur_ms'],3)} | {f(r['tflops_kernel'])} | "
                         f"{f(100*r['busy'] if r['busy'] is not None else None)}% | {f(r['gap_us'],1)} |")
            L.append("")
        # automatic reading
        if "n1_714" in summ and "n8_714" in summ:
            ref = summ["n1_714"][0]
            n8 = summ["n8_714"]
            kr = [r["dur_ms"] / ref["dur_ms"] for r in n8 if r["dur_ms"] and ref["dur_ms"]]
            bz = [r["busy"] for r in n8 if r["busy"] is not None]
            same_k = all(r["kernel"] == ref["kernel"] for r in n8)
            L += ["### Reading", "",
                  f"- 7.14, 8 GPUs one process vs 1 GPU: kernel duration {min(kr):.2f}–{max(kr):.2f}× the 1-GPU kernel; "
                  f"busy {100*min(bz):.0f}–{100*max(bz):.0f}% (1 GPU: {100*ref['busy']:.0f}%); "
                  f"{'same' if same_k else 'DIFFERENT'} GEMM kernel as on 1 GPU."]
            if max(kr) > 1.15:
                L.append("- The kernels themselves run slower with 8 GPUs in one process → device-side cause "
                         "(clocks/power or kernel selection); see the clock table above.")
            if bz and min(bz) < 0.85 * ref["busy"]:
                L.append("- The GPUs sit idle between kernels → host-side cause: the single rvs process does not "
                         "launch work fast enough for all 8 GPUs (launch/synchronisation contention).")
            L.append("")
        pt = s2 / "perf_n8_714" / "perf_top.txt"
        if pt.exists():
            top = [l for l in pt.read_text(errors="replace").splitlines() if l.strip() and not l.startswith("#")][:25]
            L += ["### Host hot spots, 7.14, 8 GPUs one process (perf, top 25)", "", "```"] + top + ["```", ""]
    else:
        L += ["## 2. Profile", "", "*Not run (verdict not confirmed or stage 2 not reached).*", ""]
    Path(a.o).write_text("\n".join(L) + "\n")
    print("wrote", a.o)


if __name__ == "__main__":
    main()
