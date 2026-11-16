#!/usr/bin/env python3
"""Nov 12 rerun: summarize candidates (base / mega / ep) and rocprofv3 profiles into part B of
hyperloom/rerun-report.md.

  analyze_candidates.py <runs file>     (lines "<arm> <rep> <out dir>" and "prof <users> <out dir>")
"""
import csv
import glob
import json
import os
import statistics as stat
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
USERS = [1, 8, 64, 128, 256]
GPUS = 8
NAMES = {"base": "Baseline", "mega": "a8w4 (`--moe-backend mega`)", "ep": "Expert parallel"}


def load_runs(path):
    arms, profs = {}, {}
    for line in open(path):
        p = line.split()
        if len(p) != 3 or p[2] == "none":
            continue
        if p[0] == "prof":
            profs[int(p[1])] = p[2]
        else:
            arms.setdefault(p[0], []).append(p[2])
    return arms, profs


def bench(dirs):
    """{users: {"tps": [...], "tpot": [...], "ttft": [...]}} over repeats."""
    out = {}
    for d in dirs:
        for c in USERS:
            f = os.path.join(d, "k0", f"c{c}.json")
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if not r.get("completed"):
                continue
            o = out.setdefault(c, {"tps": [], "tpot": [], "ttft": []})
            o["tps"].append(r["output_throughput"] / GPUS)
            o["tpot"].append(r["median_tpot_ms"])
            o["ttft"].append(r["median_ttft_ms"])
    return out


def gsm8k(dirs):
    for d in dirs:
        fs = sorted(glob.glob(os.path.join(d, "k0", "gsm8k", "**", "results_*.json"), recursive=True))
        if fs:
            r = json.load(open(fs[-1]))["results"]["gsm8k"]
            return r["exact_match,strict-match"]
    return None


def ms(v):
    if not v:
        return "–"
    m = stat.mean(v)
    return f"{m:,.1f} ± {stat.stdev(v):,.1f}" if len(v) > 1 else f"{m:,.1f}"


def group(name):
    n = name.lower()
    if any(k in n for k in ("moe", "a16w4", "a8w4", "fmoe", "expert")):
        return "MoE expert GEMMs"
    if any(k in n for k in ("cross_device_reduce", "allreduce", "all_reduce", "reduce_scatter",
                            "all_gather", "allgather", "rccl", "nccl", "all2all", "mori")):
        return "Communication (all-reduce etc.)"
    if any(k in n for k in ("attn", "mla", "flash", "paged", "decode_attention")):
        return "Attention"
    if any(k in n for k in ("gemm", "cijk", "hipblaslt", "matmul")):
        return "Other GEMMs"
    return "Everything else"


def profile(d):
    """Sum rocprofv3 kernel stats over all ranks: (total ns, {name: ns}, n files)."""
    per = {}
    files = glob.glob(os.path.join(d, "k0", "rocprof", "**", "*kernel_stats.csv"), recursive=True)
    for f in files:
        for row in csv.DictReader(open(f)):
            try:
                per[row["Name"]] = per.get(row["Name"], 0) + float(row["TotalDurationNs"])
            except (KeyError, ValueError):
                continue
    return sum(per.values()), per, len(files)


def main():
    arms, profs = load_runs(sys.argv[1])
    L = ["### Candidates (our script, node6101, ATOM image nightly_202610011450, no MTP)", "",
         "Output tok/s per GPU, mean ± standard deviation over the repeats. 1024 tokens in, 1024 out, "
         "10 × users prompts, max 64 sequences (as in the step 3 baseline). "
         "TPOT = time per output token; TTFT = time to first token (medians).", ""]
    res = {a: bench(arms.get(a, [])) for a in ("base", "mega", "ep")}
    acc = {a: gsm8k(arms.get(a, [])) for a in ("base", "mega", "ep")}
    L.append("| Arm | Repeats | GSM8K (strict) | " + " | ".join(f"{c} users" for c in USERS) + " |")
    L.append("|---|---:|---:|" + "---:|" * len(USERS))
    for a in ("base", "mega", "ep"):
        nrep = len(arms.get(a, []))
        cells = [ms(res[a].get(c, {}).get("tps")) for c in USERS]
        g = f"{acc[a]:.3f}" if acc[a] is not None else "–"
        L.append(f"| {NAMES[a]} | {nrep} | {g} | " + " | ".join(cells) + " |")
    L += ["", "Change against the baseline (mean tok/s per GPU; TPOT and TTFT at 64 users):", "",
          "| Arm | " + " | ".join(f"{c} users" for c in USERS) + " | TPOT 64 (ms) | TTFT 64 (ms) |",
          "|---|" + "---:|" * (len(USERS) + 2)]
    for a in ("base", "mega", "ep"):
        cells = []
        for c in USERS:
            b, x = res["base"].get(c, {}).get("tps"), res[a].get(c, {}).get("tps")
            cells.append(f"{(stat.mean(x) / stat.mean(b) - 1) * 100:+.1f}%" if b and x else "–")
        t = res[a].get(64, {})
        cells.append(f"{stat.mean(t['tpot']):.2f}" if t.get("tpot") else "–")
        cells.append(f"{stat.mean(t['ttft']):.0f}" if t.get("ttft") else "–")
        L.append(f"| {NAMES[a]} | " + " | ".join(cells) + " |")
    if acc["base"] is not None:
        L += ["", f"Accuracy rule: a candidate counts only if its GSM8K is within 0.01 of the baseline "
              f"({acc['base']:.3f}); Hyperloom's own baseline gate measured 0.958."]
    L += ["", "Run folders:"] + [f"- {a}: " + ", ".join(f"`{os.path.relpath(d, '/orcd/data/orcd/022/benchmarks/amd-ubuntu')}`" for d in ds)
                                 for a, ds in arms.items()]

    L += ["", "### Profiles of the baseline (rocprofv3 kernel trace, all 8 GPUs, 5 s window during the benchmark)", ""]
    groups = ["MoE expert GEMMs", "Communication (all-reduce etc.)", "Other GEMMs", "Attention", "Everything else"]
    prof = {c: profile(d) for c, d in sorted(profs.items())}
    if not any(p[2] for p in prof.values()):
        L.append("No rocprofv3 kernel stats were written (see `logs/hyperloom/candidates-node6101.state`).")
    else:
        L.append("| Kernel group | " + " | ".join(f"{c} users" for c in prof) + " |")
        L.append("|---|" + "---:|" * len(prof))
        for g in groups:
            row = []
            for c, (tot, per, _) in prof.items():
                s = sum(v for k, v in per.items() if group(k) == g)
                row.append(f"{s / tot * 100:.1f}%" if tot else "–")
            L.append(f"| {g} | " + " | ".join(row) + " |")
        for c, (tot, per, nf) in prof.items():
            if not tot:
                continue
            L += ["", f"Top 10 kernels at {c} users ({nf} stats files, {tot / 1e9:.2f} s of GPU time summed over GPUs):", "",
                  "| Kernel | Group | Share |", "|---|---|---:|"]
            for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:10]:
                L.append(f"| `{k[:90]}` | {group(k)} | {v / tot * 100:.1f}% |")
    tmp = os.path.join("/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/hyperloom", "candidates-section.md")
    open(tmp, "w").write("\n".join(L) + "\n")
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "rerun_report.py"), "set", "B", tmp], check=True)
    print("report part B written")


if __name__ == "__main__":
    main()
