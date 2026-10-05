#!/usr/bin/env python3
"""B200 vs B300 single-node comparison across all benchmarks.

B200 data:  ../b200-nodes/out-gpu-fryer, out-nccl-1node, out-ibwrite, output-megatron
            (newest file per node), plus B200 baseline runs of the "max" Megatron
            config that land in ./output/ (any node that is not a B300 node).
B300 data:  ./out-gpu-fryer, ./out-nccl-1node, ./out-ibwrite, ./output-megatron, ./output

Missing data shows as "—", so this can run at any time (it is re-run after each
benchmark finishes). Writes ./COMPARISON-b200-vs-b300.md.

Usage:  ./compare-b200-b300.py
"""
import glob
import importlib.util
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
B200_DIR = os.path.join(os.path.dirname(HERE), "b200-nodes")
# env overrides are only for testing the report against other data
B300_NODES = set(os.environ.get("B300_NODES", "node5900-c1").split(","))
DATA = os.environ.get("B300_DATA_DIR", HERE)
OUT_MD = os.environ.get("COMPARE_OUT", os.path.join(HERE, "COMPARISON-b200-vs-b300.md"))


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fryer = load("fryer", "analyze-gpu-fryer.py")
nccl = load("nccl", "analyze-nccl-1node.py")
meg = load("meg", "analyze-megatron.py")


# ---------------------------------------------------------------- helpers
def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def f(x, nd=1):
    return "—" if x is None else "{:,.{}f}".format(x, nd)


def ratio(b300, b200):
    if b300 is None or not b200:
        return "—"
    return "**{:.2f}x**".format(b300 / b200)


def newest_per_node(files, parse):
    out = {}
    for p in sorted(files, key=os.path.getmtime):
        try:
            r = parse(p)
        except Exception as e:  # a half-written file must not kill the report
            print("skip {}: {}".format(p, e))
            continue
        if r:
            out[r["node"]] = r
    return out


def table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    L = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return L


# ---------------------------------------------------------------- gpu-fryer
def section_fryer():
    L = ["## 2. gpu-fryer — per-GPU matmul throughput (TFLOP/s)", ""]
    b2 = newest_per_node(glob.glob(os.path.join(B200_DIR, "out-gpu-fryer", "*.out")),
                         fryer.parse_file)
    b3 = newest_per_node(glob.glob(os.path.join(DATA, "out-gpu-fryer", "*.out")),
                         fryer.parse_file)
    b2 = {n: r for n, r in b2.items() if n not in B300_NODES}
    b3 = {n: r for n, r in b3.items() if n in B300_NODES}
    if not b3:
        return L + ["_B300 result not available yet._", ""]

    def node_mean(r, p):
        d = r["data"].get(p)
        return mean(list(d.values())) if d else None

    precs = ["FP32", "BF16", "FP8"]
    rows = []
    for p in precs:
        b2_means = [node_mean(r, p) for r in b2.values()]
        b2_means = [x for x in b2_means if x is not None]
        b2m = mean(b2_means)
        for n, r in sorted(b3.items()):
            b3m = node_mean(r, p)
            d = r["data"].get(p) or {}
            rows.append([p, f(b2m, 0),
                         "{}–{}".format(f(min(b2_means), 0), f(max(b2_means), 0)) if b2_means else "—",
                         f(b3m, 0),
                         "{}–{}".format(f(min(d.values()), 0), f(max(d.values()), 0)) if d else "—",
                         ratio(b3m, b2m)])
    L += ["B200 = mean over {} B200 node(s) of the per-node mean (newest run per node); "
          "B300 = {} ({}). Converged value per GPU, 300 s per precision.".format(
              len(b2), ", ".join(sorted(b3)), ", ".join(sorted({r['gpu'] for r in b3.values()}))), ""]
    L += table(["Precision", "B200 mean", "B200 node range", "B300 mean",
                "B300 GPU range", "B300 / B200"], rows)
    thr = [n for n, r in b3.items() if r["throttled"]]
    L += ["", "Throttling reported on B300: {}".format(", ".join(thr) if thr else "none"), ""]

    # per-GPU detail for the B300 node(s)
    for n, r in sorted(b3.items()):
        L += ["### B300 per-GPU detail — {}".format(n), ""]
        gpus = sorted({g for p in precs for g in (r["data"].get(p) or {})})
        L += table(["GPU"] + precs,
                   [["GPU{}".format(g)] + [f((r["data"].get(p) or {}).get(g), 0) for p in precs]
                    for g in gpus])
        L.append("")
    return L


# ---------------------------------------------------------------- cublaslt
CUBLASLT_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cuBLASLt", "summary.md")


def section_cublaslt():
    """Pull sections 1, 3 and 4 of ../cuBLASLt/summary.md (its own analyzer, own jobs)."""
    L = ["## 3. cuBLASLt GEMM — per-GPU throughput incl. FP4 (TFLOP/s)", ""]
    if not os.path.exists(CUBLASLT_MD):
        return L + ["No results yet (`../cuBLASLt/summary.md` missing).", ""]
    lines = open(CUBLASLT_MD).read().splitlines()
    head = [x for x in lines if x.startswith(("- Generated", "- Nodes", "- Library"))]
    L += ["From `../cuBLASLt/summary.md` (separate benchmark: `../cuBLASLt/job-lt-bench.sh`, "
          "8 GPUs at once, tuned shape per GPU type; full method and per-shape results there)."]
    L += head + [""]
    keep, cur = {"1", "3", "4"}, None
    for x in lines:
        m = re.match(r"## (\d+)\. (.*)", x)
        if m:
            cur = m.group(1)
            if cur in keep:
                L.append("### " + m.group(2))
            continue
        if x.startswith("# ") or cur not in keep:
            continue
        L.append(x)
    if L[-1] != "":
        L.append("")
    L += ["### Why FP4 is 1.20x measured but 1.5x on paper", "",
          "(Explanation written from the 2026-10-01 cuBLASLt run; the numbers below are from the "
          "tables above.)", "",
          "The gap is mainly the power limit. Both GPUs hit their power cap during the FP4 test, "
          "and B300 then has to run at a much lower clock. NVIDIA's spec-sheet numbers assume "
          "full clocks.", "",
          "1. **Both GPUs are power-capped.** In the sustained FP4 run B200 drew 993 W against its "
          "1,000 W limit and B300 drew 1,091 W against its 1,100 W limit. At the cap the GPU "
          "lowers its clock.",
          "2. **B300 loses more clock than B200.** The 1.5x gain comes from 1.5x more FP4 math per "
          "clock cycle, which also costs more energy per cycle. B300 has only 10% more power than "
          "B200, so it slows down further.", ""]
    L += table(["FP4, sustained", "B200", "B300"], [
        ["SM clock", "1,314 MHz", "1,099 MHz (0.84x of B200)"],
        ["Share of the unthrottled clock (~1,965 / ~2,032 MHz, from the FP64 run)", "67%", "54%"],
        ["Share of spec-sheet FP4 peak", "63%", "51%"],
        ["FP4 work per watt (TFLOP/s per W)", "5.75", "6.30 (1.10x)"],
    ])
    L += ["",
          "3. **The numbers add up.** 1.5x per clock x 0.84x clock = ~1.25x, close to the measured "
          "1.20x. Both GPUs reach about 94% of what their actual clock allows (63/67 and 51/54), "
          "so the GEMM kernels are not the limit. Equivalently: 1.10x more work per watt x 1.10x "
          "more power = ~1.2x.",
          "4. **Spec-sheet context.** NVIDIA's 15 PFLOP/s dense FP4 figure for Blackwell Ultra is "
          "for parts running at up to 1,400 W (GB300-class systems), not this 1,100 W HGX board.",
          "5. **Other precisions.** FP4 is the only precision where B300 has more peak compute per "
          "clock. In BF16, FP8 and TF32 both GPUs have the same peak and run at similar clocks, so "
          "they measure about 1.0x.",
          "6. **What would change it.** A higher GPU power limit should narrow the gap; that is "
          "set by the node administrators, not by user jobs.", ""]
    return L


# ---------------------------------------------------------------- nccl
def section_nccl():
    L = ["## 4. NCCL 1-node — intra-node NVLink bus bandwidth (GB/s)", ""]
    b2 = newest_per_node(glob.glob(os.path.join(B200_DIR, "out-nccl-1node", "*.out")),
                         nccl.parse_file)
    b3 = newest_per_node(glob.glob(os.path.join(DATA, "out-nccl-1node", "*.out")),
                         nccl.parse_file)
    b2 = {n: r for n, r in b2.items() if n not in B300_NODES}
    b3 = {n: r for n, r in b3.items() if n in B300_NODES}
    if not b3:
        return L + ["_B300 result not available yet._", ""]

    def by_coll(r):
        return {c["coll"]: c for c in r["collectives"] if c["converged"] is not None}

    b2c = [by_coll(r) for r in b2.values()]
    L += ["Converged busbw = busbw at the largest message (16 GiB), best of out-of-place / "
          "in-place. B200 = mean over the B200 nodes that ran that collective "
          "(newest run per node); 8 GPUs, 1 MPI task.", ""]
    for n, r in sorted(b3.items()):
        c3 = by_coll(r)
        rows = []
        for coll in sorted(c3, key=nccl.coll_sort_key):
            vals = [c[coll]["converged"] for c in b2c if coll in c]
            b2m = mean(vals)
            rows.append([coll, f(b2m), len(vals), f(c3[coll]["converged"]),
                         f(c3[coll]["peak"]), ratio(c3[coll]["converged"], b2m),
                         "ok" if c3[coll]["ok"] else "**CHECK**"])
        L += ["### {} ({} x {})".format(n, r["ngpu"], r["gpu_name"]), ""]
        L += table(["Collective", "B200 busbw", "#B200 nodes", "B300 busbw", "B300 peak",
                    "B300 / B200", "B300 correctness"], rows)
        L.append("")

        # all_reduce busbw vs size
        if "all_reduce" in c3:
            b2rows = {}
            for c in b2c:
                if "all_reduce" in c:
                    for row in c["all_reduce"]["rows"]:
                        b2rows.setdefault(row["size"], []).append(row["oop_busbw"])
            rows = []
            for row in c3["all_reduce"]["rows"]:
                b2m = mean(b2rows.get(row["size"], []))
                rows.append([nccl.fmt_size(row["size"]), f(b2m), f(row["oop_busbw"]),
                             ratio(row["oop_busbw"], b2m)])
            L += ["#### all_reduce busbw vs message size (out-of-place)", ""]
            L += table(["Size", "B200", "B300", "B300 / B200"], rows)
            L.append("")
    return L


# ---------------------------------------------------------------- ib_write_bw
IB_LINE = re.compile(r"^(host mem -> host mem|NIC reads from GPU|NIC writes into GPU|GPU -> GPU)\s+([\d.]+)\s+Gb/s")
IB_SWEEP = re.compile(r"^\s+(\d+)\s+([\d.]+) Gb/s")
IB_DEV = re.compile(r"^\s+(client|server)\s+:\s+(\S+)")


def parse_ib(path):
    r = {"tests": {}, "sweep": {}, "dev": {}}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = IB_LINE.match(line)
            if m:
                r["tests"][m.group(1)] = float(m.group(2))
                continue
            m = IB_SWEEP.match(line)
            if m:
                r["sweep"][int(m.group(1))] = float(m.group(2))
                continue
            m = IB_DEV.match(line)
            if m:
                r["dev"][m.group(1)] = m.group(2)
    return r if r["tests"] else None


def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    for p in reversed(files):
        r = parse_ib(p)
        if r:
            r["path"] = p
            return r
    return None


def section_ib():
    L = ["## 5. ib_write_bw — GPUDirect RDMA, two rails of one node (Gb/s)", ""]
    b2 = newest(os.path.join(B200_DIR, "out-ibwrite", "ibwrite-1node-*.out"))
    b3 = newest(os.path.join(DATA, "out-ibwrite", "ibwrite-1node-*.out"))
    if not b3:
        return L + ["_B300 result not available yet._", ""]
    L += ["64 MiB RDMA write, 200 iterations. B200: `{}` (client {}, server {}). "
          "B300: `{}` (client {}, server {}).".format(
              os.path.relpath(b2["path"], HERE) if b2 else "—",
              b2["dev"].get("client", "?") if b2 else "?", b2["dev"].get("server", "?") if b2 else "?",
              os.path.relpath(b3["path"], HERE),
              b3["dev"].get("client", "?"), b3["dev"].get("server", "?")), ""]
    rows = []
    for t in ["host mem -> host mem", "NIC reads from GPU", "NIC writes into GPU", "GPU -> GPU"]:
        v2 = b2["tests"].get(t) if b2 else None
        v3 = b3["tests"].get(t)
        rows.append([t, f(v2, 2), f(v3, 2), ratio(v3, v2)])
    L += table(["Test", "B200", "B300", "B300 / B200"], rows)
    L += ["", "### Size sweep, NIC reads from GPU", ""]
    sizes = sorted(set(b3["sweep"]) | set(b2["sweep"] if b2 else []))
    sizes = [s for s in sizes if s >= 4096]
    rows = [[nccl.fmt_size(s), f(b2["sweep"].get(s) if b2 else None, 2),
             f(b3["sweep"].get(s), 2), ratio(b3["sweep"].get(s), b2["sweep"].get(s) if b2 else None)]
            for s in sizes]
    L += table(["Size", "B200", "B300", "B300 / B200"], rows)
    L.append("")
    return L


# ---------------------------------------------------------------- megatron
def megatron_runs(d):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "megatron-1node-*-g*")), key=os.path.getmtime):
        if not meg.FNAME_RE.search(os.path.basename(p)) or "-max-" in os.path.basename(p):
            continue
        r = meg.parse_output(p)
        if r["gpus"] is not None and r["ok"]:
            runs[(r["node"], r["gpus"])] = r
    return runs


def section_megatron():
    L = ["## 6. Megatron-LM 1-node — reference ~7B GPT (TFLOP/s/GPU)", ""]
    b2 = {k: v for k, v in megatron_runs(os.path.join(B200_DIR, "output-megatron")).items()
          if k[0] not in B300_NODES}
    b3 = {k: v for k, v in megatron_runs(os.path.join(DATA, "output-megatron")).items()
          if k[0] in B300_NODES}
    L += ["Same config on both: 36 layers, hidden 4096, FFN 14336, seq 2048, bf16, micro-batch 4, "
          "global batch = 128 x GPUs, 100 iters, no recompute, TP=PP=1. Metric = last-iteration "
          "throughput per GPU. B200 = mean over B200 nodes with that GPU count (newest run).", ""]
    if not b3:
        return L + ["_B300 results not available yet._", ""]
    rows = []
    for g in range(1, 9):
        v2 = [r for (n, gg), r in b2.items() if gg == g]
        v3 = [r for (n, gg), r in b3.items() if gg == g]
        t2 = mean([r["tflops"] for r in v2])
        t3 = mean([r["tflops"] for r in v3])
        i2 = mean([r["iter_ms"] for r in v2])
        i3 = mean([r["iter_ms"] for r in v3])
        rows.append([g, 128 * g, f(t2), len(v2), f(t3), f(t3 * g if t3 else None, 0),
                     f(i2, 0), f(i3, 0), ratio(t3, t2)])
    L += table(["#GPUs", "GBS", "B200 TFLOP/s/GPU", "#B200 nodes", "B300 TFLOP/s/GPU",
                "B300 aggregate", "B200 iter ms", "B300 iter ms", "B300 / B200"], rows)
    L.append("")
    return L


SWEEP_RE = re.compile(r"max-(b200|b300)-(\S+?)-(5b|13b)-mb(\d+)-(none|selective|full)-(bf16|fp8)\.(\d+)$")
MEM_RE = re.compile(r"max allocated:\s*([\d.]+)")
OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory|out of memory", re.IGNORECASE)


def parse_sweep(path):
    m = SWEEP_RE.search(os.path.basename(path))
    if not m:
        return None
    gpu, node, model, mb, rc, prec, jid = m.groups()
    r = meg.parse_output(path)
    mem, oom = None, False
    with open(path, errors="replace") as fh:
        for line in fh:
            mm = MEM_RE.search(line)
            if mm:
                mem = max(mem or 0, float(mm.group(1)) / 1024.0)  # -> GiB
            if not oom and OOM_RE.search(line):
                oom = True
    if r["ok"]:
        status = "ok"
    elif oom:
        status = "OOM"
    elif r["last_iter"]:
        status = "partial"
    else:
        status = "failed/running"
    return {"gpu": gpu.upper(), "node": node, "model": model, "mb": int(mb), "rc": rc,
            "prec": prec, "tflops": r["tflops"], "iter_ms": r["iter_ms"], "mem": mem,
            "status": status}


def section_megatron_max():
    L = ["## 7. Megatron-LM 1-node — tuned max throughput, best per GPU type (TFLOP/s/GPU)", ""]
    runs = {}
    for p in sorted(glob.glob(os.path.join(DATA, "output-max-sweep", "max-*")), key=os.path.getmtime):
        r = parse_sweep(p)
        if r:
            runs[(r["gpu"], r["model"], r["mb"], r["rc"], r["prec"])] = r  # newest wins
    L += ["Each GPU type gets its own grid sweep (8 GPUs, seq 4096, distributed optimizer, "
          "overlapped grad-reduce/param-gather, TP=PP=1, grad-acc 4, 20 iters): "
          "5B (24L, h4096) micro 4/8/16 x recompute none/selective/full, and "
          "13B (40L, h5120) micro 2/4/8 x recompute none/full, each in bf16 and fp8. "
          "The configs are NOT forced to match: the best point per GPU type and precision is "
          "compared. TFLOP/s/GPU is Megatron's model-FLOP throughput (recompute work not "
          "counted). Peak memory = max allocated by PyTorch on rank 0.", ""]
    if not runs:
        return L + ["_Sweep results not available yet._", ""]

    def best(gpu, prec):
        c = [r for r in runs.values() if r["gpu"] == gpu and r["prec"] == prec and r["status"] == "ok"]
        return max(c, key=lambda r: r["tflops"]) if c else None

    def cfg(r):
        return "{} mb{} {}".format(r["model"], r["mb"], r["rc"]) if r else "—"

    rows = []
    for prec in ("bf16", "fp8"):
        b2, b3 = best("B200", prec), best("B300", prec)
        rows.append([prec, f(b2["tflops"]) if b2 else "—", cfg(b2), f(b2["mem"], 0) if b2 else "—",
                     f(b3["tflops"]) if b3 else "—", cfg(b3), f(b3["mem"], 0) if b3 else "—",
                     ratio(b3["tflops"] if b3 else None, b2["tflops"] if b2 else None)])
    L += table(["Precision", "B200 best", "B200 config", "B200 mem GiB",
                "B300 best", "B300 config", "B300 mem GiB", "B300 / B200"], rows,
               ["---", "---:", "---", "---:", "---:", "---", "---:", "---:"])
    n_ok = {g: sum(1 for r in runs.values() if r["gpu"] == g and r["status"] == "ok") for g in ("B200", "B300")}
    n_all = {g: sum(1 for r in runs.values() if r["gpu"] == g) for g in ("B200", "B300")}
    L += ["", "Grid points finished OK: B200 {}/{}, B300 {}/{} (of 30 each).".format(
        n_ok["B200"], n_all["B200"], n_ok["B300"], n_all["B300"]), ""]

    L += ["### Full sweep grid (TFLOP/s/GPU, peak GiB)", ""]
    keys = sorted({k[1:] for k in runs}, key=lambda k: (k[3], k[0] != "5b", k[1], ["none", "selective", "full"].index(k[2])))
    rows = []
    for model, mb, rc, prec in keys:
        cells = []
        for g in ("B200", "B300"):
            r = runs.get((g, model, mb, rc, prec))
            if not r:
                cells.append("—")
            elif r["status"] == "ok":
                cells.append("{} ({} GiB)".format(f(r["tflops"]), f(r["mem"], 0)))
            else:
                cells.append(r["status"])
        r2, r3 = runs.get(("B200", model, mb, rc, prec)), runs.get(("B300", model, mb, rc, prec))
        rows.append([prec, model, mb, rc] + cells + [
            ratio(r3["tflops"] if r3 and r3["status"] == "ok" else None,
                  r2["tflops"] if r2 and r2["status"] == "ok" else None)])
    L += table(["Prec", "Model", "Micro", "Recompute", "B200", "B300", "B300 / B200"], rows,
               ["---", "---", "---:", "---", "---:", "---:", "---:"])
    L.append("")
    return L


# ---------------------------------------------------------------- on paper
def section_paper():
    """Static: official NVIDIA specs (checked 2026-10-01), not measured here."""
    L = ["## 1. On paper — official NVIDIA specs, B300 vs B200", "",
         "Sources: [NVIDIA HGX platform page](https://www.nvidia.com/en-us/data-center/hgx/) "
         "(HGX B300 vs HGX B200 table), "
         "[Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) "
         "(NVIDIA technical blog), [DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/) and "
         "[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) product pages. "
         "Per-GPU values are the HGX 8-GPU numbers divided by 8; NVIDIA lists tensor-core "
         "numbers with 2:4 sparsity, and dense = 1/2 sparse except where NVIDIA gives a "
         "dense number (FP4).", ""]
    L += table(["Per GPU (HGX, dense)", "B200 (Blackwell)", "B300 (Blackwell Ultra)", "B300 / B200"], [
        ["FP4 tensor (NVFP4)", "9 PFLOP/s", "13.5 PFLOP/s", "**1.50x**"],
        ["FP8 / FP6 tensor", "4.5 PFLOP/s", "4.5 PFLOP/s", "1.00x"],
        ["BF16 / FP16 tensor", "2.25 PFLOP/s", "2.25 PFLOP/s", "1.00x"],
        ["TF32 tensor", "1.125 PFLOP/s", "1.125 PFLOP/s", "1.00x"],
        ["FP32 (non-tensor)", "75 TFLOP/s", "75 TFLOP/s", "1.00x"],
        ["INT8 tensor", "4.5 POP/s", "~0.19 POP/s", "**~0.04x**"],
        ["FP64 / FP64 tensor", "37 TFLOP/s", "1.25 TFLOP/s", "**~0.03x**"],
        ["Attention softmax (SFU exp)", "5 T exp/s", "10.7 T exp/s", "**2.14x**"],
        ["HBM3E capacity", "180 GB (8-high stacks)", "288 GB (8 x 12-high stacks; ~270 GB usable on HGX)", "**1.5x**"],
        ["HBM bandwidth", "8 TB/s", "8 TB/s", "1.00x"],
        ["NVLink 5 GPU-to-GPU", "1.8 TB/s", "1.8 TB/s", "1.00x"],
        ["Scale-out NIC per GPU", "ConnectX-7, 400 Gb/s", "ConnectX-8, 800 Gb/s (PCIe Gen6)", "**2.0x**"],
        ["Max GPU power (HGX)", "1,000 W", "1,100 W (up to 1,400 W in GB300 systems)", "1.10x"],
        ["Process / transistors", "TSMC 4NP, 208 B", "TSMC 4NP, 208 B (160 SMs)", "same"],
    ], align=["---", "---", "---", "---:"])
    L += ["",
          "B300 advantages on paper:", "",
          "- **1.5x memory per GPU** (288 GB vs 180-192 GB). This allows larger models or "
          "longer contexts per GPU, bigger micro-batches, less activation recompute, and "
          "bigger KV caches for inference.",
          "- **1.5x dense FP4 (NVFP4)**, useful for FP4 inference and FP4 training recipes.",
          "- **~2x attention-layer exponent throughput**, which speeds up softmax-heavy "
          "attention, mainly at long context.",
          "- **2x scale-out network per GPU** (ConnectX-8 800 Gb/s vs ConnectX-7 400 Gb/s), "
          "which helps multi-node training and inference.",
          "- Higher power limit (1,100 W vs 1,000 W per GPU), so clocks may hold up better under "
          "sustained load.", "",
          "B300 disadvantages on paper:", "",
          "- **FP64 drops ~30x** (37 TFLOP/s to 1.25 TFLOP/s per GPU). B300 is a poor fit for "
          "double-precision HPC codes (CFD, MD with FP64, dense linear algebra, HPL).",
          "- **INT8 tensor drops ~24x**. Legacy INT8 inference paths should move to FP8/FP4.",
          "- **No gain for FP8, BF16, TF32 or FP32 dense math, HBM bandwidth, or NVLink.** "
          "Standard BF16/FP8 training throughput is expected to be similar to B200. Any gain "
          "comes from extra memory (bigger batches, no recompute), faster attention, and power "
          "headroom.",
          "- More power and heat per GPU (+10% on HGX), plus a newer stack: the CUDA 12.8+/13 "
          "toolchain, compute capability 10.3, and a newer driver.", "",
          "What this predicts for the measurements below: gpu-fryer BF16/FP8 and NCCL NVLink "
          "close to 1.0x; ib_write_bw up to ~2x if the B300 links run at 800 Gb/s; Megatron "
          "reference config ~1.0x; tuned Megatron somewhat above B200 thanks to the "
          "larger memory.", ""]
    return L


# ---------------------------------------------------------------- main
def main():
    L = ["# B200 vs B300 — single-node benchmark comparison", "",
         "- Generated: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")),
         "- B300 node: {} (mit_testing, 8 x B300); B200 data from `../b200-nodes/`".format(
             ", ".join(sorted(B300_NODES))),
         "- Ratios are B300 / B200; > 1.00x means B300 is faster.",
         "- Per-benchmark B300 summaries: `out-gpu-fryer/summary.md`, `out-nccl-1node/summary.md`, "
         "`output-megatron/summary.md`", ""]
    for sec in (section_paper, section_fryer, section_cublaslt, section_nccl, section_ib, section_megatron, section_megatron_max):
        try:
            L += sec()
        except Exception as e:
            L += ["## {} failed: {}".format(sec.__name__, e), ""]
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("Written to {}".format(OUT_MD))


if __name__ == "__main__":
    main()
