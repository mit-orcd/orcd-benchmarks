#!/usr/bin/env python3
"""Build the two sets of reports from the per-node results. One md per benchmark in each set:

  results/ubuntu/         amd-ubuntu only (node6100, node6101), written as if no other system
                          existed: absolute numbers, node-to-node agreement, scaling, and for
                          Kimi-K3 what AMD's vLLM recipe improves over ATOM.
  results/vs-amd-cloud/   the same benchmarks side by side with ../amd-benchmarks/amd-cloud.

Reads whatever exists, so it can run after every stage (auto_run.sh does). Inputs are the CSVs
and PRIMUS_REPORT.md tables the ported analyzers write under results/<node>/ and, for Kimi-K3,
results/<node>/kimi-cloud/ (ATOM, amd-cloud's exact images) and kimi-recipe/ (vLLM recipe).

    $PY report.py
"""
import datetime, os, re, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
CLOUD = HERE.parent / "amd-benchmarks" / "amd-cloud" / "results"
ROOT = Path(os.environ.get("RESULTS_ROOT", HERE / "results"))   # override for testing
NODES = ["node6100", "node6101"]
RES = {n: ROOT / n for n in NODES}
OUT_U = ROOT / "ubuntu"
OUT_C = ROOT / "vs-amd-cloud"
LINE_RATE_GBPS = 400.0   # 8 x 400 Gb/s ionic per node
NOW = f"{datetime.datetime.now():%Y-%m-%d %H:%M}"
ATOM_IMAGES = ("`atom-dev:nightly_202608111555` (most experiments), `nightly_202608191459` "
               "(isl4096, repeats arm A), MAD `rocm7.2.4_..._20260727_kimi_k3` (mad, single_stream, "
               "repeats arm B, ep_matched)")
RECIPE = ("AMD's vLLM recipe (recipes.vllm.ai, 2026-09-25): `vllm/vllm-openai-rocm:nightly-rocm100` "
          "(digest e76a953f), DSpark speculative decoding up to C=14, decode-context-parallel 8 + CPU KV "
          "offload above, server re-tuned per concurrency")


# ---------------------------------------------------------------- helpers
def load(path, keys=None):
    """CSV -> DataFrame, last row per key (later runs win)."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if keys:
        if not set(keys) <= set(df.columns):
            return None
        df = df.drop_duplicates(subset=keys, keep="last")
    return df


def num(x, nd=1):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:,.{nd}f}"


def key(v):
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def ratio(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return "—"
    r = a / b
    s = f"{r:.2f}x"
    return f"**{s}**" if abs(r - 1) > 0.05 else s


def table(df, keys, cols, labels=None, nd=1, ratios=()):
    """Markdown table. cols: value columns; ratios: (label, num_col, den_col)."""
    labels = labels or cols
    hdr = keys + list(labels) + [r[0] for r in ratios]
    L = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(keys) + "---:|" * (len(hdr) - len(keys))]
    for _, r in df.iterrows():
        row = [key(r[k]) for k in keys] + [num(r.get(c), nd) for c in cols]
        row += [ratio(r.get(a), r.get(b)) for _, a, b in ratios]
        L.append("| " + " | ".join(row) + " |")
    return L


def nodes_frame(name, keys, val, sub=""):
    """Outer-join `val` from results/<node>/<sub>/<name> for both nodes; columns = node names."""
    base = None
    for n in NODES:
        d = load(RES[n] / sub / name, keys)
        if d is None or val not in d.columns:
            continue
        d = d[keys + [val]].rename(columns={val: n})
        base = d if base is None else base.merge(d, on=keys, how="outer")
    return base


def node_ratio_cols(df):
    have = [n for n in NODES if n in df.columns]
    return have, ([("6100/6101", NODES[0], NODES[1])] if len(have) == 2 else [])


def write(path, title, intro, sections):
    path.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# {title}", "", f"Generated {NOW} by `report.py`.", ""] + intro
    for h, body in sections:
        L += ["", f"## {h}", ""] + body
    text = "\n".join(L) + "\n"
    # spell out RVS the first time it appears in each file
    if "ROCm Validation Suite" not in text:
        text = re.sub(r"\bRVS\b", "RVS (ROCm Validation Suite)", text, count=1)
    path.write_text(text)
    return path


PENDING = ["*Pending: no results yet.*"]
SYSTEM_U = ["System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, "
            "2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 for RVS and rccl-tests, "
            "containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G "
            "AMD Pollara (ionic) RoCEv2 rails.", ""]
SYSTEM_C = ["- **amd-cloud**: one 8 × MI355X node, ROCm 7.14, Ubuntu 22.04.5, docker "
            "([results](../../../amd-benchmarks/amd-cloud/results/SUMMARY.md)).",
            "- **amd-ubuntu**: node6100 and node6101, 8 × MI355X each, same amdgpu driver and CPUs, "
            "host ROCm 7.2.4, Ubuntu 24.04.5, apptainer. Same scripts, analyzers and (for the "
            "containers) the same image digests.",
            "- ratio = amd-ubuntu / amd-cloud; **bold** = more than 5% off. For latency (TTFT, TPOT) "
            "below 1 is better.", ""]


# ---------------------------------------------------------------- Part A: RVS
def rvs_u():
    df = nodes_frame("rvs_tflops.csv", ["gpus", "precision"], "aggregate_tflops")
    if df is None:
        return PENDING
    have, rr = node_ratio_cols(df)
    L = ["Aggregate `gst` TFLOPS: hipBLASLt GEMM on every selected GPU at once, peak per-GPU "
         "sample summed over GPUs. Each GPU runs its own GEMM, so scaling should be linear; "
         "losses are power or thermals.", ""]
    for N in (8, 1):
        L += [f"### N = {N}", ""] + table(df[df.gpus == N], ["precision"], have, nd=1, ratios=rr) + [""]
    # scaling efficiency N=8 vs 8 x N=1
    piv = {n: df.pivot_table(index="precision", columns="gpus", values=n) for n in have}
    rows = []
    for p in sorted(df.precision.unique()):
        r = {"precision": p}
        for n in have:
            t = piv[n]
            if 1 in t.columns and 8 in t.columns and p in t.index:
                r[n] = 100 * t.loc[p, 8] / (8 * t.loc[p, 1])
        rows.append(r)
    L += ["### Scaling efficiency, N = 8 vs 8 × N = 1 (%)", ""] + table(pd.DataFrame(rows), ["precision"], have, nd=1)
    return L


# ---------------------------------------------------------------- Part B: RCCL
def rccl_u():
    df = nodes_frame("rccl.csv", ["collective", "config", "gpus"], "busbw_at_max_GBps")
    if df is None:
        return PENDING
    have, rr = node_ratio_cols(df)
    d = df[df.config == "default"]
    L = ["busbw (GB/s) at the largest message size, XGMI inside one node.", ""]
    for n in have:
        piv = d.pivot_table(index="collective", columns="gpus", values=n).reset_index()
        cols = [c for c in piv.columns if c != "collective"]
        piv.columns = ["collective"] + [f"N={c}" for c in cols]
        L += [f"### {n}: busbw vs N", ""] + table(piv, ["collective"], [f"N={c}" for c in cols]) + [""]
        if 4 in cols and 5 in cols and 8 in cols:
            p2 = piv[["N=4", "N=8"]].min(axis=1)
            mid = piv[[f"N={k}" for k in (5, 6, 7) if k in cols]].max(axis=1)
            drop = (100 * (1 - mid / p2)).round(0)
            worst = piv.assign(drop=drop).sort_values("drop", ascending=False).head(3)
            L += [f"Largest N=5..7 dip below min(N=4, N=8): " +
                  ", ".join(f"{r.collective} {r.drop:.0f}%" for r in worst.itertuples()) + ".", ""]
    L += ["### N = 8, node6100 vs node6101", ""] + table(d[d.gpus == 8], ["collective"], have, ratios=rr)
    cfg = df[(df.config != "default") & (df.gpus == 8)]
    if not cfg.empty:
        L += ["", "### Config sweep, N = 8", ""] + table(cfg, ["collective", "config"], have, ratios=rr)
    return L


# ---------------------------------------------------------------- ROCm stacks (7.2.4 vs 7.14)
def stacks_frame(name, keys, val):
    """Host ROCm 7.2.4 results (columns node6100, node6101) and ROCm 7.14 results from
    results/<node>/rocm7.14/ (columns 'node6100 r7.14', 'node6101 r7.14'), outer-joined."""
    df = nodes_frame(name, keys, val)
    d2 = nodes_frame(name, keys, val, sub="rocm7.14")
    if d2 is not None:
        d2 = d2.rename(columns={n: f"{n} r7.14" for n in NODES})
        df = d2 if df is None else df.merge(d2, on=keys, how="outer")
    return df


STACK_COLS = [(n, f"{n[-4:]} ROCm 7.2.4") for n in NODES] + [(f"{n} r7.14", f"{n[-4:]} ROCm 7.14") for n in NODES]


def cloud_table(df, keys, nd=1):
    """amd-cloud column + every amd-ubuntu stack column, each with a ratio to amd-cloud."""
    cols = [(c, l) for c, l in STACK_COLS if c in df.columns]
    # apple-to-apple ROCm 7.14 columns first, host ROCm 7.2.4 after
    cols = sorted(cols, key=lambda cl: "r7.14" not in cl[0])
    return table(df, keys, ["cloud"] + [c for c, _ in cols], ["amd-cloud (7.14)"] + [l for _, l in cols], nd,
                 [(f"{l.replace('ROCm ', '')}/cloud", c, "cloud") for c, l in cols])


def rvs_c():
    df = stacks_frame("rvs_tflops.csv", ["gpus", "precision"], "aggregate_tflops")
    c = load(CLOUD / "rvs_tflops.csv", ["gpus", "precision"])
    if df is None:
        return PENDING
    df = c[["gpus", "precision", "aggregate_tflops"]].rename(columns={"aggregate_tflops": "cloud"}).merge(df, how="right")
    L = ["Aggregate `gst` TFLOPS. The GEMMs are independent per GPU, so a gap is clocks, power or "
         "the ROCm stack, never the interconnect. amd-cloud ran ROCm 7.14; amd-ubuntu ran host ROCm "
         "7.2.4 and, separately, the same ROCm 7.14 user space (`ROCM_STACK=7.14`): the 7.14 columns "
         "are the apple-to-apple comparison.", ""]
    for N in (8, 1):
        L += [f"### N = {N}", ""] + cloud_table(df[df.gpus == N], ["precision"]) + [""]
    return L


def rccl_c():
    df = stacks_frame("rccl.csv", ["collective", "config", "gpus"], "busbw_at_max_GBps")
    c = load(CLOUD / "rccl.csv", ["collective", "config", "gpus"])
    if df is None:
        return PENDING
    df = c[["collective", "config", "gpus", "busbw_at_max_GBps"]].rename(
        columns={"busbw_at_max_GBps": "cloud"}).merge(df, how="right")
    d = df[df.config == "default"]
    L = ["busbw (GB/s) at the largest message size. amd-cloud ran ROCm/RCCL 7.14; the amd-ubuntu "
         "7.14 columns use the same ROCm 7.14 user space (apple-to-apple), the 7.2.4 columns the host ROCm.", ""]
    for N, note in ((8, "full XGMI mesh"), (5, "inside the N=5..7 dip"), (2, "one link")):
        L += [f"### N = {N} ({note})", ""] + cloud_table(d[d.gpus == N], ["collective"]) + [""]
    cfg = df[(df.config != "default") & (df.gpus == 8)]
    if not cfg.empty:
        L += ["### Config sweep, N = 8", ""] + cloud_table(cfg, ["collective", "config"])
    return L


def rocm_u():
    """Host ROCm 7.2.4 vs ROCm 7.14 user space, same nodes, same driver."""
    L = ["Same nodes, same amdgpu driver (6.19.14), same benchmark scripts; only the user-space ROCm "
         "differs: host ROCm 7.2.4 (`/opt/rocm`, packaged RVS) vs ROCm 7.14.0 (TheRock tarball in "
         "`amd-software/rocm-7.14.0`, RVS built from source). Ratio = 7.14 / 7.2.4; bold = more than 5%.", ""]
    found = False
    for title, name, keys, val, filt, splits in [
        ("RVS gst aggregate TFLOPS", "rvs_tflops.csv", ["gpus", "precision"], "aggregate_tflops", None,
         [("N = 8", lambda d: d[d.gpus == 8], ["precision"]), ("N = 1", lambda d: d[d.gpus == 1], ["precision"])]),
        ("RCCL busbw at the largest size (GB/s), single node", "rccl.csv", ["collective", "config", "gpus"],
         "busbw_at_max_GBps", lambda d: d[d.config == "default"],
         [("N = 8", lambda d: d[d.gpus == 8], ["collective"]), ("N = 5", lambda d: d[d.gpus == 5], ["collective"])]),
        ("RCCL busbw at the largest size (GB/s), 2 nodes", "rccl_2node.csv", ["collective", "ppn"],
         "busbw_max_size", None, [("all PPN", lambda d: d, ["collective", "ppn"])]),
    ]:
        df = stacks_frame(name, keys, val)
        if df is None or not any(f"{n} r7.14" in df.columns for n in NODES):
            continue
        found = True
        if filt is not None:
            df = filt(df)
        L += [f"### {title}", ""]
        for sub, sel, k in splits:
            cols, labs, rr = [], [], []
            for n in NODES:
                for c, l in ((n, f"{n[-4:]} 7.2.4"), (f"{n} r7.14", f"{n[-4:]} 7.14")):
                    if c in df.columns:
                        cols.append(c); labs.append(l)
                if n in df.columns and f"{n} r7.14" in df.columns:
                    rr.append((f"{n[-4:]} 7.14/7.2.4", f"{n} r7.14", n))
            L += [f"#### {sub}", ""] + table(sel(df), k, cols, labs, 1, rr) + [""]
    return L if found else L + ["*Pending: no ROCm 7.14 results yet.*"]


# ---------------------------------------------------------------- network + 2-node
def net_u():
    L = []
    for n in NODES:
        for f in ("ib_bw.md", "ib_bw_nonuma.md"):
            p = RES[n] / f
            if p.exists():
                txt = p.read_text().splitlines()
                hdr = next((l for l in txt if l.startswith("perftest")), "")
                L += [f"### `{n}/{f}`", "", hdr, ""] + [l for l in txt if l.startswith("|")] + [""]
    if not L:
        return PENDING
    return ["Host-memory RDMA write bandwidth per ionic rail (perftest `ib_write_bw`), alone and "
            "all 8 at once. Line rate is 400 Gb/s per rail, 3,200 Gb/s (400 GB/s) per node. Rails "
            "0-3 hang off NUMA node 0 and 4-7 off NUMA node 1; each process must be bound to its "
            "NIC's NUMA node to reach line rate on all 8 at once.", ""] + L


def rccl2_u():
    rows = []
    for n in NODES:
        d = load(RES[n] / "rccl_2node.csv", ["collective", "ppn"])
        if d is not None:
            rows.append(d)
    if not rows:
        return PENDING
    d = pd.concat(rows)
    one = nodes_frame("rccl.csv", ["collective", "config", "gpus"], "busbw_at_max_GBps")
    L = [f"RCCL across node6100 + node6101 over the 8 ionic rails. Inter-node line rate is "
         f"{LINE_RATE_GBPS:.0f} GB/s per node; single-node N=8 busbw (XGMI) for scale.", "",
         "| collective | PPN | ranks | busbw @max (GB/s) | % of line rate | single node N=8 | status |",
         "|---|---:|---:|---:|---:|---:|---|"]
    for _, r in d.sort_values(["collective", "ppn"]).iterrows():
        sn = None
        if one is not None and "node6100" in one.columns:
            m = one[(one.collective == r.collective) & (one.config == "default") & (one.gpus == 8)]
            sn = m["node6100"].iloc[0] if not m.empty else None
        bw = r.get("busbw_max_size")
        pct = num(100 * bw / LINE_RATE_GBPS, 0) + "%" if pd.notna(bw) else "—"
        L.append(f"| {r.collective} | {r.ppn} | {r.ranks} | {num(bw)} | {pct} | {num(sn)} | {r.status} |")
    return L


# ---------------------------------------------------------------- Part C / M: Primus
def primus_tables(path):
    """PRIMUS_REPORT.md -> {section: {N: value}}: §1.1 compute TF/s/GPU, §2.1 GEMM mean,
    §1.2 megatron-ref MI355X TF/s/GPU at N=8."""
    if not path.exists():
        return None
    txt = path.read_text()
    out, sec = {}, None
    for line in txt.splitlines():
        if line.startswith("### 1.1 "):
            sec = "megatron"
        elif line.startswith("### 2.1 "):
            sec = "gemm"
        elif line.startswith("#"):
            sec = None
        elif sec and re.match(r"\|\s*\d+\s*\|", line):
            c = [x.strip() for x in line.strip("|").split("|")]
            try:
                out.setdefault(sec, {})[int(c[0])] = float(c[2] if sec == "megatron" else c[1])
            except (ValueError, IndexError):
                pass
    # §1.2's first table is the static Dell Cloud reference (790.4), not this host;
    # this host's own run is the "amd-ubuntu MI355X" row of the auto-generated §1.2a.
    m = re.search(r"\|\s*\**amd-ubuntu MI355X\**\s*\|\s*\**([\d.]+)\**\s*\|", txt)
    if m:
        out["megatron_ref"] = {8: float(m.group(1))}
    return out


PRIMUS_SECS = [("megatron", "Megatron-LM llama2-7B BF16 via Primus, compute TF/s/GPU"),
               ("gemm", "Primus GEMM microbench, mean TF/s/GPU"),
               ("megatron_ref", "megatron-ref GPT-15.6B (Megatron-LM v26.1), TF/s/GPU")]


def primus_frame(sec, with_cloud):
    nd = {n: primus_tables(RES[n] / "PRIMUS_REPORT.md") for n in NODES}
    c = (primus_tables(CLOUD / "PRIMUS_REPORT.md") or {}) if with_cloud else {}
    Ns = sorted(set(c.get(sec, {})) | {k for v in nd.values() if v for k in v.get(sec, {})})
    if not any(v and v.get(sec) for v in nd.values()):
        return None
    d = {"N": Ns}
    if with_cloud:
        d["cloud"] = [c.get(sec, {}).get(k) for k in Ns]
    for n in NODES:
        if nd[n]:
            d[n] = [nd[n].get(sec, {}).get(k) for k in Ns]
    return pd.DataFrame(d)


def primus_u(secs):
    L = []
    for sec, title in secs:
        df = primus_frame(sec, False)
        if df is None:
            continue
        have, rr = node_ratio_cols(df)
        L += [f"### {title}", ""] + table(df, ["N"], have, ratios=rr) + [""]
    return L or PENDING


def primus_c(secs):
    L = []
    for sec, title in secs:
        df = primus_frame(sec, True)
        if df is None:
            continue
        have = [n for n in NODES if n in df.columns]
        L += [f"### {title}", ""] + table(df, ["N"], ["cloud"] + have, ["amd-cloud"] + have, 1,
                                          [(f"{n[-4:]}/cloud", n, "cloud") for n in have]) + [""]
    return L or PENDING


# ---------------------------------------------------------------- Part D: ATOM tiers 1-2
ATOM_VALS = [("output_throughput", "Output tok/s", 0), ("median_ttft_ms", "Median TTFT (ms)", 1),
             ("median_tpot_ms", "Median TPOT (ms)", 2)]


def atom_u():
    L = []
    for val, title, nd in ATOM_VALS:
        df = nodes_frame("atom.csv", ["model", "max_concurrency"], val)
        if df is None:
            return PENDING
        have, rr = node_ratio_cols(df)
        L += [f"### {title}", ""] + table(df, ["model", "max_concurrency"], have, nd=nd, ratios=rr) + [""]
    return ["ATOM serving, ISL/OSL 1024/1024, image `atom-dev:nightly_202608111555`. Qwen3-8B-FP8 "
            "on 1 GPU (TP1), Llama-3.1-70B-FP8 on 8 GPUs (TP8). Kimi-K3 is in `kimi.md`.", ""] + L


def atom_c():
    c = load(CLOUD / "atom.csv", ["model", "max_concurrency"])
    L = []
    for val, title, nd in ATOM_VALS:
        df = nodes_frame("atom.csv", ["model", "max_concurrency"], val)
        if df is None:
            return PENDING
        df = c[["model", "max_concurrency", val]].rename(columns={val: "cloud"}).merge(df, how="right")
        df = df[df.model != "Kimi-K3"]
        have = [n for n in NODES if n in df.columns]
        L += [f"### {title}", ""] + table(df, ["model", "max_concurrency"], ["cloud"] + have,
                                          ["amd-cloud"] + have, nd,
                                          [(f"{n[-4:]}/cloud", n, "cloud") for n in have]) + [""]
    return ["Same image digest on both systems (`atom-dev:nightly_202608111555`, amd-cloud's "
            "`:latest` of 2026-08-14). Kimi-K3 is in `kimi.md`.", ""] + L


# ---------------------------------------------------------------- Kimi-K3
# ATOM experiments: (title, csv, keys, [(col, label, decimals)], row filter)
KIMI = [
    ("base: ATOM recipe, max-num-seqs 64", "atom.csv", ["max_concurrency"],
     [("output_throughput", "tok/s", 0), ("median_tpot_ms", "TPOT ms", 1)], lambda d: d[d.model == "Kimi-K3"]),
    ("maxseqs: max-num-seqs 256", "kimi-k3-maxseqs.csv", ["conc"], [("tps", "tok/s", 0), ("tpot", "TPOT ms", 1)], None),
    ("mad: MAD recipe, max-num-seqs 64", "kimi-k3-mad.csv", ["conc"], [("tps", "tok/s", 0), ("tpot", "TPOT ms", 1)], None),
    ("max-num-seqs 512", "kimi-k3-maxseqs512.csv", ["conc"], [("tps", "tok/s", 0), ("tpot", "TPOT ms", 1)], None),
    ("max-num-seqs 1024", "kimi-k3-maxseqs1024.csv", ["conc"], [("tps", "tok/s", 0), ("tpot", "TPOT ms", 1)], None),
    ("max-num-seqs 2048", "kimi-k3-maxseqs2048.csv", ["conc"], [("tps", "tok/s", 0), ("tpot", "TPOT ms", 1)], None),
    ("isl4096: ISL 4096 / OSL 1024", "kimi-k3-isl4096.csv", ["conc"], [("tps", "tok/s", 0), ("ttft", "TTFT ms", 0)], None),
    ("single_stream: latency arms", "kimi-k3-single-stream.csv", ["arm", "concurrency"],
     [("median_tpot_ms", "TPOT ms", 2), ("aggregate_tok_s", "tok/s", 1)], None),
]
# Experiments that cannot produce data, with the reason (shown instead of a silently missing table).
KIMI_NOTE = {
    "kimi-k3-maxseqs2048.csv":
        "*Not runnable on 8 × MI355X at TP8, on amd-cloud (2026-08-20) and amd-ubuntu (2026-10-02, "
        "2026-10-04) alike.* ATOM reserves Kimi-K3's KDA recurrent state (FP32) per sequence slot "
        "before the paged KV cache: 107 GiB per GPU for 2048 slots, against ~58 GiB left after the "
        "190 GiB of weights and activations at `--gpu-memory-utilization 0.93` (ATOM: \"would need "
        "1.10\"). max-num-seqs 1024 (54 GiB of state) is the largest power of two that fits.",
}
# 1024/1024 sweeps searched for the best ATOM result at a given concurrency.
ATOM_1K = [("atom.csv", "max_concurrency", "output_throughput", "median_tpot_ms", lambda d: d[d.model == "Kimi-K3"]),
           ("kimi-k3-maxseqs.csv", "conc", "tps", "tpot", None), ("kimi-k3-mad.csv", "conc", "tps", "tpot", None),
           ("kimi-k3-maxseqs512.csv", "conc", "tps", "tpot", None),
           ("kimi-k3-maxseqs1024.csv", "conc", "tps", "tpot", None),
           ("kimi-k3-maxseqs2048.csv", "conc", "tps", "tpot", None)]


def kimi_dir(s):
    """results/<node>/kimi-<s>/ with results (cloud runs on node6100, recipe on node6101)."""
    for n in NODES:
        p = RES[n] / f"kimi-{s}"
        if p.is_dir() and any(p.glob("*.csv")):
            return p
    return None


def kimi_frame(csv, keys, col, filt, srcs):
    """srcs: [(label, dir)] -> keys + one column per source."""
    base = None
    for lab, d in srcs:
        if d is None or not (d / csv).exists():
            continue
        df = pd.read_csv(d / csv)
        if filt is not None:
            df = filt(df)
        if col not in df.columns or not set(keys) <= set(df.columns):
            continue
        df = df.drop_duplicates(subset=keys, keep="last")[keys + [col]].rename(columns={col: lab})
        base = df if base is None else base.merge(df, on=keys, how="outer")
    return base


def best_atom(d):
    """Best ATOM output tok/s per concurrency over the 1024/1024 sweeps in d -> {C: (tps, tpot, sweep)}."""
    best = {}
    if d is None:
        return best
    for csv, kc, tc, pc, filt in ATOM_1K:
        if not (d / csv).exists():
            continue
        df = pd.read_csv(d / csv)
        df = filt(df) if filt is not None else df
        name = "base" if csv == "atom.csv" else csv[len("kimi-k3-"):-len(".csv")]
        for _, r in df.iterrows():
            c = int(r[kc])
            if pd.notna(r[tc]) and (c not in best or r[tc] > best[c][0]):
                best[c] = (r[tc], r[pc], name)
    return best


def recipe_rows(d):
    if d is None or not (d / "kimi-k3-recipe.csv").exists():
        return None
    return pd.read_csv(d / "kimi-k3-recipe.csv")


def recipe_vs_atom(rec, atom, lab):
    L = [f"| C | recipe tok/s | {lab} best tok/s (sweep) | tok/s ratio | recipe TPOT ms | {lab} TPOT ms | TPOT ratio |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in rec.iterrows():
        c = int(r.conc)
        a = atom.get(c)
        if a:
            L.append(f"| {c} | {num(r.tps, 0)} | {num(a[0], 0)} ({a[2]}) | {ratio(r.tps, a[0])} | "
                     f"{num(r.tpot, 2)} | {num(a[1], 2)} | {ratio(r.tpot, a[1])} |")
        else:
            L.append(f"| {c} | {num(r.tps, 0)} | — | — | {num(r.tpot, 2)} | — | — |")
    return L


def repeats_frame(d):
    if d is None or not (d / "kimi-k3-repeats.csv").exists():
        return None
    r = pd.read_csv(d / "kimi-k3-repeats.csv")
    return r.groupby("config").tok_s.agg(["mean", "std", "count"]).reset_index()


def kimi_u():
    dc, dr = kimi_dir("cloud"), kimi_dir("recipe")
    L = ["Kimi-K3 on 8 × MI355X, TP8, weights from node-local `/scratch/Kimi-K3`, ISL/OSL 1024/1024 "
         "unless noted. Two stacks:", "",
         f"- **ATOM** (`kimi-cloud`, {dc.parent.name if dc else 'not run yet'}): {ATOM_IMAGES}.",
         f"- **vLLM recipe** (`kimi-recipe`, {dr.parent.name if dr else 'not run yet'}): {RECIPE}.", ""]
    if not dc and not dr:
        return L + PENDING
    rec = recipe_rows(dr)
    L += ["### What improved: vLLM recipe vs best ATOM result at the same concurrency", ""]
    if rec is not None and dc:
        L += recipe_vs_atom(rec, best_atom(dc), "ATOM") + [
            "", "tok/s ratio above 1 and TPOT ratio below 1 favour the recipe; bold = more than 5%. "
            "ATOM has no run at C = 10, 12, 14, 44, 48, 70. The recipe also changes the stack "
            "(vLLM, ROCm 10.0 userspace), so this measures recipe and stack together.", ""]
    else:
        L += ["*Needs both the ATOM and the recipe runs.*", ""]
    if rec is not None:
        L += ["### vLLM recipe sweep", "", f"Detail: `{dr.relative_to(ROOT)}/kimi-k3-recipe.md`.", "",
              "| C | draft K | DCP | out tok/s | TPOT med ms | TTFT med ms | completed |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in rec.iterrows():
            done = key(r.completed) if pd.notna(r.completed) else "—"
            L.append(f"| {int(r.conc)} | {key(r.draft_k)} | {key(r.dcp)} | {num(r.tps, 0)} | "
                     f"{num(r.tpot, 2)} | {num(r.ttft, 0)} | {done} |")
        L.append("")
    if dc:
        L += ["### ATOM experiments", ""]
        for exp, csv, keys, vals, filt in KIMI:
            df, labs = None, []
            for col, lab, nd in vals:
                d2 = kimi_frame(csv, keys, col, filt, [(lab, dc)])
                if d2 is not None:
                    df = d2 if df is None else df.merge(d2, on=keys, how="outer")
                    labs.append(lab)
            if df is not None:
                L += [f"#### {exp}", ""] + table(df.sort_values(keys), keys, labs, nd=1) + [""]
            elif csv in KIMI_NOTE:
                L += [f"#### {exp}", "", KIMI_NOTE[csv], ""]
        r = repeats_frame(dc)
        if r is not None:
            L += ["#### repeats: run-to-run spread (tok/s, c=256)", "", "| config | mean | std | n |",
                  "|---|---:|---:|---:|"]
            L += [f"| {x.config} | {x.mean:,.0f} | {x.std:,.0f} | {x.count} |" for x in r.itertuples()] + [""]
        L += [f"Per-experiment detail (server logs, profiler, EP): `{dc.relative_to(ROOT)}/kimi-k3-*.md`."]
    return L


def kimi_c():
    dc, dr = kimi_dir("cloud"), kimi_dir("recipe")
    L = ["- **apple-to-apple**: amd-cloud vs amd-ubuntu ATOM with **the same image digests** "
         f"({ATOM_IMAGES}), same weights, flags and workload.",
         f"- **recipe**: amd-ubuntu with {RECIPE}, vs amd-cloud's best ATOM result at the same concurrency.", ""]
    if not dc and not dr:
        return L + PENDING
    rec = recipe_rows(dr)
    if rec is not None:
        L += ["### vLLM recipe (amd-ubuntu) vs best ATOM (amd-cloud)", ""]
        L += recipe_vs_atom(rec, best_atom(CLOUD), "amd-cloud ATOM") + [""]
    if dc:
        L += ["### Apple-to-apple: same images", ""]
        for exp, csv, keys, vals, filt in KIMI:
            parts = []
            for col, lab, nd in vals:
                df = kimi_frame(csv, keys, col, filt, [("amd-cloud", CLOUD), ("amd-ubuntu", dc)])
                if df is None or not {"amd-cloud", "amd-ubuntu"} <= set(df.columns):
                    continue
                parts += [f"**{lab}**", ""] + table(df.sort_values(keys), keys, ["amd-cloud", "amd-ubuntu"],
                                                    nd=nd, ratios=[("ubuntu/cloud", "amd-ubuntu", "amd-cloud")]) + [""]
            if parts:
                L += [f"#### {exp}", ""] + parts
            elif csv in KIMI_NOTE:
                L += [f"#### {exp}", "", KIMI_NOTE[csv], ""]
        c, u = repeats_frame(CLOUD), repeats_frame(dc)
        if c is not None and u is not None:
            def cell(r, cfg):
                m = r[r.config == cfg]
                return f"{m['mean'].iloc[0]:,.0f} ± {m['std'].iloc[0]:,.0f} (n={m['count'].iloc[0]})" if len(m) else "—"
            L += ["#### repeats: mean ± std tok/s (c=256)", "", "| config | amd-cloud | amd-ubuntu |", "|---|---:|---:|"]
            L += [f"| {cfg} | {cell(c, cfg)} | {cell(u, cfg)} |" for cfg in sorted(set(c.config) | set(u.config))]
    return L


# ---------------------------------------------------------------- status / index
STATUS = [("RVS gst TFLOPS", "rvs_tflops.csv"), ("RCCL single node", "rccl.csv"),
          ("RVS gst TFLOPS, ROCm 7.14", "rocm7.14/rvs_tflops.csv"), ("RCCL single node, ROCm 7.14", "rocm7.14/rccl.csv"),
          ("RCCL 2-node, ROCm 7.14", "rocm7.14/rccl_2node.csv"),
          ("RDMA per rail", "ib_bw.md"), ("RCCL 2-node", "rccl_2node.csv"),
          ("Primus / Megatron", "PRIMUS_REPORT.md"), ("ATOM tiers 1-2", "atom.csv"),
          ("Kimi-K3 ATOM (Aug-2026 images)", "kimi-cloud/images.txt"), ("Kimi-K3 vLLM recipe", "kimi-recipe/kimi-k3-recipe.csv")]


def status():
    L = ["| Benchmark | " + " | ".join(NODES) + " |", "|---|" + "---|" * len(NODES)]
    for label, f in STATUS:
        cells = []
        for n in NODES:
            p = RES[n] / f
            cells.append(f"✅ {datetime.datetime.fromtimestamp(p.stat().st_mtime):%m-%d %H:%M}"
                         if p.exists() else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    return L


def guard(fn, *a):
    try:
        return fn(*a)
    except Exception as e:   # one broken input must not hide the rest
        return [f"*Report failed: {type(e).__name__}: {e}*"]


def analysis(kind, f):
    """Hand-written conclusions kept outside the generated files (results/analysis/<kind>/<f>),
    so regenerating the tables never loses them. Placed before the tables."""
    p = ROOT / "analysis" / kind / f
    return [("Analysis", p.read_text().rstrip().splitlines())] if p.exists() else []


def main():
    U = [("rvs.md", "RVS gst TFLOPS", rvs_u, ()),
         ("rccl.md", "RCCL collectives, single node (XGMI)", rccl_u, ()),
         ("rocm.md", "Host ROCm 7.2.4 vs ROCm 7.14 (RVS, RCCL)", rocm_u, ()),
         ("net.md", "Network: RDMA per ionic rail", net_u, ()),
         ("rccl_2node.md", "RCCL across two nodes", rccl2_u, ()),
         ("primus.md", "Primus: Megatron-LM llama2-7B and GEMM microbench", primus_u, (PRIMUS_SECS[:2],)),
         ("megatron_ref.md", "Megatron-LM GPT-15.6B", primus_u, (PRIMUS_SECS[2:],)),
         ("atom.md", "ATOM LLM serving (Qwen3-8B, Llama-3.1-70B)", atom_u, ()),
         ("kimi.md", "Kimi-K3 serving: ATOM and the AMD vLLM recipe", kimi_u, ())]
    C = [("rvs.md", "RVS gst TFLOPS: amd-ubuntu vs amd-cloud", rvs_c, ()),
         ("rccl.md", "RCCL single node: amd-ubuntu vs amd-cloud", rccl_c, ()),
         ("primus.md", "Primus: amd-ubuntu vs amd-cloud", primus_c, (PRIMUS_SECS[:2],)),
         ("megatron_ref.md", "Megatron-LM GPT-15.6B: amd-ubuntu vs amd-cloud", primus_c, (PRIMUS_SECS[2:],)),
         ("atom.md", "ATOM serving: amd-ubuntu vs amd-cloud", atom_c, ()),
         ("kimi.md", "Kimi-K3: amd-ubuntu vs amd-cloud", kimi_c, ())]
    for f, title, fn, a in U:
        write(OUT_U / f, f"amd-ubuntu — {title}", SYSTEM_U, analysis("ubuntu", f) + [("Results", guard(fn, *a))])
    for f, title, fn, a in C:
        write(OUT_C / f, title, SYSTEM_C, analysis("vs-amd-cloud", f) + [("Results", guard(fn, *a))])
    links = lambda lst: [f"- [{t}]({f})" for f, t, *_ in lst]
    write(OUT_U / "README.md", "amd-ubuntu benchmark results", SYSTEM_U,
          [("Status", status()), ("Reports", links(U)),
           ("Raw", ["Per-node analyzer reports, CSVs and plots: `results/<node>/`; Kimi-K3 per image set: "
                    "`results/<node>/kimi-<set>/`; logs: `logs/<node>/`."])])
    write(OUT_C / "README.md", "amd-ubuntu vs amd-cloud", SYSTEM_C,
          [("Status", status()), ("Reports", links(C)),
           ("Not compared", ["RDMA per rail and 2-node RCCL: amd-cloud was a single node. See "
                             "[`../ubuntu/net.md`](../ubuntu/net.md), [`../ubuntu/rccl_2node.md`](../ubuntu/rccl_2node.md)."])])
    old = ROOT / "COMPARISON.md"
    if old.exists():
        old.unlink()   # superseded by vs-amd-cloud/
    print(f"wrote {OUT_U}/ and {OUT_C}/")


if __name__ == "__main__":
    sys.exit(main())
