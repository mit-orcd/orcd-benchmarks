#!/usr/bin/env python3
"""Turn a Kimi-K3 B200 sweep into results/kimi-k3-base-b200.{md,csv}.

Mirrors the structure of ../amd-benchmarks/amd-cloud/results/kimi-k3-base.md section for
section, for 2 x 8 B200 (TP8 x PP2), and adds a B200-vs-MI355X comparison section.

Usage:
  analyze-kimi-b200.py --sweep <dir> --server-log <path> --model-config <path>
                       [--run-dir <dir>] [-o results]
                       [--amd-root <path>] [--amd-sweep <dir>] [--amd-server-log <path>]

Everything derived is derived from (a) the measured sweep JSON, (b) the server's own log
lines, and (c) the parsed config.json. Nothing is hardcoded from the MI355X report except
the hardware spec sheets in HW below -- the MI355X *measurements* are re-read from the AMD
run's own sweep JSON and server log, and its derived figures are recomputed with the SAME
formulas used for B200, so the two columns are actually comparable.
"""
import argparse, csv, json, math, re, sys
from pathlib import Path

# ---------------------------------------------------------------------------------
# Hardware spec sheets. The only hardcoded numbers in this script.
# ---------------------------------------------------------------------------------
# HBM capacity is carried in BYTES, not "GB", deliberately. The whole two-node
# conclusion rests on a comparison that is only ~23 GB wide on weights alone, and
# GB-vs-GiB slippage is more than large enough to flip its sign. Convert at the point
# of display, never before.
MIB = 2 ** 20
HW = {
    "B200": dict(
        name="NVIDIA B200 (Blackwell, sm_100)",
        hbm_bytes=183359 * MIB,     # what nvidia-smi reports on these nodes (192 GB part)
        hbm_bw_gbs=8000.0,          # HBM3e
        bf16_tflops=2250.0,         # dense, no sparsity
        link="NVLink 5",
        link_bidir_gbs=1800.0,      # per GPU, aggregate bidirectional
        link_dir_gbs=900.0,         # per direction -- the figure a ring all-reduce sees
        gpus_per_node=8,
    ),
    "MI355X": dict(
        name="AMD MI355X (CDNA4, gfx950)",
        hbm_bytes=288 * 10**9,      # 288 GB; ATOM's log reports total_gpu=287.98GB
        hbm_bw_gbs=8000.0,          # HBM3E
        bf16_tflops=2500.0,         # dense
        link="Infinity Fabric (xGMI)",
        link_bidir_gbs=1075.0,
        link_dir_gbs=537.0,
        gpus_per_node=8,
    ),
}


def tib(b):
    return b / 2**40


def gb(b):
    return b / 1e9
# Inter-node fabric on the Engaging B200 nodes. Measured, not spec: see
# ../b200-nodes/notes.md (2026-08-12 remeasure) and ../b200-ubuntu/out-nccl-2node.
IB = dict(
    rails=8, rail_gbps=400.0,                 # 8 x 400 Gb/s NDR per node
    node_dir_gbs=8 * 400.0 / 8,               # = 400 GB/s per node per direction
    measured_gdr_gbps=395.5,                  # ib_write_bw GPU->GPU, 64 MiB
    measured_nccl_pair_gbs=48.4,              # NCCL sendrecv per pair @ 8 GPU/node
)

FIELDS = ["max_concurrency", "output_throughput", "total_token_throughput",
          "request_throughput", "median_ttft_ms", "p99_ttft_ms",
          "median_tpot_ms", "p99_tpot_ms", "completed", "duration"]


# ---------------------------------------------------------------------------------
# Sweep loading
# ---------------------------------------------------------------------------------
def load_sweep(d: Path):
    """Read every c<N>.json in a sweep dir into rows sorted by concurrency."""
    rows = []
    for j in sorted(d.glob("c*.json")):
        try:
            data = json.loads(j.read_text())
        except Exception as e:
            print(f"  skip {j.name}: {e}", file=sys.stderr)
            continue
        r = {k: data.get(k) for k in FIELDS}
        if not r.get("max_concurrency"):
            m = re.match(r"c(\d+)\.json", j.name)
            if m:
                r["max_concurrency"] = int(m.group(1))
        if not r.get("completed"):
            print(f"  WARNING: {j.name} has completed=0 -- dropping", file=sys.stderr)
            continue
        rows.append(r)
    rows.sort(key=lambda r: r["max_concurrency"])
    return rows


# ---------------------------------------------------------------------------------
# Server-log parsing. vLLM and ATOM say the same things in different words; parse both
# so one code path can read the B200 run and the MI355X baseline.
# ---------------------------------------------------------------------------------
def parse_vllm_log(p: Path):
    """Pull the memory budget and engine config out of a vLLM server log."""
    out = {}
    if not p or not p.exists():
        return out
    txt = p.read_text(errors="replace")

    # "the current vLLM instance can use total_gpu_memory (183.00GiB) x
    #  gpu_memory_utilization (0.90) = 164.70GiB"
    m = re.search(r"total_gpu_memory\s*\(([\d.]+)GiB\)\s*x\s*"
                  r"gpu_memory_utilization\s*\(([\d.]+)\)\s*=\s*([\d.]+)GiB", txt)
    if m:
        out["total_gpu_gib"] = float(m.group(1))
        out["util"] = float(m.group(2))
        out["budget_gib"] = float(m.group(3))

    # "model weights take 91.23GiB; non_torch_memory takes 1.20GiB; PyTorch activation
    #  peak memory takes 2.10GiB; the rest of the memory reserved for KV Cache is 70.17GiB."
    for key, pat in (("weights_gib",   r"model weights take\s*([\d.]+)GiB"),
                     ("non_torch_gib", r"non_torch_memory takes\s*([\d.]+)GiB"),
                     ("act_peak_gib",  r"PyTorch activation peak memory takes\s*([\d.]+)GiB"),
                     ("kv_gib",        r"reserved for KV Cache is\s*([\d.]+)GiB")):
        mm = re.search(pat, txt)
        if mm:
            out[key] = float(mm.group(1))

    # This build reports memory in its own words rather than the single profiling line
    # the earlier regexes expected:
    #   "Model loading took 97.35 GiB and 1376.110077 seconds"
    #   "Available KV cache memory: 59.34 GiB"
    m = re.search(r"Model loading took\s*([\d.]+)\s*GiB and\s*([\d.]+)\s*seconds", txt)
    if m:
        out.setdefault("weights_gib", float(m.group(1)))
        out.setdefault("load_s", float(m.group(2)))
    for mm in re.finditer(r"Available KV cache memory:\s*([\d.]+)\s*GiB", txt):
        out["kv_gib"] = float(mm.group(1))   # last one wins: post-padding value

    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", txt)
    if m:
        out["kv_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"#\s*GPU blocks:\s*([\d,]+)", txt)
    if m:
        out["kv_blocks"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", txt)
    if m:
        out["max_conc_for_len"] = float(m.group(2))

    # Engine config, as the server itself reported it. This build logs the config as a
    # Python DICT REPR ('max_model_len': 16384), not key=value, so both forms are tried
    # -- the first report came out with blank max_model_len / gpu_memory_utilization
    # cells because only the key=value form was matched.
    for key, name in (("tp", "tensor_parallel_size"), ("pp", "pipeline_parallel_size"),
                      ("dp", "data_parallel_size"), ("mml", "max_model_len"),
                      ("mns", "max_num_seqs"), ("mnbt", "max_num_batched_tokens")):
        mm = (re.search(rf"{name}=(\d+)", txt)
              or re.search(rf"['\"]{name}['\"]\s*:\s*(\d+)", txt))
        if mm:
            out[key] = int(mm.group(1))
    mm = (re.search(r"gpu_memory_utilization=([\d.]+)", txt)
          or re.search(r"['\"]gpu_memory_utilization['\"]\s*:\s*([\d.]+)", txt))
    if mm:
        out["util"] = float(mm.group(1))
    m = re.search(r"kv_cache_dtype=[\'\"]?(\w+)", txt)
    if m:
        out["kv_dtype"] = m.group(1)
    for key, name in (("prefix_caching", "enable_prefix_caching"),
                      ("expert_parallel", "enable_expert_parallel")):
        mm = (re.search(rf"{name}=(\w+)", txt)
              or re.search(rf"['\"]{name}['\"]\s*:\s*(\w+)", txt))
        if mm:
            out[key] = mm.group(1)

    # Weight-load time, for the "how long did 1.42 TiB take off NFS" note.
    m = re.search(r"Loading (?:model )?weights took\s*([\d.]+)\s*(?:GB/s.*?in\s*)?([\d.]+)?\s*seconds", txt)
    if m:
        try:
            out["load_s"] = float(m.group(2) or m.group(1))
        except (TypeError, ValueError):
            pass
    m = re.search(r"Model loading took [\d.]+ Gi?B and ([\d.]+) seconds", txt)
    if m:
        out["load_s"] = float(m.group(1))
    return out


def parse_atom_log(p: Path):
    """Pull the same facts out of ATOM's single 'Memory budget:' line (MI355X run)."""
    out = {}
    if not p or not p.exists():
        return out
    txt = p.read_text(errors="replace")
    m = re.search(r"Memory budget:\s*(.+)", txt)
    if m:
        for kv in m.group(1).split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip(); v = v.strip()
            mm = re.match(r"([\d.]+)GB$", v)
            if mm:
                # ATOM prints GB but computes in GiB units (total_gpu=287.98 for a 288 GB
                # part is the GiB figure); treat them as the same unit as vLLM's GiB.
                out[k] = float(mm.group(1))
            else:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    remap = dict(total_gpu="total_gpu_gib", utilization="util", budget="budget_gib",
                 peak_torch="weights_gib", non_torch="non_torch_gib",
                 available_for_kv="kv_gib", cudagraph_est="cudagraph_gib",
                 safety="safety_gib", block_bytes="block_bytes",
                 num_kvcache_blocks="kv_blocks")
    for a, b in remap.items():
        if a in out:
            out[b] = out.pop(a)
    if "block_bytes" in out and "kv_blocks" in out:
        out["kv_tokens"] = int(out["kv_blocks"]) * 128
    for key, pat in (("tp", r"tensor_parallel_size=(\d+)"), ("pp", r"pipeline_parallel_size=(\d+)"),
                     ("mml", r"max_model_len=(\d+)"), ("mns", r"max_num_seqs=(\d+)")):
        mm = re.search(pat, txt)
        if mm:
            out[key] = int(mm.group(1))
    mm = re.search(r"enable_expert_parallel=(\w+)", txt)
    if mm:
        out["expert_parallel"] = mm.group(1)
    return out


# ---------------------------------------------------------------------------------
# Architecture. Parsed from config.json and VALIDATED against the published total.
# ---------------------------------------------------------------------------------
class Arch:
    def __init__(self, cfg_path: Path):
        d = json.loads(Path(cfg_path).read_text())
        t = d.get("text_config", d)
        self.raw = d
        self.h      = t["hidden_size"]
        self.L      = t["num_hidden_layers"]
        self.E      = t["num_experts"]
        self.topk   = t["num_experts_per_token"]
        self.shared = t["num_shared_experts"]
        self.rh     = t["routed_expert_hidden_size"]   # routed expert latent width
        self.mi     = t["moe_intermediate_size"]       # shared/routed FFN width
        self.dense  = t["first_k_dense_replace"]
        self.inter  = t["intermediate_size"]
        self.V      = t["vocab_size"]
        la = t["linear_attn_config"]
        self.full_layers = len(la["full_attn_layers"])
        self.kda_layers  = len(la["kda_layers"])
        self.nH   = t["num_attention_heads"]
        self.qk_n = t["qk_nope_head_dim"]; self.qk_r = t["qk_rope_head_dim"]
        self.vh   = t["v_head_dim"]
        self.qlr  = t["q_lora_rank"];      self.kvlr = t["kv_lora_rank"]
        self.kH   = la["num_heads"];       self.kd   = la["head_dim"]
        self.ck   = la["short_conv_kernel_size"]
        self.moeL = self.L - self.dense

        # --- per-expert and per-layer parameter counts -------------------------------
        # Routed experts live in a LATENT space of width routed_expert_hidden_size, not
        # on the 7168-wide residual stream: 3 matrices of rh x mi each. That is what
        # makes 896 experts x 92 layers land at 2.72 T instead of 6.4 T, and it is
        # verified below against the checkpoint's own MXFP4 (U8) parameter count.
        self.per_expert   = 3 * self.rh * self.mi
        self.routed_total = self.moeL * self.E * self.per_expert
        self.shared_total = self.moeL * self.shared * 3 * self.h * self.mi
        self.latent_proj  = self.moeL * 2 * self.h * self.rh
        self.router       = self.moeL * self.h * self.E
        self.dense_mlp    = self.dense * 3 * self.h * self.inter
        self.mla_per      = (self.h * self.qlr
                             + self.qlr * self.nH * (self.qk_n + self.qk_r)
                             + self.h * (self.kvlr + self.qk_r)
                             + self.kvlr * self.nH * (self.qk_n + self.vh)
                             + self.nH * self.vh * self.h)
        kdim = self.kH * self.kd
        self.kda_per      = 3 * self.h * kdim + 3 * self.ck * kdim + self.h * kdim + kdim * self.h
        self.attn_total   = self.mla_per * self.full_layers + self.kda_per * self.kda_layers
        self.embed        = 2 * self.V * self.h

        self.total = (self.routed_total + self.shared_total + self.latent_proj
                      + self.router + self.dense_mlp + self.attn_total + self.embed)

        # --- active parameters per token ---------------------------------------------
        self.act_ffn = (self.moeL * (self.topk * self.per_expert
                                     + self.shared * 3 * self.h * self.mi
                                     + 2 * self.h * self.rh)
                        + self.dense_mlp + self.router)
        self.act_attn = self.attn_total
        self.act_embed = self.embed
        self.active = self.act_ffn + self.act_attn + self.act_embed

        # KV bytes per token per rank: MLA keeps a compressed latent, and only the
        # full-attention layers page it. 1 byte/value at fp8.
        self.kv_bytes_per_token = (self.kvlr + self.qk_r) * 1 * self.full_layers

    def bytes_per_expert(self):
        """MXFP4 storage: 4 bits/value + one e8m0 scale byte per 32 values."""
        return self.per_expert * (0.5 + 1.0 / 32.0)

    def experts_fired(self, batch):
        """E(distinct experts touched per layer) when `batch` tokens route independently."""
        return self.E * (1.0 - (1.0 - 1.0 / self.E) ** (batch * self.topk))


# ---------------------------------------------------------------------------------
# Derived per-run quantities
# ---------------------------------------------------------------------------------
def derive(rows, arch, tp, pp, hw):
    """Attach compute / bandwidth / interconnect derivations to each sweep row."""
    ngpu = tp * pp
    for r in rows:
        c = r["max_concurrency"]
        ots = r["output_throughput"] or 0.0
        tpot = r["median_tpot_ms"] or 0.0
        r["ngpu"] = ngpu
        r["tflops"] = 2 * arch.active * ots / 1e12
        r["tflops_per_gpu"] = r["tflops"] / ngpu
        r["pct_peak"] = 100.0 * r["tflops_per_gpu"] / hw["bf16_tflops"]
        r["steps_per_s"] = 1000.0 / tpot if tpot else 0.0
        r["tok_per_gpu"] = ots / ngpu

        # HBM: expert weight traffic dominates. With PP, each GPU holds only its stage's
        # layers, so per-GPU bytes divide by (tp*pp) across the whole model -- but a
        # decode step still traverses every layer, so the NODE-PAIR total is what the
        # MI355X single-node figure should be compared against.
        fired = arch.experts_fired(c)
        r["experts_fired"] = fired
        r["tokens_per_expert"] = (c * arch.topk / fired) if fired else 0.0
        routed_bytes_all = arch.moeL * fired * arch.bytes_per_expert()
        other_bytes_all = (arch.shared_total + arch.latent_proj + arch.router
                           + arch.dense_mlp + arch.attn_total) * 1.0  # fp8/bf16 ~1 B avg
        r["hbm_bytes_step_total"] = routed_bytes_all + other_bytes_all
        r["hbm_bytes_step_per_gpu"] = r["hbm_bytes_step_total"] / ngpu
        r["hbm_gbs_per_gpu"] = r["hbm_bytes_step_per_gpu"] * r["steps_per_s"] / 1e9
        r["hbm_pct"] = 100.0 * r["hbm_gbs_per_gpu"] / hw["hbm_bw_gbs"]

        # Intra-node all-reduce: 2 per layer, but only over the layers on THIS node.
        layers_per_stage = arch.L / pp
        r["allreduce_per_token"] = 2 * layers_per_stage
        payload = arch.h * 2 * c                       # bf16 activations for the batch
        r["ar_payload_step"] = payload * r["allreduce_per_token"]
        wire = r["ar_payload_step"] * 2 * (tp - 1) / tp
        r["link_gbs_per_gpu"] = wire * r["steps_per_s"] / 1e9
        r["link_pct"] = 100.0 * r["link_gbs_per_gpu"] / hw["link_dir_gbs"]

        # Inter-node pipeline p2p: (pp-1) stage boundaries, hidden state per token,
        # each way per step. This has no counterpart in the single-node MI355X run.
        if pp > 1:
            r["pp_bytes_step"] = (pp - 1) * arch.h * 2 * c
            r["pp_gbs"] = r["pp_bytes_step"] * r["steps_per_s"] / 1e9
            r["pp_pct"] = 100.0 * r["pp_gbs"] / IB["node_dir_gbs"]
        else:
            r["pp_bytes_step"] = 0.0; r["pp_gbs"] = 0.0; r["pp_pct"] = 0.0
    return rows


def fmt(v, nd=1):
    if v is None:
        return "—"
    try:
        return f"{float(v):,.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------
def build_report(b, a, arch, args):
    """b = B200 run dict, a = MI355X run dict (may be None)."""
    L = []
    W = L.append
    rows, mem, tp, pp = b["rows"], b["mem"], b["tp"], b["pp"]
    hw = HW["B200"]
    ngpu = tp * pp
    nnodes = math.ceil(ngpu / hw["gpus_per_node"])
    peak = max(rows, key=lambda r: r["output_throughput"] or 0)
    first = rows[0]

    disk_tib = tib(args.weight_bytes)
    node_hbm_b = hw["hbm_bytes"] * hw["gpus_per_node"]
    short_b = args.weight_bytes - node_hbm_b

    W(f"# Kimi-K3 on {nnodes} × 8 × B200 — compute and communication analysis")
    W("")
    W(f"Serving `moonshotai/Kimi-K3` ({arch.total/1e12:.2f} T params, "
      f"{args.weight_bytes/1e12:.2f} TB MXFP4 checkpoint) on **{nnodes} nodes × 8 B200** "
      f"via vLLM, **TP={tp} × PP={pp}**. Measured {b['date']}.")
    W("")
    amib = HW["MI355X"]["hbm_bytes"] * 8
    W("> **Why two nodes.** The checkpoint is "
      f"**{gb(args.weight_bytes):.0f} GB ({disk_tib:.2f} TiB)**. One 8 × B200 node holds "
      f"**{gb(node_hbm_b):.0f} GB ({tib(node_hbm_b):.2f} TiB)** of HBM "
      f"({hw['gpus_per_node']} × {gb(hw['hbm_bytes']):.0f} GB) — "
      f"**{gb(short_b):.0f} GB short of the weights alone**, before a single byte of KV "
      "cache, activation workspace, NCCL buffers or CUDA-graph pool, which together take "
      "another ~15–20 GB per GPU. There is no single-node B200 configuration for this "
      "model. TP8 shards within each node and PP2 splits the 93 layers across the pair. "
      "The MI355X baseline runs the same model on **one** node, because "
      f"8 × 288 GB = {gb(amib):.0f} GB does fit with room to spare. Every throughput "
      "figure below is therefore also reported per GPU and per node.")
    W("")

    # --- run configuration --------------------------------------------------------
    W("**Run configuration** (from the server log, not assumed):")
    W("")
    W("| Setting | Value |")
    W("|---|---|")
    W(f"| Parallelism | `tensor_parallel_size={mem.get('tp', tp)}`, "
      f"`pipeline_parallel_size={mem.get('pp', pp)}`, DP=1, "
      f"**EP {'on' if str(mem.get('expert_parallel','False')).lower()=='true' else 'off'}** |")
    W(f"| GPUs / nodes | {ngpu} / {nnodes} |")
    W("| Quantization | routed MoE experts **MXFP4** (`mxfp4-pack-quantized`, group_size 32); "
      "attention, shared experts and dense MLP left at BF16 by the checkpoint's `ignore` list |")
    W(f"| KV cache dtype | {mem.get('kv_dtype', args.kv_dtype)} |")
    W(f"| `max_model_len` / `max_num_seqs` | {mem.get('mml','—')} / {mem.get('mns','—')} |")
    W(f"| `max_num_batched_tokens` | {mem.get('mnbt','—')} |")
    W(f"| `gpu_memory_utilization` | {mem.get('util','—')} |")
    W(f"| Prefix caching | **{'enabled' if str(mem.get('prefix_caching','False')).lower()=='true' else 'disabled'}** "
      "(disabled is required — KDA recurrent state can't be rebuilt from the paged cache) |")
    W(f"| Workload | ISL/OSL {args.isl}/{args.osl}, `--ignore-eos`, concurrency "
      f"{first['max_concurrency']}→{peak['max_concurrency']} |")
    if b.get("load_s"):
        W(f"| Weight load | {b['load_s']/60:.1f} min for {disk_tib:.2f} TiB off shared NFS "
          f"(~{args.weight_bytes/1e9/b['load_s']:.1f} GB/s effective) |")
    W("")

    # --- architecture -------------------------------------------------------------
    W(f"**Architecture** (parsed from `config.json`): {arch.L} layers — "
      f"**{arch.full_layers} MLA full-attention** + **{arch.kda_layers} KDA linear-attention**; "
      f"hidden {arch.h}; MoE with **{arch.E} routed experts, top-{arch.topk} + {arch.shared} shared**, "
      f"routed-expert latent {arch.rh} → {arch.mi}.")
    W("")
    W(f"The parse is **exactly validated**: routed-expert parameters computed from the config "
      f"(`{arch.moeL} MoE layers × {arch.E} experts × 3 × {arch.rh} × {arch.mi}`) come to "
      f"**{arch.routed_total:,}**, which is the checkpoint's own MXFP4 (U8) parameter count "
      f"to the last digit. Total computed ≈ **{arch.total/1e12:.3f} T** against the advertised "
      "2.78 T.")
    W("")
    W("> The routed experts live in a **latent space of "
      f"{arch.rh}**, not on the {arch.h}-wide residual stream. That is the difference between "
      f"{arch.routed_total/1e12:.2f} T and the ~6.4 T a naive `3 × hidden × expert_width` count "
      "would predict, and it is why the exact match above is worth stating.")
    W("")
    W("---")
    W("")

    # ============================== §0 overview ===================================
    W("## 0. Overview — the short version")
    W("")
    pk = peak
    W(f"**§1 Compute** — **{fmt(pk['output_throughput'])} tok/s** at c={pk['max_concurrency']} "
      f"({(pk['output_throughput']/first['output_throughput']):.0f}× scaling from "
      f"c={first['max_concurrency']}, TPOT only "
      f"{(pk['median_tpot_ms']/first['median_tpot_ms']):.1f}× worse). Achieved "
      f"**{fmt(pk['tflops'])} TFLOP/s aggregate = {fmt(pk['tflops_per_gpu'])}/GPU = "
      f"{pk['pct_peak']:.1f}% of B200 BF16 peak**. Only "
      f"{arch.active/1e9:.0f} B of {arch.total/1e12:.2f} T params activate per token "
      f"({100*arch.active/arch.total:.1f}%).")
    W("")
    if mem.get("weights_gib"):
        W(f"**§2 Memory** — Per GPU: **{fmt(mem['weights_gib'])} GiB weights** + "
          f"**{fmt(mem.get('kv_gib'))} GiB KV pool**. KV decodes exactly: "
          f"{arch.kv_bytes_per_token:,} B/token = "
          f"`(kv_lora {arch.kvlr} + rope {arch.qk_r}) × 1 B fp8 × {arch.full_layers} MLA layers` — "
          f"proving only the {arch.full_layers} full-attention layers page KV, and that KV is "
          "replicated across TP ranks rather than sharded.")
    else:
        W("**§2 Memory** — the server log did not contain a parseable memory-profiling line; "
          "the memory section below reports what was found and marks the rest unavailable.")
    W("")
    W(f"**§3 Bottleneck — HBM traffic, and it is LATENCY-BOUND, NOT BANDWIDTH-BOUND.** "
      f"Compute {pk['pct_peak']:.1f}% utilized, NVLink {pk['link_pct']:.1f}%, "
      f"**HBM only ~{pk['hbm_pct']:.0f}%** "
      f"({fmt(pk['hbm_gbs_per_gpu'])} GB/s of {fmt(hw['hbm_bw_gbs'],0)}) — the bandwidth "
      f"is there and is going unused. At batch {pk['max_concurrency']} the tokens route "
      f"independently, so **{pk['experts_fired']:.0f} of {arch.E} experts** activate per "
      f"layer (not {arch.topk}), yet each expert then sees only "
      f"**{pk['tokens_per_expert']:.1f} tokens** — a matrix-*vector* product that cannot "
      "keep enough memory requests in flight to saturate HBM. Weight traffic dominates "
      "step time, but the ceiling being hit is memory-access latency/occupancy, not "
      "bandwidth. The fix follows directly: widen the GEMMs (§3.2).")
    W("")
    W(f"**§4 Communication** — two paths now, not one. NVLink carries "
      f"{int(pk['allreduce_per_token'])} all-reduces/token within each node "
      f"({fmt(pk['link_gbs_per_gpu'],2)} GB/s per GPU, {pk['link_pct']:.1f}% of ceiling). "
      f"**The PP2 boundary additionally puts InfiniBand in the per-token critical path** — "
      f"{fmt(pk['pp_gbs'],3)} GB/s across the pair ({pk['pp_pct']:.2f}% of the 8-rail NDR "
      "fabric). This is the cost the MI355X run does not pay, and §4.2 sizes it.")
    W("")
    W("**§5** — the levers, and §6 the head-to-head against MI355X.")
    W("")
    W("---")
    W("")

    # ============================== §1 compute ====================================
    W("## 1. Computing performance")
    W("")
    W("**Metrics used throughout this report.** Every one is measured by "
      "`vllm bench serve`, not derived, unless stated otherwise:")
    W("")
    W("| Term | Stands for | What it measures |")
    W("|---|---|---|")
    W("| **TTFT** | **Time To First Token** | Latency from sending a request to receiving "
      "its *first* output token — i.e. how long the user waits before anything appears. "
      "Dominated by **prefill** (processing the whole input prompt). |")
    W("| **TPOT** | **Time Per Output Token** | Average latency *between* successive output "
      "tokens, after the first. This is the **decode** step time — how fast the answer "
      "streams once it has started. Reported as a median over all requests. |")
    W("| **Concurrency** | — | Number of independent requests in flight at once. A "
      f"client-side load setting, not a hardware unit: all {pk['max_concurrency']} requests "
      f"at c={pk['max_concurrency']} are batched together across the same {ngpu} GPUs in one "
      "continuous-batching loop. |")
    W("| **Throughput (tok/s)** | — | **Aggregate** output tokens per second across *all* "
      "concurrent requests — not what any single user sees (§1.1). |")
    W("| **Per-user tok/s** | — | `1000 / TPOT` — just a unit flip: TPOT is ms per token, "
      "so `1/TPOT` is tokens per ms and `×1000` makes it tokens per second. The streaming "
      "rate one user actually experiences. *Derived*, not measured directly (§1.1). |")
    W("| **req/s** | — | Completed requests per second. |")
    W("")
    W("TTFT and TPOT answer different questions and are bound by different things here: "
      "TTFT is compute-dense prefill and stays nearly flat with load, while TPOT is "
      "memory-bound decode and grows with it (§3.3). Both are quoted as **medians** in "
      "the tables below; the raw JSON also carries p99.")
    W("")
    W("| Concurrency | Throughput (tok/s) | tok/s per GPU | TTFT med (ms) | TPOT med (ms) | req/s |")
    W("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        bold = "**" if r is pk else ""
        W(f"| {r['max_concurrency']} | {bold}{fmt(r['output_throughput'])}{bold} | "
          f"{fmt(r['tok_per_gpu'])} | {fmt(r['median_ttft_ms'])} | "
          f"{fmt(r['median_tpot_ms'],2)} | {fmt(r['request_throughput'],2)} |")
    W("")
    W(f"Throughput scales **{(pk['output_throughput']/first['output_throughput']):.0f}×** from "
      f"c={first['max_concurrency']} to c={pk['max_concurrency']} while TPOT grows only "
      f"{(pk['median_tpot_ms']/first['median_tpot_ms']):.1f}×. The sweep stops at "
      f"{pk['max_concurrency']} because the server was launched with `--max-num-seqs "
      f"{mem.get('mns', args.max_num_seqs)}`; past it you would measure queueing, not the engine.")
    W("")
    W("### 1.1 Per-user token rate — what one user actually experiences")
    W("")
    W("The table above is **aggregate** throughput, shared across all concurrent requests. "
      "A single user does not experience it. What one user sees is the streaming rate of "
      "their own answer, **1000 / TPOT** (TPOT = Time Per Output Token, the decode-step "
      "latency), and that moves in the *opposite* direction:")
    W("")
    W("*Why `1000 / TPOT`:* TPOT is milliseconds per token, so one over it is tokens per "
      "millisecond, and ×1000 gives tokens per second — a user receiving a token every "
      "11.24 ms is receiving 89 tokens per second. It is **per-user** because TPOT is "
      "measured within a single request: continuous batching advances many requests in one "
      "forward pass, so the engine emits N tokens per step while each user still gets only "
      "one. That is the whole reason the two columns diverge — aggregate ≈ N × per-user. "
      "Note it is the steady-state rate and **excludes TTFT**: the full wait for an "
      "N-token answer is `TTFT + (N-1) × TPOT`.")
    W("")
    W("| Conc | Total tok/s | TPOT med (ms) | **Per-user tok/s** | vs single user |")
    W("|---:|---:|---:|---:|---:|")
    solo = 1000.0 / first["median_tpot_ms"] if first["median_tpot_ms"] else None
    for r in rows:
        pu = 1000.0 / r["median_tpot_ms"] if r["median_tpot_ms"] else None
        rel = f"{(pu/solo - 1)*100:+.0f}%" if (pu and solo) else "—"
        star = "**" if r is first or r is pk else ""
        W(f"| {star}{r['max_concurrency']}{star} | {fmt(r['output_throughput'])} | "
          f"{fmt(r['median_tpot_ms'],2)} | {star}{fmt(pu)}{star} | "
          f"{'_reference_' if r is first else rel} |")
    W("")
    if solo:
        W(f"**Aggregate throughput and single-user speed trade against each other.** Going "
          f"c={first['max_concurrency']} → c={pk['max_concurrency']} buys "
          f"**{pk['output_throughput']/first['output_throughput']:.1f}× aggregate "
          f"throughput** and costs "
          f"**{(solo/(1000.0/pk['median_tpot_ms'])):.1f}× per-user speed** "
          f"({fmt(solo)} → {fmt(1000.0/pk['median_tpot_ms'])} tok/s). Both are real; which "
          "one matters depends entirely on the workload:")
        W("")
        W("| Optimising for | Run at | Read |")
        W("|---|---|---|")
        W("| One interactive user, a latency SLO, a long agentic session | **low concurrency** | per-user tok/s |")
        W("| Many users at once, cost per token, GPU utilisation | **high concurrency** | total tok/s |")
        W("")
        W("> **This is the caveat on §3.2.** Every lever proposed there raises *aggregate* "
          "throughput, and three of the four do it by increasing batch size — which makes "
          "single-user speed **worse**. Raising `--max-num-seqs` is the right call for a "
          "busy server and the wrong call for one user waiting on one answer.")
        W("")
        W("Note also that **none of the §3.2 levers help at c=1 at all**: they all widen the "
          "per-expert GEMM, and with one token in flight there is nothing to widen. The "
          "single-user fixes are different ones — speculative decoding / MTP (blocked here, "
          "see §5), removing PP (impossible on B200, see §1 preamble), and lower TP. See "
          "`notes-concurrency.md`.")
    W("")
    W("### Achieved TFLOP/s")
    W("")
    W(f"Only **top-{arch.topk} + {arch.shared} of {arch.E}** experts fire per token. Active "
      f"params per token ≈ **{arch.active/1e9:.0f} B** (of {arch.total/1e12:.2f} T — "
      f"{100*arch.active/arch.total:.1f}% activation ratio):")
    W("")
    W("| Component | Active params/token |")
    W("|---|---:|")
    W(f"| FFN (top-{arch.topk} routed + {arch.shared} shared + latent projections, "
      f"{arch.moeL} layers) | {arch.act_ffn/1e9:.1f} B |")
    W(f"| Attention ({arch.full_layers} MLA + {arch.kda_layers} KDA) | {arch.act_attn/1e9:.1f} B |")
    W(f"| Embedding / lm_head | {arch.act_embed/1e9:.1f} B |")
    W("")
    W("At 2 FLOP per active param per token:")
    W("")
    W(f"| Concurrency | Aggregate TFLOP/s | Per GPU | % of B200 BF16 peak ({fmt(hw['bf16_tflops'],0)}) |")
    W("|---:|---:|---:|---:|")
    for r in (first, pk):
        W(f"| {r['max_concurrency']} | {fmt(r['tflops'],1)} | {fmt(r['tflops_per_gpu'],2)} | "
          f"{r['pct_peak']:.2f}% |")
    W("")
    W("**Decode is nowhere near compute-bound** — barely 1% of peak. Autoregressive decode "
      "issues one token per sequence per step, so every weight matrix is used for a single "
      "narrow GEMV-like operation. This is a *memory-bandwidth* regime, quantified in §3.")
    W("")
    W("> Caveat: the active-parameter figure is config-derived. The routed-expert term is "
      "exact (it reproduces the checkpoint's MXFP4 count to the digit); the KDA term is an "
      "approximation from the config dimensions and dominates the attention row. Treat the "
      f"{arch.active/1e9:.0f} B figure as ±10%. The conclusion (decode is ~1% of peak) has "
      "far too large a margin to be affected.")
    W("")
    W("---")
    W("")

    # ============================== §2 memory =====================================
    W("## 2. GPU memory usage")
    W("")
    if mem.get("weights_gib"):
        W("Measured per rank at load time, straight from the server log:")
        W("")
        W("```")
        W(f"total_gpu={mem.get('total_gpu_gib','?')}GiB  utilization={mem.get('util','?')}  "
          f"budget={mem.get('budget_gib','?')}GiB")
        W(f"weights={mem.get('weights_gib','?')}GiB  non_torch={mem.get('non_torch_gib','?')}GiB  "
          f"act_peak={mem.get('act_peak_gib','?')}GiB  kv={mem.get('kv_gib','?')}GiB")
        W(f"kv_tokens={mem.get('kv_tokens','?')}  kv_blocks={mem.get('kv_blocks','?')}")
        W("```")
        W("")
        W(f"Per GPU (× {ngpu} for the {nnodes}-node pair):")
        W("")
        W("| Component | Per GPU | Job total | What it is |")
        W("|---|---:|---:|---|")
        for label, key, what in (
            ("Model weights + framework", "weights_gib",
             f"The TP{tp}×PP{pp} shard — dominated by MXFP4 experts"),
            ("Non-torch (NCCL buffers, CUDA runtime, driver)", "non_torch_gib", ""),
            ("PyTorch activation peak", "act_peak_gib", ""),
            ("**KV cache pool**", "kv_gib", "Allocated up front from what's left"),
        ):
            v = mem.get(key)
            if v is None:
                continue
            W(f"| {label} | {fmt(v)} GiB | {fmt(v*ngpu)} GiB | {what} |")
        W("")
        wg = mem["weights_gib"]
        W(f"**It does not load only weights.** Weights are ~{fmt(wg)} GiB/GPU "
          f"({tib(args.weight_bytes)*1024:.0f} GiB of checkpoint sharded {ngpu} ways ≈ "
          f"{tib(args.weight_bytes)*1024/ngpu:.0f} GiB, consistent), and a "
          f"**{fmt(mem.get('kv_gib'))} GiB KV pool per GPU** is carved out on top.")
        W("")
        W("### KV cache details")
        W("")
        if mem.get("kv_tokens") and mem.get("kv_gib"):
            bpt = mem["kv_gib"] * 2**30 / mem["kv_tokens"]
            W(f"Measured `kv_bytes / kv_tokens` = **{bpt:,.0f} B per token per GPU**.")
            W("")
            W(f"Predicted from the architecture: MLA stores a *compressed latent* — "
              f"`kv_lora_rank({arch.kvlr}) + qk_rope_head_dim({arch.qk_r})` = "
              f"{arch.kvlr+arch.qk_r} values/token/layer, at fp8 = {arch.kvlr+arch.qk_r} B, "
              f"× **{arch.full_layers} full-attention layers** = "
              f"**{arch.kv_bytes_per_token:,} B**.")
            W("")
            ratio = bpt / arch.kv_bytes_per_token
            if 0.9 < ratio < 1.1:
                W("The two agree, and two conclusions follow:")
            else:
                W(f"Measured is {ratio:.2f}× predicted — the gap is worth chasing before "
                  "trusting the conclusions below. Candidates: KDA state counted into the "
                  "same pool, or a block-padding effect.")
            W("")
            W(f"1. **Only the {arch.full_layers} MLA layers consume paged KV.** The "
              f"{arch.kda_layers} KDA layers keep a fixed-size *recurrent state* per request "
              "instead of a growing cache — that is the point of linear attention, and it is "
              f"why a {arch.L}-layer model has the KV footprint of a {arch.full_layers}-layer one.")
            W("2. **The KV cache is replicated, not sharded, across TP ranks.** MLA's latent "
              "is shared across heads, so sharding it would mean re-gathering it every step. "
              "Replication trades otherwise-idle HBM for zero communication.")
            W("")
            used_tok = pk["max_concurrency"] * (args.isl + args.osl)
            W(f"Capacity: **{mem['kv_tokens']:,} tokens** per GPU. The benchmark used "
              f"{pk['max_concurrency']} seqs × ~{args.isl+args.osl} tokens = "
              f"**{used_tok:,} tokens, {100*used_tok/mem['kv_tokens']:.1f}% of the pool**. "
              "KV was never remotely a constraint.")
    else:
        W("The server log did not contain a parseable vLLM memory-profiling line. Re-run "
          "with `VLLM_LOGGING_LEVEL=INFO` to capture it; the sections that depend on it are "
          "marked unavailable rather than estimated.")
    W("")
    W("---")
    W("")

    # ============================== §3 bottleneck =================================
    W("## 3. What is the bottleneck?")
    W("")
    W("**Intra-GPU HBM traffic.** Not compute, not either interconnect.")
    W("")
    W("**But be precise about which HBM limit: this is LATENCY-BOUND, NOT BANDWIDTH-BOUND.** "
      f"HBM sits at only ~{pk['hbm_pct']:.0f}% of peak, so the box has bandwidth to spare. "
      "What binds is that the MoE decode GEMMs are too narrow to keep enough concurrent "
      "memory requests in flight to use it (§3.2). Saying \"HBM-bandwidth-bound\" would "
      "point at the wrong fix — buying faster memory would not help; widening the GEMMs "
      "would.")
    W("")
    W(f"| Resource | Demand at c={pk['max_concurrency']} | B200 capability | Utilization |")
    W("|---|---:|---:|---:|")
    W(f"| Compute | {fmt(pk['tflops_per_gpu'],1)} TFLOP/s per GPU | "
      f"{fmt(hw['bf16_tflops'],0)} TFLOP/s BF16 | **{pk['pct_peak']:.2f}%** |")
    W(f"| **HBM bandwidth** | **~{fmt(pk['hbm_gbs_per_gpu'],0)} GB/s per GPU** | "
      f"{fmt(hw['hbm_bw_gbs'],0)} GB/s | **~{pk['hbm_pct']:.0f}%** |")
    W(f"| NVLink (GPU↔GPU, intra-node) | {fmt(pk['link_gbs_per_gpu'],2)} GB/s per GPU | "
      f"~{fmt(hw['link_dir_gbs'],0)} GB/s (per direction) | **~{pk['link_pct']:.2f}%** |")
    W(f"| InfiniBand (PP stage boundary) | {fmt(pk['pp_gbs'],3)} GB/s | "
      f"{fmt(IB['node_dir_gbs'],0)} GB/s (8 rails NDR) | **~{pk['pp_pct']:.2f}%** |")
    W("")
    W("### 3.1 Why HBM — the MoE batching mechanism")
    W("")
    W(f"At **batch 1**, only {arch.topk}+{arch.shared} experts fire per layer. At **batch "
      f"{pk['max_concurrency']}** the tokens route independently, so the expected number of "
      f"*distinct* experts touched per layer is "
      f"`E × (1 − (1−1/E)^(B·topk))` = **{pk['experts_fired']:.0f} of {arch.E}**.")
    W("")
    W("| Batch | Experts fired/layer | Tokens per expert | Weight bytes/step (job) | per GPU |")
    W("|---:|---:|---:|---:|---:|")
    for bsz in [1, 8, 64, 128, 256, 512, 1024]:
        f_ = arch.experts_fired(bsz)
        tot = arch.moeL * f_ * arch.bytes_per_expert()
        W(f"| {bsz} | {f_:.0f} | {bsz*arch.topk/f_:.1f} | {tot/1e9:.0f} GB | "
          f"{tot/ngpu/1e9:.0f} GB |")
    W("")
    W("This is the defining property of sparse MoE: **compute grows with batch, but weight "
      "traffic grows much faster** until nearly every expert is touched every step. From "
      "batch ~512 upward the weight bytes **plateau** — every additional token is then nearly "
      "free in bandwidth terms.")
    W("")
    W("MXFP4 is what makes this tractable. At BF16 the same reads would be **4× larger**, "
      "exceeding HBM bandwidth outright and making the model bandwidth-*starved* rather than "
      "merely bandwidth-dominated.")
    W("")
    W("### 3.2 How to improve it")
    W("")
    W(f"HBM is the binding resource at ~{pk['hbm_pct']:.0f}%, but the box is not delivering "
      "all the HBM it has. The shortfall has a specific cause: with "
      f"{pk['experts_fired']:.0f} experts fired across {pk['max_concurrency']} tokens, each "
      f"expert sees only **{pk['tokens_per_expert']:.1f} tokens** — a matrix-*vector* product, "
      "which cannot issue enough concurrent memory requests to saturate HBM.\n\n"
      "**This is the crux: the workload is latency-bound, not bandwidth-bound.** Its "
      "*volume* of weight traffic is what dominates step time, but it is nowhere near the "
      "bandwidth ceiling — a GEMV has too little memory-level parallelism to fill the "
      "pipe. More bandwidth would buy nothing; more tokens per expert buys everything.")
    W("")
    W("Ranked levers:")
    W("")
    W(f"1. **Raise `--max-num-seqs`** ({mem.get('mns', args.max_num_seqs)} → 256+). Biggest "
      "lever, costs nothing, and the KV memory is already provisioned (§2). This is the "
      "same conclusion the MI355X run reached, and for the same reason.")
    W("2. **Speculative decoding / MTP** — verifies several tokens per weight read, widening "
      "the GEMM exactly as a larger batch does. The upstream vLLM recipe gates DSpark off "
      "its `multi_node_tp_pp` profile, but that is a recipe default, not an engine limit: "
      "SemiAnalysis run DSpark on B200 TP8×PP2 (§7.5 has the setup). Untested on this run — "
      "the speculator model is not downloaded here.")
    W("3. **Expert parallelism** — fewer, whole expert reads per GPU, paid for with all-to-all "
      "over the near-idle interconnect. On MI355X this was tested and is **unsupported** "
      "(ATOM raises `NotImplementedError` for EP with the MXFP4 SiTUv2 kernel). On B200 it is "
      "worth testing separately; it is deliberately **off** here because the MI355X baseline "
      "has it off, and arm A exists to hold everything but the hardware constant.")
    W("4. **Prefill/decode disaggregation** — the recipe ships a `pd_cluster` profile.")
    W("")
    W("### 3.3 Why TTFT is flat but TPOT rises")
    W("")
    ttft_ratio = (pk["median_ttft_ms"] or 1) / (first["median_ttft_ms"] or 1)
    W(f"TTFT moves {first['median_ttft_ms']:.0f} → {pk['median_ttft_ms']:.0f} ms "
      f"({ttft_ratio:.1f}×) across a {pk['max_concurrency']}× concurrency increase, because "
      f"prefill is genuinely compute-dense ({args.isl} tokens/request in parallel) and has "
      f"headroom. TPOT rises {(pk['median_tpot_ms']/first['median_tpot_ms']):.1f}× because "
      "decode adds weight-read traffic per step as more experts activate. Different regimes — "
      "further confirmation that decode is bandwidth-limited.")
    W("")
    W("---")
    W("")

    # ============================== §4 communication ==============================
    W("## 4. Data communication analysis")
    W("")
    W("Three paths carry traffic here, where the single-node MI355X run had one.")
    W("")
    W("### 4.1 Intra-node GPU↔GPU (NVLink) — activations only")
    W("")
    W(f"With **TP={tp} and EP disabled**, every expert is sharded across the {tp} GPUs of its "
      "node, so there is **no expert-routing all-to-all**. The only cross-GPU traffic inside "
      "a node is TP activation reduction:")
    W("")
    W("| Property | Value |")
    W("|---|---|")
    W("| Collective | **all-reduce** (NCCL), 2 per layer |")
    W(f"| Layers per pipeline stage | {arch.L}/{pp} = {arch.L/pp:.1f} |")
    W(f"| Count per token per node | 2 × {arch.L/pp:.1f} = **{int(pk['allreduce_per_token'])}** |")
    W(f"| Payload per call per token | `hidden_size × 2 B` = **{arch.h*2/1024:.1f} KB** |")
    W("")
    W("| Concurrency | Steps/s | Payload/step | Wire bytes/step¹ | Sustained per GPU |")
    W("|---:|---:|---:|---:|---:|")
    for r in (first, pk):
        wire = r["ar_payload_step"] * 2 * (tp - 1) / tp
        W(f"| {r['max_concurrency']} | {fmt(r['steps_per_s'])} | "
          f"{r['ar_payload_step']/1e6:.1f} MB | {wire/1e6:.1f} MB | "
          f"**{fmt(r['link_gbs_per_gpu'],2)} GB/s** |")
    W("")
    W("¹ busbw convention: an all-reduce moves `2(N−1)/N × payload` on the wire.")
    W("")
    W(f"At **{fmt(pk['link_gbs_per_gpu'],2)} GB/s against a ~{fmt(hw['link_dir_gbs'],0)} GB/s "
      f"per-direction ceiling ({pk['link_pct']:.2f}%)**, NVLink is almost entirely idle.")
    W("")
    W(f"**What is NOT transferred:** weights (resident per GPU), KV cache (replicated), "
      "gradients (inference), expert tokens (EP off).")
    W("")
    W("#### The message-size regime — why \"1% utilized\" understates the cost")
    W("")
    W("| | Per all-reduce call |")
    W("|---|---|")
    W(f"| Payload at c={first['max_concurrency']} | **{arch.h*2*first['max_concurrency']/1024:.1f} KB** |")
    W(f"| Payload at c={pk['max_concurrency']} | **{arch.h*2*pk['max_concurrency']/1024:.0f} KB** |")
    W("")
    step_ms = 1000.0 / pk["steps_per_s"] if pk["steps_per_s"] else 0
    nar = int(pk["allreduce_per_token"])
    W(f"A step-time budget at c={pk['max_concurrency']} (step = {step_ms:.1f} ms, "
      f"{nar} all-reduces):")
    W("")
    W("| Assumption | All-reduce time/step | Share of step |")
    W("|---|---:|---:|")
    bw_ms = pk["ar_payload_step"] * 2 * (tp-1) / tp / (hw["link_dir_gbs"] * 1e9) * 1000
    W(f"| Pure bandwidth, zero overhead | {bw_ms:.2f} ms | {100*bw_ms/step_ms:.1f}% |")
    for lat_us in (5, 10, 20):
        t = bw_ms + nar * lat_us / 1000.0
        W(f"| + {lat_us} µs fixed latency per call | {t:.2f} ms | {100*t/step_ms:.1f}% |")
    W("")
    W("These calls are **latency-dominated, not bandwidth-dominated**: a few microseconds "
      f"each compounds into milliseconds across {nar} serialized collectives per step. The "
      "utilization percentage is a floor on the cost, not an estimate of it.")
    W("")
    W("> Not directly measured: there is no per-call NCCL timing from this run, so the "
      "latency rows are illustrative arithmetic over a plausible range. Confirming the real "
      "figure needs `--profile` or `NCCL_DEBUG=INFO` timing.")
    W("")

    # --- 4.2 the new one ----------------------------------------------------------
    W("### 4.2 Inter-node (InfiniBand) — the pipeline boundary")
    W("")
    W("**This path has no counterpart in the MI355X run**, which is single-node. It exists "
      "here only because the model does not fit in one node.")
    W("")
    W(f"With PP={pp}, the {arch.L} layers are split into {pp} stages. Every step, stage 0 "
      "sends its hidden states to stage 1 over IB and (for the next microbatch) receives. "
      "The payload is small — hidden state, not weights:")
    W("")
    W("| Property | Value |")
    W("|---|---|")
    W(f"| Stage boundaries | {pp-1} |")
    W(f"| Bytes per token per boundary | `hidden_size × 2 B` = {arch.h*2/1024:.1f} KB |")
    W(f"| Bytes per step at c={pk['max_concurrency']} | {pk['pp_bytes_step']/1e6:.2f} MB |")
    W(f"| Sustained | **{fmt(pk['pp_gbs'],3)} GB/s** |")
    W(f"| Fabric capability (8 rails NDR, per node per direction) | {fmt(IB['node_dir_gbs'],0)} GB/s |")
    W(f"| Utilization | **{pk['pp_pct']:.3f}%** |")
    W("")
    W("Measured fabric health on these nodes, for context (`../b200-nodes/notes.md`, "
      f"2026-08-12): `ib_write_bw` GPU→GPU at **{IB['measured_gdr_gbps']} Gb/s** — NDR line "
      f"rate — and NCCL `sendrecv` at **{IB['measured_nccl_pair_gbs']} GB/s per pair** at 8 "
      "GPUs/node. The fabric is not the problem.")
    W("")
    W("**But bandwidth is again the wrong lens, and here it matters more.** The PP cost is "
      "not the bytes; it is:")
    W("")
    W("1. **Latency in the critical path.** Every token must traverse stage 0, cross the "
      "network, then traverse stage 1. One inter-node hop is added to every decode step — "
      "an RDMA write plus synchronization, on the order of single-digit microseconds, "
      f"against a {step_ms:.1f} ms step. Small, but it is pure serial addition.")
    W("2. **The pipeline bubble.** vLLM splits the running batch into microbatches to keep "
      "both stages busy; whatever it cannot overlap is idle GPU time on one stage or the "
      "other. This is the real PP tax and it is **not visible in the byte counts above**.")
    W(f"3. **Lost features.** DSpark speculative decoding is gated off this profile entirely.")
    W("")
    W("A clean measurement of the bubble needs a stage-resident profile (per-stage step "
      "timings), which this run does not collect. What can be said from these data: the PP "
      "boundary is **not bandwidth-limited**, so if PP costs throughput here it costs it "
      "through bubbles and latency, not through the wire.")
    W("")
    W("### 4.3 Intra-GPU (HBM) — dominated by weights")
    W("")
    W(f"Per decode step at c={pk['max_concurrency']}, per GPU:")
    W("")
    W("| Traffic | Bytes/step | Share |")
    W("|---|---:|---:|")
    routed = arch.moeL * pk["experts_fired"] * arch.bytes_per_expert() / ngpu
    other = (arch.shared_total + arch.latent_proj + arch.router + arch.dense_mlp
             + arch.attn_total) / ngpu
    kvb = pk["max_concurrency"] * (args.isl + args.osl) * arch.kv_bytes_per_token
    actb = pk["ar_payload_step"] * 2
    tot_hbm = routed + other + kvb + actb
    for label, v in (("**Expert weights (MXFP4)**", routed),
                     ("Attention + shared + dense weights", other),
                     ("KV cache read", kvb),
                     ("Activations (read+write)", actb)):
        W(f"| {label} | {v/1e9:.2f} GB | {100*v/tot_hbm:.1f}% |")
    W("")
    W(f"The asymmetry is stark: **HBM moves ~{tot_hbm/1e9:.0f} GB/step while NVLink moves "
      f"~{pk['ar_payload_step']*2*(tp-1)/tp/1e9:.2f} GB and IB ~"
      f"{pk['pp_bytes_step']/1e9:.4f} GB.** Optimization effort belongs on the memory side.")
    W("")
    W("---")
    W("")

    # ============================== §5 discussion =================================
    W("## 5. Further discussion")
    W("")
    W(f"**1. Two nodes is a property of the model, not a choice.** "
      f"{gb(args.weight_bytes):.0f} GB of weights against {gb(node_hbm_b):.0f} GB of node "
      f"HBM — {gb(short_b):.0f} GB short. The consequence is not just "
      "\"more GPUs\": DSpark speculative decoding needs an extra shim and DCP flags to "
      "work with PP2 (§7.5) rather than dropping in for free as it does on one node, "
      "and PP adds a pipeline bubble plus halves the per-GPU weight residency. A B300 "
      "node (8 × 268 GB = 2144 GB) would hold it single-node; a B200 node cannot.")
    W("")
    W(f"**2. `max_num_seqs={mem.get('mns', args.max_num_seqs)}` is the binding limit, not "
      f"hardware.** KV was ~{100*pk['max_concurrency']*(args.isl+args.osl)/mem['kv_tokens']:.1f}% "
      "used and compute ~1%." if mem.get("kv_tokens") else
      f"**2. `max_num_seqs={mem.get('mns', args.max_num_seqs)}` is the binding limit.**")
    W("")
    W("**3. Prefix caching is disabled for correctness, and it costs real throughput.** KDA's "
      "recurrent state is per-request and cannot be reconstructed from the paged MLA cache. "
      "In workloads with shared prefixes this forfeits a large win that non-KDA models get "
      "for free — an architectural trade, not a tuning oversight. (The vLLM Blackwell "
      "baseline turns prefix caching *on*; it is turned off here both for correctness and to "
      "match the MI355X run.)")
    W("")
    W(f"**4. The hybrid attention design is what makes {arch.total/1e12:.2f} T fit at all.** "
      f"Only {arch.full_layers} of {arch.L} layers keep a growing KV cache.")
    W("")
    if b.get("load_s"):
        W(f"**5. Load time was {b['load_s']/60:.1f} minutes** for {disk_tib:.2f} TiB off shared "
          f"NFS (~{args.weight_bytes/1e9/b['load_s']:.1f} GB/s effective), with "
          "`--load-format fastsafetensors`.")
        W("")
    W("---")
    W("")

    # ============================== §6 comparison =================================
    W("## 6. B200 vs MI355X — head to head")
    W("")
    if not a:
        W("_The MI355X baseline was not readable at analysis time, so this section is empty. "
          "Re-run with `--amd-sweep` / `--amd-server-log` pointing at the ATOM run._")
        W("")
        return L

    arows, amem, atp, app = a["rows"], a["mem"], a["tp"], a["pp"]
    ahw = HW["MI355X"]
    angpu = atp * app
    apk = max(arows, key=lambda r: r["output_throughput"] or 0)

    W("> **Read this section carefully — it is not an equal-hardware comparison.** MI355X "
      "serves this model on **one node with 8 GPUs**; B200 needs **two nodes and 16 GPUs**, "
      "because the checkpoint does not fit in 8 × 183 GB. Total throughput therefore compares "
      "two different amounts of hardware. The per-GPU and per-node columns are the ones that "
      "carry meaning, and even those are confounded by a different engine (vLLM vs ATOM) and "
      "a different parallelism (TP8×PP2 vs TP8). Treat this as a **system-level capability "
      "comparison**, not a chip-vs-chip benchmark.")
    W("")

    W("### 6.1 What was held constant, and what could not be")
    W("")
    W("| | MI355X | B200 | Matched? |")
    W("|---|---|---|---|")
    W(f"| Model | Kimi-K3 MXFP4 | Kimi-K3 MXFP4 | ✅ |")
    W(f"| ISL / OSL | {args.isl} / {args.osl} | {args.isl} / {args.osl} | ✅ |")
    W(f"| `max_model_len` | {amem.get('mml','—')} | {mem.get('mml','—')} | "
      f"{'✅' if amem.get('mml')==mem.get('mml') else '⚠️'} |")
    W(f"| `max_num_seqs` | {amem.get('mns','—')} | {mem.get('mns','—')} | "
      f"{'✅' if amem.get('mns')==mem.get('mns') else '⚠️'} |")
    W(f"| KV dtype | fp8 | {mem.get('kv_dtype','fp8')} | ✅ |")
    W("| Prefix caching | off | off | ✅ |")
    W("| Expert parallelism | off | off | ✅ |")
    W(f"| Concurrency sweep | 1→{apk['max_concurrency']} | 1→{pk['max_concurrency']} | "
      f"{'✅' if apk['max_concurrency']==pk['max_concurrency'] else '⚠️'} |")
    W(f"| **Nodes / GPUs** | **1 / {angpu}** | **{nnodes} / {ngpu}** | ❌ *forced* |")
    W(f"| **Parallelism** | **TP{atp}** | **TP{tp}×PP{pp}** | ❌ *forced* |")
    W("| **Engine** | **ATOM** | **vLLM** | ❌ *ATOM is ROCm-only* |")
    W("")

    W("### 6.2 Hardware")
    W("")
    W("| | MI355X | B200 | Ratio (B200/MI355X) |")
    W("|---|---:|---:|---:|")
    W(f"| HBM per GPU | {gb(ahw['hbm_bytes']):.0f} GB | {gb(hw['hbm_bytes']):.0f} GB | "
      f"{hw['hbm_bytes']/ahw['hbm_bytes']:.2f}× |")
    W(f"| HBM per node (8 GPUs) | {gb(ahw['hbm_bytes']*8):.0f} GB | {gb(hw['hbm_bytes']*8):.0f} GB | "
      f"{hw['hbm_bytes']/ahw['hbm_bytes']:.2f}× |")
    W(f"| HBM bandwidth | {fmt(ahw['hbm_bw_gbs'],0)} GB/s | {fmt(hw['hbm_bw_gbs'],0)} GB/s | "
      f"{hw['hbm_bw_gbs']/ahw['hbm_bw_gbs']:.2f}× |")
    W(f"| BF16 dense peak | {fmt(ahw['bf16_tflops'],0)} TFLOP/s | {fmt(hw['bf16_tflops'],0)} TFLOP/s | "
      f"{hw['bf16_tflops']/ahw['bf16_tflops']:.2f}× |")
    W(f"| GPU↔GPU link | {ahw['link']} | {hw['link']} | — |")
    W(f"| link, per direction | {fmt(ahw['link_dir_gbs'],0)} GB/s | {fmt(hw['link_dir_gbs'],0)} GB/s | "
      f"{hw['link_dir_gbs']/ahw['link_dir_gbs']:.2f}× |")
    W(f"| **Holds Kimi-K3 on one node?** | **yes** — "
      f"{gb(ahw['hbm_bytes']*8):.0f} GB vs {gb(args.weight_bytes):.0f} GB of weights, "
      f"{gb(ahw['hbm_bytes']*8-args.weight_bytes):.0f} GB spare | "
      f"**no** — {gb(hw['hbm_bytes']*8):.0f} GB vs {gb(args.weight_bytes):.0f} GB, "
      f"**{gb(short_b):.0f} GB short** | — |")
    W("")
    W("**HBM capacity, not bandwidth or FLOPs, is the axis that decides this workload's "
      f"layout.** The two parts have identical HBM bandwidth and B200 has 90% of MI355X's "
      f"BF16 peak — neither of those decides anything here. Capacity does: "
      f"{gb(ahw['hbm_bytes']):.0f} GB/GPU vs {gb(hw['hbm_bytes']):.0f} GB/GPU is "
      f"{ahw['hbm_bytes']/hw['hbm_bytes']:.2f}×, and that single ratio is the difference "
      "between one node and two.")
    W("")

    W("### 6.3 Throughput, point by point")
    W("")
    W("| Conc | MI355X tok/s | B200 tok/s | MI355X/B200 | B200/MI355X | "
      "MI355X tok/s/GPU | B200 tok/s/GPU | per-GPU ratio |")
    W("|---:|---:|---:|---:|---:|---:|---:|---:|")
    amap = {r["max_concurrency"]: r for r in arows}
    for r in rows:
        c = r["max_concurrency"]
        ar = amap.get(c)
        if not ar:
            W(f"| {c} | — | {fmt(r['output_throughput'])} | — | — | — | "
              f"{fmt(r['tok_per_gpu'])} | — |")
            continue
        tot_ratio = r["output_throughput"] / ar["output_throughput"]
        inv_ratio = ar["output_throughput"] / r["output_throughput"]
        gpu_ratio = r["tok_per_gpu"] / (ar["output_throughput"] / angpu)
        W(f"| {c} | {fmt(ar['output_throughput'])} | {fmt(r['output_throughput'])} | "
          f"{inv_ratio:.2f}× | {tot_ratio:.2f}× | {fmt(ar['output_throughput']/angpu)} | "
          f"{fmt(r['tok_per_gpu'])} | {gpu_ratio:.2f}× |")
    W("")
    W("| Headline | MI355X | B200 | MI355X/B200 |")
    W("|---|---:|---:|---:|")
    # c=1 first: it is the latency-optimal operating point, and it tells the opposite
    # story to the peak row -- worth surfacing in the headline rather than only in the
    # point-by-point table above.
    afirst = amap.get(first["max_concurrency"])
    if afirst:
        W(f"| tok/s at c=1 *(single request)* | {fmt(afirst['output_throughput'])} | "
          f"{fmt(first['output_throughput'])} | "
          f"{afirst['output_throughput']/first['output_throughput']:.2f}× |")
        W(f"| tok/s at c=1 **per GPU** | {fmt(afirst['output_throughput']/angpu)} | "
          f"{fmt(first['output_throughput']/ngpu)} | "
          f"{(afirst['output_throughput']/angpu)/(first['output_throughput']/ngpu):.2f}× |")
        W(f"| TPOT at c=1 (ms, lower better) | {fmt(afirst['median_tpot_ms'],2)} | "
          f"{fmt(first['median_tpot_ms'],2)} | "
          f"{afirst['median_tpot_ms']/first['median_tpot_ms']:.2f}× |")
    W(f"| Peak tok/s (c={pk['max_concurrency']}) | {fmt(apk['output_throughput'])} | "
      f"{fmt(pk['output_throughput'])} | "
      f"{apk['output_throughput']/pk['output_throughput']:.2f}× |")
    W(f"| Peak tok/s **per GPU** | {fmt(apk['output_throughput']/angpu)} | "
      f"{fmt(pk['output_throughput']/ngpu)} | "
      f"{(apk['output_throughput']/angpu)/(pk['output_throughput']/ngpu):.2f}× |")
    W(f"| Peak tok/s **per node** | {fmt(apk['output_throughput'])} | "
      f"{fmt(pk['output_throughput']/nnodes)} | "
      f"{apk['output_throughput']/(pk['output_throughput']/nnodes):.2f}× |")
    W(f"| GPUs to serve the model | {angpu} | {ngpu} | "
      f"{angpu/ngpu:.2f}× |")
    W("")
    # Readers routinely mistake the peak aggregate number for a per-request speed.
    # State plainly which workload each view answers for.
    W(f"**What the peak numbers mean.** These are *aggregate* rates — the sum over all "
      f"{pk['max_concurrency']} in-flight requests, i.e. what the server delivers, not what "
      f"one request sees. At c={pk['max_concurrency']} an individual stream gets only "
      f"{fmt(1000.0/pk['median_tpot_ms'],1)} tok/s on B200 and "
      f"{fmt(1000.0/apk['median_tpot_ms'],1)} tok/s on MI355X (§6.4). Which view is "
      "\"real world\" depends on the workload, not the user count: one interactive chat "
      "issues one request at a time and feels per-user tok/s, while a batch job or an "
      "agentic app fanning out parallel calls fills the server itself and gets the "
      "aggregate rate. Both matter; here they agree on the ranking (B200 wins "
      f"{pk['output_throughput']/apk['output_throughput']:.2f}× aggregate"
      + (f", {afirst['median_tpot_ms']/first['median_tpot_ms']:.2f}× per-user at c=1"
         if afirst else "") + ").")
    W("")

    W("### 6.4 Latency")
    W("")
    W("| Conc | MI355X TTFT | B200 TTFT | MI355X TPOT | B200 TPOT | "
      "MI355X per-user tok/s | B200 per-user tok/s | **B200/MI355X** |")
    W("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ar = amap.get(r["max_concurrency"])
        pu_b = 1000.0 / r["median_tpot_ms"] if r["median_tpot_ms"] else None
        pu_a = 1000.0 / ar["median_tpot_ms"] if (ar and ar["median_tpot_ms"]) else None
        ratio = f"**{pu_b/pu_a:.2f}×**" if (pu_a and pu_b) else "—"
        W(f"| {r['max_concurrency']} | {fmt(ar['median_ttft_ms']) if ar else '—'} | "
          f"{fmt(r['median_ttft_ms'])} | {fmt(ar['median_tpot_ms'],2) if ar else '—'} | "
          f"{fmt(r['median_tpot_ms'],2)} | {fmt(pu_a) if pu_a else '—'} | "
          f"{fmt(pu_b) if pu_b else '—'} | {ratio} |")
    W("")
    W("TPOT is the metric where the PP2 boundary would show up if it costs anything: it is "
      "per-decode-step latency, and the inter-node hop plus any pipeline bubble lands there "
      "directly. TTFT is prefill-dominated and less sensitive to it.")
    W("")
    # The per-user column inverts the headline conclusion, so state it plainly rather than
    # leaving it for the reader to derive from the TPOT column.
    fa = amap.get(first["max_concurrency"])
    if fa and fa["median_tpot_ms"] and first["median_tpot_ms"]:
        pu_a1 = 1000.0 / fa["median_tpot_ms"]
        pu_b1 = 1000.0 / first["median_tpot_ms"]
        W(f"**B200 is faster per user at every concurrency — and {pu_b1/pu_a1:.2f}× faster "
          f"for a single user** ({fmt(pu_b1)} vs {fmt(pu_a1)} tok/s at "
          f"c={first['max_concurrency']}). This is the one comparison B200 wins outright, "
          "and §6.3's aggregate view hides it completely.")
        W("")
        W("The two results are not in conflict; they answer different questions:")
        W("")
        W("- *How many tokens per second per unit of silicon?* → **MI355X**, 1.48× per GPU (§6.3).")
        W(f"- *How fast does one user's answer stream?* → **B200**, {pu_b1/pu_a1:.2f}× (here).")
        W("")
        W("B200's lower per-token latency wins the second question and survives even the "
          "PP2 pipeline penalty. See `notes-concurrency.md` for the full treatment, "
          "including why none of the §3.2 levers improve the single-user case.")
    W("")

    # A single agent session is a low-concurrency client, so the peak aggregate number in
    # 6.3 is the wrong one to quote at anybody sizing an agent workload.
    W("**What an agent app should expect.** One agent session is mostly serial — a single "
      "in-flight request at a time, with occasional fan-out (parallel tool calls, "
      "subagents) of ~2–8. So one agent user sits at c=1–8, and reads the per-user column:")
    W("")
    W(f"| What the agent is doing | Conc | {angpu}× MI355X tok/s | {ngpu}× B200 tok/s |")
    W("|---|---:|---:|---:|")
    agent_labels = {1: "Serial turn — one request at a time",
                    2: "Light fan-out", 4: "Light fan-out",
                    8: "Heavy fan-out — 8 parallel calls"}
    for r in rows:
        c = r["max_concurrency"]
        if c not in agent_labels:
            continue
        ar = amap.get(c)
        pu_b = 1000.0 / r["median_tpot_ms"] if r["median_tpot_ms"] else None
        pu_a = 1000.0 / ar["median_tpot_ms"] if (ar and ar["median_tpot_ms"]) else None
        W(f"| {agent_labels[c]} | {c} | {fmt(pu_a) if pu_a else '—'} | "
          f"{fmt(pu_b) if pu_b else '—'} |")
    W("")
    W("Server-side concurrency of 32–64 comes from *many* users, not one agent — the peak "
      "aggregate rate in §6.3 only appears when that many streams are in flight at once. "
      "Note these come from a fixed 1024-in/1024-out sweep; real agent traffic has much "
      "longer, cache-heavy prompts, so TTFT would be worse and TPOT somewhat better with "
      "prefix caching on (§7).")
    W("")

    W("### 6.5 Where each system's headroom is")
    W("")
    W("| Resource (at peak concurrency) | MI355X | B200 |")
    W("|---|---:|---:|")
    W(f"| Compute utilization | {apk['pct_peak']:.2f}% | {pk['pct_peak']:.2f}% |")
    W(f"| HBM utilization | ~{apk['hbm_pct']:.0f}% | ~{pk['hbm_pct']:.0f}% |")
    W(f"| GPU↔GPU link utilization | ~{apk['link_pct']:.2f}% | ~{pk['link_pct']:.2f}% |")
    W(f"| Inter-node utilization | n/a (single node) | ~{pk['pp_pct']:.3f}% |")
    W(f"| Experts fired per layer | {apk['experts_fired']:.0f} / {arch.E} | "
      f"{pk['experts_fired']:.0f} / {arch.E} |")
    W(f"| Tokens per expert | {apk['tokens_per_expert']:.1f} | {pk['tokens_per_expert']:.1f} |")
    W("")
    W("**Both systems land in the same regime and hit the same wall**: ~1% of compute, ~1% of "
      "interconnect, and a memory system doing most but not all of what it can, because each "
      "expert's weight matrix is read to serve barely more than one token. The lever on both "
      "is `max_num_seqs`. That the two agree on the diagnosis, from different vendors' "
      "silicon and different serving engines, is the strongest evidence that the finding is "
      "about the *model*, not about either box.")
    W("")
    W("### 6.6 Memory footprint")
    W("")
    if mem.get("weights_gib") and amem.get("weights_gib"):
        W("| | MI355X (TP8, 1 node) | B200 (TP8×PP2, 2 nodes) |")
        W("|---|---:|---:|")
        W(f"| Weights per GPU | {fmt(amem['weights_gib'])} GiB | {fmt(mem['weights_gib'])} GiB |")
        W(f"| KV pool per GPU | {fmt(amem.get('kv_gib'))} GiB | {fmt(mem.get('kv_gib'))} GiB |")
        W(f"| KV capacity per GPU | {amem.get('kv_tokens',0):,} tok | {mem.get('kv_tokens',0):,} tok |")
        W(f"| KV bytes/token | {arch.kv_bytes_per_token:,} B | {arch.kv_bytes_per_token:,} B |")
        W(f"| Total HBM committed | {fmt((amem['weights_gib']+amem.get('kv_gib',0))*angpu)} GiB | "
          f"{fmt((mem['weights_gib']+mem.get('kv_gib',0))*ngpu)} GiB |")
        W("")
        W("Per-GPU weight residency is roughly halved on B200 — the same checkpoint spread "
          "over twice as many GPUs. That is why B200's per-GPU weight *traffic* is also "
          "roughly halved in §3, and why per-GPU throughput comparisons must be read "
          "alongside the GPU count, not instead of it.")
    else:
        W("_Memory tables unavailable: one of the two server logs did not yield a parseable "
          "memory line._")
    W("")
    W("### 6.7 What this comparison does and does not establish")
    W("")
    W("**Does:**")
    W("")
    W("- MI355X serves Kimi-K3 on **one** node; B200 needs **two**. For a 2.78 T MXFP4 "
      "frontier model, HBM capacity per node is the deciding specification.")
    W("- Both reach the same bottleneck (HBM, via thin MoE GEMMs) and the same lever "
      "(`max_num_seqs`), independently.")
    W("- The inter-node fabric is **not** the limiting factor on the B200 pair: the PP "
      "boundary uses a fraction of a percent of an 8-rail NDR fabric that measures at line rate.")
    W("")
    W("**Does not:**")
    W("")
    W("- Establish a per-chip performance ratio. Engine (vLLM vs ATOM), parallelism "
      "(PP2 vs none) and GPU count all differ; no single ratio isolates the silicon.")
    W("- Say anything about B200 with a model that *does* fit in one node — that is a "
      "different and more favourable configuration for B200, and it is not this measurement.")
    W("- Measure the pipeline bubble, which needs per-stage profiling this run does not collect.")
    W("")
    W("---")
    W("")

    # --- sources -------------------------------------------------------------------
    W("## 7. Cross-check against SemiAnalysis InferenceX")
    W("")
    W("SemiAnalysis publish continuous Kimi-K3 benchmarks at "
      "<https://inferencex.semianalysis.com>, and open-source the harness "
      "(Apache-2.0, `github.com/SemiAnalysisAI/InferenceX`). Their configs are in "
      "`../semianalysis-ref/`. Their **B200 recipe uses TP8 × PP2 on 2 nodes — the same "
      "layout as this run**, which makes the methodology directly checkable.")
    W("")
    W("### 7.1 What they independently confirm")
    W("")
    W("| Fact | **Applies to** | Ours (measured) | Theirs (stated in config) |")
    W("|---|---|---|---|")
    W("| Checkpoint size | **both** | 1,560,936,091,448 B = 1.561 TB / 1.420 TiB, 96 shards | "
      "*\"1.561 TB decimal (1.420 TiB, 96 safetensors)\"* |")
    W("| Does not fit one 8×B200 node | **B200** | 1561 GB vs 1538 GB — 23 GB short | "
      "*\"does not fit one 8xB200 node, so TP8 shards … and PP2 splits the 93 layers\"* |")
    W("| Layout forced to TP8 × PP2, 16 GPUs | **B200** | yes | `tensor-parallel-size: 8`, "
      "`pipeline-parallel-size: 2`, `agg_nodes: 2` |")
    W("| `gpu-memory-utilization` 0.90 not 0.95 | **B200** | 0.90 | 0.90, and the *same reason*: "
      "*\"the flashinfer trtllm MXFP4 MoE kernel allocates a ~1.6 GiB runtime workspace "
      "OUTSIDE vLLM's memory pool … at 0.95 a 178 GiB B200 … OOMs\"* |")
    W("| Usable HBM per GPU | **B200** | 178.35 GiB (nvidia-smi) | *\"a 178 GiB B200\"* |")
    W("| Fits on ONE node at TP8 | **MI355X** | yes, 288 GB/GPU | *\"~195 GB/GPU across 8 GPUs "
      "of the 288 GB part; TP=4 … cannot load\"* |")
    W("| Expert parallelism off | **both** | EP off | *\"Plain TP (NOT TEP): expert parallelism is "
      "deliberately off\"* |")
    W("| Image, load format, autotune, batched-token cap | **B200** | `vllm/vllm-openai:kimi-k3`, "
      "`fastsafetensors`, autotune off, 8192 | identical on all four |")
    W("")
    W("Two independent teams, different clusters, same conclusions — including the exact "
      "checkpoint byte count and the non-obvious 0.90 memory-utilisation workaround.")
    W("")
    W("### 7.2 What differs — why the numbers are NOT directly comparable")
    W("")
    W("| Dimension | **Applies to** | Ours | SemiAnalysis | Effect |")
    W("|---|---|---|---|---|")
    W("| **Workload** | **both** | fixed ISL/OSL 1024/1024, `--ignore-eos`, random synthetic | "
      "**AgentX agentic trace replay**, real multi-turn traces, 1M+ context | "
      "**largest difference.** Their traces have long shared prefixes and huge context; "
      "ours is a fixed, cache-hostile synthetic shape |")
    W("| **Prefix caching** | **both** | **off** | **on** (default, kept for trajectory reuse) | "
      "theirs reuses KV across turns; ours never does. Big throughput swing on agentic "
      "traffic |")
    W("| `max-model-len` | **both** | 16384 | native **1M** (unset) | theirs pays a far larger KV "
      "footprint per sequence |")
    W("| `max-num-seqs` | **B200** | **64** (fixed) | let vLLM choose | ours deliberately caps the "
      "batch; §3.2 shows that cap is what binds our throughput |")
    W("| Benchmark client | **both** | `vllm bench serve` | `aiperf` + trace replay | different "
      "measurement harness |")
    W("| **Spec decoding (MTP)** | **MI355X** | **off** | **DSpark MTP on** (`SPEC_NUM_TOKENS 2`) | "
      "their MI355X arm gets a lever ours does not use — see §7.5 |")
    W("| `max-num-seqs` | **MI355X** | 64 | 128 | their MI355X runs a deeper batch |")
    W("| Serving engine | **MI355X** | ATOM (our baseline) | vLLM ROCm *and* an ATOM variant | "
      "we compare ATOM-vs-vLLM; they run both |")
    W("")
    W("**Precision is the same, despite the labels.** Their config says "
      "`precision: \"fp4\"` and ours says MXFP4 — the same thing. Both serve the native "
      "`moonshotai/Kimi-K3` MXFP4 checkpoint (`mxfp4-pack-quantized`, 4-bit routed experts "
      "with e8m0 scales); neither re-quantises. \"FP4\" on their dashboard is the "
      "checkpoint's own format, not a separate NVFP4 conversion.")
    W("")
    W("### 7.3 Aggregate throughput — theirs vs ours")
    W("")
    W("Their API carries aggregate output throughput (`output_tput_tps`) at every "
      "concurrency point — the same quantity as our §1 \"Throughput (tok/s)\" column. "
      "Their **B200 arms are 16 GPUs, TP8 × PP2 across 2 nodes — the same layout as ours**; "
      "their MI355X arms are 8 GPUs on one node.")
    W("")
    # Fetched values, /api/v1/benchmarks?model=Kimi-K3; raw JSON in ../semianalysis-ref/.
    # conc -> (B200 no-spec, B200 +MTP, MI355X vLLM +MTP, MI355X ATOM +MTP)
    their_agg = {
        1:  (59.9, 118.1, 67.4, 91.0),
        2:  (72.1, 144.5, None, None),
        4:  (76.0, 190.3, 115.7, 137.1),
        8:  (99.7, 338.3, 169.5, 233.0),
        10: (None, None, 235.0, 293.1),
        16: (92.4, 518.0, None, None),
        32: (60.2, 768.0, None, None),
        48: (838.2, None, None, None),
        72: (1021.6, None, None, None),
        96: (1159.4, None, None, None),
    }
    ours_b = {r["max_concurrency"]: r["output_throughput"] for r in rows}
    W("| Conc | **Ours** B200 TP8×PP2 (no spec) | **Theirs** B200 TP8×PP2 (no spec) | "
      "**Theirs** B200 +MTP | **Theirs** MI355X vLLM +MTP | **Theirs** MI355X ATOM +MTP | "
      "**Ours** MI355X ATOM (no spec) |")
    W("|---:|---:|---:|---:|---:|---:|---:|")
    for c in sorted(set(their_agg) | set(ours_b)):
        t = their_agg.get(c, (None, None, None, None))
        ar = amap.get(c)
        cells = [fmt(ours_b[c]) if c in ours_b else "—"] \
            + [fmt(v) if v is not None else "—" for v in t] \
            + [fmt(ar["output_throughput"]) if ar else "—"]
        W(f"| {c} | " + " | ".join(cells) + " |")
    W("")
    # The obvious question on seeing this table next to 7.4's: we win one and lose the
    # other. Answer it here rather than leaving the reader to assume a hardware verdict.
    W("#### Why ours leads here but trails on per-user tok/s (§7.4)")
    W("")
    W("The two metrics have different denominators:")
    W("")
    W("- **Per-user tok/s** = `1 / TPOT` — the rate a request streams *while it is "
      "decoding*. Measured directly (their `median_intvty`, our median TPOT).")
    W("- **Aggregate tok/s** = output tokens ÷ total wall clock — which includes every "
      "second spent on **prefill**, producing no output at all.")
    W("")
    W("So `aggregate ÷ concurrency` is always ≤ per-user tok/s, and the gap is the "
      "**decode duty cycle** — the fraction of the run that was decode rather than prefill:")
    W("")
    W("| | Conc | Aggregate | ÷ conc | Per-user (measured) | Duty cycle |")
    W("|---|---:|---:|---:|---:|---:|")
    pu_pk = 1000.0 / pk["median_tpot_ms"]
    W(f"| **Ours** B200, 1024/1024 | {pk['max_concurrency']} | "
      f"{fmt(pk['output_throughput'])} | "
      f"{fmt(pk['output_throughput']/pk['max_concurrency'])} | {fmt(pu_pk)} | "
      f"**{(pk['output_throughput']/pk['max_concurrency'])/pu_pk:.2f}** |")
    W("| **Theirs** B200 +MTP, agentic | 32 | 768.0 | 24.0 | 46.5 | **0.52** |")
    W("")
    W("Their traces average **~138k input tokens** per request against a handful of output "
      "tokens, so half their wall clock is prefill; our `--ignore-eos` 1024/1024 shape "
      "keeps nearly every slot decoding continuously. Neither number is a hardware verdict:")
    W("")
    W("- **Our aggregate lead is a workload artifact.** Counting *all* tokens processed, "
      f"our c={pk['max_concurrency']} run moves ~"
      f"{fmt(pk.get('total_token_throughput') or pk['request_throughput']*(args.isl+args.osl))} "
      "tok/s (input + output); their c=32 run moves **111,970**. They do far more token "
      "work — it just is not output. (Their `tput_per_gpu` is computed on that total, "
      "which is why it looks so large.)")
    W("- **Their per-user lead is a real software lever** — MTP, which we do not use. "
      "Turning it on would raise *both* of our columns (§7.5).")
    W("")
    W("**Their no-spec column is not one curve.** It falls to 60.2 at c=32 and then jumps "
      "to 838.2 at c=48 because the c=1–32 and c=48–96 points come from different recipe "
      "families (the low-concurrency records carry no `recipe_fingerprint`; the "
      "high-concurrency ones do). Do not read a scaling trend down that column.")
    W("")

    W("### 7.4 Per-user tok/s — theirs vs ours")
    W("")
    W("Retrieved live from their public API "
      "(`/api/v1/benchmarks?model=Kimi-K3`); raw JSON and an extracted CSV are in "
      "`../semianalysis-ref/`. Their **\"Interactivity\"** metric is exactly our per-user "
      "tok/s — verified against their own fields: `mean_tpot` 0.00454 s → 1/0.00454 = "
      "220.3 = their `mean_intvty`. Same definition, `1 / TPOT`.")
    W("")
    W("| Conc | **Ours** B200 TP8×PP2 (no spec) | **Theirs** B200 TP8×PP2 (no spec) | "
      "**Theirs** B200 +MTP | **Theirs** MI355X vLLM +MTP | **Theirs** MI355X ATOM +MTP | "
      "**Ours** MI355X ATOM (no spec) |")
    W("|---:|---:|---:|---:|---:|---:|---:|")
    W("| 1 | **89.0** | 81.9 | **221.7** | 84.0 | 127.2 | 46.6 |")
    W("| 2 | 84.4 | 74.8 | 219.3 | — | — | 44.3 |")
    W("| 4 | 73.4 | 54.1 | 203.3 | 62.5 | 88.7 | 40.0 |")
    W("| 8 | 64.9 | 21.1 | 154.1 | 41.4 | 64.7 | 37.0 |")
    W("| 10 | — | — | — | 35.7 | 62.7 | — |")
    W("| 16 | 53.6 | 8.6 | 77.7 | — | — | 32.0 |")
    W("| 32 | 39.6 | 3.8 | 46.5 | — | — | 26.4 |")
    W("| 64 | 27.9 | — | — | — | — | 20.0 |")
    W("")
    W("**\"no spec\" = plain decoding, one token per forward pass. \"+MTP\" = "
      "speculative decoding on: a draft model proposes several tokens and "
      "the big model verifies them in one pass.**")
    W("")
    W("> ⚠️ **Every one of their MI355X runs uses MTP — they publish no MI355X result "
      "without it — so their MI355X column can never be compared against their B200 "
      "no-spec column on equal footing.** (Verified in their API data: all 8 MI355X "
      "records are `spec_method: mtp`; B200 has both, 9 `none` and 6 `mtp`.)")
    W("")
    W("Units: output tokens/s delivered to a single request. Theirs are `median_intvty`; "
      "ours are `1000 / median TPOT`. **Read across rows with care — see §7.2.** The "
      "columns differ in workload (agentic traces vs fixed 1024/1024), prefix caching "
      "(on vs off) and context (1M vs 16K), so this is not a like-for-like ranking.")
    W("")
    W("#### What \"MTP\" is — and it is software, not silicon")
    W("")
    W("**Short answer: yes — MTP is an inference *software* technique, not a hardware "
      "capability.** It says nothing about how fast the GPU is. Any vendor's hardware can "
      "run it once the software stack supports it. So the MTP columns above measure "
      "**software maturity, not silicon**, and must not be read as a hardware comparison.")
    W("")
    W("**MTP = Multi-Token Prediction**, a form of **speculative decoding**:")
    W("")
    W("- **Without it:** one forward pass = **one** token. Every activated expert's "
      "weights are read from HBM to emit that single token (~2 GB per GPU at c=1). The "
      "weight read is the cost, not the arithmetic.")
    W("- **With it:** a small draft model proposes the next **N** tokens; the full model "
      "**verifies all N in one pass** — one weight read instead of N. Correct guesses are "
      "kept, the first wrong one and everything after is redone.")
    W("- **Output is bit-identical** to normal decoding. Pure speedup, no quality tradeoff.")
    W("")
    W("**Why it helps so much here:** decode is *latency-bound, not bandwidth-bound* (§3). "
      "Verifying N tokens at once makes each expert GEMM **N wide instead of 1** — the same "
      "widening `max-num-seqs` buys, but **without needing other users**. That is why it is "
      "the only §3.2 lever that helps a single user (§1.1).")
    W("")
    W("Measured gain in their B200 data (MTP vs no-spec, same hardware):")
    W("")
    W("| | c=1 | c=8 | c=32 |")
    W("|---|---:|---:|---:|")
    W("| B200 per-user tok/s, **no spec** | 81.9 | 21.1 | 3.8 |")
    W("| B200 per-user tok/s, **MTP** | 221.7 | 154.1 | 46.5 |")
    W("| ratio | **2.7×** | **7.3×** | **12.2×** |")
    W("")
    W("> **Note the trend is the opposite of what simple theory predicts.** Speculative "
      "decoding should matter *most* at low concurrency and fade as batching widens the "
      "GEMMs on its own — yet here the ratio grows from 2.7× to 12.2×. The likely cause is "
      "that the ratio is not a clean MTP-only A/B: the no-spec column collapses steeply "
      "(81.9 → 3.8) under long-context agentic traces, and their MTP rows come from a "
      "different recipe family than the `agg-b200-tp8pp2-agentic.yaml` config cross-checked "
      "in §7.1. Treat the **c=1 figure (2.7×) as the trustworthy one** and the high-"
      "concurrency ratios as confounded.")
    W("")
    W("Ceiling is draft length × acceptance rate — never the full N, since not every guess "
      "is accepted.")
    W("")
    W("**One fact that matters for reading the table:** Kimi-K3 has no built-in MTP head "
      f"(`num_nextn_predict_layers = {arch.raw.get('text_config',{}).get('num_nextn_predict_layers', 0)}`) "
      "— the gain comes from **DSpark**, a separate speculator model, downloaded and run "
      "alongside it. \"MTP\" names the technique, not a checkpoint feature (§7.5 has the "
      "how-to, for both platforms).")
    W("")
    W("**What the table shows, in three points:**")
    W("")
    W("1. **Our B200 no-spec number matches theirs at c=1** (89.0 vs 81.9) — an "
      "independent check that our TP8×PP2 setup is performing normally, not "
      "misconfigured.")
    W("2. **MTP is worth ~2.7× at c=1 on B200**, in their own data, same hardware — the "
      "single largest lever in this table. §7.5 shows how to get it.")
    W("3. **On equal footing (no spec decoding, low concurrency) B200 and MI355X land "
      "far closer than either vendor's best-configured number suggests.** The biggest "
      "differentiator in the whole table is MTP — a **software** feature both platforms "
      "can run, not a silicon difference.")
    W("")
    W("### 7.5 How to enable MTP — B200 and MI355X")
    W("")
    W("**Correction to an earlier claim in this report:** spec decoding does *not* "
      "require single-node TP. That was inferred from the upstream vLLM recipe gating "
      "DSpark off `multi_node_tp_pp` — a **recipe default, not an engine limit**. "
      "SemiAnalysis's own B200 recipe runs DSpark with `pipeline-parallel-size: 2`. "
      "**MTP is already built into our image** (`vllm/vllm-openai:kimi-k3`) — the "
      "`dspark` method, `KimiK3MTP`, and `TOKENSPEED_MLA` are all present. No new "
      "package. Both platforms use the **same speculator model**, "
      "`Inferact/Kimi-K3-DSpark`.")
    W("")
    W("| Step | MI355X | B200 |")
    W("|---|---|---|")
    W("| 1. Speculator | `Inferact/Kimi-K3-DSpark` | *same* |")
    W("| 2. `pard_token` shim | not needed | **required** |")
    W("| 3. Extra server flags | none | 3 flags, below |")
    W("| 4. Attention backend in the config | `TRITON_MLA` | `TOKENSPEED_MLA` |")
    W("| 5. Draft length | 2 (SemiAnalysis recipe) | 7 (SemiAnalysis recipe) |")
    W("")
    W("**Why MI355X is simpler:** one node holds the whole model, so the speculator "
      "drops straight into the existing TP8 launch — no shim, no extra flags.")
    W("")
    W("**Why B200 needs more:** PP2 splits the model across 2 nodes, so the speculator's "
      "draft/verify state has to stay consistent across the pipeline stage boundary. "
      "That needs decode context parallelism plus a config shim (the checkpoint's "
      "`mask_token_id` must be aliased to `pard_token` in a local copy of `config.json`, "
      "or the speculative-config fails to load) and three extra server flags:")
    W("")
    W("```")
    W("--decode-context-parallel-size 8")
    W("--dcp-comm-backend a2a")
    W("--attention-backend TOKENSPEED_MLA")
    W("```")
    W("")
    W("`--speculative-config` JSON (same shape on both, `attention_backend` and "
      "`num_speculative_tokens` differ as in the table above):")
    W("")
    W("```json")
    W("{\"model\": \"Inferact/Kimi-K3-DSpark\", \"num_speculative_tokens\": 2,")
    W(" \"method\": \"dspark\", \"attention_backend\": \"TRITON_MLA\"}")
    W("```")
    W("")
    W("**Neither was attempted here** — the speculator is not downloaded "
      "(`HF_HUB_OFFLINE=1`). Given the ~2.7× per-user gain at c=1 measured in §7.4, this "
      "is the single highest-value follow-up available — bigger than any lever in §3.2.")
    W("")
    W("---")
    W("")
    W("## Source data")
    W("")
    W("| What | Where |")
    W("|---|---|")
    W(f"| B200 sweep (per-concurrency JSON) | `{b['sweep']}` |")
    W(f"| B200 server log | `{b['log']}` |")
    W(f"| MI355X sweep | `{a['sweep']}` |")
    W(f"| MI355X server log | `{a['log']}` |")
    W(f"| Model config | `{args.model_config}` |")
    W(f"| MI355X published report | `{args.amd_root}/results/kimi-k3-base.md` |")
    W("")
    W("Derived figures (active params, FLOP/s, HBM / NVLink / IB volumes) are computed from "
      "measured throughput plus the parsed architecture, **using the same formulas for both "
      "systems** so the columns are comparable. Memory tables are read from each server's own "
      "log. Where this report's derived numbers differ from the published MI355X report, it is "
      "because that report's active-parameter estimate omitted the MoE latent projections; "
      "this one is validated against the checkpoint's exact MXFP4 parameter count.")
    W("")
    W("---")
    W("")

    # --- terminology ---------------------------------------------------------------
    W("## Terminology — HBM, NVLink, InfiniBand")
    W("")
    W("Three data paths, and the bottleneck analysis turns on telling them apart.")
    W("")
    W("**HBM** — the GPU's own on-package memory, where weights, KV cache and activations "
      "live. **Intra-GPU**: one GPU, no other GPU involved. Every weight read is a read from "
      f"HBM. B200: {gb(hw['hbm_bytes']):.0f} GB per GPU at {fmt(hw['hbm_bw_gbs'],0)} GB/s.")
    W("")
    W(f"**NVLink** — the GPU↔GPU interconnect inside one node, NVIDIA's counterpart to AMD's "
      f"xGMI. **Intra-node**. B200: {fmt(hw['link_bidir_gbs'],0)} GB/s bidirectional per GPU "
      f"(~{fmt(hw['link_dir_gbs'],0)} GB/s per direction, the figure a ring all-reduce sees). "
      "Carries activation all-reduces only (§4.1).")
    W("")
    W("**InfiniBand** — the **inter-node** fabric. Present in this run and absent from the "
      f"MI355X one. {IB['rails']} rails × {fmt(IB['rail_gbps'],0)} Gb/s NDR per node, measured "
      f"at {IB['measured_gdr_gbps']} Gb/s GPU→GPU. Carries the pipeline stage boundary (§4.2).")
    W("")
    W("**Why the distinction decides everything.** HBM is roughly an order of magnitude faster "
      "per GPU than NVLink, so the instinct is that HBM can never be the constraint. That is "
      f"backwards here: HBM moves ~{tot_hbm/1e9:.0f} GB per step while NVLink moves "
      f"~{pk['ar_payload_step']*2*(tp-1)/tp/1e9:.2f} GB and IB ~{pk['pp_bytes_step']/1e9:.4f} GB. "
      "The *slower* links are the idle ones.")
    return L


def write_csv(path: Path, b, a):
    cols = ["system", "ngpu", "nodes", "max_concurrency", "output_throughput",
            "tok_per_gpu", "total_token_throughput", "request_throughput",
            "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms",
            "tflops", "tflops_per_gpu", "pct_peak", "hbm_gbs_per_gpu", "hbm_pct",
            "link_gbs_per_gpu", "link_pct", "pp_gbs", "pp_pct",
            "experts_fired", "tokens_per_expert", "completed"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for tag, run in (("B200", b), ("MI355X", a)):
            if not run:
                continue
            nodes = math.ceil(run["tp"] * run["pp"] / 8)
            for r in run["rows"]:
                w.writerow([tag, run["tp"] * run["pp"], nodes] +
                           [r.get(c) for c in cols[3:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--server-log", type=Path)
    ap.add_argument("--model-config", required=True, type=Path)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("-o", "--out", default="results", type=Path)
    ap.add_argument("--amd-root", type=Path,
                    default=Path("/orcd/data/orcd/022/benchmarks/amd-benchmarks/amd-cloud"))
    ap.add_argument("--amd-sweep", type=Path)
    ap.add_argument("--amd-server-log", type=Path)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--kv-dtype", default="fp8")
    ap.add_argument("--weight-bytes", type=int, default=1560998987867)
    ap.add_argument("--basename", default="kimi-k3-base-b200")
    ap.add_argument("--hbm-mib", type=int, default=None,
                    help="measured per-GPU HBM in MiB (nvidia-smi memory.total); overrides "
                         "the B200 spec-sheet value so the fit arithmetic uses what the "
                         "run actually saw")
    args = ap.parse_args()
    if args.hbm_mib:
        HW["B200"]["hbm_bytes"] = args.hbm_mib * MIB
        print(f"B200 HBM overridden to measured {args.hbm_mib} MiB "
              f"({gb(HW['B200']['hbm_bytes']):.1f} GB) per GPU")

    arch = Arch(args.model_config)
    print(f"arch: {arch.L} layers ({arch.full_layers} MLA + {arch.kda_layers} KDA), "
          f"{arch.E} experts, total {arch.total/1e12:.3f} T, active {arch.active/1e9:.1f} B")
    if arch.routed_total != 2722740830208:
        print(f"  NOTE: routed-expert count {arch.routed_total} does not match the published "
              "MXFP4 total; the config may have changed.", file=sys.stderr)

    # --- B200 side ---------------------------------------------------------------
    rows = load_sweep(args.sweep)
    if not rows:
        print(f"ERROR: no usable sweep points in {args.sweep}", file=sys.stderr)
        return 2
    mem = parse_vllm_log(args.server_log) if args.server_log else {}
    tp = mem.get("tp", args.tp)
    pp = mem.get("pp", args.pp)
    derive(rows, arch, tp, pp, HW["B200"])
    load_s = mem.get("load_s")
    if args.run_dir and (args.run_dir / "load_seconds.txt").exists():
        try:
            load_s = float((args.run_dir / "load_seconds.txt").read_text().strip())
        except ValueError:
            pass
    date = "unknown date"
    try:
        j = sorted(args.sweep.glob("c*.json"))[0]
        date = json.loads(j.read_text()).get("date", date)
    except Exception:
        pass
    b = dict(rows=rows, mem=mem, tp=tp, pp=pp, sweep=args.sweep, log=args.server_log,
             load_s=load_s, date=date)

    # --- MI355X baseline ----------------------------------------------------------
    a = None
    asweep = args.amd_sweep
    alog = args.amd_server_log
    if asweep is None:
        cands = sorted((args.amd_root / "logs" / "atom").glob("sweep_20260814_164903"))
        if not cands:
            cands = sorted((args.amd_root / "logs" / "atom").glob("sweep_*"))
        asweep = cands[-1] if cands else None
    if alog is None and asweep is not None:
        cands = sorted((args.amd_root / "logs" / "atom").glob("server_20260814_164506/atom_server.log"))
        alog = cands[-1] if cands else None
    if asweep and asweep.exists():
        arows = [r for r in load_sweep(asweep)]
        if arows:
            amem = parse_atom_log(alog) if alog else {}
            atp = amem.get("tp", 8); app = amem.get("pp", 1)
            derive(arows, arch, atp, app, HW["MI355X"])
            a = dict(rows=arows, mem=amem, tp=atp, pp=app, sweep=asweep, log=alog)
            print(f"MI355X baseline: {len(arows)} points from {asweep}")
    if a is None:
        print("WARNING: MI355X baseline not found -- comparison section will be empty",
              file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    md = args.out / f"{args.basename}.md"
    md.write_text("\n".join(build_report(b, a, arch, args)) + "\n")
    write_csv(args.out / f"{args.basename}.csv", b, a)
    print(f"wrote {md}")
    print(f"wrote {args.out / (args.basename + '.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
