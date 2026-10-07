#!/usr/bin/env python3
"""Build results/ubuntu/kimi-amd-recipe.md: AMD's Kimi-K3 recipe PDF (vLLM v0.29.0) vs the
PDF's own table (128K/1K) and vs the current recipes at 1K/1K (AMD vLLM recipe from
recipes.vllm.ai, atom/run_kimi_recipe.sh; ATOM base / max-num-seqs 256 / 512).

Reads results/<node>/kimi-amd-recipe/{isl128k,isl1k}.csv from either node, so it can run after
each arm; a missing arm is shown as pending. Run by run_kimi_amdrecipe.sh.
"""
import csv
from pathlib import Path

ROOT = Path("/orcd/data/orcd/022/benchmarks/amd-ubuntu/results")
OUT = ROOT / "ubuntu" / "kimi-amd-recipe.md"
# amd-kimi-k3-recipe.pdf, "Benchmark Results": total_token_throughput / 8, 128K/1K
PDF = {1: 874, 2: 1090, 4: 1235, 8: 1199, 16: 1179, 32: 1118, 64: 1096, 128: 1110}


def rd(p):
    if not p.exists():
        return None
    return {int(r["conc"]): r for r in csv.DictReader(open(p)) if r.get("tps")}


def arm(name):
    for node in ("node6100", "node6101"):
        d = rd(ROOT / node / "kimi-amd-recipe" / f"{name}.csv")
        if d:
            return node, d
    return None, None


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return None


def f(x, nd=0):
    return "—" if x is None else f"{x:,.{nd}f}"


def ratio(a, b, bold_lo=0.95, bold_hi=1.05):
    if a is None or b is None or b == 0:
        return "—"
    v = a / b
    s = f"{v:.2f}x"
    return f"**{s}**" if v < bold_lo or v > bold_hi else s


def main():
    L = ["# amd-ubuntu — Kimi-K3 with AMD's recipe PDF (vLLM v0.29.0)", ""]
    L += ["Recipe: `amd-kimi-k3-recipe.pdf` — `vllm/vllm-openai-rocm:v0.29.0`, TP8, one server for the sweep, "
          "`max-num-seqs 128`, `max-num-batched-tokens 4096`, `gpu-memory-utilization 0.95`, cudagraph "
          "`FULL_DECODE_ONLY`, `+fused_rms_norm_gated`, AITER MXFP4 MoE (`VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1`), "
          "no speculative decoding. Scripts: `atom/run_kimi_amdrecipe.sh` (`isl128k` on node6100, `isl1k` on node6101), "
          "weights from `/scratch/Kimi-K3`, run under apptainer.", ""]

    # ---- 1. reproduce the PDF -----------------------------------------------------------
    node, a = arm("isl128k")
    L += ["## 1. AMD's workload (ISL/OSL 128K/1K): ours vs the PDF", ""]
    if not a:
        L += ["*Pending: the `isl128k` arm has not produced results yet.*", ""]
    else:
        L += [f"Total tok/s per GPU = total_token_throughput / 8 (the PDF's metric). Ours: {node}. "
              "Apple-to-apple: same image, server flags and client settings; different machine.", "",
              "| C | ours total tok/s/GPU | PDF total tok/s/GPU | ours / PDF | out tok/s | TTFT med ms | TPOT med ms |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for c in sorted(set(PDF) | set(a)):
            r = a.get(c, {})
            t = num(r, "ttps_gpu")
            L.append(f"| {c} | {f(t)} | {f(PDF.get(c))} | {ratio(t, PDF.get(c))} | {f(num(r,'tps'),1)} | "
                     f"{f(num(r,'ttft'))} | {f(num(r,'tpot'),2)} |")
        L.append("")

    # ---- 2. vs current recipes at 1K/1K ---------------------------------------------------
    node, b = arm("isl1k")
    rec = rd(ROOT / "node6101" / "kimi-recipe" / "kimi-k3-recipe.csv") or {}
    atom = {}
    p = ROOT / "node6100" / "kimi-cloud" / "atom.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            atom[int(r["max_concurrency"])] = {"tps": r["output_throughput"], "tpot": r["median_tpot_ms"],
                                                "ttft": r["median_ttft_ms"], "cfg": "base"}
    for fn, cfg in (("kimi-k3-maxseqs.csv", "max-num-seqs 256"), ("kimi-k3-maxseqs512.csv", "max-num-seqs 512")):
        for c, r in (rd(ROOT / "node6100" / "kimi-cloud" / fn) or {}).items():
            if c not in atom or float(r["tps"]) > float(atom[c]["tps"]):
                atom[c] = {"tps": r["tps"], "tpot": r["tpot"], "ttft": r["ttft"], "cfg": cfg}
    L += ["## 2. Our workload (ISL/OSL 1K/1K): AMD PDF recipe vs current recipes", ""]
    if not b:
        L += ["*Pending: the `isl1k` arm has not produced results yet.*", ""]
    else:
        L += [f"PDF recipe: {node}, range ratio 0.8 and 10 × C prompts like the earlier Kimi runs. "
              "Current vLLM recipe: `results/node6101/kimi-recipe/` (DSpark speculative decoding up to C=14, "
              "DCP 8 + CPU KV offload above, server re-tuned per C). ATOM: best of base / max-num-seqs 256 / 512 "
              "(`results/node6100/kimi-cloud/`). Ratios are PDF recipe / other: tok/s above 1 and TPOT below 1 favour the PDF recipe; "
              "bold = more than 5%.", "",
              "| C | PDF tok/s | vLLM recipe tok/s | ATOM tok/s (config) | PDF / vLLM recipe | PDF / ATOM | PDF TPOT ms | vLLM recipe TPOT ms | ATOM TPOT ms | TPOT PDF / vLLM recipe | TPOT PDF / ATOM | PDF TTFT ms |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in sorted(set(b) | set(rec) | set(atom)):
            x, y, z = b.get(c, {}), rec.get(c, {}), atom.get(c, {})
            if not x:
                continue
            xt, yt, zt = num(x, "tps"), num(y, "tps"), num(z, "tps")
            xp, yp, zp = num(x, "tpot"), num(y, "tpot"), num(z, "tpot")
            L.append(f"| {c} | {f(xt)} | {f(yt)} | {f(zt)}{' (' + z['cfg'] + ')' if z else ''} | "
                     f"{ratio(xt, yt)} | {ratio(xt, zt)} | {f(xp,2)} | {f(yp,2)} | {f(zp,2)} | "
                     f"{ratio(xp, yp)} | {ratio(xp, zp)} | {f(num(x,'ttft'))} |")
        L += ["", "C = 256 is above the PDF recipe's `max-num-seqs 128`, so half the requests queue; it is kept to line up with ATOM.", ""]

    L += ["## Is this apple-to-apple?", "",
          "- **§1 (vs the PDF): nearly.** Same image, server flags and client settings on the same GPU type. Differences: "
          "this machine, apptainer instead of docker, weights from local disk, and one short warm-up before the sweep.",
          "- **§2 (vs current recipes): no, recipe vs recipe.** Same hardware, model, ISL/OSL, concurrency and prompt count, "
          "but each recipe has its own vLLM/ATOM version and server settings (the current vLLM recipe uses speculative "
          "decoding, the PDF recipe does not; ATOM is a different engine).", "",
          "Per-arm detail: `results/<node>/kimi-amd-recipe/{isl128k,isl1k}.md`; logs: "
          "`logs/<node>/kimi-amd-recipe/atom/kimi_amdrecipe_<arm>_<ts>/`."]
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
