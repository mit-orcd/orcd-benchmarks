#!/usr/bin/env python3
"""Build two files:
  results/ubuntu/kimi-amd-recipe.md       the new AMD recipe 2026-10 (`amd-kimi-k3-recipe.pdf`, vLLM v0.29.0):
                                          ours vs the PDF's own table (128K/1K) and its 1K/1K results
  results/ubuntu/kimi-recipe-old-vs-new.md the new AMD recipe 2026-10 vs the old recipes at 1K/1K (vLLM recipe from
                                          recipes.vllm.ai, atom/run_kimi_recipe.sh; ATOM base / max-num-seqs 256 / 512)

Reads results/<node>/kimi-amd-recipe/{isl128k,isl1k}.csv from either node, so it can run after
each arm; a missing arm is shown as pending. Run by run_kimi_amdrecipe.sh.
"""
import csv
from pathlib import Path

ROOT = Path("/orcd/data/orcd/022/benchmarks/amd-ubuntu/results")
OUT = ROOT / "ubuntu" / "kimi-amd-recipe.md"
OUT_CMP = ROOT / "ubuntu" / "kimi-recipe-old-vs-new.md"
# amd-kimi-k3-recipe.pdf, "Benchmark Results": total_token_throughput / 8, 128K/1K
PDF = {1: 874, 2: 1090, 4: 1235, 8: 1199, 16: 1179, 32: 1118, 64: 1096, 128: 1110}

# hand-written reading of the old-vs-new table (2026-10-08 results)
CMP_NOTES = [
    "## Reading", "",
    "- **Old vLLM recipe vs new AMD recipe 2026-10: the old vLLM recipe is faster at every load where both ran** "
    "(new / old 0.76x at 1 user, 0.86–0.93x at 4–128, 0.69x at 256); its speculative decoding gives the largest "
    "gain at low load.",
    "- **ATOM vs new AMD recipe 2026-10: the new AMD recipe 2026-10 is faster up to 32 users** (1.10–1.20x at 1–8 users, 1.03x at 16–32) "
    "and slightly slower at 64–128 (0.94–0.96x).",
    "- **At 256 users the new AMD recipe 2026-10 falls behind both (0.67–0.69x)**: its `max-num-seqs 128` makes half the "
    "requests wait.",
    "- **Fastest per load on our nodes:** old vLLM recipe at 1, 4, 8, 64 and 128 users; new AMD recipe 2026-10 at 2, 16 and 32 "
    "(the old vLLM recipe was not run there); ATOM (`max-num-seqs` 256–512) at 256 users and above.",
    "- **Suggested setup:** the old vLLM recipe for interactive use (1–128 users); ATOM with `max-num-seqs` 256–512 "
    "for 256 users or more. The new AMD recipe 2026-10 is a good single setting for 2–32 users without speculative decoding; "
    "raise its `max-num-seqs` above 128 for heavier load.",
]


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
    L = ["# amd-ubuntu — Kimi-K3 with new AMD recipe 2026-10 (vLLM v0.29.0)", ""]
    L += ["Recipe: `amd-kimi-k3-recipe.pdf` — `vllm/vllm-openai-rocm:v0.29.0`, TP8, one server for the sweep, "
          "`max-num-seqs 128`, `max-num-batched-tokens 4096`, `gpu-memory-utilization 0.95`, cudagraph "
          "`FULL_DECODE_ONLY`, `+fused_rms_norm_gated`, AITER MXFP4 MoE (`VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1`), "
          "no speculative decoding. Scripts: `atom/run_kimi_amdrecipe.sh` (`isl128k` on node6100, `isl1k` on node6101), "
          "weights from `/scratch/Kimi-K3`, run under apptainer.", ""]

    # ---- 1. reproduce the PDF -----------------------------------------------------------
    node, a = arm("isl128k")
    L += ["## 1. AMD's workload (ISL/OSL 128K/1K): our nodes vs AMD's published numbers", "", "Our runs already use this same recipe: the new AMD recipe 2026-10 from `amd-kimi-k3-recipe.pdf` (same image `vllm/vllm-openai-rocm:v0.29.0`, server flags and client settings). AMD's published numbers are the results table printed in that PDF.", ""]
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
    L += ["## 2. Our workload (ISL/OSL 1K/1K)", ""]
    if not b:
        L += ["*Pending: the `isl1k` arm has not produced results yet.*", ""]
    else:
        L += [f"{node}, range ratio 0.8 and 10 × C prompts like the earlier Kimi runs. "
              "Comparison with the old recipes: [kimi-recipe-old-vs-new.md](kimi-recipe-old-vs-new.md).", "",
              "| C | out tok/s | total tok/s | TTFT med ms | TTFT p99 ms | TPOT med ms | TPOT p99 ms |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for c in sorted(b):
            x = b[c]
            L.append(f"| {c} | {f(num(x,'tps'))} | {f(num(x,'total_tps'))} | {f(num(x,'ttft'))} | "
                     f"{f(num(x,'ttft99'))} | {f(num(x,'tpot'),2)} | {f(num(x,'tpot99'),2)} |")
        L += ["", "C = 256 is above the recipe's `max-num-seqs 128`, so half the requests queue (TTFT jumps).", ""]

    # ---- 2b. old vs new recipes (separate file) ----------------------------------------
    C = ["# amd-ubuntu — Kimi-K3: old recipes vs the new AMD recipe 2026-10", "",
         "**Compared: the new AMD recipe 2026-10 (`amd-kimi-k3-recipe.pdf`, vLLM v0.29.0, no speculative decoding) vs the two old recipes "
         "(the vLLM recipe with speculative decoding, and ATOM), all on our nodes, ISL/OSL 1K/1K, 8 GPUs (TP8).** "
         "Recipe vs recipe, not hardware. New-recipe results alone: [kimi-amd-recipe.md](kimi-amd-recipe.md).", ""]
    if not b:
        C += ["*Pending: the `isl1k` arm has not produced results yet.*", ""]
    else:
        C += [f"new AMD recipe 2026-10: {node}, range ratio 0.8 and 10 × C prompts like the earlier Kimi runs. "
              "Old vLLM recipe: `results/node6101/kimi-recipe/` (DSpark speculative decoding up to C=14, "
              "DCP 8 + CPU KV offload above, server re-tuned per C). ATOM: best of base / max-num-seqs 256 / 512 "
              "(`results/node6100/kimi-cloud/`). Ratios are new AMD recipe 2026-10 / old recipe: tok/s above 1 and TPOT below 1 favour the new AMD recipe 2026-10; "
              "bold = more than 5%.", "",
              "| C | new tok/s | old vLLM recipe tok/s | ATOM tok/s (config) | new / old vLLM recipe | new / ATOM | new TPOT ms | old vLLM recipe TPOT ms | ATOM TPOT ms | TPOT new / old vLLM recipe | TPOT new / ATOM | new TTFT ms |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in sorted(set(b) | set(rec) | set(atom)):
            x, y, z = b.get(c, {}), rec.get(c, {}), atom.get(c, {})
            if not x:
                continue
            xt, yt, zt = num(x, "tps"), num(y, "tps"), num(z, "tps")
            xp, yp, zp = num(x, "tpot"), num(y, "tpot"), num(z, "tpot")
            C.append(f"| {c} | {f(xt)} | {f(yt)} | {f(zt)}{' (' + z['cfg'] + ')' if z else ''} | "
                     f"{ratio(xt, yt)} | {ratio(xt, zt)} | {f(xp,2)} | {f(yp,2)} | {f(zp,2)} | "
                     f"{ratio(xp, yp)} | {ratio(xp, zp)} | {f(num(x,'ttft'))} |")
        C += ["", "C = 256 is above the new AMD recipe 2026-10's `max-num-seqs 128`, so half the requests queue; it is kept to line up with ATOM.", ""]

    L += ["## Is this apple-to-apple?", "",
          "- **§1 (vs the PDF): nearly.** Same image, server flags and client settings on the same GPU type. Differences: "
          "this machine, apptainer instead of docker, weights from local disk, and one short warm-up before the sweep.",
          "",
          "Per-arm detail: `results/<node>/kimi-amd-recipe/{isl128k,isl1k}.md`; logs: "
          "`logs/<node>/kimi-amd-recipe/atom/kimi_amdrecipe_<arm>_<ts>/`."]
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")
    if b:
        C += ["## Is this apple-to-apple?", "",
              "**No, recipe vs recipe.** Same hardware, model, ISL/OSL, concurrency and prompt count, but each recipe has its "
              "own vLLM/ATOM version and server settings (the old vLLM recipe uses speculative decoding, the new AMD recipe 2026-10 does "
              "not; ATOM is a different engine).", ""]
        C += CMP_NOTES
        OUT_CMP.write_text("\n".join(C) + "\n")
        print(f"wrote {OUT_CMP}")


if __name__ == "__main__":
    main()
