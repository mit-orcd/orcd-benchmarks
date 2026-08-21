#!/usr/bin/env python3
"""Summarise the section-3.2 improvement experiments into results/kimi-k3-improve-b200.md.

Writes ONLY that file (plus a .csv). Never touches kimi-k3-base-b200.* or RUN-SUMMARY.md.

Usage:
  analyze-improve-b200.py --run-dir logs/improve_<ts> [-o results]
                          [--baseline-sweep logs/kimi_base_.../sweep]
"""
import argparse, csv, json, re, sys
from pathlib import Path

FIELDS = ["max_concurrency", "output_throughput", "total_token_throughput",
          "request_throughput", "median_ttft_ms", "p99_ttft_ms",
          "median_tpot_ms", "p99_tpot_ms", "completed", "duration"]

# Architecture constants for the expert-firing model. Derived and validated in
# analyze-kimi-b200.py (routed-expert params reproduce the checkpoint's MXFP4 count
# exactly); restated here so this script stands alone and cannot perturb that one.
E, TOPK = 896, 16


def experts_fired(batch):
    return E * (1.0 - (1.0 - 1.0 / E) ** (batch * TOPK))


def tokens_per_expert(batch):
    f = experts_fired(batch)
    return (batch * TOPK / f) if f else 0.0


def load_sweep(d: Path):
    rows = []
    if not d or not d.exists():
        return rows
    for j in sorted(d.glob("c*.json")):
        try:
            data = json.loads(j.read_text())
        except Exception:
            continue
        r = {k: data.get(k) for k in FIELDS}
        if not r.get("max_concurrency"):
            m = re.match(r"c(\d+)\.json", j.name)
            if m:
                r["max_concurrency"] = int(m.group(1))
        if not r.get("completed"):
            continue
        rows.append(r)
    rows.sort(key=lambda r: r["max_concurrency"])
    return rows


def status_of(run: Path, tag: str):
    f = run / f"{tag}.status"
    return f.read_text().strip() if f.exists() else "(no status file)"


def fmt(v, nd=1):
    try:
        return f"{float(v):,.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("-o", "--out", default=Path("results"), type=Path)
    ap.add_argument("--baseline-sweep", type=Path)
    ap.add_argument("--basename", default="kimi-k3-improve-b200")
    args = ap.parse_args()
    run = args.run_dir

    base = load_sweep(args.baseline_sweep) if args.baseline_sweep else []
    base_peak = max(base, key=lambda r: r["output_throughput"] or 0) if base else None

    arms = {t: load_sweep(run / f"{t}_sweep") for t in
            ("lever1_mns256", "lever1_mns512", "lever3_ep", "lever2_spec")}

    L = []
    W = L.append
    W("# Kimi-K3 on 2 × 8 × B200 — improvement experiments")
    W("")
    W("Tests the four levers proposed in section 3.2 of `kimi-k3-base-b200.md`. The "
      "baseline there is `max_num_seqs=64`, TP8 × PP2, peak "
      f"**{fmt(base_peak['output_throughput']) if base_peak else '1,696.4'} tok/s** at c=64, "
      "with HBM at only ~23% of peak because each expert saw just **1.7 tokens** per step.")
    W("")
    W("**The question every arm below is testing:** the baseline is *latency-bound, not "
      "bandwidth-bound* — so does widening the per-expert GEMM actually convert the idle "
      "77% of HBM into throughput?")
    W("")
    W(f"Run: `{run}`")
    W("")
    W("---")
    W("")

    # ---------- verdict table ----------
    W("## 0. Verdict — what to actually do")
    W("")
    W("| Lever | Recommended? | Status | Result |")
    W("|---|---|---|---|")

    def peak(tag):
        rs = arms.get(tag) or []
        return max(rs, key=lambda r: r["output_throughput"] or 0) if rs else None

    p256, p512, pep = peak("lever1_mns256"), peak("lever1_mns512"), peak("lever3_ep")
    b = base_peak["output_throughput"] if base_peak else None

    def gain(p):
        if p and b:
            return f"{p['output_throughput']/b:.2f}× baseline ({fmt(p['output_throughput'])} tok/s)"
        return "no data"

    # Best across BOTH cap arms, not whichever happens to be non-empty first --
    # cap 512 can out-perform cap 256 and the verdict must not under-report it.
    l1_arms = [x for x in (p256, p512) if x]
    l1_best = max(l1_arms, key=lambda r: r["output_throughput"]) if l1_arms else None
    W(f"| **1. Raise `max_num_seqs`** | ⭐ **YES — do this first** | "
      f"{'ran' if l1_best else 'no data'} | {gain(l1_best)} |")
    W(f"| 3. Expert parallelism | ⭐ worth testing (2nd priority) | "
      f"{'ran' if pep else status_of(run,'lever3_ep')} | {gain(pep)} |")
    W(f"| 2. Speculative decoding | ✗ unavailable on this layout | "
      f"{status_of(run,'lever2_spec')} | gated off `multi_node_tp_pp`; PP is mandatory on B200 |")
    W("| 4. Prefill/decode disagg | ✗ needs 2× the hardware | not run | "
      "≥32 GPUs (4 nodes) required; 16 available |")
    W("")
    W("---")
    W("")

    # ---------- lever 1 ----------
    W("## 1. ⭐ Raise `--max-num-seqs` — the recommended lever")
    W("")
    if not (arms["lever1_mns256"] or arms["lever1_mns512"]):
        W("_No data: neither arm produced sweep points._")
    else:
        W("| Cap | Conc | tok/s | vs baseline peak | TTFT med (ms) | TPOT med (ms) | "
          "experts fired | tokens/expert |")
        W("|---:|---:|---:|---:|---:|---:|---:|---:|")
        if base_peak:
            c = base_peak["max_concurrency"]
            W(f"| 64 *(baseline)* | {c} | {fmt(base_peak['output_throughput'])} | 1.00× | "
              f"{fmt(base_peak['median_ttft_ms'])} | {fmt(base_peak['median_tpot_ms'],2)} | "
              f"{experts_fired(c):.0f} | {tokens_per_expert(c):.1f} |")
        for tag, cap in (("lever1_mns256", 256), ("lever1_mns512", 512)):
            for r in arms[tag]:
                c = r["max_concurrency"]
                rel = f"{r['output_throughput']/b:.2f}×" if b else "—"
                W(f"| {cap} | {c} | {fmt(r['output_throughput'])} | {rel} | "
                  f"{fmt(r['median_ttft_ms'])} | {fmt(r['median_tpot_ms'],2)} | "
                  f"{experts_fired(c):.0f} | {tokens_per_expert(c):.1f} |")
        W("")
        W("**How to read this.** `tokens/expert` is the quantity that was binding: at the "
          "baseline's 1.7 tokens each expert GEMM is a matrix-*vector* product with too "
          "little memory-level parallelism to saturate HBM. Every row that raises it is "
          "buying back the idle bandwidth. Weight bytes plateau above batch ~512, so "
          "beyond that additional tokens are close to free.")
    W("")

    # ---------- lever 3 ----------
    W("## 2. Expert parallelism — matched A/B against lever 1")
    W("")
    W("EP is compared against the `max_num_seqs=256` arm at the **same cap and the same "
      "concurrencies**, so EP is the only variable. This matters: the MI355X study's first "
      "EP result compared an EP arm at cap 256 against a TP-only arm at cap 64, and only "
      "one row of it was a valid comparison.")
    W("")
    if not arms["lever3_ep"]:
        W(f"_Did not produce data. Status: `{status_of(run,'lever3_ep')}`._")
        W("")
        W("On MI355X, EP is hard-blocked for this model (ATOM raises `NotImplementedError` "
          "for EP with the MXFP4 SiTUv2 kernel). If it also fails here, the mechanism is "
          "unavailable on both vendors' stacks for MXFP4 MoE.")
    else:
        W("| Conc | TP-only (cap 256) | EP (cap 256) | EP/TP | TP TPOT | EP TPOT |")
        W("|---:|---:|---:|---:|---:|---:|")
        tp_map = {r["max_concurrency"]: r for r in arms["lever1_mns256"]}
        for r in arms["lever3_ep"]:
            c = r["max_concurrency"]
            t = tp_map.get(c)
            ratio = f"{r['output_throughput']/t['output_throughput']:.2f}×" if t else "—"
            W(f"| {c} | {fmt(t['output_throughput']) if t else '—'} | "
              f"{fmt(r['output_throughput'])} | {ratio} | "
              f"{fmt(t['median_tpot_ms'],2) if t else '—'} | {fmt(r['median_tpot_ms'],2)} |")
        W("")
        W("**Why EP could help even though the interconnect is idle.** Under TP each GPU "
          "reads a *thin slice* of every activated expert and computes a partial GEMM. "
          "Under EP each GPU holds *whole* experts and computes complete GEMMs — fewer, "
          "larger, more contiguous weight reads with more memory-level parallelism per "
          "read. That attacks the same bottleneck as lever 1 by a different route, which "
          "is why it is worth measuring rather than dismissing on the bandwidth numbers.")
    W("")

    # ---------- lever 2 ----------
    W("## 3. Speculative decoding (DSpark) — unavailable here")
    W("")
    W(f"Status: `{status_of(run,'lever2_spec')}`")
    W("")
    log = run / "lever2_spec_server" / "vllm_server.log"
    if log.exists():
        errs = sorted(set(re.findall(
            r"(?:ValueError|NotImplementedError|RuntimeError|AssertionError|TypeError): .{0,200}",
            log.read_text(errors="replace"))))[:4]
        if errs:
            W("```")
            for e in errs:
                W(e)
            W("```")
            W("")
    W("The vLLM recipe gates DSpark off the `multi_node_tp_pp` profile — it does not "
      "compose with pipeline parallelism yet (vllm-project/vllm#50098). On B200 that is "
      "decisive rather than inconvenient: **PP is mandatory**, because the 1561 GB "
      "checkpoint does not fit one node's 1538 GB. So the single most promising "
      "throughput lever after batch size is structurally unavailable on this hardware — "
      "and it is available on MI355X, which serves the model on one node with no PP.")
    W("")
    W("> This is a real, quantifiable cost of needing two nodes, separate from the "
      "pipeline bubble.")
    W("")

    # ---------- lever 4 ----------
    W("## 4. Prefill/decode disaggregation — needs 2× the hardware")
    W("")
    st = (run / "lever4_pd.status")
    alloc = "16 GPUs / 2 nodes"
    if st.exists():
        m = re.search(r"allocated_gpus=(\d+)", st.read_text())
        if m:
            alloc = f"{m.group(1)} GPUs / {int(m.group(1))//8} nodes"
    W("Not run, on arithmetic rather than preference:")
    W("")
    W("| | |")
    W("|---|---|")
    W("| P/D needs | 2 independent engine instances (prefill pool + decode pool) |")
    W("| Weights per instance | 1561 GB — each holds a **full** copy |")
    W("| GPUs per instance | 16 (2 nodes), since one node's 1538 GB cannot hold it |")
    W("| **Minimum for P/D** | **32 GPUs / 4 nodes** |")
    W(f"| This allocation | {alloc} |")
    W("")
    W("Testing it would require doubling the reservation. Worth revisiting only if the "
      "model is ever served on hardware where one instance fits in a single node.")
    W("")
    W("---")
    W("")

    # ---------- recommendation ----------
    W("## 5. Recommendation")
    W("")
    if p256 or p512:
        best = max([x for x in (p256, p512) if x], key=lambda r: r["output_throughput"])
        cap = 256 if best is p256 else 512
        W(f"**Raise `--max-num-seqs` to {cap}.** Best measured: "
          f"**{fmt(best['output_throughput'])} tok/s** at c={best['max_concurrency']}"
          + (f", {best['output_throughput']/b:.2f}× the cap-64 baseline" if b else "")
          + f", at a median TPOT of {fmt(best['median_tpot_ms'],2)} ms. One flag, no extra "
            "hardware, and the KV memory was already provisioned.")
    else:
        W("**Raise `--max-num-seqs`** remains the recommendation on mechanism, but this "
          "run produced no data for it — re-run before acting.")
    W("")
    W("Then, in order: measure EP (§2) if it loads; treat spec decoding (§3) as blocked "
      "until vLLM composes it with PP; leave P/D (§4) until there is 4-node capacity.")
    W("")
    W("---")
    W("")
    W("## Source data")
    W("")
    W("| What | Where |")
    W("|---|---|")
    for tag in ("lever1_mns256", "lever1_mns512", "lever3_ep", "lever2_spec"):
        W(f"| {tag} | `{run}/{tag}_sweep/`, `{run}/{tag}_server/` |")
    if args.baseline_sweep:
        W(f"| baseline (cap 64) | `{args.baseline_sweep}` |")
    W(f"| driver log | `{run}/STATE.txt` |")
    W("")
    W("Baseline report: `kimi-k3-base-b200.md` (unmodified by this run).")

    args.out.mkdir(parents=True, exist_ok=True)
    md = args.out / f"{args.basename}.md"
    md.write_text("\n".join(L) + "\n")

    with open(args.out / f"{args.basename}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "max_num_seqs", "max_concurrency", "output_throughput",
                    "median_ttft_ms", "median_tpot_ms", "request_throughput",
                    "experts_fired", "tokens_per_expert", "completed"])
        if base_peak:
            for r in base:
                c = r["max_concurrency"]
                w.writerow(["baseline", 64, c, r["output_throughput"], r["median_ttft_ms"],
                            r["median_tpot_ms"], r["request_throughput"],
                            f"{experts_fired(c):.0f}", f"{tokens_per_expert(c):.2f}",
                            r["completed"]])
        for tag, cap in (("lever1_mns256", 256), ("lever1_mns512", 512),
                         ("lever3_ep", 256), ("lever2_spec", 64)):
            for r in arms.get(tag) or []:
                c = r["max_concurrency"]
                w.writerow([tag, cap, c, r["output_throughput"], r["median_ttft_ms"],
                            r["median_tpot_ms"], r["request_throughput"],
                            f"{experts_fired(c):.0f}", f"{tokens_per_expert(c):.2f}",
                            r["completed"]])
    print(f"wrote {md}")
    print(f"wrote {args.out / (args.basename + '.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
