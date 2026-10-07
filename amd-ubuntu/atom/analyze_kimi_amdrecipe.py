#!/usr/bin/env python3
"""Summarize run_kimi_amdrecipe.sh (AMD's Kimi-K3 recipe PDF) -> <arm>.{csv,md}.

    analyze_kimi_amdrecipe.py <arm> <sweep_dir> [<sweep_dir> ...] -o <results_dir>

Input: <sweep>/Kimi-K3-MXFP4_isl*_osl*_c<C>.json (vllm bench serve). Per concurrency the last
sweep (argument order) with a result wins. CSV columns follow the other Kimi-K3 CSVs plus
ttps_gpu = total_token_throughput / 8, the PDF's metric.
"""
import argparse, csv, json, re
from pathlib import Path

COLS = ["conc", "tps", "total_tps", "ttps_gpu", "rps", "ttft", "ttft99", "tpot", "tpot99",
        "completed", "in_tokens", "out_tokens"]


def load(j: Path):
    x = json.load(open(j))
    c = int(re.search(r"_c(\d+)\.json$", j.name).group(1))
    t = x.get("total_token_throughput")
    return {"conc": c, "tps": x.get("output_throughput"), "total_tps": t,
            "ttps_gpu": t / 8 if t else None, "rps": x.get("request_throughput"),
            "ttft": x.get("median_ttft_ms"), "ttft99": x.get("p99_ttft_ms"),
            "tpot": x.get("median_tpot_ms"), "tpot99": x.get("p99_tpot_ms"),
            "completed": x.get("completed"), "in_tokens": x.get("total_input_tokens"),
            "out_tokens": x.get("total_output_tokens")}


def f(x, nd):
    return "—" if x in ("", None) else f"{x:,.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("sweep", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for s in a.sweep:
        for j in sorted(Path(s).glob("Kimi-K3-MXFP4_isl*_osl*_c*.json")):
            r = load(j); rows[r["conc"]] = r
    rows = [rows[c] for c in sorted(rows)]
    with open(out / f"{a.arm}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader(); w.writerows(rows)
    md = [f"# Kimi-K3, AMD recipe PDF (vLLM v0.29.0), arm {a.arm}", "",
          "Sweeps: " + ", ".join(f"`{Path(s).name}`" for s in a.sweep), "",
          "| C | out tok/s | total tok/s | total tok/s/GPU | TTFT med ms | TTFT p99 ms | TPOT med ms | TPOT p99 ms | completed |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['conc']} | {f(r['tps'],1)} | {f(r['total_tps'],1)} | {f(r['ttps_gpu'],0)} | "
                  f"{f(r['ttft'],0)} | {f(r['ttft99'],0)} | {f(r['tpot'],2)} | {f(r['tpot99'],2)} | {r['completed']} |")
    (out / f"{a.arm}.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} points -> {out}/{a.arm}.{{csv,md}}")


if __name__ == "__main__":
    main()
