#!/usr/bin/env python3
"""Summarize run_kimi_recipe.sh -> kimi-k3-recipe.{md,csv}.

Input: <sweep>/c<C>/{c<C>.json, server_cmd.sh}. CSV columns follow the other Kimi-K3 CSVs
(conc,tps,total_tps,rps,ttft,ttft99,tpot,tpot99,completed) plus the recipe settings per point
(draft_k, max_num_seqs, dcp, offload) and the spec-decode acceptance if the server logged it.

    analyze_kimi_recipe.py <sweep_dir> [<sweep_dir> ...] -o <results_dir>

Several sweeps (e.g. a rerun of the failed points): per concurrency the last sweep (in argument
order) that has a result wins; a point without a result in any sweep is kept as a "—" row.
"""
import argparse, csv, json, re
from pathlib import Path

COLS = ["conc", "tps", "total_tps", "rps", "ttft", "ttft99", "tpot", "tpot99", "completed",
        "draft_k", "max_num_seqs", "dcp", "offload", "accept_len"]


def point(d: Path):
    c = int(d.name[1:])
    r = {"conc": c}
    hdr = (d / "server_cmd.sh").read_text().splitlines()[1] if (d / "server_cmd.sh").exists() else ""
    for k, pat in (("draft_k", r"K=(\d+)"), ("max_num_seqs", r"max-num-seqs=(\d+)"),
                   ("dcp", r"dcp=(\d+)"), ("offload", r"offload=(\d+)")):
        m = re.search(pat, hdr)
        r[k] = int(m.group(1)) if m else ""
    j = d / f"c{c}.json"
    if j.exists():
        x = json.load(open(j))
        r.update(tps=x.get("output_throughput"), total_tps=x.get("total_token_throughput"),
                 rps=x.get("request_throughput"), ttft=x.get("median_ttft_ms"),
                 ttft99=x.get("p99_ttft_ms"), tpot=x.get("median_tpot_ms"),
                 tpot99=x.get("p99_tpot_ms"), completed=x.get("completed"))
    # vLLM logs "Mean acceptance length: X" for spec decoding; keep the last value.
    log = d / "server.log"
    if log.exists():
        m = re.findall(r"[Mm]ean acceptance length:\s*([\d.]+)", log.read_text(errors="replace"))
        r["accept_len"] = float(m[-1]) if m else ""
    return r


def f(x, nd):
    return "—" if x in ("", None) else f"{x:,.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    sweeps, out = [Path(s) for s in a.sweep], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best, src = {}, {}
    for sweep in sweeps:
        for d in sweep.glob("c*"):
            if not (d.is_dir() and d.name[1:].isdigit()):
                continue
            r = point(d)
            prev = best.get(r["conc"])
            if prev is None or r.get("tps") not in ("", None) or prev.get("tps") in ("", None):
                best[r["conc"]], src[r["conc"]] = r, sweep.name
    rows = sorted(best.values(), key=lambda r: r["conc"])
    sweep = ", ".join(sorted({src[r["conc"]] for r in rows}))
    with open(out / "kimi-k3-recipe.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})
    image = (out / "images.txt").read_text().strip() if (out / "images.txt").exists() else "?"
    L = ["# Kimi-K3 — AMD vLLM recipe (recipes.vllm.ai, MI355X, 2026-09-25)", "",
         f"Image: `{image}`. TP8 on one node, MXFP4 experts, FP8 KV cache, prefix caching, "
         "one server per concurrency point with the recipe's per-point settings (DSpark "
         "speculative decoding up to C=14, decode-context-parallel 8 + CPU KV offload above). "
         "Workload: random ISL/OSL 1024/1024, `--ignore-eos`, 10×C prompts, `vllm bench serve`.",
         f"Raw logs: `{sweep}`.", "",
         "| C | draft K | max-num-seqs | DCP | KV offload | out tok/s | tok/s per user | TTFT med (ms) | TPOT med (ms) | TPOT p99 (ms) | accept len | completed |",
         "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        per_user = (1000 / r["tpot"]) if r.get("tpot") else ""
        L.append(f"| {r['conc']} | {r['draft_k']} | {r['max_num_seqs']} | {r['dcp']} | "
                 f"{'yes' if r.get('offload') == 1 else 'no'} | {f(r.get('tps'), 0)} | {f(per_user, 1)} | "
                 f"{f(r.get('ttft'), 0)} | {f(r.get('tpot'), 2)} | {f(r.get('tpot99'), 2)} | "
                 f"{f(r.get('accept_len'), 2)} | {r.get('completed', '—') or '—'} |")
    missing = [r["conc"] for r in rows if r.get("tps") in ("", None)]
    if missing:
        L += ["", f"No result at C = {', '.join(map(str, missing))}: see `c<C>/server.log` and `bench.log`."]
    L += ["", "tok/s per user = 1000 / median TPOT. With speculative decoding TPOT counts accepted "
          "tokens, so it already includes the draft's benefit."]
    (out / "kimi-k3-recipe.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out}/kimi-k3-recipe.md ({len(rows)} points)")


if __name__ == "__main__":
    main()
