#!/usr/bin/env python3
"""NVIDIA HPL (FP64) on B200 vs rocHPL on MI355X.

  analyze-hpl-b200.py <run dir with summary.txt>

Writes ../results/hpl-b200.md (B200 only) and fills section 2 (between the HPL-B200-START/END
markers) plus the "FP64 HPL" summary row of ../../amd-ubuntu/results/mi355x-vs-b200.md, and the same in
../../amd-ubuntu/results/mi355x-rocm10.1-vs-b200.md with the ROCm 10.1 rocHPL results.
MI355X: ../../amd-ubuntu/results/node6100/rocm7.14/rochpl.csv (rocHPL, ROCm 7.14).
FP64 GEMM per GPU (section 3 of mi355x-vs-b200.md): MI355X RVS 77, B200 cuBLASLt 36.
"""
import csv
import os
import re
import sys

B = "/orcd/data/orcd/022/benchmarks"
OUT_MD = os.environ.get("HPL_MD", f"{B}/b200-nodes/results/hpl-b200.md")
CMP_MD = os.environ.get("CMP_MD", f"{B}/amd-ubuntu/results/mi355x-vs-b200.md")
ROC = os.environ.get("ROC_CSV", f"{B}/amd-ubuntu/results/node6100/rocm7.14/rochpl.csv")
LABEL = os.environ.get("ROC_LABEL", "ROCm 7.14")
SRC = os.environ.get("ROC_SRC", "[rochpl-vs-rvs.md](rochpl-vs-rvs.md)")
PEAK = {"mi": 78.6, "b2": 37.0}   # FP64 TFLOPS per GPU, datasheet, dense
GEMM = {"mi": 77.0, "b2": 36.0}   # measured FP64 GEMM per GPU (RVS, cuBLASLt), section 3


def main():
    run = sys.argv[1].rstrip("/")
    best = {}
    for line in open(f"{run}/summary.txt"):
        f = line.split()
        if len(f) < 11:
            continue
        r = dict(G=int(f[0]), P=int(f[1]), Q=int(f[2]), N=int(f[3]), NB=int(f[4]), time=f[5],
                 tf=float(f[6]), ok=f[7] == "1", frac=f[8], nodes=f[9], job=f[10])
        b = best.get(r["G"])
        if b is None or (r["ok"], r["tf"]) > (b["ok"], b["tf"]):
            best[r["G"]] = r
    roc = {}
    if os.path.exists(ROC):
        for r in csv.DictReader(open(ROC)):
            if r["passed"] == "1":
                roc[int(r["gpus"])] = float(r["hpl_tflops"])
    ok = {g: r for g, r in best.items() if r["ok"]}
    failed = sorted(g for g, r in best.items() if not r["ok"])
    one = ok.get(1, {}).get("tf")

    # ---- B200-only results file -------------------------------------------------------
    rows = ["| GPUs | Nodes | P x Q | N | Time (s) | HPL (TFLOPS) | Per GPU | % of FP64 GEMM (36) | % of peak (37) "
            "| Scaling vs 1 GPU | Job |", "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for g in sorted(best):
        r = best[g]
        if r["ok"]:
            pg = r["tf"] / g
            sc = f"{r['tf'] / one:.2f}x ({100 * r['tf'] / one / g:.0f}%)" if one else "-"
            rows.append(f"| {g} | {2 if g > 8 else 1} | {r['P']} x {r['Q']} | {r['N']:,} | {r['time']} | **{r['tf']:,.1f}** "
                        f"| {pg:.1f} | {100 * pg / GEMM['b2']:.0f}% | {100 * pg / PEAK['b2']:.0f}% | {sc} | {r['job']} |")
        else:
            rows.append(f"| {g} | {2 if g > 8 else 1} | {r['P']} x {r['Q']} | {r['N']:,} | - | failed | - | - | - | - | {r['job']} |")
    md = f"""# NVIDIA HPL (FP64) on B200

Run: `{run}`. Container `nvidia-hpc-benchmarks/hpc-benchmarks_25.04.sif` (NGC HPC-Benchmarks 25.04,
CUDA 12.8), Slurm `mit_testing`, whole nodes (`--exclusive`), one MPI rank per GPU, `srun --mpi=pmi2`.
Job script: `b200-nodes/hpl/job-hpl-b200.sh`; submit: `b200-nodes/hpl/submit-hpl-b200.sh`.

Same rules as rocHPL on MI355X: N fills about 75% of the usable HBM per GPU (178 GiB; 60% on a
retry), the same N twice and the fastest run is reported, the residual check must pass. NB = 1024
and P x Q (P >= Q) are NVIDIA's own defaults from the container's sample decks.

{chr(10).join(rows)}

FP64 GEMM = 36 TFLOPS per GPU measured with cuBLASLt; peak = 37 TFLOPS per GPU (datasheet, dense).
""" + (f"\nFailed points: {', '.join(map(str, failed))} GPUs (see the logs in the run dir).\n" if failed else "")
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w").write(md)

    # ---- section 2 of mi355x-vs-b200.md -----------------------------------------------------
    t = ["| GPUs | MI355X rocHPL (TFLOPS) | MI355X per GPU | B200 HPL (TFLOPS) | B200 per GPU | MI355X / B200 "
         "| MI355X % of FP64 GEMM | B200 % of FP64 GEMM |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ratios, pm, pb = [], [], []
    for g in sorted(set(roc) | set(best)):
        m = roc.get(g)
        b = ok.get(g)
        ms = (f"{m:,.1f}", f"{m / g:.1f}", f"{100 * m / g / GEMM['mi']:.0f}%") if m else ("-", "-", "-")
        if b:
            bs = (f"{b['tf']:,.1f}", f"{b['tf'] / g:.1f}", f"{100 * b['tf'] / g / GEMM['b2']:.0f}%")
            pb.append(100 * b["tf"] / g / GEMM["b2"])
        else:
            bs = ("failed" if g in best else "-", "-", "-")
        if m:
            pm.append(100 * m / g / GEMM["mi"])
        rt = f"**{m / b['tf']:.2f}x**" if m and b else "-"
        if m and b:
            ratios.append(m / b["tf"])
        t.append(f"| {g}{' (2 nodes)' if g > 8 else ''} | {ms[0]} | {ms[1]} | {bs[0]} | {bs[1]} | {rt} | {ms[2]} | {bs[2]} |")
    rr = f"{min(ratios):.2f}–{max(ratios):.2f}x" if ratios else "pending"
    rng = lambda v: f"{min(v):.0f}–{max(v):.0f}%" if v else "-"
    sec = f"""TFLOP/s, FP64. MI355X = rocHPL on {LABEL}, node6100 (16 GPUs: node6100 + node6101), from
{SRC}. B200 = NVIDIA HPL (NGC HPC-Benchmarks 25.04), from
`../../b200-nodes/results/hpl-b200.md`. Both: N ≈ 75% of HBM per GPU, best of 2 runs, residual check passed.

{chr(10).join(t)}

% of FP64 GEMM = HPL per GPU ÷ the FP64 GEMM measured in §3 (MI355X RVS 77, B200 cuBLASLt 36 TFLOP/s per GPU).

**How HPL differs from the FP64 GEMM in §3.** RVS and cuBLASLt run only a matrix multiply, each GPU
alone. HPL solves a full dense system: the same GEMM, plus panel factorization (small, serial work),
row swaps and broadcasts between GPUs and nodes. So HPL lands below the GEMM rate: MI355X keeps
{rng(pm)} of its FP64 GEMM, B200 keeps {rng(pb)}. MI355X / B200 is {rr} on HPL, against 2.14x on the FP64
GEMM and 2.12x on paper.

**Apple-to-apple: nearly.**
- Same benchmark and the same rules (problem size, best of 2 runs, residual check).
- Each vendor's own build: rocHPL (rocBLAS) vs NVIDIA HPL (cuBLAS); NB and process grid are each vendor's default (512 vs 1024).
- MI355X has 288 GB per GPU, B200 178 GiB usable, so MI355X runs a larger N at the same 75%, which helps it a little.
""" + (f"- B200 failed points: {', '.join(map(str, failed))} GPUs.\n" if failed else "")
    s = open(CMP_MD).read()
    s2 = re.sub(r"<!-- HPL-B200-START -->.*?<!-- HPL-B200-END -->",
                lambda _: "<!-- HPL-B200-START -->\n" + sec + "<!-- HPL-B200-END -->", s, flags=re.S)
    if s2 == s and "<!-- HPL-B200-START -->" not in s:
        print("WARN: HPL-B200 markers not found in", CMP_MD)
    s2 = re.sub(r"^\| FP64 HPL \(Linpack\) \(§2\) \| [^|]+ \|",
                f"| FP64 HPL (Linpack) (§2) | {rr} |", s2, flags=re.M)
    open(CMP_MD, "w").write(s2)
    print(f"B200 HPL: {len(ok)} passing points, failed {failed or 'none'}; MI355X/B200 {rr}; wrote {OUT_MD}, {CMP_MD}")


if __name__ == "__main__":
    main()
    # also section 2 of the ROCm 10.1 file (same B200 data, MI355X rocHPL on ROCm 10.1)
    ALT = f"{B}/amd-ubuntu/results/mi355x-rocm10.1-vs-b200.md"
    if "CMP_MD" not in os.environ and os.path.exists(ALT):
        CMP_MD, LABEL = ALT, "ROCm 10.1"
        ROC = f"{B}/amd-ubuntu/results/node6100/rocm10.1/rochpl.csv"
        SRC = "`node6100/rocm10.1/rochpl.csv`"
        main()
