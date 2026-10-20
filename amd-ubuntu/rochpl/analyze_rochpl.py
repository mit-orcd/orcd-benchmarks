#!/usr/bin/env python3
"""rocHPL (fp64) vs RVS gst fp64 on MI355X.

  analyze_rochpl.py <run dir with summary.txt>

Writes results/node6100/rocm7.14/rochpl.csv and results/rochpl-vs-rvs.md.
RVS reference: results/node6100/rocm7.14/rvs_tflops.csv (1..8 GPUs, ROCm 7.14, same stack as rocHPL);
16 GPUs = node6100 8-GPU + node6101 8-GPU aggregate (RVS runs per node, no cross-node work).
"""
import csv
import os
import sys

BENCH = "/orcd/data/orcd/022/benchmarks/amd-ubuntu"
RES = os.environ.get("RES_DIR", f"{BENCH}/results")
PEAK_FP64 = 78.6  # TFLOPS per GPU, AMD spec, dense fp64 (matrix = vector on CDNA 4)


def rvs_fp64(node):
    out = {}
    path = f"{RES}/{node}/rocm7.14/rvs_tflops.csv"
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        if r["precision"] == "fp64":
            out[int(r["gpus"])] = float(r["aggregate_tflops"])
    return out


def main():
    run = sys.argv[1].rstrip("/")
    rows = []
    for line in open(f"{run}/summary.txt"):
        f = line.split()
        if len(f) < 11:
            continue
        g, p, q, n, nb, t, tf, ok, rc, frac, log = f[:11]
        rows.append(dict(G=int(g), P=int(p), Q=int(q), N=int(n), NB=int(nb), time=t,
                         tflops=float(tf), ok=ok == "1" and rc == "0", rc=rc, frac=frac, log=log))
    # keep the best passing row per G (a retry at mem_frac 0.6 adds a second row)
    best = {}
    for r in rows:
        b = best.get(r["G"])
        if b is None or (r["ok"], r["tflops"]) > (b["ok"], b["tflops"]):
            best[r["G"]] = r
    r100, r101 = rvs_fp64("node6100"), rvs_fp64("node6101")
    rvs = dict(r100)
    if 8 in r100 and 8 in r101:
        rvs[16] = r100[8] + r101[8]

    os.makedirs(f"{RES}/node6100/rocm7.14", exist_ok=True)
    with open(f"{RES}/node6100/rocm7.14/rochpl.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "gpus", "nodes", "P", "Q", "N", "NB", "time_s", "hpl_tflops",
                    "hpl_per_gpu", "rvs_fp64_tflops", "hpl_pct_of_rvs", "hpl_pct_of_peak", "passed"])
        for g in sorted(best):
            r = best[g]
            rv = rvs.get(g)
            w.writerow([os.path.basename(run), g, 2 if g > 8 else 1, r["P"], r["Q"], r["N"], r["NB"],
                        r["time"], r["tflops"], round(r["tflops"] / g, 2), rv or "",
                        round(100 * r["tflops"] / rv, 1) if rv and r["ok"] else "",
                        round(100 * r["tflops"] / (PEAK_FP64 * g), 1) if r["ok"] else "", int(r["ok"])])

    one = best.get(1, {}).get("tflops") if best.get(1, {}).get("ok") else None
    tab = ["| GPUs | Nodes | P x Q | N | Time (s) | rocHPL (TFLOPS) | rocHPL per GPU | RVS fp64 (TFLOPS) "
           "| rocHPL / RVS | rocHPL % of peak | rocHPL scaling vs 1 GPU |",
           "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in sorted(best):
        r = best[g]
        rv = rvs.get(g)
        rvs_s = (f"{rv:,.1f}" + (" (8 + 8)" if g == 16 else "")) if rv else "-"
        if r["ok"]:
            scal = f"{r['tflops'] / one:.2f}x ({100 * r['tflops'] / one / g:.0f}%)" if one else "-"
            tab.append(f"| {g} | {2 if g > 8 else 1} | {r['P']} x {r['Q']} | {r['N']:,} | {r['time']} "
                       f"| **{r['tflops']:,.1f}** | {r['tflops'] / g:.1f} | {rvs_s} "
                       f"| **{100 * r['tflops'] / rv:.1f}%** | {100 * r['tflops'] / (PEAK_FP64 * g):.1f}% | {scal} |"
                       if rv else
                       f"| {g} | {2 if g > 8 else 1} | {r['P']} x {r['Q']} | {r['N']:,} | {r['time']} "
                       f"| **{r['tflops']:,.1f}** | {r['tflops'] / g:.1f} | - | - "
                       f"| {100 * r['tflops'] / (PEAK_FP64 * g):.1f}% | {scal} |")
        else:
            tab.append(f"| {g} | {2 if g > 8 else 1} | {r['P']} x {r['Q']} | {r['N']:,} | - | failed (rc {r['rc']}) "
                       f"| - | {rvs_s} | - | - | - |")

    ok = [best[g] for g in sorted(best) if best[g]["ok"]]
    pct = [100 * r["tflops"] / rvs[r["G"]] for r in ok if r["G"] in rvs]
    failed = [g for g in sorted(best) if not best[g]["ok"]]
    vers = open(f"{run}/versions.txt").read().strip() if os.path.exists(f"{run}/versions.txt") else ""
    vl = []
    for l in vers.splitlines()[:6]:
        l = l.strip()
        if l.startswith("mpirun (Open MPI)"):
            vl.append("Open MPI " + l.split()[-1])
        elif l.startswith("# Library version:"):
            vl.append("UCX " + l.split()[-1])
        elif l.startswith("rocHPL"):
            vl.append("rocHPL " + l.split()[1])
        elif l.startswith("ROCm"):
            vl.append("ROCm " + l.split()[-1])
    vline = ", ".join(vl)

    md = f"""# rocHPL vs RVS fp64 — MI355X

System: 8 x AMD Instinct MI355X (gfx950) per node, node6100 (1–8 GPUs) and node6100 + node6101
(16 GPUs), ROCm 7.14 (`amd-software/rocm-7.14.0`), Ubuntu 24.04. rocHPL run: `{os.path.basename(run)}`.
RVS run: `results/node6100/rocm7.14/rvs_tflops.md` and `results/node6101/rocm7.14/rvs_tflops.md`
(same nodes, same ROCm 7.14).

## What each test does

| | RVS `gst` | rocHPL |
|---|---|---|
| Work | One large fp64 GEMM per GPU, repeated | Solves a dense fp64 system Ax = b (LU factorization), checks the residual |
| GPU-to-GPU traffic | None. Each GPU works alone. | Yes. Panel broadcasts and row swaps through MPI |
| Across nodes | No (16 GPUs = sum of the two 8-GPU node results) | Yes. GPU-aware MPI (Open MPI + UCX with ROCm) over RoCE, no RCCL |
| Result | Peak GEMM TFLOPS | Sustained TFLOPS for the whole solve (the Top500 metric) |

Both use the same precision (fp64), so RVS fp64 is the upper bound for rocHPL on the same GPUs.
Peak fp64 = {PEAK_FP64} TFLOPS per GPU (AMD spec, dense).

## rocHPL vs RVS fp64

{chr(10).join(tab)}

rocHPL / RVS = rocHPL TFLOPS ÷ RVS fp64 TFLOPS at the same GPU count.
Scaling can go above 100%: every run fills the same share of HBM per GPU, so N grows with the
GPU count. The GEMM work grows as N³ but the panel work only as N², so a larger run spends a
larger share of its time in the GEMM.
"""
    if pct:
        md += f"\nrocHPL reaches **{min(pct):.0f}–{max(pct):.0f}%** of the RVS fp64 rate.\n"
    if failed:
        md += f"\nFailed points: {', '.join(map(str, failed))} GPUs (see the run logs).\n"
    md += f"""
## Why rocHPL is below RVS

- RVS runs only the GEMM. rocHPL also runs the panel factorization, which is small, serial work
  that does not fill the GPU.
- rocHPL moves data: panel broadcasts along each row of the P x Q grid and row swaps along each
  column. Inside a node this goes over XGMI; across nodes over the RoCE NICs.
- The trailing update (the GEMM part) shrinks as the solve goes on, so the last part of the run
  is less efficient.
- A larger N (more of the 288 GB HBM) gives a larger GEMM share and a higher result.

## Setup

- rocHPL: built from github.com/ROCm/rocHPL for gfx950 with ROCm 7.14 hipcc and rocBLAS
  (`../rochpl`, `setup/build-rochpl.sh`).
- MPI: Open MPI 5.0.8 + UCX 1.18.0, both built with ROCm 7.14 (GPU-aware). `--mca pml ucx`.
  1 node: UCX shared memory + ROCm IPC (GPU-to-GPU over XGMI, zero-copy), no NIC.
  2 nodes: plus RC over the 8 ionic RoCEv2 rails (GID index 1); large GPU messages use
  zero-copy RDMA reads (GPU-direct, dmabuf), small ones go through host memory. TCP only sets up
  the connections. UD is not used: it times out on the ionic NICs.
- One rank per GPU, NB = {ok[0]['NB'] if ok else 512}, N fills about 75% of HBM (60% on a retry),
  rocHPL `--it 2` (fastest run reported), split fraction `-f 0.3` (rocHPL default).
- Versions: {vline or '-'}

## How to run

```bash
# build (once, on an AMD node)
ssh node6100.inband 'bash /orcd/data/orcd/022/benchmarks/rochpl/setup/build-rochpl.sh'
# sweep 1-8 GPUs on node6100, then 16 GPUs on node6100 + node6101, then this file
./launch.sh node6100 rochpl/run_rochpl_all.sh
```
"""
    open(f"{RES}/rochpl-vs-rvs.md", "w").write(md)
    print(f"rocHPL: {len(ok)} passing points, failed {failed or 'none'}; wrote {RES}/rochpl-vs-rvs.md")


if __name__ == "__main__":
    main()
