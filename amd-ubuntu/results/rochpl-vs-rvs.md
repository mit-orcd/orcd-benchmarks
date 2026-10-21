# rocHPL vs RVS fp64 — MI355X

System: 8 x AMD Instinct MI355X (gfx950) per node, node6100 (1–8 GPUs) and node6100 + node6101
(16 GPUs), ROCm 7.14 (`amd-software/rocm-7.14.0`), Ubuntu 24.04. rocHPL run: `rochpl_20261020_204342`.
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
Peak fp64 = 78.6 TFLOPS per GPU (AMD spec, dense).

## rocHPL vs RVS fp64

| GPUs | Nodes | P x Q | N | Time (s) | rocHPL (TFLOPS) | rocHPL per GPU | RVS fp64 (TFLOPS) | rocHPL / RVS | rocHPL % of peak | rocHPL scaling vs 1 GPU |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 x 1 | 163,840 | 48.82 | **60.1** | 60.1 | 76.5 | **78.6%** | 76.5% | 1.00x (100%) |
| 2 | 1 | 1 x 2 | 231,424 | 66.57 | **124.1** | 62.0 | 153.8 | **80.7%** | 78.9% | 2.06x (103%) |
| 3 | 1 | 1 x 3 | 284,160 | 81.12 | **188.6** | 62.9 | 230.8 | **81.7%** | 80.0% | 3.14x (105%) |
| 4 | 1 | 2 x 2 | 327,680 | 93.36 | **251.2** | 62.8 | 308.2 | **81.5%** | 79.9% | 4.18x (104%) |
| 5 | 1 | 1 x 5 | 366,080 | 102.67 | **318.6** | 63.7 | 385.6 | **82.6%** | 81.1% | 5.30x (106%) |
| 6 | 1 | 2 x 3 | 402,432 | 113.17 | **383.9** | 64.0 | 462.4 | **83.0%** | 81.4% | 6.39x (106%) |
| 7 | 1 | 1 x 7 | 433,664 | 121.34 | **448.1** | 64.0 | 538.7 | **83.2%** | 81.4% | 7.46x (107%) |
| 8 | 1 | 2 x 4 | 462,848 | 128.25 | **515.4** | 64.4 | 616.1 | **83.6%** | 82.0% | 8.58x (107%) |
| 16 | 2 | 4 x 4 | 655,360 | 192.14 | **976.7** | 61.0 | 1,231.2 (8 + 8) | **79.3%** | 77.7% | 16.25x (102%) |

rocHPL / RVS = rocHPL TFLOPS ÷ RVS fp64 TFLOPS at the same GPU count.
Scaling can go above 100%: every run fills the same share of HBM per GPU, so N grows with the
GPU count. The GEMM work grows as N³ but the panel work only as N², so a larger run spends a
larger share of its time in the GEMM.

rocHPL reaches **79–84%** of the RVS fp64 rate.

## Why rocHPL is below RVS

- RVS runs only the GEMM. rocHPL also runs the panel factorization, which is small, serial work
  that does not fill the GPU.
- rocHPL moves data: panel broadcasts along each row of the P x Q grid and row swaps along each
  column. Inside a node this goes over XGMI; across nodes over the RoCE NICs.
- The trailing update (the GEMM part) shrinks as the solve goes on, so the last part of the run
  is less efficient.
- A larger N (more of the 288 GB HBM) gives a larger GEMM share and a higher result.

## What each test measures, and which real work it is close to

| | RVS `gst` | rocHPL |
|---|---|---|
| Main thing measured | Peak fp64 speed of the matrix units, one GEMM per GPU | Sustained fp64 speed of a full dense solver on many GPUs |
| What limits it | Compute only | Compute, plus panel factorization, data movement (XGMI, RoCE) and memory |
| GPUs work together | No | Yes |
| Question it answers | Is each GPU healthy and at spec? | How fast does the whole system solve a large dense problem? |
| Real work it is close to | The inner GEMM loop of any fp64 code — a ceiling, not an app | Dense linear algebra: direct (LU) solvers, boundary element methods (electromagnetics, acoustics), dense electronic-structure and quantum-chemistry solves |
| Real work it is NOT like | Anything with communication or memory traffic | Sparse and memory-bound codes (most CFD, climate, graph work: closer to HPCG or STREAM); AI training and inference (bf16, fp8, fp4: closer to RVS at those precisions, RCCL, and real model runs) |

**rocHPL is closer to real-world computation**: it runs a full solver with real communication across
GPUs and nodes, and is the Top500 metric. But it is still a best case: a dense, compute-bound fp64
problem. Most real science codes are sparse or memory-bound and reach a much smaller share of peak.
RVS is a hardware ceiling and health check, not a model of any application.

## Setup

- rocHPL: built from github.com/ROCm/rocHPL for gfx950 with ROCm 7.14 hipcc and rocBLAS
  (`../rochpl`, `setup/build-rochpl.sh`).
- MPI: Open MPI 5.0.8 + UCX 1.18.0, both built with ROCm 7.14 (GPU-aware). `--mca pml ucx`.
  1 node: UCX shared memory + ROCm IPC (GPU-to-GPU over XGMI, zero-copy), no NIC.
  2 nodes: plus RC over the 8 ionic RoCEv2 rails (GID index 1); large GPU messages use
  zero-copy RDMA reads (GPU-direct, dmabuf), small ones go through host memory. TCP only sets up
  the connections. UD is not used: it times out on the ionic NICs.
- One rank per GPU, NB = 512, N fills about 75% of HBM (60% on a retry),
  rocHPL `--it 2` (fastest run reported), split fraction `-f 0.3` (rocHPL default).
- Versions: Open MPI 5.0.8, UCX 1.18.0, rocHPL d76d0c3, ROCm 7.14.60850-0000000

## How to run

```bash
# build (once, on an AMD node)
ssh node6100.inband 'bash /orcd/data/orcd/022/benchmarks/rochpl/setup/build-rochpl.sh'
# sweep 1-8 GPUs on node6100, then 16 GPUs on node6100 + node6101, then this file
./launch.sh node6100 rochpl/run_rochpl_all.sh
```
