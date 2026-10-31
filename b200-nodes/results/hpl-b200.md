# NVIDIA HPL (FP64) on B200

Run: `/orcd/data/orcd/022/benchmarks/b200-nodes/hpl/out-hpl/run_20261021_163151`. Container `nvidia-hpc-benchmarks/hpc-benchmarks_25.04.sif` (NGC HPC-Benchmarks 25.04,
CUDA 12.8), Slurm `mit_testing`, whole nodes (`--exclusive`), one MPI rank per GPU, `srun --mpi=pmi2`.
Job script: `b200-nodes/hpl/job-hpl-b200.sh`; submit: `b200-nodes/hpl/submit-hpl-b200.sh`.

Same rules as rocHPL on MI355X: N fills about 75% of the usable HBM per GPU (178 GiB; 60% on a
retry), the same N twice and the fastest run is reported, the residual check must pass. NB = 1024
and P x Q (P >= Q) are NVIDIA's own defaults from the container's sample decks.

| GPUs | Nodes | P x Q | N | Time (s) | HPL (TFLOPS) | Per GPU | % of FP64 GEMM (36) | % of peak (37) | Scaling vs 1 GPU | Job |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1 x 1 | 133,120 | 45.21 | **34.8** | 34.8 | 97% | 94% | 1.00x (100%) | 27372151 |
| 2 | 1 | 2 x 1 | 188,416 | - | failed | - | - | - | - | 27372152 |
| 3 | 1 | 3 x 1 | 230,400 | - | failed | - | - | - | - | 27372153 |
| 4 | 1 | 2 x 2 | 266,240 | - | failed | - | - | - | - | 27372154 |
| 5 | 1 | 5 x 1 | 296,960 | - | failed | - | - | - | - | 27372155 |
| 6 | 1 | 3 x 2 | 325,632 | - | failed | - | - | - | - | 27372156 |
| 7 | 1 | 7 x 1 | 351,232 | - | failed | - | - | - | - | 27372157 |
| 8 | 1 | 4 x 2 | 376,832 | - | failed | - | - | - | - | 27372158 |
| 16 | 2 | 4 x 4 | 524,288 | - | failed | - | - | - | - | 26533183 |

FP64 GEMM = 36 TFLOPS per GPU measured with cuBLASLt; peak = 37 TFLOPS per GPU (datasheet, dense).

Failed points: 2, 3, 4, 5, 6, 7, 8, 16 GPUs (see the logs in the run dir).
