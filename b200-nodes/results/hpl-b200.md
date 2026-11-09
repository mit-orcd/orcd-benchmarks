# NVIDIA HPL (FP64) on B200

Run: `/orcd/data/orcd/022/benchmarks/b200-nodes/hpl/out-hpl/run_20261109_162850`. Container `nvidia-hpc-benchmarks/hpc-benchmarks_25.04.sif` (NGC HPC-Benchmarks 25.04,
CUDA 12.8), Slurm `mit_testing`, whole nodes (`--exclusive`), one MPI rank per GPU, `srun --mpi=pmi2`.
Job script: `b200-nodes/hpl/job-hpl-b200.sh`; submit: `b200-nodes/hpl/submit-hpl-b200.sh`.

Same rules as rocHPL on MI355X: N fills about 75% of the usable HBM per GPU (178 GiB; 60% on a
retry), the same N twice and the fastest run is reported, the residual check must pass. NB = 1024
and P x Q (P >= Q) are NVIDIA's own defaults from the container's sample decks.

| GPUs | Nodes | P x Q | N | Time (s) | HPL (TFLOPS) | Per GPU | % of FP64 GEMM (36) | % of peak (37) | Scaling vs 1 GPU | Job |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1 x 1 | 133,120 | 45.22 | **34.8** | 34.8 | 97% | 94% | 1.00x (100%) | 28512784 |
| 2 | 1 | 2 x 1 | 188,416 | 64.14 | **69.5** | 34.8 | 97% | 94% | 2.00x (100%) | 28512785 |
| 3 | 1 | 3 x 1 | 230,400 | 78.51 | **103.9** | 34.6 | 96% | 94% | 2.99x (100%) | 28512786 |
| 4 | 1 | 2 x 2 | 266,240 | 91.45 | **137.6** | 34.4 | 96% | 93% | 3.95x (99%) | 28512787 |
| 5 | 1 | 5 x 1 | 296,960 | 101.69 | **171.7** | 34.3 | 95% | 93% | 4.93x (99%) | 28512788 |
| 6 | 1 | 3 x 2 | 325,632 | 111.49 | **206.5** | 34.4 | 96% | 93% | 5.93x (99%) | 28512789 |
| 7 | 1 | 7 x 1 | 351,232 | 120.65 | **239.4** | 34.2 | 95% | 92% | 6.88x (98%) | 28512790 |
| 8 | 1 | 4 x 2 | 376,832 | 129.79 | **274.9** | 34.4 | 95% | 93% | 7.90x (99%) | 28512291 |
| 16 | 2 | 4 x 4 | 524,288 | 172.50 | **557.0** | 34.8 | 97% | 94% | 16.01x (100%) | 28512292 |

FP64 GEMM = 36 TFLOPS per GPU measured with cuBLASLt; peak = 37 TFLOPS per GPU (datasheet, dense).
