# mpi-p2p

OSU micro-benchmarks measuring inter-node MPI point-to-point bandwidth
(`osu_bw`) and latency (`osu_latency`).

## Setup

`run/env.sh` loads the MPI module stack used by the job. Adjust for your
site if needed.

## Run (single benchmark)

```bash
cd run
sbatch run.sh       # schedules a 2-node job and runs osu_bw / osu_latency
```

## Analyze

```bash
cd run
./get-results.sh    # prints bandwidth (MB/s) and latency (us) at 4 MiB
```

## Run via py-all-bench

```bash
cd ../py-all-bench
module load miniforge/24.3.0-0
python bench_submit.py  mpi-p2p --partition mit_normal_gpu --nodes 3506 3507
python bench_analyze.py mpi-p2p --partition mit_normal_gpu --num-results 2
```

`py-all-bench` automatically schedules every pair of the supplied nodes.
