# Plan — benchmark the new B200 nodes (gpu-fryer, NCCL 1-node, NCCL 2-node)

**Nothing has been run. No job has been submitted.** This document is the plan only.

Working dir: `/orcd/data/orcd/022/benchmarks/b200-nodes`.
No existing script is modified — every command below either uses an existing
script as-is or overrides its Slurm directives on the `sbatch` command line.

---

## 1. The nodes

`nodes` holds a Slurm hostlist; expanded it is **7 nodes**:

```
node5600-c1  node5601-c1  node5602-c1  node5702-c1  node5800-c1  node5801-c1  node5802-c1
```

Verified with `sinfo`: every one is `idle` in partition **`mit_testing`**
(also in `sched_system_all`), each with `gpu:b200:8` and 2 TB RAM.

Three chassis groups — `56xx` (3 nodes), `57xx` (1), `58xx` (3). The grouping
only matters for the 2-node NCCL pairing (§4).

**The old nodes are gone.** `node5500/5501/5502`, which every current
`summary.md` describes, are no longer in Slurm. Their `.out` files and their
summary tables stay exactly as they are — the new nodes are folded into the
existing summaries as a short verdict section, not as a rewrite (see §6).

---

## 0. Probe first (2 minutes, before any benchmark)

Three things are hard-coded against the old nodes and must be confirmed before
committing hours of GPU time:

```bash
srun -p mit_testing -w node5600-c1 -N 1 --gpus-per-node=b200:1 --mem=8G -t 5 \
  bash -lc 'hostname; nvidia-smi -L; ibstat -l; \
            module load apptainer/1.4.2 && singularity --version; \
            module load nvhpc/26.1 && ls /orcd/data/orcd/022/benchmarks/nccl-tests/build-nvhpc-26.1/sendrecv_perf'
```

| What | Why it matters | If it differs |
|---|---|---|
| `hostname` — `node5600-c1` or `node5600`? | It names the output files and is the key the analyze scripts group by | Nothing breaks either way; just note which form appears in the summaries |
| `ibstat -l` — the 8 IB rails | `job-nccl-2node.sh` pins `NCCL_IB_HCA=mlx5_4,7,8,9,10,13,14,15` (measured on node5500) | If the new nodes enumerate differently, copy the script to a **new** file `job-nccl-2node-c1.sh` with the corrected rail list — do not edit the original |
| `apptainer/1.4.2`, `nvhpc/26.1`, nccl-tests binaries | gpu-fryer needs the SIF + apptainer; NCCL needs nvhpc and the prebuilt binaries | Stop and report — the benchmarks cannot run as written |

`run-gpu-fryer.sh` already auto-detects whether FUSE mounting works and falls
back to `--unsquash`, so a restrictive `/dev/fuse` on the new nodes is handled.

### 0.1 Probe result — one blocker, one workaround (2026-08-24)

Two of the three checks passed; one failed hard.

**Passed.** `hostname` reports the full `node5600-c1` form. `apptainer/1.4.2` and
`nvhpc/26.1` load (`/orcd/software/core/001` is mounted). The IB rails are
**identical to the old nodes** — of the 16 `mlx5` devices, exactly
`mlx5_4,7,8,9,10,13,14,15` are Active at 400 Gb/s, so the `NCCL_IB_HCA` list
hard-coded in `job-nccl-2node.sh` is already correct. No `job-nccl-2node-c1.sh`
copy is needed.

**Blocker: `/orcd/data/orcd/022` does not mount on any of the 7 new nodes.**

```
node5600-c1: 022=MISSING home=OK software=OK      (same on all 7 nodes)
```

`/orcd/data/orcd` is an autofs indirect map; on a login node `022` resolves to
`hstor013-n1:/group/orcd/022` over **RDMA** (`proto=rdma,port=20049`), but on the
new nodes the mount never materializes — `stat` on the directory returns
`No such file or directory`, and Slurm itself logs
`error: couldn't chdir to /orcd/data/orcd/022/... going to /tmp instead`.
Every script, the gpu-fryer SIF and the nccl-tests binaries live under that path,
so nothing in this repo can run on the new nodes as written. `/orcd/scratch` is
missing there too; `/home` (TCP-mounted), `/orcd/software/core/001` and
`/orcd/pool/004` are fine.

This is a site/config issue on the new nodes, not something the benchmarks can
fix. Worth raising with ORCD: the `auto.orcd.data` map appears not to serve
`022` (or the RDMA route to `hstor013-n1` is unavailable) from the `10.1.56.x`
side these nodes sit on.

**Workaround: stage everything on `/orcd/pool/004/shaohao/b200-newnodes`,**
which *is* mounted on all 7 nodes.

| Staged item | Source | Size |
|---|---|---|
| `gpu-fryer_1.1.0.sif` | `benchmarks/gpu-fryer/` | 5.3 GB (byte-identical) |
| `bin/*_perf` | `nccl-tests/build-nvhpc-26.1/` | 13 MB — `ldd` shows they need only `/orcd/software` + system libs |
| `run-gpu-fryer.sh`, `run-nccl-1node.sh`, `job-nccl-2node.sh` | this repo | copies with **only** the `GPUFRYER_DIR` / `BUILD_DIR` lines repointed |

The originals in this repo are untouched, as required. Output dirs
`out-gpu-fryer/`, `out-nccl-1node/`, `out-nccl-2node/` live under the staging dir
and the `.out` files are copied back into this repo's `out-*/` dirs for analysis,
which still runs on the login node where `022` is mounted.

Both staged scripts were smoke-tested on node5600-c1 before the full run
(gpu-fryer 15 s: all 8 GPUs healthy, ~3.9 PFLOP/s fp8 aggregate; NCCL sendrecv:
678 GB/s converged busbw — both in family with the old nodes).

---

## 2. gpu-fryer (7 single-node jobs, run first)

```bash
./job-gpu-fryer.sh node5600-c1,node5601-c1,node5602-c1,node5702-c1,node5800-c1,node5801-c1,node5802-c1 300
```

- Existing wrapper, used unmodified: one `--exclusive` job per node,
  `-p mit_testing --gpus-per-node=b200:8 -t 60`.
- 300 s each for fp32 / bf16 / fp8 → ~16–17 min per node (measured: 16:26 on
  node5500). All 7 run in parallel if the partition is free → **~20 min wall**.
- Output: `out-gpu-fryer/gpu-fryer-<host>-<timestamp>.out`; Slurm stdout to
  `slurm-logs/`.

Gate: all 7 `.out` files present and each ends with the fp8 section.

---

## 3. NCCL 1-node (7 single-node jobs, after gpu-fryer finishes)

```bash
./job-nccl-1node.sh node5600-c1,node5601-c1,node5602-c1,node5702-c1,node5800-c1,node5801-c1,node5802-c1 all
```

- `all` = all 10 collectives, all 8 GPUs — matches how the current
  `out-nccl-1node/summary.md` was produced, so the new numbers stay comparable.
- ~4–5 min per node (measured: 4:25), parallel → **~10 min wall**.
- Output: `out-nccl-1node/nccl-1node-<host>-<timestamp>.out`.
- `hypercube` failed on every old node and will very likely fail again; that is
  a known nccl-tests issue, not a node fault. The analyzer records it as FAIL.

**Ordering constraint:** gpu-fryer and NCCL must never run on the same node at
the same time (`README.md`). Since both use `--exclusive`, Slurm enforces this,
but §2 is still run to completion first so a queued NCCL job never inherits a
hot GPU.

---

## 4. NCCL 2-node (5 pair jobs)

`job-nccl-2node.sh` is an sbatch script pinned to `#SBATCH -w node5500,node5502`.
Command-line flags override `#SBATCH` directives, so it is reused as-is with
`-w`, `-J` and `-t` supplied at submit time:

```bash
for pair in 5600-c1:5601-c1 5800-c1:5801-c1 5602-c1:5702-c1 5702-c1:5802-c1 5602-c1:5802-c1; do
  a=node${pair%%:*}; b=node${pair##*:}
  sbatch -p mit_testing -w "$a,$b" -N 2 -t 30 \
         -J "nccl-2node-${a#node}-${b#node}" job-nccl-2node.sh all 8
done
```

Pairs, chosen to cover all 7 nodes and every chassis-group combination:

| Pair | Nodes | Tests |
|---|---|---|
| 1 | node5600-c1 + node5601-c1 | intra-56xx |
| 2 | node5800-c1 + node5801-c1 | intra-58xx |
| 3 | node5602-c1 + node5702-c1 | 56xx ↔ 57xx |
| 4 | node5702-c1 + node5802-c1 | 57xx ↔ 58xx |
| 5 | node5602-c1 + node5802-c1 | 56xx ↔ 58xx |

- `all 8` = every collective at 8 GPUs/node = 16 ranks, the configuration the
  current 2-node summary uses.
- ~6 min per pair; pairs 3–5 share nodes so Slurm serializes them →
  **~20–30 min wall**.
- Output: `out-nccl-2node/nccl-2node-<pair>-<jobid>.out` (the `-J` name feeds
  `%x` in the script's `-o`).
- Risk: the old 2-node runs needed the explicit `NCCL_IB_HCA` rail pinning and
  the TCP-only MPI bootstrap to connect at 16 ranks. If a pair fails to
  bootstrap, the first check is `NCCL_DEBUG=WARN` output in the `.out` file
  against the probe's `ibstat -l` (§0), then a `job-nccl-2node-c1.sh` copy with
  the right rails.

---

## 5. Failure handling

Any node that fails a benchmark is re-run once alone. If it fails again it stays
in the summary with its failure noted rather than being dropped — a node that
cannot run the test is a result. Nothing is retried more than once without
reporting back first.

---

## 6. Updating the summary `.md` files

**The existing summaries are kept and amended, not rebuilt.** The current
`summary.md` files (node5500–5502, with all their tables and commentary) stay as
they are. Each gains one short section on the new nodes whose content depends on
what the runs show — a verdict, not a data dump.

### 6.1 Get the new numbers without overwriting anything

The analyze scripts always rewrite `<out-dir>/summary.md`, so they are run
against a backup and their stdout captured for reading only:

```bash
NEW="node5600-c1 node5601-c1 node5602-c1 node5702-c1 node5800-c1 node5801-c1 node5802-c1"
SC=<scratch>            # not in this repo

for d in out-gpu-fryer out-nccl-1node out-nccl-2node; do cp $d/summary.md $SC/$d.orig; done

./analyze-gpu-fryer.py  $NEW                    > $SC/gpu-fryer-new.md
./analyze-nccl-1node.py $NEW                    > $SC/nccl-1node-new.md
./analyze-nccl-2node.py out-nccl-2node/nccl-2node-5[68]*.out > $SC/nccl-2node-new.md

for d in out-gpu-fryer out-nccl-1node out-nccl-2node; do cp $SC/$d.orig $d/summary.md; done
```

The originals are restored immediately; the `$SC/*-new.md` files are working
material that never lands in the repo. (`analyze-gpu-fryer.py` also rewrites
`gpu-fryer-speedup.svg` — it is backed up and restored the same way, since the
plot belongs to the node5500 run the current summary discusses.)

### 6.2 What gets written into each summary

One new section per file, appended after the main comparison table, headed
**"New nodes (node5600/5601/5602/5702/5800/5801/5802)"**. Which of three forms it
takes is decided by the numbers, not in advance:

| Outcome | What the section says |
|---|---|
| New nodes track the current ones (worst-case per-node deviation within ~5%) | Two or three sentences: the 7 nodes were run with the same config, every figure of merit lands within X% of the node5500–5502 results, so the existing tables describe them too. One compact table at most — a single headline column per node (gpu-fryer: FP32/BF16/FP8 mean; NCCL 1-node: converged busbw for 2–3 representative collectives; NCCL 2-node: sendrecv + all_reduce per pair). No per-GPU tables, no per-message-size tables. |
| One or more nodes is off, or a benchmark fails | Name the node and the symptom concretely — which precision or collective, the measured value, the gap vs. the other nodes, and what the log points to. Healthy nodes are still covered by the one-line "the rest match" statement. |
| A benchmark cannot run at all (e.g. IB rail mismatch blocks 2-node NCCL) | State that plainly in that summary, with the probe evidence, rather than leaving the section out. |

The deviation threshold and the exact numbers quoted come from the runs; the
"within X%" figure is computed, never rounded to a comforting number.

Also added, one line in each summary's header block: node5500–5502 are no longer
in Slurm, so their tables are a historical baseline and the new nodes are the
current hardware.

### 6.3 Regenerate the PDFs

```bash
./md-to-pdf.py out-gpu-fryer/summary.md
./md-to-pdf.py out-nccl-1node/summary.md
./md-to-pdf.py out-nccl-2node/summary.md
```

(`out-nccl-2node-sharp/summary.md` has no PDF today and is not in scope.)

## 7. Out of scope unless asked

- `README.md`, `notes.md` and `claude.md` still describe node5500/node5502 as
  *the* B200 test nodes. They need rewriting for the new fleet, but that is a
  separate task from "update the summary md files".
- Megatron-LM, SHARP, and the ibwrite benchmarks are not part of this run.

---

## 8. Total cost

| Phase | Jobs | Wall (parallel) | Node-hours |
|---|---:|---:|---:|
| Probe | 1 | 2 min | ~0 |
| gpu-fryer | 7 | ~20 min | ~2.0 |
| NCCL 1-node | 7 | ~10 min | ~0.6 |
| NCCL 2-node | 5 | ~30 min | ~1.0 |
| Analysis | — | ~5 min | 0 |
| **Total** | **20** | **~70 min** | **~3.6** |

All on `mit_testing`, all `--exclusive`, all currently idle.

---

## 9. Execution log (2026-08-24, run in background)

Because of the mount blocker in §0.1, the repo wrappers `job-gpu-fryer.sh` /
`job-nccl-1node.sh` could not be used — their `--wrap` runs out of the repo dir,
which the new nodes cannot see. The same work was submitted directly with
`sbatch --chdir=<stage>` against the staged scripts; the benchmark configuration
(300 s per precision; `all` collectives; `all 8` for 2-node) is exactly as
planned in §2–§4, as is the 5-pair layout.

Phases are chained with `--dependency=afterany`, so gpu-fryer fully finishes
before any NCCL job starts (§3's ordering constraint), and 1-node finishes before
2-node.

| Phase | Job IDs | Nodes |
|---|---|---|
| gpu-fryer, 300 s | 21125200–21125206 | all 7, one job each |
| NCCL 1-node, `all` | 21125207–21125213 | all 7, one job each |
| NCCL 2-node, `all 8` | 21125214–21125218 | 5 pairs (§4 table) |

Staging dir: `/orcd/pool/004/shaohao/b200-newnodes`
(`logs/` = Slurm stdout, `out-*/` = benchmark output, `watch.sh` = phase watcher).

### 9.1 Results (2026-08-24)

| Phase | Slurm state | Verdict |
|---|---|---|
| gpu-fryer | 7/7 COMPLETED | All healthy, within 1.7% of the old nodes on every precision |
| NCCL 1-node | 7/7 COMPLETED | All PASS except `hypercube` (fails on old nodes too), within 2.8% of old |
| NCCL 2-node | 5/5 "FAILED" | **Not a real failure** — see below. 3 pairs healthy, 2 pairs slow |

**Why the 2-node jobs show FAILED.** `job-nccl-2node.sh` runs the collectives in
one loop and `hypercube_perf` is last. It reports `Out of bounds values : 32
FAILED` and exits 1, mpirun aborts, and the job's exit status becomes non-zero —
so `sacct` marks the whole job FAILED even though the 9 preceding collectives
completed with valid data and `PASS` correctness. The old node5500-5502 runs
failed identically (job 19791438, also at 00:05:56), and the existing summary was
built from exactly such "FAILED" runs. No data was lost.

**Real finding, sharpened by a confirmation run:** the four ring collectives run
at ~half bandwidth (~198 vs ~375-380 GB/s) specifically on the
node5602-c1<->node5802-c1 and node5702-c1<->node5802-c1 routes. A targeted
confirmation run (node5800-c1+node5802-c1, job 21137578) shows node5802-c1 is
**not** degraded in general — 380.5/377.0 GB/s, full bandwidth, when paired with
node5800-c1 (same 58xx chassis group). The fault isolates to node5802-c1's
cross-chassis fabric path(s) to the 56xx/57xx switches, not to the node itself —
likely a routing/cabling issue on that specific hop. Rails, driver, GPUDirect
config, correctness and NCCL logs all check out clean on both ends. Details and
the full confirmation table are in `out-nccl-2node/summary.md`.
