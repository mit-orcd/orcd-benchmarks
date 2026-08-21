# Kimi-K3 on B200 (Engaging) — serving benchmark

Reproduces, on Engaging's B200 nodes, the Kimi-K3 serving characterization that
`../amd-benchmarks/amd-cloud/atom` did on 8 × MI355X, and generates a report with the
same structure as `../amd-benchmarks/amd-cloud/results/kimi-k3-base.md` plus a
**B200 vs MI355X** comparison section.

**Status: complete.** The full 1→64 concurrency sweep ran on 2 × 8 B200 (TP8 × PP2) and
`results/kimi-k3-base-b200.md` holds the finished analysis. Peak **1,696 tok/s** at c=64.

---

## Quick start

### I just want the results

Already generated — nothing to run:

| File | What it is |
|---|---|
| `results/kimi-k3-base-b200.md` | **The report.** Compute, memory, bottleneck, communication, and the B200 vs MI355X comparison (§6) |
| `results/kimi-k3-base-b200.csv` | Same sweep data, machine-readable |
| `results/RUN-SUMMARY.md` | Run-level summary: stage outcomes, hardware, key finding, known failure modes |
| `results/kimi-k3-improve-b200.md` | The four §3.2 improvement levers (written by `job-improve-b200.sh`) |

```bash
cat results/RUN-SUMMARY.md              # start here
cat results/kimi-k3-base-b200.md        # full analysis
```

### Run the whole benchmark from scratch

One command. It submits a Slurm dependency chain and returns immediately — no
babysitting process, so it survives logout:

```bash
./chain.sh
```

That queues, in order:

```
gate    (1 node,  ~1 min)   image + GPU + IB + KimiK3 arch + ray
verify  (2 nodes, ~2 min)   ray 16-GPU cluster, model inspection, JIT filelocks
1node   (1 node,  ~4 min)   TP8 x PP1 -- EXPECTED to OOM; that failure is the measurement
base    (2 nodes, ~35 min)  TP8 x PP2: serve -> 1..64 sweep -> analysis
summary (CPU,     ~5 s)     -> results/RUN-SUMMARY.md
```

`base` runs the analyzer itself, so **`results/kimi-k3-base-b200.md` exists the moment
the job ends** — there is no separate analysis step to remember.

Job IDs land in `logs/CHAIN.txt`. Check on it with:

```bash
squeue -u $USER
sacct -j $(grep -oE '[0-9]+' logs/CHAIN.txt | tr '\n' ',' | sed 's/,$//') \
      -o JobID,JobName%18,State,Elapsed,ExitCode
```

### Run one stage only

```bash
./submit.sh                 # prints the stages, reservation and parallelism
./submit.sh probe           # driver / HBM / model-mount check on every B200 node
./submit.sh gate            # cheap 1-node sanity gate
./submit.sh base            # the 2-node measurement run
./submit.sh --alt base      # same, retargeted to the non-reserved Rocky nodes
```

### Re-analyze without re-running the benchmark

The expensive part is the 2-node run; regenerating the report from saved sweep data
takes seconds. Use this after editing `analyze-kimi-b200.py`:

```bash
source common/env.sh
module load apptainer/1.5.2
B=logs/kimi_base_20260821_130024          # or: ls -dt logs/kimi_base_* | head -1

apptainer exec $(apt_args) "$VLLM_SIF" $PY_C analyze-kimi-b200.py \
  --sweep "$B/sweep" --server-log "$B/server/vllm_server.log" \
  --model-config "$MKIMI/config.json" --run-dir "$B" \
  --tp 8 --pp 2 --isl 1024 --osl 1024 --max-num-seqs 64 --kv-dtype fp8 \
  --hbm-mib 183359 --weight-bytes 1560936091448 \
  -o results
```

It re-reads the MI355X baseline out of `../amd-benchmarks/amd-cloud/logs/atom/` on every
run, so the comparison section stays in sync automatically.

### Run the improvement experiments

Tests the four levers from §3.2 of the report. Writes **only**
`results/kimi-k3-improve-b200.{md,csv}` — it never touches the baseline report:

```bash
sbatch job-improve-b200.sh      # 2 nodes, ~2 h, all four levers in one allocation
```

### Test the analyzers without burning an allocation

Both analyzers run against synthetic sweep data plus the **real** MI355X baseline.
Do this before any change that touches report generation:

```bash
./selftest-analyze.sh           # exercises analyze-kimi-b200.py end to end
```

### Before anything else, if the environment may have changed

```bash
./submit.sh probe               # ~1 min/node: driver r580+? checkpoint visible?
```

The image is pulled **once** by `./pull-image.sh`; every other script calls
`check_image()` and fails rather than re-pulling. The Kimi-K3 weights are **never**
downloaded — they are read in place from `/orcd/compute/orcd/025/models/Kimi-K3`.

---

## The constraint that shapes everything

Measured from the HF repo, not assumed:

```
moonshotai/Kimi-K3   96 safetensors shards, 1,560,998,987,867 bytes = 1561 GB (1.42 TiB)
                     2,779,931,837,184 params, MXFP4 "mxfp4-pack-quantized", ungated
```

| Box | HBM/GPU | HBM/node | Holds 1561 GB of weights? |
|---|---:|---:|---|
| MI355X | 288 GB | 2304 GB | **yes** — 743 GB spare |
| **B200** | **192 GB** | **1538 GB** | **no — 23 GB short on weights alone** |

And that 23 GB is the optimistic figure: it assumes 100% of HBM goes to weights. Add the
KV pool, activation workspace, NCCL buffers and CUDA-graph pool (~15–20 GB/GPU) and one
B200 node is short by ~150–180 GB. The vLLM recipe agrees: `single_node_tp` is not a
usable strategy for this model on B200.

**So the B200 run is 2 nodes / 16 GPUs, TP8 × PP2** — the recipe's `multi_node_tp_pp`
strategy, listed `b200: verified`. This is forced, not chosen, and the report says so
in every section that could be misread as a chip-vs-chip result.

---

## Scripts

| Script | What it does | Where it runs |
|---|---|---|
| `chain.sh` | **Submits the whole benchmark as one Slurm dependency chain** and returns | login node |
| `submit.sh` | Submits any single stage with the reservation/account/nodes applied | login node |
| `job-pull-image.sh` | Batch wrapper for `pull-image.sh` (compute nodes have outbound net) | `sbatch`, CPU |
| `job-summary.sh` | Aggregates the run into `results/RUN-SUMMARY.md`, whatever the outcome | `sbatch`, CPU |
| `common/env.sh` | Shared paths, run config, NCCL env, apptainer args | sourced |
| `lib/kimi-run.sh` | Shared body of the two Kimi runs (preflight → serve → sweep → analyze) | sourced |
| `pull-image.sh` | `vllm/vllm-openai:kimi-k3` → `imag/*.sif`, **once, atomically** | login node |
| `job-probe-drivers.sh` | **Driver / HBM / model-mount check on every B200 node** | `sbatch`, 1 GPU, ~1 min |
| `download-kimi.sh` | Smoke-tier model only — **refuses** to fetch Kimi-K3 | `sbatch`, CPU partition |
| `job-gate-b200.sh` | Image / GPU / IB / matmul / arch-registration gate | `sbatch`, **1** B200 node |
| `job-kimi-1node.sh` | **Step 1**: Kimi-K3 TP8 on one node — expected to OOM | `sbatch`, **1** B200 node |
| `job-kimi-base.sh` | **Main run**: TP8×PP2, server → sweep → teardown → analysis | `sbatch`, **2** B200 nodes |
| `run-vllm-server.sh` | Ray cluster + `vllm serve`, waits for HTTP *and* real VRAM | inside allocation |
| `run-vllm-bench.sh` | Concurrency sweep via `vllm bench serve` | inside allocation |
| `stop-vllm-server.sh` | Tear down this job's server and Ray session only | inside allocation |
| `analyze-kimi-b200.py` | → `results/kimi-k3-base-b200.{md,csv}` | inside allocation |
| `selftest-analyze.sh` | Exercises the analyzer on synthetic data + the real MI355X baseline | login node |

### Slurm targeting and parallelism (from `./notes`)

Both are now baked into the `#SBATCH` headers, so a bare `sbatch job-kimi-base.sh`
targets the right thing:

```
#SBATCH -p mit_testing
#SBATCH --reservation=rres_joohye_2026-08-20_lj4j2ya3
#SBATCH -A rres_acc_joohye_2026-08-20_lj4j2ya3
#SBATCH -w node5700-c1,node5701-c1
#SBATCH -N 2 --ntasks-per-node=1 --gpus-per-node=b200:8 --exclusive
```

Verified against `scontrol show res`: the reservation is ACTIVE 2026-08-20 → 2026-08-27
on exactly those two nodes in `mit_testing`. The **account flag is required** — the
reservation is account-restricted, so a job charged elsewhere is refused entry.

**QOS is deliberately left at the default (`normal`).** The reservation carries
`rres_qos_joohye_2026-08-20_lj4j2ya3`, but it is flagged `RequiresReservation` +
`OverPartQOS` and exists to lift partition limits we never reach — `mit_testing` already
allows `MaxTime=7-00:00:00` with `AllowQos=normal,unlimited`, and the longest job here
is 6 h. If Slurm ever refuses a submission, add
`#SBATCH -q rres_qos_joohye_2026-08-20_lj4j2ya3`; the reason is written into each header.

| Stage | Nodes | GPUs | Parallelism |
|---|---:|---:|---|
| `1node` | 1 (`node5700-c1`) | 8 | **TP8 × PP1** — tensor-parallel across the node's 8 GPUs, no pipeline stage |
| `base` | 2 (`node5700-c1`, `node5701-c1`) | 16 | **TP8 × PP2** — TP8 *within* each node on NVLink, PP2 *across* the pair on InfiniBand |

The TP/PP axis assignment is deliberate: TP all-reduces twice per layer and must stay on
NVLink, while PP crosses a stage boundary only once per step. Reversing them would put
186 all-reduces per token onto InfiniBand. `lib/kimi-run.sh` asserts at runtime that the
allocation matches the requested parallelism exactly — a 2-node allocation for a PP1 job
aborts rather than silently idling and billing 8 GPUs.

### Order of work

```
./submit.sh probe        # ~1 min/node    <- SETTLES THE DRIVER QUESTION FIRST
./pull-image.sh          # ~20 min, login node
./submit.sh gate         # 1 node, ~20 min
./submit.sh download     # CPU job, ~5 min  (smoke model only; Kimi-K3 is pre-staged)
./submit.sh 1node        # 1 node, <2 h    <- the "try one node first" step from notes
./submit.sh base         # 2 nodes, 3-4 h  -> results/ written by the job itself
```

### Running it unattended

`./chain.sh` submits everything as one **Slurm dependency chain** and returns
immediately. There is no babysitting process — no `nohup`, no polling loop, nothing that
dies with the session. Slurm holds the chain, so it completes after logout and would
survive a login-node reboot.

```
pull  (CPU)      fetch the image once
  |  afterok
gate  (1 node)   image + GPU + IB + arch registration
  |  afterok
1node (1 node)   TP8 x PP1 -- EXPECTED to OOM; the failure is the measurement
  |  afterANY  <-- not afterok, precisely because it is expected to fail
base  (2 nodes)  TP8 x PP2 + automatic analysis -> results/kimi-k3-base-b200.md
  |  afterany
summary (CPU)    -> results/RUN-SUMMARY.md, written whatever the outcome
```

The `afterany` on the single-node attempt is the load-bearing detail: that job is
*expected* to fail with an OOM, and `afterok` would cancel the entire rest of the chain
on exactly the outcome we predicted. The final summary is `afterany` too — a summary
that only appears on success is useless to someone reading the directory the next
morning.

Job IDs are recorded in `logs/CHAIN.txt`.

`submit.sh` with no argument prints the stages, the reservation and the parallelism.
`./submit.sh --alt <stage>` retargets onto the Rocky nodes: it rewrites the header into
a temp copy with the reservation and account lines deleted, rather than trying to clear
an `#SBATCH` value from the command line — Slurm's behaviour when overriding a header
reservation with an empty one is not something to discover on a 2-node allocation.

The probe and gate run first on purpose. The `vllm:kimi-k3` image is a **CUDA 13
(cu130) build with no cu129 tag**, so the host driver must be **r580+**. Minutes on one
node settle that before a 2-node hold is spent on it.

### ✅ Node check — resolved

`./notes` gives reservation `rres_joohye_2026-08-20_lj4j2ya3` on **node5700-c1,
node5701-c1**. Verified live: ACTIVE 2026-08-20 → 2026-08-27, partition `mit_testing`,
account `rres_acc_joohye_2026-08-20_lj4j2ya3`.

An earlier concern here — that these were the Ubuntu nodes on driver **570.211.01**,
too old for the cu130 image — **is retired**. `./submit.sh probe` (jobs 20849845-48)
checked all four candidate nodes on 2026-08-20:

| | node5700-c1 | node5701-c1 | node5500-c1 | node5501-c1 |
|---|---|---|---|---|
| OS | Rocky 8.10 | Rocky 8.10 | Rocky 8.10 | Rocky 8.10 |
| Driver | **590.48.01** | **590.48.01** | **590.48.01** | **590.48.01** |
| CUDA | 13.1 | 13.1 | 13.1 | 13.1 |
| HBM/GPU | 183359 MiB | 183359 MiB | 183359 MiB | 183359 MiB |
| Checkpoint visible | ✅ 96/96 | ✅ 96/96 | ✅ 96/96 | ✅ 96/96 |

The reserved pair was reinstalled from Ubuntu to Rocky since the 2026-08-12 survey in
`../b200-ubuntu/ubuntu-nccl.md`, which is what made that warning stale. **r590 ≥ r580,
so the cu130 image runs**, and the benchmark uses the reserved nodes as `notes`
specifies. `--alt` remains available but is not needed.

It also confirmed the capacity arithmetic on real hardware: **8 × 183359 MiB = 1538 GB
per node against 1561 GB of weights — 23 GB short**, before any KV cache.

### Why vLLM and not ATOM

ATOM is ROCm-only. vLLM mainline registers `KimiK3ForConditionalGeneration` and ships an
NVIDIA-specific K3 path. Crucially the **measurement** path is unchanged:
`vllm bench serve` and `atom.benchmarks.benchmark_serving` are the same code lineage and
emit the same JSON keys, which is what lets **one analyzer read both runs** and compare
them without hand-transcribing numbers.

### Launch flags are recipe-derived, not tuned

From the vLLM recipe's Blackwell baseline + `multi_node_tp_pp` override. Each has a
stated reason in `job-kimi-base.sh`; the two that would silently corrupt results if
changed:

- `--gpu-memory-utilization 0.90` (not the 0.95 baseline) — the flashinfer TRTLLM MXFP4
  MoE kernel takes a ~1.6 GiB workspace *outside* vLLM's pool on the first forward, and
  at 0.95 a B200 OOMs during warmup.
- `--no-enable-prefix-caching` — **correctness, not tuning**. KDA recurrent state is
  per-request and cannot be rebuilt from the paged MLA cache, so prefix reuse would be
  silently wrong. It also matches the MI355X run.

Deliberately **not** set: DSpark speculative decoding (the recipe gates it *off*
`multi_node_tp_pp` — it does not compose with PP yet), expert parallelism (off on
MI355X too), and `--moe-backend` (recipe says let vLLM auto-select on B200).

### The checkpoint is pre-staged — nothing downloads it

Kimi-K3 is already on the cluster and is used in place:

```
/orcd/compute/orcd/025/models/Kimi-K3 -> safetensors/Kimi-K3/
```

Verified 2026-08-20: **96 safetensors shards, 1,560,936,091,448 bytes (1.420 TiB)**; the
index declares **497,220 tensors across exactly 96 shards**; `quantization_config.format`
is `mxfp4-pack-quantized` under `text_config` — the same checkpoint the MI355X run
served. The analyzer's self-test passes against this exact `config.json`.

Two properties drive how the scripts treat it:

- **It belongs to another user** (`lincolnb`, mode 644) on `fstor025.ib:/compute/orcd/025`.
  The container bind is `:ro` and nothing here ever writes inside it.
- **The path is a symlink** into `$MODEL_STORE/safetensors/`. Binding the symlink alone
  would leave a dangling link inside the container, so `apt_args` binds the **store**
  (`/orcd/compute/orcd/025/models`), which covers both the link and its target.

`check_model()` verifies shard count and exact byte total — metadata only, not a byte of
the 1.42 TiB is read. `download-kimi.sh kimi` **refuses** and points at that check
instead.

> **Open question the probe now answers.** The *login* node mounts that export; that says
> nothing about the B200 compute nodes. `./submit.sh probe` reports `MODEL VISIBLE` /
> `MODEL NOT VISIBLE` per node alongside the driver verdict, and `job-gate-b200.sh`
> aborts if the node it lands on cannot read the checkpoint. If the B200 nodes do not
> mount it, either ORCD mounts it there or the weights get staged somewhere reachable —
> and that has to surface before a 2-node allocation, not during one.

### The image is pulled exactly once

`./pull-image.sh` is the **only** thing that fetches the image. Every other script calls
`check_image()` and fails if it is missing — a GPU job must never spend allocation time
downloading 15 GB, and two jobs must never race to write the same `.sif`.

Getting "once" right needs more than an `if [ -f ]` test, because `apptainer pull`
writes its destination **in place**: interrupt it and a truncated file is left at the
final path, which a naive presence check accepts. The failure then surfaces as an
inscrutable engine crash twenty minutes into a 2-node allocation. So:

| Mechanism | What it prevents |
|---|---|
| `flock` on `imag/*.lock` for the whole script | two concurrent pulls fighting over one destination |
| pull to `imag/.pull-<tag>.$$.sif`, `mv` only on success | a truncated `.sif` ever appearing at the real path |
| `apptainer inspect` before publishing | publishing a file that downloaded but is not a valid image |
| `*.manifest` written **last** | its existence is the completion record — a `.sif` without one is a failed pull and is discarded on the next run |
| size check against the manifest in `check_image()` | a `.sif` truncated *after* the pull |
| `sha256` recorded once; `--verify` to recheck | content drift, without hashing 20 GB on every job start (that would be real I/O on shared NFS for no reason) |

```bash
./pull-image.sh                # no-op if already present and valid
./pull-image.sh --verify       # recompute sha256 and compare
./pull-image.sh --force        # discard and refetch
./pull-image.sh --keep-cache   # keep the OCI layer cache (deleted by default)
```

The OCI layer cache is a second full copy of the image, useful only for resuming an
interrupted pull, so it is deleted once the `.sif` exists. Both it and the build tmpdir
are forced off `$HOME` (441/500 GB used) — the default location would blow the quota
mid-pull and leave exactly the corrupt `.sif` this design exists to prevent.

### Safety

These are not wrappers around upstream launch scripts, for the same reason the ATOM
wrappers were not:

- Nothing is ever `pkill`ed. `stop-vllm-server.sh` acts only on the PID this job
  recorded and the Ray session this job started.
- `job-kimi-base.sh` traps `EXIT/TERM/INT`, so a walltime kill or `scancel` still tears
  down the server and Ray rather than leaving 16 GPUs held by orphan workers.
- The server readiness check requires **HTTP 200 *and* resident VRAM**. With 1.42 TiB
  coming off NFS, "process is up" and "model is loaded" are tens of minutes apart.
- The sweep treats **`completed == 0` as fatal**. `vllm bench serve` exits 0 even when
  every request fails — it warns and writes zeros — so without that check a dead server
  yields a full sweep of `0.00` rows and a cheerful `rc=0`.
- Every GPU step is an sbatch job, and so is the image pull — nothing heavy runs on
  the login node. The 1.56 TB checkpoint is never downloaded at all; it is read in
  place from the pre-staged copy.

### Storage

Kimi-K3 is read in place from `/orcd/compute/orcd/025/models` and costs us no quota at
all. Only the image and the smoke-tier model land in
`/orcd/data/orcd/022/benchmarks/b200-kimi/` (77 TB free). Never `$HOME` (441/500 GB used)
and never `$SCRATCH` (1 TB quota). `pull-image.sh`
forces `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` off `$HOME` for the same reason — the
default would blow the quota mid-pull and leave a corrupt `.sif`.

---

## The analyzer

`analyze-kimi-b200.py` runs automatically at the end of `job-kimi-base.sh` and writes
`results/kimi-k3-base-b200.{md,csv}`. It reads:

1. the B200 sweep JSONs,
2. the vLLM server log (memory profile + engine config — *"from the server log, not
   assumed"*, as the MI355X report puts it),
3. `config.json`,
4. and the **real MI355X run** — its sweep JSON and ATOM server log, straight out of
   `../amd-benchmarks/amd-cloud/logs/atom/`.

Both systems' derived figures (active params, TFLOP/s, HBM / interconnect volumes) are
recomputed with the **same formulas**, so the comparison columns are actually
comparable rather than one side being transcribed from a published table.

### Architecture parse is exactly validated

The routed experts live in a **latent space of 3584**, not on the 7168-wide residual
stream. Getting that wrong predicts ~6.4 T parameters instead of 2.72 T. The parse is
checked against the checkpoint's own MXFP4 parameter count:

```
92 MoE layers × 896 experts × 3 × 3584 × 3072 = 2,722,740,830,208
                    safetensors U8 param count = 2,722,740,830,208   ✅ exact
                               computed total ≈ 2.777 T vs 2.78 T advertised
```

The analyzer prints a warning if this stops matching (i.e. the checkpoint changed).

> This is also where this report will differ from the published MI355X one: that report
> gives 84 B active params/token, this one computes 103 B, because the MI355X figure
> omitted the MoE latent projections. Only derived TFLOP/s depends on it — all measured
> throughput and latency numbers are unaffected — and the analyzer applies its own
> formula to *both* sides so the comparison stays internally consistent.

### Report structure

Sections 0–5 mirror `kimi-k3-base.md` (overview, compute, memory, bottleneck,
communication, discussion), with one addition: **§4.2 covers the inter-node PP
boundary**, a path the single-node MI355X run does not have. §6 is new and holds the
comparison: what was held constant and what could not be (6.1), hardware (6.2),
throughput point by point with per-GPU and per-node normalization (6.3), latency (6.4),
where each system's headroom is (6.5), memory footprint (6.6), and an explicit
**"what this does and does not establish"** (6.7).

### Testing it before the run

```bash
./selftest-analyze.sh
```

Generates a synthetic B200 sweep and a vLLM server log in the exact formats the
analyzer parses, runs it against the **real** MI355X baseline, and checks the output.
It proves the plumbing — parse → derive → render — not the physics; the numbers in its
output are meaningless. What it catches is the class of bug that otherwise surfaces at
the end of a 4-hour job: a regex that never matches, a divide-by-zero at c=1, a missing
key. Currently passes, producing a ~400-line report.

---

## Open items

1. **Driver version on the reserved nodes** — must be r580+; last seen at r570. This
   is the one open item that can invalidate the whole approach. See the warning above;
   `./submit.sh probe` settles it in minutes.
2. **Walltime** — whether a 2-node × ~4 h hold inside the reservation needs
   coordination with its owner.
3. **Model export on the compute nodes** — the checkpoint lives on
   `fstor025.ib:/compute/orcd/025`, mounted on the login node. Whether the B200 nodes
   mount it is unverified; `./submit.sh probe` reports it per node.
4. **Load time** — `scontrol` reports `TmpDisk=0`, so 1.42 TiB is read from NFS on every
   server start. Budget 20–40 min per load.
5. **Arm B** (B200-native tuning: prefix caching on, `max_num_seqs` 256, longer context)
   is designed but not scripted — arm A, the MI355X-matched comparison, is the
   deliverable here.
6. **`notes` says MI300X in passing** — the baseline in this repo is **MI355X**
   (CDNA4/gfx950, 288 GB). Everything here compares against that. MI300X is a different
   part (192 GB, CDNA3) and there is no MI300X Kimi-K3 run to compare with.
