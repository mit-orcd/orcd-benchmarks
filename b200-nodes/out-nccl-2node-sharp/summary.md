# NCCL all_reduce — SHARP vs Ring (2-node A/B)

- Generated: 2026-08-07 13:42:20
- Runs: node5500+node5501, node5500+node5502, node5501+node5502
- 8 GPUs/node x 2 nodes, all_reduce, 1 MiB - 16 GiB
- Both legs run back-to-back in ONE allocation (same nodes, same NICs)
- Reference (`results_b200.md`): Ring **170** -> SHARP **357** GB/s = **2.2x**

## Converged busbw: Ring vs SHARP

| Node pair | Ring (GB/s) | SHARP (GB/s) | speed-up | SHARP status | correctness |
|-----------|------------:|-------------:|---------:|--------------|:-----------:|
| node5500+node5501 | 239.1 | — | — | UNAVAILABLE (run aborted) | PASS |
| node5500+node5502 | 234.4 | — | — | UNAVAILABLE (run aborted) | PASS |
| node5501+node5502 | 235.7 | — | — | UNAVAILABLE (run aborted) | PASS |

## Did SHARP actually engage?

NCCL falls back to Ring **silently** when the CollNet/SHARP path cannot be set up, so the speed-up column above is only meaningful once this is confirmed. Evidence from the NCCL `INIT,ENV,NET` debug output:

- **node5500+node5501** — UNAVAILABLE (run aborted): `[node5500][Aug 06 20:01:54 585255][SR     ][39045][error] - no AM service record found(SA query)`
- **node5500+node5502** — UNAVAILABLE (run aborted): `[node5500][Aug 06 20:04:30 366765][SR     ][40282][error] - no AM service record found(SA query)`
- **node5501+node5502** — UNAVAILABLE (run aborted): `[node5501][Aug 07 13:24:37 368414][SR     ][44772][error] - no AM service record found(SA query)`

## Verdict: are these the expected SHARP results?

**No — there are no SHARP results at all.** Not poor numbers: *zero* measurements. The SHARP leg aborted during initialisation, before running a single message size.

**What did work:** the Ring baseline, at **234-239 GB/s** across 3 run(s), validation-clean and consistent with the standalone 2-node all_reduce in `out-nccl-2node/summary.md`. The A/B harness is sound; only half of it can execute.

**What a good result would look like:** the reference cluster measured Ring 170 -> SHARP 357 GB/s = **2.2x**. The upside here is plausibly larger: our Ring all_reduce sits at ~239 GB/s, only ~60% of the 400 GB/s hardware ceiling, while every other ring collective on this fabric already runs at 92-96%. That gap *is* the two-pass Ring penalty SHARP exists to remove, and it is the single largest piece of unrealised inter-node performance in these benchmarks — it directly gates multi-node DDP gradient sync.

**Why we cannot get it:** `No Aggregation Manager (sharp_am) detected`. Confirmed three independent ways — through NCCL with a plain CollNet setup, through NCCL with the AICR environment recipe (`job-nccl-2node-sharp-aicr.sh`), and through `sharp_hello` standalone with NCCL entirely out of the picture. The node-side stack is complete and correct: the plugin loads, CollNet channels are allocated, and the SHARP client library runs. The fabric simply has no Aggregation Manager to register a SHARP job with.

**Nothing further can be done from the job side.** Getting SHARP results requires the InfiniBand admins to run `sharp_am` on the subnet manager / UFM host and provision aggregation trees for these nodes — see `sharp.md` for the full diagnosis, the hardware assessment, and the questions to ask. Once that is done, `job-nccl-2node-sharp-aicr.sh` runs unchanged and this table will populate itself.

## Bus bandwidth vs message size (GB/s)

### node5500+node5501

| Message size | Ring | SHARP | speed-up |
|-------------:|-----:|------:|---------:|
| 1 MiB | 4.1 | — | — |
| 4 MiB | 8.3 | — | — |
| 16 MiB | 27.5 | — | — |
| 64 MiB | 51.1 | — | — |
| 256 MiB | 122.4 | — | — |
| 1 GiB | 177.1 | — | — |
| 4 GiB | 226.6 | — | — |
| 16 GiB | 239.1 | — | — |

### node5500+node5502

| Message size | Ring | SHARP | speed-up |
|-------------:|-----:|------:|---------:|
| 1 MiB | 3.7 | — | — |
| 4 MiB | 5.6 | — | — |
| 16 MiB | 24.5 | — | — |
| 64 MiB | 46.7 | — | — |
| 256 MiB | 124.8 | — | — |
| 1 GiB | 168.8 | — | — |
| 4 GiB | 225.4 | — | — |
| 16 GiB | 234.4 | — | — |

### node5501+node5502

| Message size | Ring | SHARP | speed-up |
|-------------:|-----:|------:|---------:|
| 1 MiB | 6.1 | — | — |
| 4 MiB | 6.6 | — | — |
| 16 MiB | 28.7 | — | — |
| 64 MiB | 62.0 | — | — |
| 256 MiB | 138.2 | — | — |
| 1 GiB | 187.8 | — | — |
| 4 GiB | 228.1 | — | — |
| 16 GiB | 235.7 | — | — |

busbw, best of out-of-place / in-place. SHARP offloads the reduction to the InfiniBand switches, making all_reduce a single pass instead of ReduceScatter+AllGather; the reference sees the gain grow with message size and win above ~4 MB.

