# Admin notes: node6100 / node6101 (8 × MI355X each)

Written 2026-10-08, updated 2026-10-09, from all results in this directory. Details: [SUMMARY.md](SUMMARY.md).

## Healthy, nothing to fix

- **GPUs:** on ROCm 7.14, the RVS (ROCm Validation Suite) GEMM, RCCL, Primus training and Kimi-K3 serving results are almost the same as amd-cloud's (within a few percent). Both nodes agree within 1–3%.
- **Network:** all 16 Pollara rails reach 98% of line rate. GPUDirect RDMA is active on all 8 rails, and the ring collectives across the two nodes reach 91–93% of 400 GB/s, the same as B200's InfiniBand. No extra driver module (like `nv_peermem` on B200) or ACS change is needed.
- **Apptainer** works since the AppArmor user-namespace change on 2026-10-01. Please keep that setting.

## Suggested changes

| # | What | Why (measured) | Suggested action | Documents |
|---|---|---|---|---|
| 1 | Default ROCm version | The default (host) ROCm is 7.2.4. ROCm 7.14 is newer and the same version amd-cloud uses: fp4 GEMM is ≈25% faster per GPU when each GPU has its own process, all other precisions are the same, and with `NCCL_NET=IB` (item 2) its RCCL matches or beats 7.2.4 across nodes | Change the default ROCm from 7.2.4 to 7.14 (or newer), with `NCCL_NET=IB` set by default (e.g. in `/etc/nccl.conf` or the module). Keep 7.2.4 available as a module for users who need it. A user-space copy is in `amd-software/rocm-7.14.0`. | [ubuntu/rocm.md](ubuntu/rocm.md), [vs-amd-cloud/rvs-fp4-recheck.md](vs-amd-cloud/rvs-fp4-recheck.md), [ubuntu/sendrecv-check.md](ubuntu/sendrecv-check.md) |
| 2 | RCCL settings for point-to-point across nodes | With RCCL's defaults, 2-node sendrecv reaches only 57–64% of one rail. On ROCm 7.14, RCCL's new `IB-CAST` network transport also slows alltoall (82% vs 96% on 7.2.4). With `NCCL_NET=IB` (standard transport) plus `NCCL_NCHANNELS_PER_NET_PEER=4`, sendrecv reaches 99% (same as B200); with `NCCL_NET=IB` alone, alltoall reaches 96%. The 4 channels lower alltoall by about 30% (`ubuntu/sendrecv-check.md`) | Document for users: on ROCm 7.14 set `NCCL_NET=IB` for multi-node jobs; add `NCCL_NCHANNELS_PER_NET_PEER=4` only for sendrecv-heavy jobs (not in `/etc/nccl.conf`, since it hurts alltoall). | [ubuntu/sendrecv-check.md](ubuntu/sendrecv-check.md), [ubuntu/rccl_2node.md](ubuntu/rccl_2node.md), [mi355x-vs-b200.md §4](mi355x-vs-b200.md#4-rccl-vs-nccl--two-nodes-16-gpus-8-per-node) |
| 3 | NUMA binding for the RDMA NICs | Without binding, 8 rails at once reach only 52% of line rate (98% with binding) | Document the mapping (rails 0–3 → NUMA 0, rails 4–7 → NUMA 1), or provide a small wrapper for users. | [ubuntu/net.md](ubuntu/net.md), [node6100/ib_bw_nonuma.md](node6100/ib_bw_nonuma.md) (no binding), [node6100/ib_bw.md](node6100/ib_bw.md) (with binding) |
| 4 | Scheduling | The nodes are ssh-only and not in Slurm; jobs from different users can collide on the GPUs | Add the nodes to Slurm (GPU GRES), or provide a shared lock or reservation method. | [../README.md](../README.md) (Introduction) |
| 5 | GPU access for new users | GPU tests fail until the user is in the `render` and `video` groups | Add GPU users to `render` and `video` by default. | [../plan.md](../plan.md) (GPU access), [../README.md](../README.md) (Installation) |
| 6 | Profiling | `kernel.perf_event_paranoid = 4` blocks `perf`, which was needed to look at host-side launch overhead | Lower it to 2 on these nodes, if the site policy allows. | [vs-amd-cloud/rvs-fp4-recheck.md](vs-amd-cloud/rvs-fp4-recheck.md); `logs/node6101/rvs/fp4_recheck_20261007_201109/STATE.txt` ("perf skipped (perf_event_paranoid=4)") |

## For user documentation

- Use 1, 2, 4 or 8 GPUs per node for collective-heavy work. With 5–7 GPUs, RCCL busbw falls to ≈47 GB/s, because those GPU counts don't use the full set of direct GPU links. amd-cloud shows the same.
- Run one process per GPU. On ROCm 7.14, a single process driving 8 GPUs leaves the GPUs idle 37–70% of the time (RVS fp4 profile).
- `$HOME` on these nodes is node-local and differs from the login nodes. Use `/orcd/data/...` paths.
- Kimi-K3 serving on one node: vLLM recipe 1 for interactive use with short prompts (fastest up to 128 users); ATOM for long prompts (fastest from 2 users at 128K/1K) and for 256+ users. Details: `ubuntu/kimi-recipe-old-vs-new.md`.
- The Kimi-K3 weights are at node-local `/scratch/Kimi-K3` on both nodes. Please keep them, or tell the user before `/scratch` is cleaned.

## To raise with AMD (optional)

- Kimi-K3 on AMD's own workload (128K/1K): with the same recipe (vLLM recipe 2) our nodes reach 0.56–0.82× AMD's tested numbers at 1–4 users and 1.01–1.10× from 8 users. All three recipes we ran (ATOM, vLLM recipe 1, vLLM recipe 2) are at 0.55–0.66× at 1 user, which points to a per-step overhead on our setup rather than one recipe. Ask which host settings and warm-up were used, and for their TTFT (time to first token) and TPOT (time per output token) at 1 user.
- Megatron-LM v26.1 image: fused RoPE crashes on gfx950. GPT-15.6B runs without it at ≈0.73× of a published MI355X result.
- RVS fp4 on ROCm 7.14 with 8 GPUs in one process runs at 0.57× of 1 GPU. This is a host-side launch limit in the test, not in the GPUs.
