# Admin notes: node6100 / node6101 (8 × MI355X each)

Written 2026-10-08 from all results in this directory. Details: [SUMMARY.md](SUMMARY.md).

## Healthy, nothing to fix

- **GPUs:** on ROCm 7.14, the RVS (ROCm Validation Suite) GEMM, RCCL, Primus training and Kimi-K3 serving results are almost the same as amd-cloud's (within a few percent). Both nodes agree within 1–3%.
- **Network:** all 16 Pollara rails reach 98% of line rate. GPUDirect RDMA is active on all 8 rails, and the ring collectives across the two nodes reach 92–95% of 400 GB/s. No extra driver module (like `nv_peermem` on B200) or ACS change is needed.
- **Apptainer** works since the AppArmor user-namespace change on 2026-10-01. Please keep that setting.

## Suggested changes

| # | What | Why (measured) | Suggested action |
|---|---|---|---|
| 1 | RCCL settings for point-to-point across nodes | 2-node sendrecv reaches 57–64% of one rail by default. `NCCL_NCHANNELS_PER_NET_PEER=4` lifts it to 94–99%, but lowers alltoall by about 30%. On ROCm 7.14, RCCL's new `IB-CAST` network transport makes alltoall and sendrecv slower; `NCCL_NET=IB` restores them (`ubuntu/sendrecv-check.md`) | Document for users: on ROCm 7.14 set `NCCL_NET=IB` for multi-node jobs; add `NCCL_NCHANNELS_PER_NET_PEER=4` only for sendrecv-heavy jobs (not in `/etc/nccl.conf`, since it hurts alltoall). |
| 2 | NUMA binding for the RDMA NICs | Without binding, 8 rails at once reach only 52% of line rate (98% with binding) | Document the mapping (rails 0–3 → NUMA 0, rails 4–7 → NUMA 1), or provide a small wrapper for users. |
| 3 | Scheduling | The nodes are ssh-only and not in Slurm; jobs from different users can collide on the GPUs | Add the nodes to Slurm (GPU GRES), or provide a shared lock or reservation method. |
| 4 | GPU access for new users | GPU tests fail until the user is in the `render` and `video` groups | Add GPU users to `render` and `video` by default. |
| 5 | Docker | AMD's recipes (Kimi-K3, ATOM, Primus) are written for docker; here they run through an apptainer shim | Either provide rootless docker/podman, or document apptainer use for AMD's images. |
| 6 | ROCm version | Host ROCm is 7.2.4. On ROCm 7.14, fp4 GEMM is ≈28% faster per GPU when each GPU has its own process; all other precisions are the same | When convenient, offer ROCm 7.14 (or newer) as a module next to 7.2.4. A user-space copy is in `amd-software/rocm-7.14.0`. |
| 7 | Profiling | `kernel.perf_event_paranoid = 4` blocks `perf`, which was needed to look at host-side launch overhead | Lower it to 2 on these nodes, if the site policy allows. |

## For user documentation

- Use 1, 2, 4 or 8 GPUs per node for collective-heavy work. With 5–7 GPUs, RCCL busbw falls to ≈47 GB/s, because those GPU counts don't use the full set of direct GPU links. amd-cloud shows the same.
- Run one process per GPU. On ROCm 7.14, a single process driving 8 GPUs leaves the GPUs idle 37–70% of the time (RVS fp4 profile).
- `$HOME` on these nodes is node-local and differs from the login nodes. Use `/orcd/data/...` paths.
- The Kimi-K3 weights are at node-local `/scratch/Kimi-K3` on both nodes. Please keep them, or tell the user before `/scratch` is cleaned.

## To raise with AMD (optional)

- Kimi-K3 with the new AMD recipe 2026-10 (our runs use that exact recipe), on AMD's own workload, is 0.56–0.82× AMD's published numbers at 1–4 users and the same or better from 8 users. Ask which host settings and warm-up were used.
- Megatron-LM v26.1 image: fused RoPE crashes on gfx950. GPT-15.6B runs without it at ≈0.73× of a published MI355X result.
- RVS fp4 on ROCm 7.14 with 8 GPUs in one process runs at 0.57× of 1 GPU. This is a host-side launch limit in the test, not in the GPUs.
