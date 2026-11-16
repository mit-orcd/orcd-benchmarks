# Hyperloom rerun report (started Nov 12)

The plan is in [notes-improve.md](notes-improve.md), "How to fix each problem". The two background jobs below write into this file as they go.

## Fixes made before the runs

1. **JSON quoting.** The real cause was not `reference_script.py` (its parser keeps the quotes). It was `remove_server_args` in `hyperloom/orchestrator/actions/executors/_grid_server_args.py`.
   - **What happens:** each time Hyperloom removes a flag (as in the explore variants and the conc sweep's best setting), it splits the flags with `shlex.split`. That strips the JSON's inner quotes. Its repair step cannot fix values with `*` or `[ ]` (`*self_attn.[qkv]_conv1d*`).
   - **Fix:** split with Hyperloom's own JSON-safe splitter first. The original file is kept as `_grid_server_args.py.orig-20261112`.
   - **Unit test:** removing `--max-num-seqs`, `--cudagraph-mode` or the quant flag now keeps valid JSON.
2. **Temp folder.** `TMPDIR=/dev/shm/shaohao-tmp` (node-local), set in `run_step4.sh` and for the candidate runs. It is cleaned after each run.
3. **Helper file access (the "sandbox").** `claude` was not on PATH inside the container, so Hyperloom quietly fell back to its in-process SDK helpers.
   - **Proof:** session A's log says `falling back to in-process`.
   - **Why it mattered:** the in-process helpers run without `--add-dir` and without `bypassPermissions`, so every read outside the working folder was refused.
   - **Fix:** `run_step4.sh` now puts `~/.local/bin` on PATH, so Hyperloom uses the CLI helpers.
   - **Helper time:** CPU helpers get a base of 20 min instead of 10 (new `HYPERLOOM_SPECIALIST_CPU_BASE_MIN` in `orchestrator/loop/dispatcher.py`, original kept as `.orig-20261112`).
4. **GEAK off** (`KERNEL=0` → `--no-kernel`).
5. **New sessions get their own log**, `step4-A-run-new-<time>.log`, so session A's `step4-A-run.log` is not overwritten.

**Candidate changes from the plan:**
- **a8w4:** ATOM takes the MoE activation type from the model's quantization config. Kimi-K3's MXFP4 config has no activation type, so activations stay BF16. The one switch ATOM offers is `--moe-backend mega` (MegaMoE, a8w4 by default), so that is the a8w4 candidate.
- **MoE tile tuning:** deferred. GEAK's best tile setting gave +0.3% at 64 users, which is inside run-to-run noise, and applying it needs a tuned-table overlay for aiter.

## Part A: Hyperloom steps on node6100 (T1 → T2 → T3 → T4)

Script: `hyperloom/rerun_pipeline.sh`. Progress: `logs/hyperloom/rerun-pipeline-node6100.state`. Cancel: `touch logs/hyperloom/AUTORESUME_STOP`.

<!-- A-BEGIN -->
- Nov 12 17:10 UTC: Pipeline started on node6100.
- Nov 12 17:16 UTC: **T1 passed** (JSON quoting + temp folder): Hyperloom's flag removal now keeps the JSON; ATOM started with TMPDIR on /dev/shm and answered ("Paris. The Eiffel Tower is located in Paris. The Louvre Muse"). Folder `logs/hyperloom/rerun-tests/t1-20261112T171014Z`.
- Nov 12 17:16 UTC: **T2 passed** (helper file access): `claude` found at `/home/shaohao/.local/bin/claude` inside the container, so Hyperloom will use the CLI helpers with `--add-dir` and `bypassPermissions`. A test helper read `/scratch/Kimi-K3/config.json` (model_type kimi_k3) and `/app/ATOM/atom/config.py` (line 2035).
- Nov 12 18:27 UTC: T3 check of session `Kimi-K3/20261112T172246Z-162a2dc3` (log `step4-A-run-new-20261112T171617Z.log`):
- Nov 12 18:27 UTC:   - JSON quoting: 0 server logs with `invalid loads value`
- Nov 12 18:27 UTC:   - Temp folder: 0 files with `Errno 39`
- Nov 12 18:27 UTC:   - Explore: 0 of 0 variants gave a benchmark number
- Nov 12 18:27 UTC:   - Helpers: 4 finished, 1 report blocked reads; claude CLI /home/shaohao/.local/bin/claude; in-process fallback no
- Nov 12 18:27 UTC:   - Multi-user check (conc sweep): 0 of 0 levels gave a number
- Nov 12 18:27 UTC:     - final.md: - **Why it stopped**: Robustness escalated: the run stopped early (not a target hit). Common triggers are an approaching deadline, a validated-gain plateau, rising crash_count, or a stale aiter JIT bu
- Nov 12 18:27 UTC:     - final.md: - baseline            : `1671.6 tok/s/GPU`
- Nov 12 18:27 UTC:     - final.md: - current_best        : `1671.6 tok/s/GPU` (action=`baseline`)
- Nov 12 18:27 UTC:     - final.md: - cumulative_gain_val : `0.00%` ⚠ never validated — no full-stack rebench ran in this session
- Nov 12 18:27 UTC:   - stop_reason: robustness_escalated
- Nov 12 18:27 UTC: T3 failed: RESULT FAIL: explore did not run; helpers still blocked (f97017705e8f41cab6c5d50df7213eaa)
- Nov 12 18:27 UTC: T3 failed once -> retry.
- Nov 12 18:55 UTC: T3 was stopped by a Claude limit (session `20261112T183144Z-6f7c0147`); it will run again when Claude is available.
- Nov 12 18:55 UTC: Claude not available (limit); checking every 60 min.
- Nov 12 20:14 UTC: Pipeline started on node6100.
- Nov 12 20:15 UTC: Why T3 failed (found Nov 12, 3:10 PM ET): Hyperloom's framework step could not write patches (no git copy of ATOM: `no_git_framework_source_root`), never handed over to explore, and used the session's time; the robustness check then closed the session. The one 'blocked' helper was a false alarm (it said a fusion was 'blocked by design' after reading the code), so the check now looks only for blocked reads. Change: framework step off (`FRAMEWORK=0` → `--no-framework-agent`) for T3 and T4. T3 restarted with the new setting at 3:14 PM ET (times above are UTC).
- Nov 12 21:31 UTC: T3 check of session `Kimi-K3/20261112T201815Z-45cc6269` (log `step4-A-run-new-20261112T201407Z.log`):
- Nov 12 21:31 UTC:   - JSON quoting: 0 server logs with `invalid loads value`
- Nov 12 21:31 UTC:   - Temp folder: 0 files with `Errno 39`
- Nov 12 21:31 UTC:   - Explore: 1 of 1 variants gave a benchmark number: v00_kda_ssm_state_fp16 1,720 tok/s
- Nov 12 21:31 UTC:   - Helpers: 2 finished, 0 report blocked reads; claude CLI /home/shaohao/.local/bin/claude; in-process fallback no
- Nov 12 21:31 UTC:   - Multi-user check (conc sweep): 0 of 0 levels gave a number
- Nov 12 21:31 UTC:     - final.md: - **Why it stopped**: Robustness escalated: the run stopped early (not a target hit). Common triggers are an approaching deadline, a validated-gain plateau, rising crash_count, or a stale aiter JIT bu
- Nov 12 21:31 UTC:     - final.md: - baseline            : `1676.7 tok/s/GPU`
- Nov 12 21:31 UTC:     - final.md: - current_best        : `1719.6 tok/s/GPU` (action=`explore`)
- Nov 12 21:31 UTC:     - final.md: - cumulative_gain_val : `2.56%` (validated_at_stack_len=1, ts=2026-11-12T21:28:21.890080+00:00)
- Nov 12 21:31 UTC:   - stop_reason: robustness_escalated
- Nov 12 21:31 UTC: **T3 passed.**
- Nov 12 21:31 UTC: T4: fresh 12-hour session started with auto-resume after Claude limits (GEAK off, framework agent off, conc sweep 1,8,64,128,256). Progress: `logs/hyperloom/step4-autoresume-node6100.state`.
<!-- A-END -->

## Part B: candidates and profiles on node6101 (no tokens)

Script: `hyperloom/run_candidates.sh`, using our own `atom/run_kimi_atom_mtp.sh`. Progress: `logs/hyperloom/candidates-node6101.state`. Cancel: `touch logs/hyperloom/CANDIDATES_STOP`.

<!-- B-BEGIN -->
### Candidates (our script, node6101, ATOM image nightly_202610011450, no MTP)

Output tok/s per GPU, mean ± standard deviation over the repeats. 1024 tokens in, 1024 out, 10 × users prompts, max 64 sequences (as in the step 3 baseline). TPOT = time per output token; TTFT = time to first token (medians).

| Arm | Repeats | GSM8K (strict) | 1 users | 8 users | 64 users | 128 users | 256 users |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3 | 0.961 | 7.9 ± 0.0 | 54.6 ± 0.1 | 206.4 ± 1.5 | 207.2 ± 0.2 | 206.3 ± 0.4 |
| a8w4 (`--moe-backend mega`) | 2 | – | – | – | – | – | – |
| Expert parallel | 2 | – | – | – | – | – | – |

Change against the baseline (mean tok/s per GPU; TPOT and TTFT at 64 users):

| Arm | 1 users | 8 users | 64 users | 128 users | 256 users | TPOT 64 (ms) | TTFT 64 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | 38.06 | 207 |
| a8w4 (`--moe-backend mega`) | – | – | – | – | – | – | – |
| Expert parallel | – | – | – | – | – | – | – |

Accuracy rule: a candidate counts only if its GSM8K is within 0.01 of the baseline (0.961); Hyperloom's own baseline gate measured 0.958.

Run folders:
- base: `logs/node6101/hyperloom-cand/atom/kimi_atom_mtp_base_r1_20261112_171020`, `logs/node6101/hyperloom-cand/atom/kimi_atom_mtp_base_r2_20261112_181351`, `logs/node6101/hyperloom-cand/atom/kimi_atom_mtp_base_r3_20261112_191012`
- mega: `logs/node6101/hyperloom-cand/atom/kimi_atom_mtp_mega_r1_20261112_181058`, `logs/node6101/hyperloom-cand/atom/kimi_atom_mtp_mega_r2_20261112_190719`
- ep: `logs/node6101/hyperloom-cand/atom/kimi_atom_mtp_ep_r1_20261112_181116`, `logs/node6101/hyperloom-cand/atom/kimi_atom_mtp_ep_r2_20261112_190738`

### Profiles of the baseline (rocprofv3 kernel trace, all 8 GPUs, 5 s window during the benchmark)

| Kernel group | 1 users | 64 users | 256 users |
|---|---:|---:|---:|
| MoE expert GEMMs | 10.3% | 36.7% | – |
| Communication (all-reduce etc.) | 2.5% | 21.7% | – |
| Other GEMMs | 35.4% | 19.5% | – |
| Attention | 16.8% | 7.6% | – |
| Everything else | 35.0% | 14.5% | – |

Top 10 kernels at 1 users (8 stats files, 34.59 s of GPU time summed over GPUs):

| Kernel | Group | Share |
|---|---|---:|
| `kernel_gemm_0` | Other GEMMs | 16.2% |
| `_attn_res_fused_kernel` | Attention | 11.9% |
| `aiter::f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128` | Other GEMMs | 11.6% |
| `void aiter::ar_ll_gfx9<std::bfloat16_t, 8, 256>(std::bfloat16_t* const*, std::bfloat16_t*,` | Everything else | 10.2% |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_situv2_16x32` | MoE expert GEMMs | 9.4% |
| `void aiter::topk_reg_kernel<hip_bfloat16, 14, true, false, true>(hip_bfloat16*, hip_bfloat` | Everything else | 3.8% |
| `hgemm_bf16_t16x32x128x6_ksd_w1x2x1_gm0_bias0_lnt_pft_0` | Other GEMMs | 3.4% |
| `void kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 16, 1>, float, std::bfloat16_t>(MlaRedu` | Attention | 3.0% |
| `_ZN5aiter37dynamic_per_token_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPT_PKfiPKii` | Everything else | 2.5% |
| `void aiter::dynamic_per_group_scaled_quant_kernel<std::bfloat16_t, opus::fp4_t, 32, 32, tr` | Everything else | 2.5% |

Top 10 kernels at 64 users (8 stats files, 40.52 s of GPU time summed over GPUs):

| Kernel | Group | Share |
|---|---|---:|
| `gemm1_a16w4_port_a16w4_h3584_i384_ne896_bm16_tn128_situv2_xcd4` | MoE expert GEMMs | 19.1% |
| `void aiter::cross_device_reduce_2stage<std::bfloat16_t, 8, false, false>(aiter::RankData*,` | Communication (all-reduce etc.) | 14.2% |
| `kernel_gemm_0` | Other GEMMs | 10.5% |
| `gemm2_a16w4_port_ne896_h3584_i384_bm16_tn128` | MoE expert GEMMs | 10.2% |
| `void aiter::cross_device_reduce_2stage<std::bfloat16_t, 8, false, true>(aiter::RankData*, ` | Communication (all-reduce etc.) | 7.3% |
| `_attn_res_fused_kernel` | Attention | 5.4% |
| `aiter::f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128` | Other GEMMs | 5.3% |
| `fused_sigmoid_gating_delta_rule_update_kernel` | Everything else | 3.9% |
| `gemm1_a16w4_port_a16w4_h3584_i384_ne896_bm32_tn128_situv2_xcd4` | MoE expert GEMMs | 3.0% |
| `hgemm_bf16_t64x32x64x5_ksd_w4x1x1_gm0_bias0_lnt_pft_0` | Other GEMMs | 1.9% |
<!-- B-END -->
