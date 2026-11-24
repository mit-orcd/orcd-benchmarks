# Kimi-K3 on 8 x MI355X (ATOM, TP8): Hyperloom MoE kernel run (Nov 18-24, 2026)

**Result: nothing was kept.** Hyperloom ended with best = baseline and a validated gain of 0.00%.
No kernel, patch, or setting was adopted. Our own A/B check did not run (`logs/hyperloom/moe-20261118/check.dirs`
does not exist), because there was nothing to check.

## 0. All Hyperloom runs for Kimi-K3 (Nov 12-24): the only improvement

Across every Hyperloom run for Kimi-K3, the only improvement is two **server settings**. Hyperloom found both, and we confirmed both with our own script.
No kernel, code patch, or framework change was kept in any run: steps 3 and 4, step 5 (DSpark, +0%), and this MoE kernel run (+0%).
Our numbers are on node6101, 8 x MI355X, TP8, image `nightly_202610011450`, 64 users, ISL/OSL 1024/1024.

| Change (settings only) | Output tok/s (8 GPUs) | TPOT median | TTFT median | GSM8K strict | Source |
|---|---|---|---|---|---|
| 1. `ATOM_GDN_SSM_DTYPE=fp16` | 1,626.1 → 1,679.3 (**+3.3%**, mean of 2 pairs) | about 3% lower | no clear change | 0.9583 → 0.9575 (same) | `hyperloom/status-20261116.md` |
| 2. Step 3 setting, on top of 1 | 1,680.7 → 1,852.8 (**+10.2%**) | 37.28 → 32.86 ms (11.9% lower) | 207 → 823 ms (**4.0x higher**) | 0.9613 → 0.9591 (same) | `hyperloom/status-chain-20261117.md` |

Together the two changes give about +13% output throughput over the plain ATOM image. Accuracy is unchanged. The cost is a 4x higher TTFT, which is still under 1 s.
Hyperloom's own measurements were +2.56% for change 1 and +13.08% for change 2.

The step 3 setting is three settings that act together:
`ATOM_PREFILL_DECODE_INTERVAL=4`, `ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000`, and `--max-num-batched-tokens 8192` (in place of 16384).

**Why change 1 helps.** Kimi-K3 has 93 layers. 69 of them are KDA linear-attention layers, and each keeps a recurrent state per request.
The default stores that state in fp32. With fp16, every decode step reads and writes half as many bytes of state, so each decode step is shorter.
Numerics stay inside the accuracy check.

**Why change 2 helps.** The gain comes from the scheduler, which decides when to run prefill and decode. No kernel runs faster.
The source is the ATOM code in the image: `atom/model_engine/prefill_delayer.py`, `engine_core.py:170-176`, and `utils/envs.py:808-860`.

- **The baseline.** `ATOM_PREFILL_DECODE_INTERVAL` defaults to 0, which switches off ATOM's PrefillDelayer.
  The scheduler then starts a prefill forward as soon as a new request arrives.
  A prefill forward has an almost fixed cost: kernel launches, padding to shape, and the MoE collectives. While it runs, the decode step of all 64 running requests waits.
  With many small prefill forwards, the decode steps are interrupted often, so TPOT goes up.
- **`ATOM_PREFILL_DECODE_INTERVAL=4`.** Any value above 0 turns the PrefillDelayer on, in local mode (TP only, DP=1, PP=1). It then does two things:
  - After every prefill forward, it reserves the next 4 forwards for decode.
  - It holds new prefill requests until together they fill a prefill forward to its target, by default 90% of the token budget.
    Decode keeps running during the hold.

  The result is fewer, fuller prefill forwards and fewer decode interruptions. TPOT is 11.9% lower, so output tok/s rises.
- **`ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000`.** A queued request is released at the latest after 5 s. This allows long holds, so prefills are well batched.
- **`--max-num-batched-tokens 8192`.** This sets the token budget per forward. With 8,192 in place of 16,384, the 90% fill target is reached with about 8 prompts of 1,024 tokens, not about 15.
  So the delayer releases sooner, and each prefill forward is shorter. This explanation is inferred from the code. It was not measured separately.
- **The cost.** New requests wait in the queue before their prefill runs, so TTFT goes from 207 to 823 ms.
  The change trades time to the first token for throughput. Numerics do not change.

Why Hyperloom found no kernel gain, here and in steps 3-4:
- The shipped aiter tuning tables are already near the best for the top kernels.
- Hyperloom cannot write the one large MoE kernel change (sections 5-6).
- In steps 3-4, the runs stopped at their target after the settings change, before the kernel agent did much work.

All paths are relative to `/orcd/data/orcd/022/benchmarks/amd-ubuntu`.
Session: `logs/hyperloom/Kimi-K3/20261118T215437Z-a29acdbf` (called `SESSION` below).
GEAK kernel-agent directory: `SESSION/geak/e2e_cycle0` (called `GEAK` below).

## 1. Setup

| Item | Value |
|---|---|
| Image | `rocm/atom-dev:nightly_202610011450` (SIF `atom-dev-nightly_202610011450.sif`; the launch log names the resolved file `atom-dev-latest.sif`). This is the same image as our step 3 baseline. |
| Hardware | 8 x MI355X (gfx950), node6100 (Nov 18), then node6101 (Nov 21 and Nov 24) |
| Model | `/scratch/Kimi-K3` (text path only; Hyperloom ignored the vision inputs) |
| Baseline recipe | `hyperloom/ref_s3.sh` = FP16 SSM state + step 3 setting: `ATOM_GDN_SSM_DTYPE=fp16`, `ATOM_PREFILL_DECODE_INTERVAL=4`, `ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000`, `--max-num-batched-tokens 8192`, `--kv_cache_dtype fp8`, `--max-num-seqs 64`, `--gpu-memory-utilization 0.93`, `--max-model-len 16384`, `--block-size 128`, online `ptpc_fp8` quant (MoE experts excluded, so they stay mxfp4) |
| Workload | 64 users, ISL/OSL 1024/1024, random range ratio 0.8, 320 prompts, 8 warmups |
| Hyperloom options | kernel agent on, framework agent off, explore off (`--no-explore --no-warm-replay`), kernel phase 75% of budget, target +25% |
| Budget | 6 h of Hyperloom time. Used: 2.43 h (Nov 18), then 5.66 h in total by the Nov 21 pause. The counter was then reset to 2.0 h used, which gave 4 h more on Nov 24. |
| Hyperloom LLM cost (GEAK only) | $67.60 estimated, 1,045 API calls (`GEAK/report/geak_run_report_Kimi-K3.md`; this covers the kernel agent only, not the orchestrator) |

Timeline (US Eastern time, from `logs/hyperloom/moe-20261118/state` and `logs/hyperloom/step4-node610*.state`):

- **Nov 18, 4:48 PM:** Start on node6100. Hyperloom measured the baseline, ran a profile and roofline, and ran GEAK.
  GEAK returned `status=error`. A robustness alert reports that the orchestrator was silent for 8,244 s.
- **Nov 18, 7:12 PM:** Stopped at the Claude 5-hour limit. Hyperloom's `reports/final.md` was written here.
- **Nov 21, 1:23 PM:** Resumed from the kernel phase. node6100 could not mount `/orcd/data/orcd/022` after a reboot,
  so the run moved to node6101. A stale lock from the old loop blocked the first relaunch.
- **Nov 21, 1:26-4:40 PM:** GEAK tuning session 1 and kernel lane 1. Paused by user request at 5.66 h used.
- **Nov 24, 12:06-2:36 AM:** Resumed with a 4 h budget (GEAK tuning session 2). Stopped at the Claude 5-hour limit.
- **Nov 24, 4:36-8:04 AM:** Resumed with 1.51 h left (GEAK tuning session 3 and kernel lane 2). Stopped at the Claude 5-hour limit
  (`SESSION/STOPPED_TOKEN_LIMIT`). Final `stop_reason=token_limit`, phase `KERNEL_AGENT`.

## 2. Result

Best = baseline, so the "best" column repeats the baseline.

| Metric | Our run, step 3 setting (Nov 18, 640 prompts) | Hyperloom baseline (Nov 18, 320 prompts) | Hyperloom best | GEAK reference (warm server, 320 prompts) |
|---|---|---|---|---|
| Output tok/s (8 GPUs) | 1,852.8 | 1,930.7 | 1,930.7 (= baseline) | 2,065.9 |
| TPOT, median | 32.86 ms | 31.20 ms (mean 30.82) | same | 30.10 ms |
| TTFT, median | 823 ms | 872 ms (mean 1,089) | same | 482 ms |
| GSM8K strict | 0.9591 (all 1,319) | 0.9606 | same | not measured |
| Validated gain | n/a | n/a | **0.00%** | best A/B +0.43%, below the 0.72% noise floor |

Sources: ours from `hyperloom/status-chain-20261117.md`.
Hyperloom baseline: `SESSION/runs/baseline/8b8334f874084228ac7c991a7d575907/measure_round/benchmark_atom_20261118_221213/` (`summary.txt`, `inferencex_result.json`).
Hyperloom GSM8K: `.../warmup_round/benchmark_atom_20261118_215519/results_2026-11-18T22-12-07.246826.json`.
GEAK reference: `GEAK/baseline/bench_summary.json`.

The three throughput numbers come from different measurement modes, so do not compare them with each other.
The GEAK "warm server" mode reuses a running server, so its TTFT is much lower.
GEAK measured its own restart-to-restart spread as 0.72% (2,065.94 / 2,080.81 / 2,070.90 / 2,068.28 tok/s).
It accepted only gains above that floor.

## 3. What was modified

Nothing. The facts file reports `kept_kinds: []`, and `SESSION/patches/`, `SESSION/findings/`, `SESSION/runs/kernel_opt/` and `SESSION/runs/integrate/` are empty.
The tuning reports end with "The installed tree is pristine". The aiter CSV md5 sums match the shipped files, and there is no overlay, env, or deploy bundle.
`SESSION/current_setting.sh` is the plain baseline launch.

## 4. How much each change improved

There are no kept changes, so there is nothing to report. Section 6 lists the measured effect of each attempt.

## 5. Why it would improve performance

Nothing was kept, so this section shows the profile and the mechanism the agent planned to use. Nothing here was measured as a gain.

Hyperloom's profile at 64 users, decode-dominated (`GEAK/strategy.md`), as a share of GPU time:

| Kernel | Share |
|---|---|
| MoE stage 1 (`gemm1_a16w4_...`, FlyDSL) | 21.49% |
| MoE stage 2 (`gemm2_a16w4_...`) | 11.47% |
| `kernel_gemm_0` (fp8 a8w8 dense) | 11.47% |
| all-reduce | 10.5% |
| `_attn_res_fused_kernel` | 6.51% |
| `f4gemm` a4w4 dense | 6.22% |

So the MoE expert GEMMs are 32.96% of GPU time. This matches the order of our own profile in `plan-hyperloom.md`: MoE 36.7%, all-reduce 21.7%, other GEMMs 19.5%, attention 7.6%.
The TraceLens window also showed 31.37% GPU idle and 11.96% exposed communication (`SESSION/reports/final.md`).

The agent's MoE analysis is in `GEAK/kernels/_exp/team_gemm1_..._20261124_120334_3502314_6253/.../roadmap.md`:

- **The problem.** At decode M=64, about 610 of 896 experts are active, with about 1.7 valid rows per 16-row tile.
  So about 90% of MFMA rows are idle. The kernel streams weights, about 1.3 GB per call.
  For each weight byte, it unpacks fp4 to bf16 and multiplies by the scale on the VALU. It then issues bf16 MFMA with K=32.
- **The planned fix ("kill the dequant").** Feed the fp4 weights directly to the gfx950 scaled MFMA (`v_mfma_scale_f32_*_f8f6f4`),
  using the e8m0 scales as the B-scale operand. The activations would be quantized inline to mxfp8.
  This removes the VALU dequantization and cuts MFMA issue by 4x per weight byte, with no extra weight memory.
  A later round would fuse stage 1 and stage 2, to remove one launch and the intermediate round trip through memory.
- **Status.** This kernel was never built or timed (see section 6, item 3).

## 6. What was tried and not kept, and why

All numbers below are Hyperloom's or GEAK's, not ours.

1. **Retune the fused-MoE tile table at token=64 (h0, MoE, 32.96%).** Rejected: −1.29% end to end.
   - The tuner picked `flydsl_moe1_..._t16x192x128_w2_xcd1` + `flydsl_moe2_..._t16x256x128_atomic_bnt2`.
     It ran at 217.86 µs in the tuner, against 231.15 µs for the shipped row in `kimik3_a16w4_tuned_fmoe.csv`.
     On the production path it ran at 225.51/225.92 µs against 230.40/230.47 µs (1.021x).
   - It was engaged in the server, but serving was slower: 2,042.98 tok/s against references of 2,070.90 and 2,068.28 tok/s.
   - Reason given: the tuner times uniform random routing. The server's real expert skew, at about 1.1 tokens per expert, favours the shipped tiling.
     The change was reverted and kept only as evidence (`GEAK/tuning/rejected_bundle_moe_t64/`, `GEAK/tuning/tuning_report.prev_session.md`).
2. **Op-level backend bake-off on MoE gemm1 (kernel lane, Nov 21 and Nov 24).** Not integrated.
   - The best result was the same retuned FlyDSL row: 0.14478 ms against 0.14902 ms live (1.0254x).
     The Amdahl ceiling for that speedup is +0.82% end to end, and the same row had already lost 1.29% in serving.
   - The other backends were not usable. aiter CK/asm has no 2-stage path for a16w4 SiTUv2.
     The Triton `moe_op_gemm_a16w4` has no SiTUv2 and uses a different weight layout.
   - Sources: `GEAK/kernels/gemm1_..._task/opbench_result.json`, `GEAK/opbench_moe/`.
3. **Author a new MoE kernel (scaled-MFMA a8w4, section 5).** Never reached timing.
   - Two attempts were made, on Nov 21 at 4:27 PM and on Nov 24 at 7:03 AM.
     Each time the baseline unit test passed correctness (`CORRECTNESS: PASS`, including CUDA-graph replay).
   - The next step, building the candidate overlay, then failed with `PermissionError: [Errno 13]`.
     On Nov 21 it could not write `workspace/_cand_overlay/_patched`. On Nov 24 it could not write `workspace/_cand_overlay/_overlay_manifest.json`.
     As a result, `extraction_baseline.json` is `usable: false` (no `ut_result.json`).
   - On Nov 24 the unit test failed at 7:52 AM. The Claude 5-hour limit stopped the run at 8:02 AM. No candidate kernel was ever timed.
   - GEAK's final status is `workflow_parse_error`, phase reached `search` (`SESSION/geak/result.json`).
4. **Retune the fp8 dense GEMM row a8w8 M=64 N=7168 K=4224 (h1).** Real per-op win, but below the noise floor.
   - The change moved this row from CK to `flydsl_bpreshuflle_64x128x128_..._ks3`: 14.10 µs against 19.53 µs (1.386x, errRatio 0.0).
   - Interleaved A/B: 2,077.15 against 2,068.22 tok/s (+0.43%; the two pairs gave +1.14% and −0.27%). This is below the 0.72% floor.
   - The agent's own estimate: at most about 0.5 ms of a 30 ms step, even if all 93 layers used this shape.
   - The bundle is parked, not installed: `GEAK/tuning/candidate_bundle_a8w8_k4224/`.
5. **Retune the a4w4 mxfp4 dense GEMM (h2, 6.22%).** No gain.
   - The shipped asm `f4gemm ... 32x128` with splitK=0 won for all 4 shapes, out of 480 candidates.
   - The CK splitK variants fail the error gate (errRatio about 0.99).
6. **Retune the bf16 router GEMM (`hgemm_bf16_...`, 2.21%).** No gain.
   - Out of 13,442 candidates, the best ran at 8.707 µs against 8.655 µs for the shipped row. The shipped table is already optimal.
7. **Extract `kernel_gemm_0` (fp8 dense, 11.47%).** Infrastructure failure.
   - The capture server failed with `/usr/bin/sleep: Transport endpoint is not connected` (the container's FUSE overlay was lost),
     then `atom readiness: /health failed`, `CAPTURE_EXIT=2` (`GEAK/logs/capture_kernel_gemm_0.log`, Nov 24, 2:36 AM).
8. **Hyperloom's own kernel-opt dispatch (Nov 18).** It attempted no kernels.
   - All top kernels are HIP-graph replays, so TraceLens found no source file (`not attempted: no_source_file`).
   - `aiter::fused_moe_` (2.76%) was below the 10% dispatch threshold (`SESSION/reports/kernel_optimization_summary.json`).
9. **Not attempted:**
   - `_attn_res_fused_kernel`, the KDA gating kernel, and the KDA decode chain were planned but never reached.
   - The shipped a8w4 and a4w4 MoE rows (221.6 and 214.0 µs, against 231.2 µs for a16w4) were not tried, because they need an accuracy gate.
   - All-reduce (10.5%) is not a kernel-agent target.

Why nothing was kept, in summary:

- Every table-selectable kernel on the top ops is already tuned by the shipped Kimi-K3 aiter CSVs.
- The one real lever, a newly written MoE kernel, never got past the overlay permission error.
- Usable time was lost to three Claude 5-hour limit stops, the node6100 mount failure, a stale lock, and the FUSE failure.
- A 0.72% restart noise floor hides gains below about 0.7%.

## 7. Is Hyperloom worth running again for Kimi-K3?

**No, not in this form.** The rule set in `hyperloom/status-chain-20261117.md` was:
"If that run keeps no kernel, stop using Hyperloom for Kimi-K3". This run kept no kernel.

Evidence from this run:

- Four working periods gave 0.00% validated gain: Nov 18 4:48-7:12 PM, Nov 21 1:26-4:40 PM, and Nov 24 12:06-2:36 AM and 4:36-8:04 AM.
  GEAK's LLM cost was $67.60 (estimated).
- The table-tuning space is exhausted on MoE, a8w8, a4w4, and the router GEMM. A rerun would repeat these dead ends.
- A new MoE kernel is the only lever with a plausible gain. The agent estimated 3-8% end to end for h0 (`GEAK/strategy.md`); nothing measured this.

A narrow rerun would only make sense if all of these hold:

1. The `_cand_overlay` permission bug in GEAK is fixed.
2. The run is limited to the MoE kernel authoring lane.
3. The Claude 5-hour limit has reset before start.

Even then, the result is uncertain, and any kernel would need our own throughput and GSM8K check.

Better uses of time, without Hyperloom:

- Test `--max-num-seqs` 128/256 and a smaller prefill delay with our own script (from the status chain; still open).
- Test a newer ATOM image. In the past this gave more than any Hyperloom run.
- The a8w4 MoE path (expert parallel, `--moe-backend mega`) is still blocked in ATOM/aiter code, which Hyperloom cannot fix.
