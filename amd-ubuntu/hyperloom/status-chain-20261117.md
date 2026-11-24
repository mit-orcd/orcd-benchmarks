# Hyperloom chain of Nov 17 (plan-hyperloom.md §11), results per step

Written by `hyperloom/run_chain.sh` when each step ends. Baseline of every step: image
`nightly_202610011450` + `ATOM_GDN_SSM_DTYPE=fp16`. Output tok/s are totals for all 8 GPUs.
Progress: `logs/hyperloom/chain-20261117/chain.state`.

## s2: server settings at 128-256 users

- Status: finished. Session: `logs/hyperloom/Kimi-K3/20261117T162001Z-1505ca2e`
- Stop reason: conc_sweep_done; target users: 256
- Baseline: 1,678.5 tok/s
- Validated gain: +0.15%
- Best setting (replay_warm_recipe): ``
- Report: `logs/hyperloom/Kimi-K3/20261117T162001Z-1505ca2e/reports/final.md`

## s3: kernel agent on the MoE

- Status: finished. Session: `logs/hyperloom/Kimi-K3/20261117T201037Z-1a94cc7f`
- Stop reason: target_reached; target users: 64
- Baseline: 1,706.9 tok/s
- Validated gain: +13.08%
- Best setting (explore): `--kv_cache_dtype fp8 --trust-remote-code --no-enable_prefix_caching --online_quant_config {"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]} --tensor-parallel-size 8 --max-num-seqs 64 --gpu-memory-utilization 0.93 --max-model-len 16384 --blo`
- Report: `logs/hyperloom/Kimi-K3/20261117T201037Z-1a94cc7f/reports/final.md`

## s4: framework agent (communication)

- Status: finished. Session: `logs/hyperloom/Kimi-K3/20261118T014451Z-39740a3e`
- Stop reason: target_reached; target users: 64
- Baseline: 1,719.9 tok/s
- Validated gain: +11.00%
- Best setting (integrate_patch): ``
- Report: `logs/hyperloom/Kimi-K3/20261118T014451Z-39740a3e/reports/final.md`

## s5: low load with DSpark speculative decoding

- Status: finished. Session: `logs/hyperloom/Kimi-K3/20261118T084134Z-b0f51cf8`
- Stop reason: global_converged; target users: 1
- Baseline: 190.3 tok/s
- Validated gain: +0.00%
- Best setting (baseline): ``
- Report: `logs/hyperloom/Kimi-K3/20261118T084134Z-b0f51cf8/reports/final.md`

## Summary (Nov 18, 3:30 PM ET)

All 4 steps finished at 2:36 PM ET Nov 18 (about 28 h of Hyperloom in total).

### Fixed

- **Step 3 setting, confirmed with our own script** (`hyperloom/run_check_s3.sh`, node6101, 64 users,
  ISL/OSL 1024/1024, 640 prompts, image `nightly_202610011450`, TP8). Setting:
  `ATOM_PREFILL_DECODE_INTERVAL=4`, `ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000`, `--max-num-batched-tokens 8192`
  (in place of 16384), on top of `ATOM_GDN_SSM_DTYPE=fp16`.

  | | FP16 baseline | Step 3 setting | Change |
  |---|---|---|---|
  | Output tok/s (8 GPUs) | 1,680.7 | 1,852.8 | **+10.2%** |
  | TPOT (time per output token), median | 37.28 ms | 32.86 ms | 11.9% lower |
  | TTFT (time to first token), median | 207 ms | 823 ms | **4.0× higher** |
  | GSM8K strict (all 1,319) | 0.9613 | 0.9591 | −0.2 points (within noise) |

  The gain is real (Hyperloom reported +13.08%) and accuracy holds, but TTFT is 4× higher.
  Output: `logs/node6101/kimi-atom-mtp/atom/kimi_atom_mtp_check-s3-{A,B}_20261118_*`.
- **Restart and limits:** the chain restarted from its checkpoint after the pause (6:29 PM ET Nov 17),
  and step 5 resumed by itself after two Claude limit stops (5-hour and weekly).

### Not fixed

- Step 5 (DSpark speculative decoding at 1 user): +0%, stop reason `global_converged`.
- No kernel or framework gain: step 3 kept no kernel; step 4 kept no patch (it found the step 3 setting again).
- Users above 64: step 2 never tried `max-num-seqs` above 64.
- Expert parallel and a8w4 (`--moe-backend mega`): still blocked by the ATOM/aiter code.

### Is a long run worth it?

No. About 28 h gave one useful result, a setting change found early in step 3; the kernel and framework
agents found no gains. A longer run would mostly search the same setting space again and use up the weekly
Claude limit (hit on Nov 18). Better next steps:

1. Test `max-num-seqs` 128/256 with our own script (`MAXSEQS` knob in `atom/run_kimi_atom_mtp.sh`; no tokens).
2. If the 4× TTFT is not acceptable, test a smaller queue delay (for example 1,000 ms).
3. Use Hyperloom again only for one narrow target (for example one MoE kernel), with a budget of 4 h or less.

## Do we gain, and is Hyperloom worth continuing? (Nov 18)

**Gain: yes, from settings only.** Both changes were found by Hyperloom and confirmed with our own runs:

| Change | Output tok/s at 64 users | Cost |
|---|---|---|
| `ATOM_GDN_SSM_DTYPE=fp16` | about +3% | none (GSM8K same) |
| Prefill delay + `--max-num-batched-tokens 8192` (step 3) | +10.2% more (1,681 → 1,853) | TTFT (time to first token) 207 → 823 ms |

Together about +13% throughput over the plain ATOM image, and TPOT (time per output token) is 12% lower.
The step 3 change is a trade-off: it holds new requests back to batch them, so TTFT is higher (still under 1 s).
No gain at 1 user (DSpark +0%), and no kernel or framework patch was kept.

**Why the kernel and framework agents found nothing:** the chain design, not only the agents.

- Step 3 stopped after about 4 h of its 12 h, because the settings change reached the +10% target, so the
  kernel agent stopped with it.
- Step 4 began again from the FP16 baseline, not the step 3 setting, so it found the same setting again and
  stopped at its target too.

The main costs in the profile are still unaddressed: MoE expert GEMMs 37% and all-reduce 22% of the time.

**Worth continuing: one more short, targeted run, not long open runs.**

1. One run of 4–6 h: the step 3 setting as the new baseline (so it cannot be found again), kernel agent only
   (`KERNEL=1`) on the MoE GEMMs, and a higher target so it does not stop early on a small settings gain.
   Run it after the weekly Claude limit resets.
2. If that run keeps no kernel, stop using Hyperloom for Kimi-K3. The remaining large gains (expert parallel,
   a8w4) are blocked by ATOM/aiter code, which Hyperloom cannot fix; a newer ATOM image has given more before
   than any Hyperloom run.
3. Meanwhile, with no tokens: test `max-num-seqs` 128/256 and a smaller prefill delay (for example 1,000 ms,
   to bring TTFT back down) with our own script.
