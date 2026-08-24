# Can we apply MTP in our runs? — CORRECTED

**Earlier note in this file was wrong.** It claimed DSpark "does not compose with
pipeline parallelism" on B200, citing the upstream vLLM recipe's `strategies` list
gating DSpark off `multi_node_tp_pp`. That is a **recipe-level default, not a hard
engine limitation** — SemiAnalysis's own B200 recipe proves it.

## What they actually do

Found in `agg-b200-tp8pp2-mooncake-c1-agentic.yaml` (and siblings c2/c4/…/c96),
fetched from `github.com/SemiAnalysisAI/InferenceX`:

```yaml
tensor-parallel-size: 8
pipeline-parallel-size: 2
decode-context-parallel-size: 8
dcp-comm-backend: a2a
speculative-config: '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":7,
                      "method":"dspark","attention_backend":"TOKENSPEED_MLA",
                      "draft_sample_method":"probabilistic"}'
```

**MTP + PP2 together, on B200, 16 GPUs.** This is what produced the `spec_method: mtp`
B200 rows in their public API data (§7.4 of `results/kimi-k3-base-b200.md`).

## Three things they do that we don't

1. **A different speculator** — `Inferact/Kimi-K3-DSpark`, not the
   `RedHatAI/Kimi-K3-speculator.dspark` the upstream vLLM recipe names.
   `num_speculative_tokens: 7`.
2. **A compat shim** (`configs/kimik3-dspark-config-compat.sh`). The Inferact DSpark
   checkpoint publishes its parallel-drafting token as `mask_token_id`; vLLM's parallel
   drafter expects `pard_token`. Their script downloads the checkpoint, builds a
   symlinked local copy, and injects `pard_token = mask_token_id` into `config.json`
   without touching the weights. **Without this shim the speculative-config simply does
   not load** — this, not PP, looks like the real reason the plain upstream recipe
   avoids the combination.
3. **`decode-context-parallel-size: 8`** with `dcp-comm-backend: a2a` and the
   `TOKENSPEED_MLA` attention backend, plus Mooncake KV offload (`offload=on` on all 6
   of their B200 MTP API records) — none of which are in our current server flags.

## Revised answer

**We can, in principle.** It is a real configuration gap, not a hardware or engine
wall:

- download `Inferact/Kimi-K3-DSpark` (blocked today by `HF_HUB_OFFLINE=1`)
- apply the `pard_token` compat shim to a local copy of it
- add `--decode-context-parallel-size 8 --dcp-comm-backend a2a
  --attention-backend TOKENSPEED_MLA` and the `speculative-config` JSON to
  `job-kimi-base.sh`'s server args

**Not done here** because it was not attempted, not because it is blocked. This
reverses the "structurally unavailable" framing that appears in §7.3/§7.4 of
`results/kimi-k3-base-b200.md` and in `notes-concurrency.md` — those still need
correcting to reflect this.

Source: `semianalysis-ref/` does not yet contain the mooncake yaml or the compat shim;
fetched directly from GitHub during this investigation, not yet saved locally.
