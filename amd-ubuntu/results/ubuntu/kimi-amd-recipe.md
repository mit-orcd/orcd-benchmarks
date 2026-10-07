# amd-ubuntu — Kimi-K3 with AMD's recipe PDF (vLLM v0.29.0)

Recipe: `amd-kimi-k3-recipe.pdf` — `vllm/vllm-openai-rocm:v0.29.0`, TP8, one server for the sweep, `max-num-seqs 128`, `max-num-batched-tokens 4096`, `gpu-memory-utilization 0.95`, cudagraph `FULL_DECODE_ONLY`, `+fused_rms_norm_gated`, AITER MXFP4 MoE (`VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1`), no speculative decoding. Scripts: `atom/run_kimi_amdrecipe.sh` (`isl128k` on node6100, `isl1k` on node6101), weights from `/scratch/Kimi-K3`, run under apptainer.

## 1. AMD's workload (ISL/OSL 128K/1K): ours vs the PDF

*Pending: the `isl128k` arm has not produced results yet.*

## 2. Our workload (ISL/OSL 1K/1K): AMD PDF recipe vs current recipes

*Pending: the `isl1k` arm has not produced results yet.*

## Is this apple-to-apple?

- **§1 (vs the PDF): nearly.** Same image, server flags and client settings on the same GPU type. Differences: this machine, apptainer instead of docker, weights from local disk, and one short warm-up before the sweep.
- **§2 (vs current recipes): no, recipe vs recipe.** Same hardware, model, ISL/OSL, concurrency and prompt count, but each recipe has its own vLLM/ATOM version and server settings (the current vLLM recipe uses speculative decoding, the PDF recipe does not; ATOM is a different engine).

Per-arm detail: `results/<node>/kimi-amd-recipe/{isl128k,isl1k}.md`; logs: `logs/<node>/kimi-amd-recipe/atom/kimi_amdrecipe_<arm>_<ts>/`.
