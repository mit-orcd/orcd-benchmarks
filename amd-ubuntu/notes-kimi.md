# Kimi notes

## Kimi-K3.1-Pro vs Kimi-K3 (2026-11-06)

Source: https://huggingface.co/moonshotai/Kimi-K3.1-Pro (model card) and the `config.json` of both models.
Not read: the K3.1 tech blog and the Kimi-K3 card. The K3.1 card lists no release date.

**Mostly the same model, with one change that matters for performance.** K3.1 Pro is K3 trained further, with the same overall structure.

**Same:**
- Total size: the K3.1 card says 2.8T parameters and 104B active. The config fields behind these match K3.
- Layers and experts: 93 layers, 896 experts with 16 selected per token, hidden size 7168.
- Quantization: MXFP4 weights in both configs. K3.1's config ignores two more MoE parts (`block_sparse_moe.gate`, `routed_expert_up/down_proj`), so those are not MXFP4.
- Limits: 1M-token context and no MTP head in the config (`num_nextn_predict_layers` = 0).

**Different:**
- **Attention:** the 24 full-attention Gated MLA layers become Gated DeepSeek Sparse Attention (DSA), with index top-k 2048 (`dsa_config`). This is the big change. It alters those layers' compute, especially for long inputs, so our tokens/s and memory results won't carry over.
- **Weights:** new, so it's a separate download. Our K3 copy (`/scratch/Kimi-K3`) is 1.5T on disk; the card doesn't state K3.1's size.
- **Model type:** `kimi_k31` instead of `kimi_k3`. Our current ATOM and vLLM images may not recognize it, so they may need newer versions (not tested).

**For our benchmarks:** treat it as a new model and re-run it, not as a drop-in swap. Our K3 results and any MTP draft model don't apply directly.
