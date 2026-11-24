#!/bin/bash
# Nov 18 reference for the MoE kernel run: ref_fp16.sh + the step 3 setting (confirmed +10.2% at 64 users,
# hyperloom/status-chain-20261117.md). Hyperloom reads the exports and the static server flags; it never runs this file.
export ATOM_GDN_SSM_DTYPE=fp16
export ATOM_PREFILL_DECODE_INTERVAL=4
export ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000
python3 -m atom.entrypoints.openai_server --model /scratch/Kimi-K3 --tensor-parallel-size 8 --kv_cache_dtype fp8 --max-num-seqs 64 --gpu-memory-utilization 0.93 --trust-remote-code --max-model-len 16384 --max-num-batched-tokens 8192 --block-size 128 --no-enable_prefix_caching --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'
