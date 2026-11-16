#!/bin/bash
# Reference launch recipe for Hyperloom step 4 session A (--reference-script). Hyperloom only reads
# the static server flags (it never runs this). Same flags as our step 3 ATOM baseline without MTP
# (atom/run_kimi_atom_mtp.sh, K=0): TP8, FP8 KV cache, max-num-seqs 64, gpu-mem-util 0.93,
# max-model-len 16384, batched 16384, block 128, no prefix caching, PTPC-FP8 online quant.
# The JSON has no spaces: Hyperloom re-joins the flags with spaces and drops the quotes.
python3 -m atom.entrypoints.openai_server --model /scratch/Kimi-K3 --tensor-parallel-size 8 --kv_cache_dtype fp8 --max-num-seqs 64 --gpu-memory-utilization 0.93 --trust-remote-code --max-model-len 16384 --max-num-batched-tokens 16384 --block-size 128 --no-enable_prefix_caching --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'
