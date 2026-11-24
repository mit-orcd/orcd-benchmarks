#!/bin/bash
# Nov 17 reference for Hyperloom (plan-hyperloom.md §11 step 1): kimi_atom_reference.sh + FP16 SSM state.
# Hyperloom reads the export and the static server flags; it never runs this file.
export ATOM_GDN_SSM_DTYPE=fp16
python3 -m atom.entrypoints.openai_server --model /scratch/Kimi-K3 --tensor-parallel-size 8 --kv_cache_dtype fp8 --max-num-seqs 64 --gpu-memory-utilization 0.93 --trust-remote-code --max-model-len 16384 --max-num-batched-tokens 16384 --block-size 128 --no-enable_prefix_caching --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'
