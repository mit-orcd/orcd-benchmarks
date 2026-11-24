#!/bin/bash
# Nov 17 reference for step 5 (low load, 1-8 users): ref_fp16.sh + ATOM DSpark speculative decoding,
# K=6 as in the InferenceX ATOM recipe at 1 user (atom/run_kimi_atom_mtp.sh). Hyperloom never runs this file.
export ATOM_GDN_SSM_DTYPE=fp16
python3 -m atom.entrypoints.openai_server --model /scratch/Kimi-K3 --tensor-parallel-size 8 --kv_cache_dtype fp8 --max-num-seqs 64 --gpu-memory-utilization 0.93 --trust-remote-code --max-model-len 16384 --max-num-batched-tokens 16384 --block-size 128 --no-enable_prefix_caching --method dspark --draft-model /orcd/data/orcd/022/benchmarks/amd-software/models/Kimi-K3-DSpark --num-speculative-tokens 6 --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'
