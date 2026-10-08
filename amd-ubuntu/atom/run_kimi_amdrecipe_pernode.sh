#!/usr/bin/env bash
# run_kimi_amdrecipe.sh with a per-node persistent container overlay.
#
# 2026-10-07 fix: the isl1k arm on node6101 failed with "No module named
# aiter.jit.module_fmha_fwd_bf16_opus" / "[Errno 28] No space left on device". Both nodes default to
# the same overlay (vllm-openai-rocm-v0.29.0.img, one container at a time); node6100's server held
# it, so node6101's fell back to the 64 MB --writable-tmpfs and AITER's JIT build ran out of space
# (same failure as the Primus llama2-7B run). A per-node overlay fixes it. This wrapper is used
# instead of editing run_kimi_amdrecipe.sh, which was still running on node6100.
#   ./launch.sh node6101 atom/run_kimi_amdrecipe_pernode.sh isl1k
export CTR_OVERLAY="vllm-v0.29.0-$(hostname -s)"
exec bash /orcd/data/orcd/022/benchmarks/amd-ubuntu/atom/run_kimi_amdrecipe.sh "$@"
