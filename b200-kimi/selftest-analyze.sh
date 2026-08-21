#!/usr/bin/env bash
# Exercise analyze-kimi-b200.py end to end on SYNTHETIC B200 data plus the REAL MI355X
# baseline, so the report generator is known-good before a 2-node allocation is spent.
#
# Usage: ./selftest-analyze.sh            (writes to a temp dir, prints the report path)
#
# Generates a plausible-but-fake sweep and a vLLM server log in the exact formats the
# analyzer parses. It proves the plumbing (parse -> derive -> render), NOT the physics:
# the numbers in the output are meaningless. What it catches is the class of bug that
# otherwise surfaces at the end of a 4-hour job -- a regex that never matches, a divide
# by zero at c=1, a missing key.
set -uo pipefail
cd "$(dirname "$0")"
source common/env.sh

T=$(mktemp -d)
trap 'echo "test tree: $T"' EXIT
SW=$T/sweep; mkdir -p "$SW"

CFG="${1:-$MKIMI/config.json}"
if [[ ! -f "$CFG" ]]; then
  echo "fetching config.json (model not downloaded yet)"
  CFG=$T/config.json
  curl -sf "https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json" -o "$CFG" \
    || { echo "no config.json available; pass one as \$1" >&2; exit 1; }
fi

python3 - "$SW" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
# Fake but monotone numbers shaped like the MI355X curve, so the derived columns land in
# a believable range and any sign/scale error in the analyzer is visible.
for c, ots, ttft, tpot in [(1, 52.0, 210.0, 19.2), (2, 99.0, 232.0, 20.1),
                           (4, 176.0, 240.0, 22.4), (8, 330.0, 244.0, 24.3),
                           (16, 572.0, 250.0, 28.0), (32, 940.0, 262.0, 34.1),
                           (64, 1440.0, 275.0, 44.8)]:
    json.dump({
        "date": "20260821-101500", "backend": "openai", "model_id": "Kimi-K3",
        "num_prompts": c * 10, "max_concurrency": c, "completed": c * 10,
        "duration": c * 10 * 1024 / ots,
        "request_throughput": ots / 1024.0, "output_throughput": ots,
        "total_token_throughput": ots * 2,
        "median_ttft_ms": ttft, "p99_ttft_ms": ttft * 3.5,
        "median_tpot_ms": tpot, "p99_tpot_ms": tpot * 1.1,
    }, open(out / f"c{c}.json", "w"))
print(f"wrote 7 synthetic sweep points to {out}")
PY

# A vLLM server log in the real format, to exercise every regex in parse_vllm_log().
cat > "$T/vllm_server.log" <<'LOG'
INFO ... Initializing a V1 LLM engine with config: model='/models/Kimi-K3', tensor_parallel_size=8, pipeline_parallel_size=2, data_parallel_size=1, max_model_len=16384, max_num_seqs=64, max_num_batched_tokens=8192, kv_cache_dtype='fp8', enable_prefix_caching=False, enable_expert_parallel=False,
INFO ... Model loading took 91.2340 GiB and 1284.551000 seconds
INFO ... Memory profiling takes 41.02 seconds
INFO ... the current vLLM instance can use total_gpu_memory (183.00GiB) x gpu_memory_utilization (0.90) = 164.70GiB
INFO ... model weights take 91.23GiB; non_torch_memory takes 2.41GiB; PyTorch activation peak memory takes 3.06GiB; the rest of the memory reserved for KV Cache is 68.00GiB.
INFO ... GPU KV cache size: 5,277,286 tokens
INFO ... Maximum concurrency for 16,384 tokens per request: 322.10x
INFO ... # GPU blocks: 41229, # CPU blocks: 0
LOG

echo "1284" > "$T/load_seconds.txt"

python3 analyze-kimi-b200.py \
  --sweep "$SW" \
  --server-log "$T/vllm_server.log" \
  --model-config "$CFG" \
  --run-dir "$T" \
  -o "$T/results" || { echo "ANALYZER FAILED" >&2; exit 1; }

echo
echo "=== checks ==="
MD="$T/results/kimi-k3-base-b200.md"
fail=0
check() { if grep -q "$1" "$MD"; then echo "  ok   : $2"; else echo "  FAIL : $2"; fail=1; fi; }
check "^# Kimi-K3 on 2 × 8 × B200"        "title"
check "^## 6. B200 vs MI355X"             "comparison section present"
check "MI355X tok/s"                      "per-point comparison table"
check "91.2"                              "memory line parsed from vLLM log"
check "5,277,286"                         "KV token count parsed"
check "2,722,740,830,208"                 "exact MXFP4 param validation"
if grep -qE '\| (—|nan|None) \|.*\| (—|nan|None) \|' "$MD"; then
  echo "  WARN : some table cells are empty"
fi
grep -c "nan" "$MD" | awk '{ if ($1>0) print "  FAIL : "$1" nan(s) in report"; else print "  ok   : no nan" }'
echo
wc -l "$MD" "$T/results/kimi-k3-base-b200.csv"
echo
echo "report: $MD"
[[ $fail -eq 0 ]] || exit 1
