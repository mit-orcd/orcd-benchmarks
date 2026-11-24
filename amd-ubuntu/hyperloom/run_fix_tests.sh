#!/bin/bash
# Nov 16 small tests to confirm the 3 Hyperloom problems are fixed (before any long run).
#   bash run_fix_tests.sh gpu      ON node6101 (no Claude tokens), in this order:
#     base    baseline, 64 users, 320 prompts, GSM8K first 250 questions (reference for a8w4+EP)
#     geak    baseline + --cudagraph-mode FULL, range ratio 1.0, 64 users, 320 prompts (GEAK's client
#             settings; Hyperloom used range ratio 0.8 -> explains 2,049.55 vs 1,661.1?)
#     a8w4ep  --moe-backend mega --enable-expert-parallel, 64 users, 320 prompts, GSM8K 250
#             (plain --enable-expert-parallel cannot run: aiter's a16w4 SiTUv2 MoE kernel raises
#             "not supported: expert-parallel masking"; mega requires expert parallel)
#     prof256 rocprofv3 profile of the baseline at 256 users, STOP_T=300 so rocprofv3 can write its
#             files (the Nov 12 run was killed after 30 s and left no kernel_stats files)
#   A run whose server does not start or has no c64.json is retried once, then skipped.
#   bash run_fix_tests.sh ssm      ON node6101: base vs ATOM_GDN_SSM_DTYPE=fp16 (see the ssm block)
#   bash run_fix_tests.sh sandbox  ON node6100 (a few tokens): the T2 helper file-access check again.
# Progress: logs/hyperloom/fixtests-<host>.state ; run list: logs/hyperloom/fixtests-runs.txt
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks; SW=$B/amd-software; H=$B/amd-ubuntu; L=$H/logs/hyperloom
ST=$L/fixtests-$(hostname -s).state; mkdir -p "$L" /dev/shm/shaohao-tmp
say() { echo "[$(date -Iseconds)] $*" >>"$ST"; }
TMPE="TMPDIR=/dev/shm/shaohao-tmp"
MODE=${1:?usage: run_fix_tests.sh gpu|sandbox}

if [[ $MODE == sandbox ]]; then
  SIF=$SW/sif/atom-dev-nightly_202610011450.sif; OV=$SW/cache/overlay/hyperloom-atom.img
  ENVF=$H/private/hyperloom/.env; LTMP=/dev/shm/shaohao-tmp
  D=$L/rerun-tests/t2-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$D" "$LTMP/t2cwd"
  LN=$(apptainer exec --cleanenv "$SIF" grep -n "moe_backend: str" /app/ATOM/atom/config.py | head -1 | cut -d: -f1)
  say "sandbox check start (expected line $LN)"
  apptainer exec --cleanenv --overlay "$OV:ro" -B /orcd/data/orcd/022 -B /scratch/Kimi-K3 --pwd "$LTMP/t2cwd" \
    --env ENVF="$ENVF" "$SIF" bash -c '
      set -a; . "$ENVF"; set +a
      export PATH="$HOME/.local/bin:$PATH"
      echo "which=$(python3 -c "import shutil;print(shutil.which(\"claude\") or \"MISSING\")")"
      timeout 600 claude --print --output-format text --permission-mode bypassPermissions \
        --add-dir /app/ATOM/atom/ --add-dir /app/aiter-test/aiter/ --add-dir /opt/rocm/ \
        --model "${CLAUDE_MODEL:-claude-opus-5-5}" \
        "Use the Read tool to read /scratch/Kimi-K3/config.json and /app/ATOM/atom/config.py. Reply with exactly two lines and nothing else: line 1 = the value of model_type in config.json; line 2 = the line number in config.py of the line that starts with \"moe_backend: str\" (after leading spaces). If a read is refused, write REFUSED and the error text." 2>&1 | tail -20
    ' >"$D/out.txt" 2>&1
  which=$(grep -m1 '^which=' "$D/out.txt" | cut -d= -f2)
  mt=$(grep -c kimi_k3 "$D/out.txt"); ln=$(grep -cx "[[:space:]]*$LN[[:space:]]*" "$D/out.txt")
  if [[ -n $which && $which != MISSING && $mt -ge 1 && $ln -ge 1 ]] && ! grep -q REFUSED "$D/out.txt"; then
    say "sandbox PASS: claude=$which model_type found, config.py line $LN found (dir $D)"
  else
    say "sandbox FAIL: claude=$which model_type_found=$mt line_found=$ln reply: $(grep -v '^which=' "$D/out.txt" | tr '\n' ' ' | cut -c1-300)"
  fi
  say "sandbox DONE"; exit 0
fi

RUNS=$L/fixtests-runs.txt
newest_out() { ls -1dt "$H"/logs/$(hostname -s)/hyperloom-fix/atom/kimi_atom_mtp_"$1"_* 2>/dev/null | head -1; }
run_one() {  # name, then env assignments for run_kimi_atom_mtp.sh
  local name=$1 try O; shift
  for try in 1 2; do
    say "$name: start (try $try)"
    env RESULTS_SUBDIR=hyperloom-fix NO_REPORT=1 OUTTAG="$name" EXTRA_ENVS="$TMPE" "$@" \
      bash "$H/atom/run_kimi_atom_mtp.sh" >/dev/null 2>&1
    O=$(newest_out "$name")
    if [[ -n $O && ! -f $O/k0/FAILED ]] && ls "$O"/k0/c*.json >/dev/null 2>&1; then
      say "$name: done dir=$O $(grep -E 'tok/s|GSM8K rc' "$O/STATE.txt" | sed 's/^\[[^]]*\] //' | tr '\n' ';')"
      echo "$name $O" >>"$RUNS"; return 0
    fi
    say "$name: FAILED (try $try) dir=$O: $(grep -m1 -E '^[A-Za-z]*Error' "$O/k0/atom_server.log" 2>/dev/null | cut -c1-250)"
  done
  echo "$name FAILED ${O:-none}" >>"$RUNS"; return 1
}

if [[ $MODE == ssm ]]; then
  # Nov 16 step 1 (plan-hyperloom.md §10): Hyperloom T3's +2.56% setting ATOM_GDN_SSM_DTYPE=fp16
  # (SSM state fp16 instead of fp32). Interleaved base/ssm16, 2 repeats, 64 users, 320 prompts;
  # full GSM8K (1,319 questions) in repeat 1 of each.
  RUNS=$L/fixtests-ssm-runs.txt; : >"$RUNS"
  say "ssm tests start"
  for r in 1 2; do
    g=0; (( r == 1 )) && g=1
    run_one "ssmbase_r$r" ARMS="0:64" NUM_PROMPTS=320 GSM8K=$g
    run_one "ssm16_r$r"   ARMS="0:64" NUM_PROMPTS=320 GSM8K=$g EXTRA_ENVS="$TMPE ATOM_GDN_SSM_DTYPE=fp16"
  done
  say "ssm tests DONE"; exit 0
fi

: >"$RUNS"
say "gpu tests start"
run_one base    ARMS="0:64" NUM_PROMPTS=320 GSM8K=1 GSM8K_LIMIT=250
run_one geak    ARMS="0:64" NUM_PROMPTS=320 RANGE_RATIO=1.0 EXTRA_ARGS="--cudagraph-mode FULL"
run_one a8w4ep  ARMS="0:64" NUM_PROMPTS=320 GSM8K=1 GSM8K_LIMIT=250 EXTRA_ARGS="--moe-backend mega --enable-expert-parallel"
run_one prof256 ARMS="0:256" NUM_PROMPTS=1536 STOP_T=300 \
  PREFIX="rocprofv3 --kernel-trace --stats --summary --summary-output-file /out/rocprof_summary.txt -f csv -d /out/rocprof -P 720:5:1 --"
O=$(newest_out prof256); n=$(find "$O/k0/rocprof" -name '*kernel_stats.csv' 2>/dev/null | wc -l)
say "prof256: kernel_stats files=$n"
find "$O/k0/rocprof" -name '*kernel_trace.csv' 2>/dev/null | sort | tail -n +2 | xargs -r rm -f
say "gpu tests DONE"
