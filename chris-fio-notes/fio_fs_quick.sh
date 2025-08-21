#!/usr/bin/env bash
# Quick fio test on a filesystem mount point (non-root)
# Usage: ./fio_fs_quick.sh /scratch
# Env overrides: RUNTIME=30 SIZE=2G JOBS=8 DIRECT=1
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "Usage: $0 <mount_point>" >&2
  exit 1
fi

TARGET="$1"
RESULTS="fio_fs_results.txt"
RUNTIME="${RUNTIME:-30}"   # seconds per test
SIZE="${SIZE:-2G}"         # per-job file size
JOBS="${JOBS:-8}"          # parallel files/jobs
DIRECT="${DIRECT:-1}"      # 1 = O_DIRECT (bypass cache), 0 = cached
IOENGINE="io_uring"
fio -ioengine-help >/dev/null 2>&1 || { echo "fio not found in PATH"; exit 2; }
fio -ioengine-help 2>/dev/null | grep -q io_uring || IOENGINE="libaio"

echo "=== Filesystem Quick Test ($(date)) ===" > "$RESULTS"
echo "Target: $TARGET" >> "$RESULTS"
echo "ioengine: $IOENGINE | runtime=${RUNTIME}s | jobs=${JOBS} | size/job=$SIZE | direct=$DIRECT" >> "$RESULTS"
echo "" >> "$RESULTS"

run_test () {
  local name="$1" rw="$2" bs="$3" iodepth="$4"
  echo "--- $name ($rw, $bs, iodepth=$iodepth) ---" | tee -a "$RESULTS"
  # run fio and capture human-readable output
  tmp="$(mktemp)"
  set +e
  fio --name="$name" --rw="$rw" --bs="$bs" --iodepth="$iodepth" \
      --numjobs="$JOBS" --size="$SIZE" --directory="$TARGET" \
      --time_based=1 --runtime="$RUNTIME" --direct="$DIRECT" --ioengine="$IOENGINE" \
      --group_reporting --unlink=1 \
      >"$tmp" 2>&1
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "fio error (rc=$rc). See $tmp" | tee -a "$RESULTS"
    echo "" >> "$RESULTS"
    return
  fi
  # Grab the final summary line: 'read:' or 'write:'
  IFS=$'\n'  line=( $( grep -i -e ' read:' -e ' write:' "$tmp" ) )
  if [ -z "$line" ]; then
    echo "No summary line found. Raw output in $tmp" | tee -a "$RESULTS"
    echo "" >> "$RESULTS"
    return
  fi
  # Extract BW token and IOPS token (e.g., BW=1234MiB/s, IOPS=567890)
  for l in "${line[@]}"; do
    echo ${l}
  done
  # rm -f "$tmp"
}

run_test "SEQ-READ"   "read"      "1M"  32
run_test "SEQ-WRITE"  "write"     "1M"  32
run_test "RAND-READ"  "randread"  "4k" 128
run_test "RAND-WRITE" "randwrite" "4k" 128

echo "Results saved to $RESULTS"

