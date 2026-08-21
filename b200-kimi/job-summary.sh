#!/usr/bin/env bash
# Final stage: aggregate everything the chain produced into one markdown summary.
#
# Runs with `--dependency=afterany` on the main run, so it reports whatever happened --
# including a failure. A summary that only appears on success is useless to someone
# reading the directory the next morning.
#
# The headline analysis (results/kimi-k3-base-b200.md) is written by job-kimi-base.sh
# itself; this adds the run-level narrative around it.
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -J kimi-summary
#SBATCH -o out/kimi-summary.%J.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source common/env.sh

OUTMD="$RESULTS/RUN-SUMMARY.md"
mkdir -p "$RESULTS"

state_of() {  # state_of <jobid>
  sacct -n -X -j "$1" -o State%20 2>/dev/null | head -1 | tr -d ' '
}
elapsed_of() {
  sacct -n -X -j "$1" -o Elapsed 2>/dev/null | head -1 | tr -d ' '
}

{
echo "# Kimi-K3 on B200 — run summary"
echo
echo "Generated $(date -Iseconds) by job \`${SLURM_JOB_ID:-?}\`."
echo
echo "Chain: \`${CHAIN_IDS:-unknown}\`"
echo
echo "## Stage outcomes"
echo
echo "| Stage | Job | State | Elapsed |"
echo "|---|---:|---|---:|"
for entry in ${CHAIN_IDS:-}; do
  IFS=: read -r name jid <<<"$entry"
  echo "| $name | $jid | $(state_of "$jid") | $(elapsed_of "$jid") |"
done
echo

echo "## Hardware and staging (from the probe)"
echo
echo '```'
grep -h -E "^=====|DRIVER (OK|TOO OLD)|MODEL (VISIBLE|NOT VISIBLE)|Driver Version|shards," \
  "$OUT_DIR"/drv-*.out 2>/dev/null | head -40
echo '```'
echo

echo "## Pre-flight verification (2 nodes, no weight load)"
echo
lastv=$(ls -dt "$LOG_ROOT"/verify_* 2>/dev/null | head -1)
if [[ -n "$lastv" && -f "$lastv/STATE.txt" ]]; then
  echo "Run dir: \`$lastv\`"
  echo
  echo '```'
  grep -E "check [0-9]|OK:|FAIL:|resolve_ipv4|ray cluster GPUs|VERIFY" "$lastv/STATE.txt" 2>/dev/null | head -20
  echo '```'
else
  echo "_No verification run found._"
fi
echo

echo "## Known failure modes already fixed"
echo
echo "| Symptom | Root cause | Fix |"
echo "|---|---|---|"
echo "| \`FATAL: \"python\": executable file not found\` | image has only \`python3\` | \`\$PY_C=python3\` everywhere |"
echo "| \`ModuleNotFoundError: ray\` | ray is not in the vLLM image | ray 2.57.0 installed \`--no-deps\` into \`pylibs/\` |"
echo "| \`transformers\` pulling in a broken TensorFlow | host \`~/.local\` site-packages shadowed the container's | \`PYTHONNOUSERSITE=1\` + explicit \`PYTHONPATH\` |"
echo "| \`bits/libc-header-start.h: No such file\` during model inspection | host Spack gcc leaked via \`PATH\`; Triton JIT used it against container headers | \`PATH\`/\`CC\`/\`CXX\` pinned to the container |"
echo "| ray GCS unreachable at \`[fe80::...]:6379\` | \`getent hosts\` returned an IPv6 link-local on the compute node | \`resolve_ipv4()\` forces a routable IPv4 |"
echo "| 2-node job launched behind a failed gate | a CANCELLED job satisfies \`afterany\` | \`base\` now needs \`afterok:verify\` too |"
echo

echo "## Single-node attempt (TP8 × PP1)"
echo
lastrun=$(ls -dt "$LOG_ROOT"/kimi_1node_* 2>/dev/null | head -1)
if [[ -n "$lastrun" && -f "$lastrun/STATE.txt" ]]; then
  echo "Run dir: \`$lastrun\`"
  echo
  echo '```'
  grep -E "VERDICT|parallelism OK|per-GPU HBM|model OK|ABORT|out of memory|OutOfMemory" \
    "$lastrun/STATE.txt" 2>/dev/null | head -25
  echo '```'
  echo
  # Pull the OOM numbers straight out of torch's own message. This is the measurement,
  # so it belongs in the report verbatim rather than as a paraphrase.
  oom=$(grep -oE "CUDA out of memory\. Tried to allocate [^)]*free\." \
        "$lastrun"/server/vllm_server.log 2>/dev/null | head -1)
  if [[ -n "$oom" ]]; then
    ngpu_oom=$(grep -c "torch.OutOfMemoryError" "$lastrun"/server/vllm_server.log 2>/dev/null)
    echo "**Result: the checkpoint does not fit on one B200 node.** This is the finding the"
    echo "single-node stage exists to establish, not a failure to fix."
    echo
    echo "\`torch.OutOfMemoryError\` was raised on **$ngpu_oom of 8 GPUs**, at model load:"
    echo
    echo '```'
    echo "$oom"
    echo '```'
    echo
    cap=$(grep -oE "total capacity of [0-9.]+ GiB" "$lastrun"/server/vllm_server.log 2>/dev/null | head -1 | grep -oE "[0-9.]+")
    used=$(grep -oE "this process has [0-9.]+ GiB memory in use" "$lastrun"/server/vllm_server.log 2>/dev/null | head -1 | grep -oE "[0-9.]+")
    if [[ -n "$cap" && -n "$used" ]]; then
      echo "| | Per GPU | × 8 (node) |"
      echo "|---|---:|---:|"
      echo "| Usable HBM | $cap GiB | $(awk -v c="$cap" 'BEGIN{printf "%.1f", c*8}') GiB |"
      echo "| Occupied before the failing allocation | $used GiB | $(awk -v u="$used" 'BEGIN{printf "%.1f", u*8}') GiB |"
      echo "| Kimi-K3 weights | — | 1454.2 GiB (1561 GB) |"
      echo
      echo "Every GPU was filled to within ~0.1 GiB of capacity and the model still did not"
      echo "fit, which is why the benchmark runs TP8 × PP2 across two nodes."
    fi
  fi
else
  echo "_No single-node run directory found._"
fi
echo

echo "## Two-node run (TP8 × PP2)"
echo
lastbase=$(ls -dt "$LOG_ROOT"/kimi_base_* 2>/dev/null | head -1)
if [[ -n "$lastbase" && -f "$lastbase/STATE.txt" ]]; then
  echo "Run dir: \`$lastbase\`"
  echo
  echo '```'
  grep -E "parallelism OK|model OK|image OK|server up|sweep rc|analyze rc|DONE|ABORT|VERDICT" \
    "$lastbase/STATE.txt" 2>/dev/null | head -30
  echo '```'
  if [[ -f "$lastbase/sweep/summary.txt" ]]; then
    echo
    echo "### Sweep"
    echo
    echo '```'
    cat "$lastbase/sweep/summary.txt"
    echo '```'
  fi
else
  echo "_No two-node run directory found._"
fi
echo

echo "## Reports"
echo
for f in "$RESULTS"/kimi-k3-base-b200*.md "$RESULTS"/kimi-k3-base-b200*.csv; do
  [[ -e "$f" ]] || continue
  echo "- \`$f\` ($(wc -l <"$f") lines, $(date -r "$f" -Iseconds))"
done
[[ -e "$RESULTS/kimi-k3-base-b200.md" ]] || {
  echo
  echo "> **The headline report was not produced.** Check the two-node run's"
  echo "> \`STATE.txt\` and \`analyze.log\` above for why."
}
echo
echo "## Comparison baseline"
echo
echo "MI355X: \`$AMD_ROOT/results/kimi-k3-base.md\`, measured 2026-08-14 (ATOM, TP8, 1 node)."
} >"$OUTMD"

echo "wrote $OUTMD"
# head FIRST, sed second: head reads only 60 lines and exits cleanly, so sed sees a
# natural EOF. The reverse order (sed | head) had sed try to WRITE all lines of
# $OUTMD while head only reads 60 and closes early -- sed then hit SIGPIPE on the
# 61st write and the whole job (20912378) was marked FAILED by Slurm even though
# $OUTMD itself had already been written out completely and correctly above.
head -60 "$OUTMD" | sed 's/^/  /'
