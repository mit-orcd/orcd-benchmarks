#!/usr/bin/env bash
# Kimi-K3 deep-dive (amd-cloud Part D tier 3 and follow-ups), strictly sequential, one server at
# a time. Replaces amd-cloud's run_queue_*.sh.
#
#   atom/run_kimi_all.sh <set> [experiments]
#     set = cloud : exactly the images amd-cloud ran each experiment with (apple-to-apple)
#           new   : the newest images (2026-10-01), same experiments, to see what improved
#   ./launch.sh node6100 atom/run_kimi_all.sh cloud
#   ./launch.sh node6101 atom/run_kimi_all.sh new          # the two sets can run on the two nodes
#   ./launch.sh node6100 atom/run_kimi_all.sh cloud "maxseqs single_stream"
#
# Weights: always $KIMI_MODEL = /scratch/Kimi-K3 (node-local copy).
# Output: results/<node>/kimi-<set>/kimi-k3-*.md|csv, logs/<node>/kimi-<set>/atom/...
# Image per experiment (from amd-cloud's logs; :latest moved during the amd-cloud campaign):
#   experiment               cloud                        new
#   base (tier 3), maxseqs,  nightly_202608111555         nightly_202610011450
#     kimi_512/1024/2048,
#     profile
#   isl4096, repeats arm A   nightly_202608191459         nightly_202610011450
#   mad, single_stream,      MAD 20260727_kimi_k3         kimi_k3_agentic_0924
#     repeats arm B, ep_matched
set -uo pipefail
SET=${1:?usage: $0 <cloud|new> [experiments]}
case $SET in cloud|new) ;; *) echo "set must be cloud or new" >&2; exit 2 ;; esac
export RESULTS_SUBDIR=kimi-$SET
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$(dirname "$0")"
EXPS="${2:-base maxseqs mad kimi_512 kimi_1024 kimi_2048 isl4096 single_stream repeats ep_matched profile}"
mkdir -p "$LOG_ROOT/atom" "$RESULTS"
STATE=$LOG_ROOT/atom/kimi_all_$(date +%Y%m%d_%H%M%S).txt
say() { echo "[$(date -Iseconds)] $HOSTNAME_S [$SET] $*" | tee -a "$STATE"; }

if [[ $SET == cloud ]]; then
  LATEST=$ATOM_IMG_CLOUD; LATEST2=$ATOM_IMG_CLOUD2; MAD=$MAD_IMG_CLOUD
else
  LATEST=$ATOM_IMG_NEW; LATEST2=$ATOM_IMG_NEW; MAD=$MAD_IMG_NEW
fi
export MKIMI=$KIMI_MODEL MAD_IMG=$MAD IMG_B=$MAD
say "start: model=$KIMI_MODEL latest=$LATEST latest2=$LATEST2 mad=$MAD exps='$EXPS'"
for i in "$LATEST" "$LATEST2" "$MAD"; do
  docker image inspect "$i" >/dev/null 2>&1 || say "WARNING: no SIF for $i (experiments using it will fail)"
done
# Record exactly what ran.
{ echo "set: $SET"; echo "model: $KIMI_MODEL"; for i in "$LATEST" "$LATEST2" "$MAD"; do
    echo "$i -> $(docker image inspect --format '{{.Id}}' "$i" 2>/dev/null)"; done; } > "$RESULTS/images.txt"

for e in $EXPS; do
  case $e in
    base)            s=run_part_d.sh;      args=tier3; img=$LATEST ;;
    maxseqs|mad)     s=run_kimi_$e.sh;     args="";    img=$LATEST ;;
    isl4096)         s=run_isl4096.sh;     args="";    img=$LATEST2 ;;
    repeats)         s=run_repeats.sh;     args="";    img=$LATEST2 ;;
    *)               s=run_$e.sh;          args="";    img=$LATEST ;;
  esac
  [[ -x $s ]] || { say "SKIP $e: no $s"; continue; }
  say "START $e ($s $args, ATOM_IMG=$img)"
  ATOM_IMG=$img IMG_A=$img ./"$s" $args > "$LOG_ROOT/atom/kimi_all_$e.log" 2>&1; rc=$?
  say "DONE  $e rc=$rc"
  $PY "$BENCH_ROOT/report.py" >> "$STATE" 2>&1 || say "report.py failed after $e"
  if docker ps | grep -q .; then
    say "WARNING: container still up after $e, stopping"
    for n in $(docker ps); do docker rm -f "$n"; done
  fi
  sleep 30   # let VRAM drain
done
say "kimi_all DONE"
