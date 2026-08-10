#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH -t 30
#SBATCH -J eng-gdr
#SBATCH -o eng-gdr-%J.out

# Engaging B200 counter-test for the AICR inter-node GDRDMA defect.
# Self-contained: no AICR paths, no hardcoded partition, no hardcoded rail names.
#
# Submit with the partition your B200 nodes live in, e.g.:
#     sbatch -p <b200-partition> run-engaging-check.sh
# Optional overrides:
#     NIC_FORCE=mlx5_4  sbatch -p ... run-engaging-check.sh     # pin a rail
#     sbatch -p ... -w node5500,node5501 run-engaging-check.sh  # pin nodes
#
# Produces eng-gdr-<jobid>.out. Send that file back; that is all that is needed.

set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1
SIZE=${SIZE:-8388608}          # 8 MiB
ITERS=${ITERS:-2000}

echo "################ ENGAGING GDRDMA COUNTER-TEST ################"
echo "date    : $(date)"
echo "job     : ${SLURM_JOB_ID:-none}"
echo "nodes   : ${SLURM_JOB_NODELIST:-none}"
echo

# ---------- locate a CUDA toolkit (for nvcc + libcudart for perftest --use_cuda) ----------
if ! command -v nvcc >/dev/null 2>&1; then
  for m in cuda nvhpc cuda/12.4 cuda/12.6 nvhpc/24.5; do
    module load "$m" >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1 && { echo "loaded module: $m"; break; }
  done
fi
if command -v nvcc >/dev/null 2>&1; then
  CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
  export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
  echo "nvcc    : $(command -v nvcc)"
else
  CUDA_ROOT=""
  echo "nvcc    : NOT FOUND -- part A will be skipped; parts B/C still work"
fi
echo "perftest: $(ib_write_bw --version 2>&1 | head -1)"
echo

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SERVER=${nodes[0]}
CLIENT=${nodes[1]:-${nodes[0]}}

# ---------- resolve the PIX rail for the allocated GPU, restricted to 400 Gb/s ----------
resolve_nic() {
  srun --overlap -N1 -n1 --nodelist="$1" bash -c '
    topo=$(nvidia-smi topo -m 2>/dev/null)
    declare -A m
    while read -r k v; do m[$k]=$v; done < <(echo "$topo" | sed -n "s/^ *\(NIC[0-9]*\): \(mlx5_[0-9]*\)$/\1 \2/p")
    row=$(echo "$topo" | awk "/^GPU0/{print; exit}")
    best=""; idx=0
    for f in $(echo "$row" | cut -f2-); do
      if [ "$f" = "PIX" ]; then
        cand=${m[NIC$((idx-1))]:-}
        if [ -n "$cand" ]; then
          r=$(ibstat "$cand" 2>/dev/null | sed -n "s/.*Rate: *\([0-9]*\).*/\1/p" | head -1)
          [ "$r" = "400" ] && best=$cand
        fi
      fi
      idx=$((idx+1))
    done
    [ -z "$best" ] && for c in $(ibstat -l 2>/dev/null); do
        r=$(ibstat "$c" 2>/dev/null | sed -n "s/.*Rate: *\([0-9]*\).*/\1/p" | head -1)
        [ "$r" = "400" ] && { best=$c; break; }
      done
    echo "$best"
  ' 2>/dev/null | tail -1
}

if [ -n "${NIC_FORCE:-}" ]; then
  NIC="$NIC_FORCE"; echo "rail    : $NIC (forced)"
else
  NIC=$(resolve_nic "$SERVER")
  echo "rail    : $NIC (auto: PIX to GPU, 400 Gb/s)"
fi
[ -z "$NIC" ] && { echo "FATAL: no 400 Gb/s rail found"; exit 1; }
# Server and client MUST use the same rail: different rails can be different IB subnets,
# which fails with 'Failed status 12' (transport retry exceeded).
echo

# =====================================================================
echo "=================== PART A: PCIe full duplex (no IB) ==================="
echo "AICR reference: H2D 57.6 / D2H 57.3 / concurrent 49.1 each way = 98.3 GB/s total"
echo
if [ -n "$CUDA_ROOT" ]; then
  nvcc -O2 -o /tmp/pcie_duplex.$$ pcie_duplex.cu 2>&1 | grep -vE "^ *int genCur|\^|Remark:|^$" | head -5
  [ -x /tmp/pcie_duplex.$$ ] && /tmp/pcie_duplex.$$ 256 16
  rm -f /tmp/pcie_duplex.$$
else
  echo "skipped (no nvcc)"
fi

# =====================================================================
echo
echo "=================== PART B: node configuration ==================="
srun --overlap -N1 -n1 --nodelist="$SERVER" bash -c '
echo "--- host: $(hostname)"
echo "--- B1. NVIDIA driver relaxed-ordering params (THE key comparison) ---"
grep -iE "relax|order" /proc/driver/nvidia/params 2>/dev/null || echo "  params not readable"
echo "--- B2. nvidia-smi relaxed ordering / BAR1 ---"
nvidia-smi -q 2>/dev/null | grep -iE "relaxed|bar1" | head -6
echo "--- B3. kernel cmdline ---"
cat /proc/cmdline
echo "--- B4. IOMMU ---"
echo "iommu dirs: $(ls /sys/class/iommu 2>/dev/null | tr "\n" " ")"
echo "--- B5. peermem ---"
lsmod | grep -iE "peermem|nvidia_p2p" || echo "  nvidia_peermem NOT loaded"
echo "--- B6. GPU/NIC PCIe link state ---"
printf "%-14s %-6s %-6s %-8s %-8s %s\n" BDF cur_w max_w cur_spd max_spd device
for d in /sys/bus/pci/devices/*; do
  b=$(basename "$d"); cls=$(cat "$d/class" 2>/dev/null)
  case "$cls" in 0x0302*|0x0300*|0x0207*|0x0c06*) ;; *) continue ;; esac
  printf "%-14s %-6s %-6s %-8s %-8s %s\n" "$b" \
    "$(cat $d/current_link_width 2>/dev/null)" "$(cat $d/max_link_width 2>/dev/null)" \
    "$(cat $d/current_link_speed 2>/dev/null | cut -d" " -f1)" \
    "$(cat $d/max_link_speed 2>/dev/null | cut -d" " -f1)" \
    "$(lspci -s ${b#0000:} 2>/dev/null | cut -d" " -f2- | cut -c1-42)"
done
echo "--- B7. PCIe bridges above the GPU ---"
g=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr "A-Z" "a-z" | sed "s/^0000\{1,\}:/0000:/")
cur="/sys/bus/pci/devices/$g"
for i in 1 2 3; do
  par=$(readlink -f "$cur/.." 2>/dev/null); bn=$(basename "$par")
  [[ "$bn" =~ ^[0-9a-f]{4}: ]] || break
  echo "  L$i: $bn  $(lspci -s ${bn#0000:} 2>/dev/null | cut -d" " -f2- | cut -c1-58)"
  cur="$par"
done
echo "--- B8. rail rates ---"
ibstat 2>/dev/null | grep -E "^CA |Rate:" | paste - - | sed "s/^/  /"
'

# =====================================================================
echo
echo "=================== PART C: RDMA matrix (THE decisive test) ==================="
echo "server=$SERVER client=$CLIENT rail=$NIC size=$SIZE iters=$ITERS"
echo "NOTE: perftest prints bidirectional rows as the SUM of both directions -- divide by 2."
echo "NOTE: perftest requests MR-level relaxed ordering BY DEFAULT;"
echo "      --disable_pcie_relaxed turns it OFF."
echo
echo "AICR reference (per direction, GB/s):"
echo "   host uni 46.3 | host bidir 47.3 | GPU uni 47.5 | GPU bidir 27.2  <-- the defect"
echo "   GPU bidir RO-on 32.0 vs RO-off 31.5  (toggling RO changes nothing on AICR)"
echo

declare -A GBPS      # test id -> raw Gb/s
declare -A BIDIR     # test id -> 1 if bidirectional

run() {
  local id="$1"; local label="$2"; local isbi="$3"; shift 3
  echo "########## $label ##########"
  srun --overlap -N1 -n1 --nodelist="$SERVER" \
    ib_write_bw -d "$NIC" -s $SIZE -n $ITERS -F --report_gbits "$@" > /tmp/.srv.$$ 2>&1 &
  local pid=$!
  sleep 4
  local out
  out=$(srun --overlap -N1 -n1 --nodelist="$CLIENT" \
    ib_write_bw -d "$NIC" -s $SIZE -n $ITERS -F --report_gbits "$@" "$SERVER" 2>&1)
  wait $pid
  echo "$out" | grep -E "^ *$SIZE |Failed|error|status"
  local bw
  bw=$(echo "$out" | awk -v s="$SIZE" '$1==s {print $4; exit}')
  if [ -n "$bw" ]; then
    GBPS[$id]=$bw; BIDIR[$id]=$isbi
  else
    GBPS[$id]="FAILED"; BIDIR[$id]=$isbi
    echo "  !! no result parsed -- server side said:"
    sed 's/^/     /' /tmp/.srv.$$ | grep -viE "^ *$" | tail -12
  fi
  rm -f /tmp/.srv.$$
}

run C1 "C1. HOST unidirectional"                    0
run C2 "C2. HOST bidirectional"                     1  -b
run C3 "C3. GPU  unidirectional"                    0  --use_cuda=0
run C4 "C4. GPU  bidirectional  (RO on, default)"   1  --use_cuda=0 -b
run C5 "C5. GPU  bidirectional  (RO DISABLED)"      1  --use_cuda=0 -b --disable_pcie_relaxed
run C6 "C6. HOST bidirectional  (RO DISABLED)"      1  -b --disable_pcie_relaxed

echo
echo "################ CONVERTED RESULTS (GB/s per direction) ################"
printf "%-4s %-38s %12s %14s %10s\n" ID TEST "raw Gb/s" "GB/s per dir" "AICR"
declare -A DESC=( [C1]="HOST unidirectional" [C2]="HOST bidirectional" \
                  [C3]="GPU  unidirectional" [C4]="GPU  bidir (RO on)" \
                  [C5]="GPU  bidir (RO OFF)" [C6]="HOST bidir (RO OFF)" )
declare -A REF=( [C1]=46.3 [C2]=47.3 [C3]=47.5 [C4]=27.2 [C5]=31.5 [C6]=44.0 )
for id in C1 C2 C3 C4 C5 C6; do
  raw=${GBPS[$id]:-MISSING}
  if [ "$raw" = "FAILED" ] || [ "$raw" = "MISSING" ]; then
    printf "%-4s %-38s %12s %14s %10s\n" "$id" "${DESC[$id]}" "$raw" "-" "${REF[$id]}"
  else
    div=8; [ "${BIDIR[$id]}" = "1" ] && div=16     # /8 for GB/s, /2 again if bidirectional
    perdir=$(awk -v b="$raw" -v d="$div" 'BEGIN{printf "%.1f", b/d}')
    printf "%-4s %-38s %12s %14s %10s\n" "$id" "${DESC[$id]}" "$raw" "$perdir" "${REF[$id]}"
  fi
done

echo
echo "################ AUTOMATIC VERDICT ################"
c4=${GBPS[C4]:-FAILED}; c5=${GBPS[C5]:-FAILED}
if [ "$c4" = "FAILED" ]; then
  echo "C4 did not produce a result -- cannot conclude. Fix C4 and re-run."
else
  c4d=$(awk -v b="$c4" 'BEGIN{printf "%.1f", b/16}')
  c5d="n/a"; [ "$c5" != "FAILED" ] && c5d=$(awk -v b="$c5" 'BEGIN{printf "%.1f", b/16}')
  echo "C4 (GPU bidirectional, RO on) = $c4d GB/s per direction   [AICR: 27.2]"
  echo "C5 (GPU bidirectional, RO off) = $c5d GB/s per direction   [AICR: 31.5]"
  echo
  healthy=$(awk -v v="$c4d" 'BEGIN{print (v>=42)?1:0}')
  if [ "$healthy" = "1" ]; then
    echo "VERDICT 1: Engaging is HEALTHY (C4 >= 42)."
    echo "  => AICR's 27.2 GB/s/dir is a genuine cluster defect, confirmed by direct contrast."
    if [ "$c5d" != "n/a" ]; then
      collapsed=$(awk -v v="$c5d" 'BEGIN{print (v<38)?1:0}')
      if [ "$collapsed" = "1" ]; then
        echo "VERDICT 2: relaxed ordering IS the lever (C5 collapsed while C4 healthy)."
        echo "  => AICR bug is 'RO never reaches the wire'. Chase, with root on AICR:"
        echo "     NIC DevCtl.RlxdOrd enable bit / ConnectX PCI_WR_ORDERING / PEX890xx switch."
      else
        echo "VERDICT 2: relaxed ordering is NOT the lever (C5 ~= C4, both healthy)."
        echo "  => AICR's difference lies elsewhere. The PART B config diff is now the lead."
      fi
    fi
  else
    echo "VERDICT 1: Engaging shows the SAME collapse (C4 < 42)."
    echo "  => This OVERTURNS the diagnosis: it is not an AICR misconfiguration."
    echo "     Report this prominently. The paper's hardware-limit model deserves a second look."
  fi
fi
echo
echo "Also record PART B1 (EnablePCIERelaxedOrderingMode). If it is 0 here AND C4 is healthy,"
echo "that NVIDIA driver parameter is exonerated permanently -- it is the vendor default."

echo
echo "################ WHAT THE ANSWER MEANS ################"
cat <<'EOT'
Convert: Gb/s / 8 = GB/s.  Bidirectional rows are the SUM -> divide by 2 again.

C4 (GPU bidir, RO on) is the number that matters:
  ~47 GB/s/dir  -> Engaging is HEALTHY. AICR's 27.2 is a defect, confirmed by contrast.
  ~27 GB/s/dir  -> Engaging shows it too => NOT an AICR misconfiguration. That would
                   overturn the whole diagnosis, and the paper's model deserves a second look.

C5 vs C4 is the fingerprint test:
  C5 collapses to ~27 while C4 is ~47 -> relaxed ordering IS the lever, and it works on
     Engaging but not on AICR. The AICR bug is then "RO never reaches the wire":
     look at NIC DevCtl.RlxdOrd / ConnectX PCI_WR_ORDERING / the PCIe switch.
  C5 == C4 (both healthy) -> RO is not the lever anywhere; the AICR difference is
     elsewhere (switch config, BIOS, firmware). Compare PART B against the AICR dumps.

PART B1 is the other key line: if Engaging also shows
  EnablePCIERelaxedOrderingMode: 0  and is healthy, that NVIDIA driver parameter is
  exonerated for good (it is the vendor default and discriminates nothing).
EOT
