#!/usr/bin/env bash
# Waits for amd-software/rocm-10.1.0.READY (setup/install-rocm-10.1.sh), then runs run_rocm10.1_all.sh.
# The driver file is read only when it starts, so it can still be edited while this waits.
#   ./launch.sh node6100 rocm10.1/start_after_install.sh
SW=/orcd/data/orcd/022/benchmarks/amd-software
for i in $(seq 1 480); do
  [[ -f $SW/rocm-10.1.0.READY ]] && break
  (( i % 5 == 0 )) && ! pgrep -f install-rocm-10.1.sh >/dev/null && [[ ! -f $SW/rocm-10.1.0.READY ]] && break
  sleep 60
done
echo "[$(date -Iseconds)] install READY=$( [[ -f $SW/rocm-10.1.0.READY ]] && echo yes || echo no ); start driver"
exec bash /orcd/data/orcd/022/benchmarks/amd-ubuntu/rocm10.1/run_rocm10.1_all.sh
