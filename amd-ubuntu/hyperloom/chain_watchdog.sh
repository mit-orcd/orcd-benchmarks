#!/bin/bash
# Hourly from cron on the login node: restart run_chain.sh on node6100 if it is not running
# (node reboot, killed process, ...). It does nothing when the chain is done or AUTORESUME_STOP is set.
# The chain itself holds /dev/shm/shaohao-chain.lock, so a second copy exits at once.
# Remove from cron when done: crontab -l | grep -v chain_watchdog.sh | crontab -
H=/orcd/data/orcd/022/benchmarks/amd-ubuntu; L=$H/logs/hyperloom; C=$L/chain-20261117
[[ -f $C/ALL_DONE || -e $L/AUTORESUME_STOP ]] && exit 0
ssh -o BatchMode=yes -o ConnectTimeout=30 node6100 \
  "flock -n /dev/shm/shaohao-chain.lock true && { echo \"[\$(date -Iseconds)] watchdog: chain not running -> start\" >>$C/chain.state; nohup bash $H/hyperloom/run_chain.sh >/dev/null 2>&1 </dev/null & }" \
  >/dev/null 2>&1
