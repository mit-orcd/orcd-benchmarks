#!/bin/bash
# Wait for the run-nccl.sh jobs to finish, then write summary.md.
# Launched detached (nohup+disown) by ./watch-and-summarize.sh so it survives
# the interactive session logging out.
cd /orcd/data/orcd/022/benchmarks/dgx-nodes || exit 1

JOBIDS="20852057 20852058 20852059 20852060 20852061 20852062 20852063 20852064 20852065"

echo "$(date) waiting on jobs: $JOBIDS" >> make-summary.log
while true; do
    pending=$(squeue -h -j "$(echo $JOBIDS | tr ' ' ',')" -o "%i" 2>/dev/null)
    [ -z "$pending" ] && break
    sleep 60
done
echo "$(date) all jobs finished, generating summary.md" >> make-summary.log

python3 - <<'PYEOF' >> make-summary.log 2>&1
import subprocess, datetime
tables = subprocess.run(["python3", "analyze-nccl.py"], capture_output=True, text=True).stdout

ceilings = """
## Hardware ceilings for reference

| Link | Theoretical peak | What it bounds |
|---|---|---|
| NVLink / NVSwitch (H100, per GPU) | ~900 GB/s | 1-node collectives (intra-node) |
| InfiniBand NDR (per HCA, x1) | ~50 GB/s (400 Gb/s) | 2-node collectives, if verbs worked |
| Ethernet fallback (measured path here) | ~10-25 GB/s typical for a single TCP stream | 2-node collectives on these nodes (verbs unavailable, see caveat below) |

**Caveat:** on all three node pairs tested, NCCL could not open any of the
12 IB HCAs (`ibv_open_device` fails, though `ibv_devinfo -l`/sysfs shows the
ports ACTIVE) -- this looks like a MOFED userspace/kernel mismatch on these
nodes rather than a cabling/switch problem. NCCL transparently fell back to
its TCP socket transport over the 10.1.x Ethernet NIC, so the 2-node numbers
below reflect Ethernet-fallback bandwidth, not the IB fabric's real ceiling.
Once verbs are fixed, expect the 2-node numbers to rise sharply toward the
~50 GB/s NDR ceiling.
"""

body = f"""# NCCL benchmark summary — DGX H100 nodes (node180[0,1], node270[0,1], node280[0,1])

Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by make-summary.sh (analyze-nccl.py over out-1node/, out-2node/).

Jobs: 1-node NCCL (NVLink/NVSwitch, 8 GPUs) on each of the 6 nodes; 2-node
NCCL (inter-node) on each bracketed pair (180[0,1], 270[0,1], 280[0,1]).
node170[0,1] excluded (not in mit_testing). Ubuntu 24.04 build
(build-utuntu-nvhpc-26.1, NVHPC 26.1 / CUDA 12.9 / NCCL 2.29.2).

{tables}
{ceilings}
"""
open("summary.md", "w").write(body)
print("wrote summary.md")
PYEOF

echo "$(date) done" >> make-summary.log
