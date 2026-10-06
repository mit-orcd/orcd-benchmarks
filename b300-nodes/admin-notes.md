# Notes for the admins

Two issues found while benchmarking B300 (node5900-c1) against B200 on mit_testing, with suggested
fixes. Checked 2026-10-05.

## 1. node5800-c1: jobs fail to launch

**Symptom.** Since 2026-10-02, every job that includes node5800-c1 (mit_testing, 8x B200) ends up
"launch failed requeued held". The B200 Megatron sweep (b300-nodes) and the b200-training jobs were
both affected. The job is allocated and its `extern` step completes, but the `batch` step on
node5800-c1 is cancelled before the script starts. The stdout file is never created.
Examples: jobs 24476123 and 24476125 (node5800-c1 + node5801-c1), 2026-10-02 12:31 and again
2026-10-05 20:13 after release.

**Likely cause: Slurm version mismatch.**

| | Slurm version |
|---|---|
| slurmctld (`scontrol version`) | 26.05.4 |
| 11 other mit_testing nodes (node5600/5601/5602/5801/5802/5900-c1, ...) | 26.05.4 |
| **node5800-c1** | **25.05.4** |

slurmd on node5800-c1 was restarted 2026-10-02 03:38, so it seems to have missed the upgrade. The
first launch failures came the same day.

**What is fine on the node (via ssh):** up 46 days, idle, `slurmd` active, 8 B200 GPUs visible
(`nvidia-smi -L`), `/tmp` (tmpfs, 202 GB) and `/` (1.8 TB, 3% used) nearly empty, /home,
/orcd/data/orcd/022 and scratch mounted and writable, group membership correct.

**Not checked:** `/var/log/slurmd` and the system journal are readable only by root, so the actual
launch error was not seen.

**Suggested fix (admin side):**
1. Drain the node: `scontrol update nodename=node5800-c1 state=drain reason="slurmd 25.05.4 vs ctld 26.05.4"`.
2. Check `/var/log/slurmd` (or `journalctl -u slurmd`) around 2026-10-05 20:13 to confirm the launch error.
3. Install the same Slurm packages as the other nodes (26.05.4), including any plugins
   (e.g. the pyxis/spank or cgroup plugins), then `systemctl restart slurmd`.
4. Confirm with `scontrol show node node5800-c1 | grep Version` (should show 26.05.4), run a short
   test job on it (`srun -w node5800-c1 -p mit_testing hostname`), then resume the node.
5. Optional: add a Slurm version check to the node health check so a node that comes back with an
   older slurmd is drained automatically.

**Workaround on the user side (in place now).** node5800-c1 is excluded (`ExcNodeList=node5800-c1`)
on all pending jobs, and `job-megatron-max-sweep.sh` uses `-x node5800-c1` for `any`-node B200 runs.
Once the node is fixed, remove the exclusion with `scontrol update job=<ids> ExcNodeList=` and drop
the `-x` default in `job-megatron-max-sweep.sh`.

## 2. GPU power cap limits B300 FP4 speedup to 1.20x (1.5x on paper)

**Symptom.** In the cuBLASLt GEMM benchmark (`../cuBLASLt/summary.md`, run 2026-10-01), B300 is
1.20x faster than B200 at FP4, but NVIDIA's datasheet says 1.5x (13.5 vs 9 PFLOP/s dense per GPU).
B300 reaches only 51% of its datasheet FP4 peak, while B200 reaches 63% of its own.

**Cause: when a GPU is at its power limit, its speed is set by power x efficiency, and B300 is
only 1.10x better than B200 on each.**

- **Where the 1.5x comes from.** Speed = work per clock x clock rate. B300 does 1.5x more FP4 work
  per clock than B200. The datasheet assumes both run at full clock, which gives 1.5x.
- **Why the clocks are not the same.** Both GPUs reach their power limit during FP4 (993 of 1,000 W
  and 1,091 of 1,100 W) and lower their clocks to stay under it. B300 does 1.5x more work in each
  clock, so each clock uses more energy, but B300 has only 10% more power. It therefore has to run
  at a lower clock than B200: 1,099 vs 1,314 MHz (0.84x).
- **The result.** 1.5x work per clock x 0.84x clock = ~1.25x, close to the measured 1.20x.
- **The same thing viewed through power.** Under a power limit, speed = power x work per watt.
  B300 has 1.10x the power (1,100 vs 1,000 W) and does 1.10x the FP4 work per watt (6.30 vs 5.75
  TFLOP/s per W). 1.10 x 1.10 = 1.21x, matching the measurement. To reach 1.5x at the same work per
  watt, B300 would need about 1.5 / 1.10 = 1.36x B200's power, i.e. ~1,360 W per GPU instead of 1,100 W.

| FP4, sustained 60 s | B200 (node5802-c1) | B300 (node5900-c1) |
|---|---:|---:|
| Power limit (current = default = max) | 1,000 W | 1,100 W |
| Power drawn | 993 W | 1,091 W |
| SM clock | 1,314 MHz | 1,099 MHz (0.84x of B200) |
| Max SM clock | 1,965 MHz | 2,032 MHz |
| Share of max clock | 67% | 54% |
| Max GPU temperature | 73 °C | 74 °C |
| Measured FP4 | 5,708 TFLOP/s | 6,877 TFLOP/s (1.20x) |

The GEMM kernels reach ~94% of what the actual clock allows on both GPUs, so the software is not the limit.
Temperatures are moderate, so this is a power limit, not a thermal limit. The other precisions
(BF16, FP8, TF32) measure ~1.0x because B300 has the same peak per clock as B200 there.

The same limit shows up in training: Megatron-LM is 1.03-1.06x (reference config) and 1.12-1.13x
(tuned, using B300's larger memory) faster on B300.

**Current settings (from `nvidia-smi`, 2026-10-05).** On both nodes every GPU already runs at its
maximum allowed power limit:

| Node | GPU | power.limit | power.default_limit | power.max_limit |
|---|---|---:|---:|---:|
| node5802-c1 | B200 | 1,000 W | 1,000 W | 1,000 W |
| node5900-c1 | B300 SXM6 AC | 1,100 W | 1,100 W | 1,100 W |

So the limit cannot be raised with `nvidia-smi -pl`; 1,100 W is the board maximum for this
air-cooled (SXM6 AC) HGX B300. NVIDIA's 15 PFLOP/s FP4 headline figure is for Blackwell Ultra parts
running at up to ~1,400 W (liquid-cooled GB300-class systems). Driver 590.48.01 on node5900-c1
does not offer the `nvidia-smi workload-power-profile` option.

**Suggested actions (admin side):**
1. Confirm there is no extra node-level cap: check the BMC / chassis power-capping settings and
   the BIOS power profile on node5900-c1 (and node5802-c1), and set them to maximum performance if
   they are lower. A quick check during a load run is
   `nvidia-smi -q -d PERFORMANCE` (the throttle reason should be "SW Power Cap" only, not
   "HW Slowdown" or "HW Power Brake").
2. Check with the vendor (or NVIDIA) whether a higher-TGP firmware or power profile exists for this
   B300 SXM6 AC board, and whether newer drivers add workload power profiles (e.g. a MAX-P profile)
   that would let the GPU hold a higher clock at the same power.
3. For future purchases: a liquid-cooled B300 / GB300 system with a ~1,400 W per-GPU limit is
   what gives the full ~1.5x FP4 gain over B200. With the current 1,100 W boards, plan for ~1.2x
   FP4 and ~1.0-1.1x for BF16/FP8 work, with B300's main benefit being the larger memory
   (288 GB vs 180 GB).

No change is needed from users: jobs cannot change the power limit, and the measured numbers are
what the hardware gives at 1,100 W.
