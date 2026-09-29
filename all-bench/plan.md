# Plan: consolidate all-bench run scripts into one module file

Status: **plan only — nothing implemented yet.** Existing `run-*.sh` /
`get-results-*.sh` stay untouched until the new driver is verified.

## 1. Current state (surveyed 2026-09-23)

- ~65 `run*.sh` presets + 8 `get-results-*.sh` in `all-bench/`.
- Every preset is the same ~25-line script: a header of 7 variables
  (`nodes partition reservation qos cpus gpu_type gpus`) + `all_bench`,
  then an identical loop:
  ```bash
  cd $root_dir/$bench/run
  ./run.sh "$nodes" $partition $reservation $qos $cpus $gpu_type $gpus
  [ -f run-2node.sh ] && ./run-2node.sh "$nodes" ...same args...
  ```
- Loop body is identical in 44 scripts; the rest differ only by
  comments/whitespace or by calling a different entry point:
  - `run-h200.sh` → `submit-h200.sh`, `run-mig.sh` / `run-mig-421.sh` → `submit-mig.sh`
  - `get-results-*.sh` → `get-results.sh $partition $lines $gpu_type` (+ `get-results-2node.sh`)
- 8 scripts owned by bdarek are mode 700 (unreadable to us):
  `run-apollo.sh run-bcs2.sh run-bcs.sh run_ccoley_cpu.sh run_eilers.sh
  run-ghoniem.sh run_ki.sh run_lhtsai_cpu.sh` — values must be obtained
  from the owner or left out.
- Bug found: `run-mit-testing.sh` never sets `all_bench`, so it runs nothing.
- The `all_bench` combos repeat by hardware class (CPU 1-node, CPU multi-node,
  L40S, H100, H200, A100, RTX, MIG).

## 2. Target design

Two files in `all-bench/`:

### 2a. `bench-presets.sh` — the module (data only, sourced)
One bash function per preset that just sets the variables, plus named
benchmark sets so the combos are defined once:

```bash
# benchmark sets
SET_CPU1="openmp mpi-calc-pi"
SET_CPUN="openmp mpi-calc-pi mpi-p2p"
SET_L40S="openmp mpi-calc-pi gpu-burn-r8 nccl-tests"
SET_H100="nccl-tests gpu-burn-r8 nvidia-hpc-benchmarks"
SET_H200="nccl-tests gpu-burn-r8 nvidia-hpc-benchmarks openmp mpi-calc-pi"
...

preset_normal()         { nodes="1605 1606"; partition=mit_normal; reservation=none; qos=unlimited; cpus=96;  gpu_type=none; gpus=none; all_bench="$SET_CPUN"; }
preset_normal_gpu_h200(){ nodes="...";       partition=mit_normal_gpu; ...;                           gpu_type=h200; gpus=4; all_bench="$SET_H200"; }
preset_pi_mbathe()      { nodes="4501 4417 4309"; partition=mit_testing; reservation=root_503; qos=normal; cpus=64; gpu_type=a100; gpus=4; all_bench="$SET_H100"; }
preset_h200_single()    { ...; runner=submit-h200.sh; }   # special entry point
preset_mig()            { ...; runner=submit-mig.sh; }
# get-results presets reuse the same function + lines=N
```

Defaults: `root_dir=/orcd/data/orcd/022/benchmarks`, `runner=run.sh`,
`runner2=run-2node.sh`, `lines=30`.

### 2b. `bench.sh` — the single driver (logic only)
```
./bench.sh list                          # list presets (grep '^preset_')
./bench.sh show   <preset>               # print resolved variables, no submit
./bench.sh run    <preset> [overrides]   # submit benchmarks
./bench.sh results <preset> [overrides]  # collect via get-results.sh
```
Overrides on the command line, e.g.
`./bench.sh run normal nodes="1612 1613" bench="mpi-p2p" dry=1`.

Driver behaviour:
1. `source bench-presets.sh`; call `preset_<name>` (error if missing).
2. Apply `key=value` overrides.
3. Validate: every bench dir + `run/<runner>` exists; `nodes` non-empty;
   `gpu_type/gpus` = `none` for CPU partitions.
4. Loop exactly like today (`run.sh`, then `run-2node.sh` if present;
   `get-results.sh` + `get-results-2node.sh` for `results`).
5. `dry=1` prints the commands instead of executing them.
6. Append one line per invocation to `log.run` (date, preset, nodes, benches).

## 3. Migration steps

1. Write `bench-presets.sh` by extracting the *active* (uncommented) header
   values from each readable `run-*.sh` / `get-results-*.sh` (script it with
   `sed`, then hand-check). Keep preset name = old filename minus `run-`/`.sh`.
2. Write `bench.sh`.
3. Verify with `dry=1` for every preset: the printed commands must match
   what the old script would run (diff old vs new command lines).
4. Real test on one small preset (e.g. single CPU node, `openmp`), compare
   `get-results` output with the old script.
5. Move old `run-*.sh` / `get-results-*.sh` into `bak/` (only ours; ask
   bdarek before touching their files).
6. Update `all-bench/README.md` (Usage/Analysis sections) to the new
   `bench.sh` commands.

## 4. Open questions

- "Module file" = sourced bash preset file (as planned above), or an
  **Lmod modulefile** (`module load all-bench/<preset>` setting env vars)?
  The bash approach is simpler and needs no module path; Lmod is possible
  if you want `module load` usage.
- Keep one function per preset, or collapse to per-hardware-class presets
  and pass `nodes`/`partition` on the command line?
- bdarek's 8 unreadable presets — include (need values) or skip?
- Should `run` optionally chain `results` via `sbatch --dependency` (the
  commented-out idea at the bottom of several scripts)?
