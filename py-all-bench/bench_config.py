#!/usr/bin/env python3
"""
Shared configuration and utilities for benchmark automation.

Provides:
- Root directory paths and benchmark registry
- Slurm query helpers (partition, nodes, CPUs, GPUs)
- Thread/process list generation for CPU benchmarks
- Common sbatch submission wrapper
"""

import os
import subprocess
import sys
import tempfile

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = "/orcd/data/orcd/022/benchmarks"

# ── Benchmark registry ───────────────────────────────────────────────────────
# Maps benchmark name -> category ("cpu", "gpu", "mpi")
BENCHMARKS = {
    "openmp":                 "cpu",
    "mpi-calc-pi":            "cpu",
    "mpi-p2p":                "mpi",
    "gpu-burn-r8":            "gpu",
    "nccl-tests":             "gpu",
    "nvidia-hpc-benchmarks":  "gpu",
    "gpu-fryer":              "gpu",
    "megatron-lm":            "gpu",
    "numpy":                  "cpu",
}

CPU_BENCHMARKS = [b for b, c in BENCHMARKS.items() if c == "cpu"]
GPU_BENCHMARKS = [b for b, c in BENCHMARKS.items() if c == "gpu"]
MPI_BENCHMARKS = [b for b, c in BENCHMARKS.items() if c == "mpi"]


# ── Slurm helpers ────────────────────────────────────────────────────────────

def run_cmd(cmd, check=True):
    """Run a shell command and return stripped stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}", file=sys.stderr)
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def query_partitions():
    """Return list of available Slurm partitions."""
    out = run_cmd("sinfo -h -o '%P'")
    return [p.rstrip("*") for p in out.splitlines() if p.strip()]


def query_nodes_in_partition(partition):
    """Return list of node names in a partition."""
    out = run_cmd(f"sinfo -h -p {partition} -o '%n'")
    return [n.strip() for n in out.splitlines() if n.strip()]


def query_cpus_on_node(node):
    """Return number of CPUs on a node."""
    out = run_cmd(f"sinfo -h -n {node} -o '%c'")
    return int(out.splitlines()[0].strip())


def query_gpus_on_node(node):
    """Return (gpu_type, gpu_count) for a node, or (None, 0) if no GPUs."""
    out = run_cmd(f"sinfo -h -n {node} -o '%G'", check=False)
    line = out.strip()
    # Format: gpu:type:count  or  gpu:count  or  (null)
    if not line or line == "(null)":
        return None, 0
    parts = line.split(":")
    if len(parts) == 3:  # gpu:type:count
        return parts[1], int(parts[2])
    elif len(parts) == 2:  # gpu:count
        return None, int(parts[1])
    return None, 0


def query_qos_list():
    """Return list of available QoS names."""
    out = run_cmd("sacctmgr -n list qos format=name", check=False)
    return [q.strip() for q in out.splitlines() if q.strip()]


# ── Thread list generation ───────────────────────────────────────────────────

def generate_thread_list(cores):
    """
    Generate the thread/process count list for CPU benchmarks.
    Matches the logic in openmp/run/run.sh and mpi-calc-pi/run/run.sh:
      1, 2, 4, 8, ..., cores/2, cores, 3*cores/2, 2*cores
    With special handling for < 6 cores.
    """
    if cores == 1:
        return [1]
    if cores == 2:
        return [1, 2]
    if cores < 6:
        return [1, 2, cores]

    thread_list = [1]
    threads = 1
    while threads * 4 < cores:
        threads *= 2
        thread_list.append(threads)
    thread_list.extend([cores // 2, cores, 3 * cores // 2, 2 * cores])
    return thread_list


# ── Sbatch submission ────────────────────────────────────────────────────────

def sbatch_submit(script_text, reservation="none", extra_flags=None):
    """
    Submit a job script via sbatch.

    Parameters
    ----------
    script_text : str
        Full batch script content (including #SBATCH directives).
    reservation : str
        Slurm reservation name, or "none" to omit.
    extra_flags : list[str] or None
        Additional sbatch CLI flags.

    Returns
    -------
    str
        sbatch output (typically "Submitted batch job <id>").
    """
    cmd = ["sbatch"]
    if reservation and reservation != "none":
        cmd.append(f"--reservation={reservation}")
    if extra_flags:
        cmd.extend(extra_flags)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_text)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            cmd + [tmp_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"sbatch failed: {result.stderr.strip()}", file=sys.stderr)
            return ""
        output = result.stdout.strip()
        print(f"  {output}")
        return output
    finally:
        os.unlink(tmp_path)


def sbatch_submit_stdin(script_text, reservation="none", extra_flags=None):
    """
    Submit a job script via sbatch on stdin (heredoc style).

    Same interface as sbatch_submit but pipes the script through stdin,
    matching the shell-script heredoc approach.
    """
    cmd = ["sbatch"]
    if reservation and reservation != "none":
        cmd.append(f"--reservation={reservation}")
    if extra_flags:
        cmd.extend(extra_flags)

    result = subprocess.run(
        cmd,
        input=script_text,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr.strip()}", file=sys.stderr)
        return ""
    output = result.stdout.strip()
    print(f"  {output}")
    return output
