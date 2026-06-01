#!/usr/bin/env python3
"""
Benchmark result analysis automation.

Replaces individual get-results.sh scripts.
Parses output files and prints key metrics for each benchmark.

Usage:
    python bench_analyze.py <benchmarks> --partition <name> [options]

Examples:
    python bench_analyze.py nccl-tests --partition mit_normal_gpu --num-results 2
    python bench_analyze.py openmp mpi-calc-pi --partition mit_normal --num-results 4
    python bench_analyze.py gpu-burn-r8 --partition mit_normal_gpu --num-results 3 --gpu-type l40s
"""

import argparse
import os
import re
import sys

from bench_config import ROOT_DIR, BENCHMARKS


# ── Helpers ──────────────────────────────────────────────────────────────────

def list_recent_files(directory, num_files):
    """
    List the most recent files in a directory, sorted by modification time
    (newest first). Matches the shell `ls -lt | head -n N` pattern.
    Pass num_files=0 to return all files.
    """
    if not os.path.isdir(directory):
        print(f"  Directory not found: {directory}", file=sys.stderr)
        return []

    entries = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            entries.append((os.path.getmtime(path), name))

    entries.sort(reverse=True)  # newest first
    if num_files == 0:
        return [name for _, name in entries]
    return [name for _, name in entries[:num_files]]


def grep_file(filepath, pattern):
    """Return all lines in filepath matching the regex pattern."""
    matches = []
    try:
        with open(filepath, "r", errors="replace") as f:
            for line in f:
                if re.search(pattern, line):
                    matches.append(line.rstrip("\n"))
    except OSError as e:
        print(f"  Cannot read {filepath}: {e}", file=sys.stderr)
    return matches


# ── Analysis functions ───────────────────────────────────────────────────────

def analyze_openmp(partition, num_results, **kwargs):
    """Analyze OpenMP results. Mirrors openmp/run/get-results.sh."""
    print("########## openmp results #########")

    out_dir = os.path.join(ROOT_DIR, "openmp", "work", partition, "output")
    files = list_recent_files(out_dir, num_results)

    for fname in files:
        filepath = os.path.join(out_dir, fname)
        print("================================")
        print(fname)
        lines = grep_file(filepath, r"(THREADS|time)")
        for line in lines:
            print(line)


def analyze_mpi_calc_pi(partition, num_results, **kwargs):
    """Analyze MPI calc-pi results. Mirrors mpi-calc-pi/run/get-results.sh."""
    print("########## mpi-calc-pi results #########")

    out_dir = os.path.join(ROOT_DIR, "mpi-calc-pi", "work", partition, "output")
    files = list_recent_files(out_dir, num_results)

    for fname in files:
        filepath = os.path.join(out_dir, fname)
        print("================================")
        print(fname)
        lines = grep_file(filepath, r"(THREADS|time)")
        for line in lines:
            print(line)


def analyze_mpi_p2p(partition, num_results, **kwargs):
    """Analyze MPI p2p results. Mirrors mpi-p2p/run/get-results.sh."""
    print("########## mpi-p2p results #########")

    out_dir = os.path.join(ROOT_DIR, "mpi-p2p", "work", partition, "output")
    files = list_recent_files(out_dir, num_results)

    for fname in files:
        filepath = os.path.join(out_dir, fname)
        print("================================")
        print(f"Bandwidth (MB/s) and Avg Latency(us) of {fname}")
        lines = grep_file(filepath, r"^4194304\b")
        for line in lines:
            # Extract second column (bandwidth or latency value)
            parts = line.split()
            if len(parts) >= 2:
                print(parts[1])


def analyze_gpu_burn(partition, num_results, gpu_type=None, **kwargs):
    """Analyze GPU burn results. Mirrors gpu-burn-r8/run/get-results.sh."""
    print("########## gpu-burn-r8 results #########")

    if not gpu_type:
        print("Error: --gpu-type required for gpu-burn-r8 analysis", file=sys.stderr)
        return

    out_dir = os.path.join(ROOT_DIR, "gpu-burn-r8", partition, f"output-{gpu_type}")
    # Original shell uses N_lines = num_results * 3 + 1 (3 files per node: tc32, std32, d64)
    total_files = num_results * 3
    files = list_recent_files(out_dir, total_files)

    for fname in files:
        filepath = os.path.join(out_dir, fname)
        print("================================")
        print(fname)
        lines = grep_file(filepath, r"100\.0%")
        for line in lines:
            # Extract the part after "100.0%"
            parts = line.split("100.0%")
            if len(parts) > 1:
                print(parts[-1].strip())


def analyze_nccl_tests(partition, num_results, **kwargs):
    """Analyze NCCL test results. Mirrors nccl-tests/run/get-results.sh."""
    print("########## nccl-tests results #########")

    bench_dir = os.path.join(ROOT_DIR, "nccl-tests")
    dirs = [
        os.path.join(bench_dir, partition, "out-1node"),
        os.path.join(bench_dir, partition, "out-2node"),
    ]

    for out_dir in dirs:
        print(f"^^^^^^^ {out_dir} ^^^^^^^^^^")
        files = list_recent_files(out_dir, num_results)

        for fname in files:
            filepath = os.path.join(out_dir, fname)
            print("=========================================================================================")
            print(fname)

            # Find sendrecv_perf sections and extract 4294967296-byte results
            in_sendrecv = False
            try:
                with open(filepath, "r", errors="replace") as f:
                    lines_after = 0
                    for line in f:
                        line = line.rstrip("\n")
                        if "sendrecv_perf" in line:
                            print(line)
                            in_sendrecv = True
                            lines_after = 0
                            continue
                        if in_sendrecv:
                            lines_after += 1
                            if "4294967296" in line:
                                print(line)
                            # The shell script uses -A 45 (45 lines after match)
                            if lines_after > 45:
                                in_sendrecv = False
            except OSError as e:
                print(f"  Cannot read {filepath}: {e}", file=sys.stderr)


def analyze_nvidia_hpc(partition, num_results, gpu_type=None, **kwargs):
    """Analyze NVIDIA HPC benchmark results. Mirrors nvidia-hpc-benchmarks/run/get-results.sh."""
    print("########## nvidia-hpc-benchmarks results #########")

    if not gpu_type:
        print("Error: --gpu-type required for nvidia-hpc-benchmarks analysis", file=sys.stderr)
        return

    out_dir = os.path.join(ROOT_DIR, "nvidia-hpc-benchmarks", partition, f"output-{gpu_type}")
    files = list_recent_files(out_dir, num_results)

    for fname in files:
        filepath = os.path.join(out_dir, fname)
        print("================================")
        print(f"GFLOPs of {fname}")
        print("n*GPUs    per GPU")

        # WC0 lines: extract columns 7 and 9 (1-indexed)
        wc0_lines = grep_file(filepath, r"WC0")
        for line in wc0_lines:
            parts = line.split()
            if len(parts) >= 9:
                print(f"{parts[6]} {parts[8]}")

        # LU GFLOPS lines (with 1 line of context before)
        try:
            with open(filepath, "r", errors="replace") as f:
                prev_line = ""
                for line in f:
                    line = line.rstrip("\n")
                    if "LU GFLOPS" in line:
                        print(prev_line)
                        print(line)
                    prev_line = line
        except OSError as e:
            print(f"  Cannot read {filepath}: {e}", file=sys.stderr)


def analyze_gpu_fryer(partition, num_results, **kwargs):
    """
    Analyze gpu-fryer results.

    Extracts GPU memory usage per precision mode (fp32, bf16, fp8).
    Output files live in gpu-fryer/output/ (not partition-specific).
    """
    print("########## gpu-fryer results #########")

    out_dir = os.path.join(ROOT_DIR, "gpu-fryer", "output")
    files = list_recent_files(out_dir, num_results)

    for fname in files:
        filepath = os.path.join(out_dir, fname)
        print("================================")
        print(fname)

        # Print precision mode headers and GPU memory lines
        lines = grep_file(filepath, r"(Run with |GPU #\d+: Using |Detected GPU #\d+)")
        for line in lines:
            print(line)


def analyze_megatron_lm(partition, num_results, **kwargs):
    """
    Analyze Megatron-LM GPT benchmark results.

    Extracts per-GPU throughput (TFLOP/s/GPU) from iteration log lines.
    Shows the last logged iteration per file to report steady-state performance.
    """
    print("########## megatron-lm results #########")

    out_dir = os.path.join(ROOT_DIR, "megatron-lm", "Megatron-LM", "output")
    # Only consider job output files (pattern: out.<node>-<jobid> or *.out)
    all_files = list_recent_files(out_dir, 0)  # 0 = unlimited, filter below
    files = [f for f in all_files
             if f.startswith("out.") or f.endswith(".out")][:num_results]

    for fname in files:
        filepath = os.path.join(out_dir, fname)
        print("================================")
        print(fname)

        # Collect all iteration lines; they contain throughput and loss
        iter_lines = grep_file(filepath, r"^\s*\[.*\] iteration\s+\d+")
        if iter_lines:
            # Print last iteration line for steady-state throughput
            last = iter_lines[-1]
            # Extract key fields: iteration, throughput, GBS, lm loss
            for field in ("iteration", "throughput per GPU", "global batch size", "lm loss:"):
                for part in last.split("|"):
                    if field in part:
                        print(" ", part.strip())
                        break
        else:
            print("  No iteration lines found (job may still be running or failed)")


def analyze_numpy(partition, num_results, **kwargs):
    """Analyze numpy matrix-multiply results. Extracts OMP_NUM_THREADS and timing."""
    print("########## numpy results #########")

    out_dir = os.path.join(ROOT_DIR, "numpy", partition, "output")
    files = list_recent_files(out_dir, num_results)

    for fname in files:
        filepath = os.path.join(out_dir, fname)
        print("================================")
        print(fname)
        lines = grep_file(filepath, r"(OMP_NUM_THREADS|seconds to matrix multiply)")
        for line in lines:
            print(line)


# ── Dispatch table ───────────────────────────────────────────────────────────

ANALYZE_DISPATCH = {
    "openmp":                analyze_openmp,
    "mpi-calc-pi":           analyze_mpi_calc_pi,
    "mpi-p2p":               analyze_mpi_p2p,
    "gpu-burn-r8":           analyze_gpu_burn,
    "nccl-tests":            analyze_nccl_tests,
    "nvidia-hpc-benchmarks": analyze_nvidia_hpc,
    "gpu-fryer":             analyze_gpu_fryer,
    "megatron-lm":           analyze_megatron_lm,
    "numpy":                 analyze_numpy,
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s nccl-tests --partition mit_normal_gpu --num-results 2
  %(prog)s openmp mpi-calc-pi --partition mit_normal --num-results 4
  %(prog)s gpu-burn-r8 --partition mit_normal_gpu --num-results 3 --gpu-type l40s
  %(prog)s nvidia-hpc-benchmarks --partition mit_normal_gpu --num-results 2 --gpu-type h200
""",
    )
    parser.add_argument(
        "benchmarks", nargs="*",
        help=f"Benchmark(s) to analyze (choices: {', '.join(BENCHMARKS.keys())})",
    )
    parser.add_argument("--all-bench", action="store_true",
                        help="Analyze every registered benchmark")
    parser.add_argument("--partition", required=True, help="Slurm partition name")
    parser.add_argument("--num-results", type=int, default=2,
                        help="Number of recent results to show (default: 2)")
    parser.add_argument("--gpu-type", help="GPU type (required for gpu-burn-r8 and nvidia-hpc-benchmarks)")
    args = parser.parse_args()

    if args.all_bench and args.benchmarks:
        parser.error("--all-bench cannot be combined with explicit benchmark names")
    if not args.all_bench and not args.benchmarks:
        parser.error("provide benchmark names or --all-bench")
    if args.all_bench:
        args.benchmarks = list(BENCHMARKS.keys())
        # Skip GPU-type-dependent analyses if no --gpu-type provided.
        if not args.gpu_type:
            args.benchmarks = [b for b in args.benchmarks
                               if b not in ("gpu-burn-r8", "nvidia-hpc-benchmarks")]
    else:
        unknown = [b for b in args.benchmarks if b not in BENCHMARKS]
        if unknown:
            parser.error(f"unknown benchmark(s): {', '.join(unknown)}")

    return args


def main():
    args = parse_args()

    for bench in args.benchmarks:
        func = ANALYZE_DISPATCH[bench]
        func(
            partition=args.partition,
            num_results=args.num_results,
            gpu_type=args.gpu_type,
        )


if __name__ == "__main__":
    main()
