#!/usr/bin/env python3
"""
Benchmark job submission automation.

Replaces all-bench/run-all.sh and individual run/run.sh scripts.
Submits Slurm jobs for one or more benchmarks across specified nodes.

Usage:
    python bench_submit.py <benchmarks> [options]

Examples:
    python bench_submit.py nccl-tests --nodes 3511 3512 --partition mit_normal_gpu --qos unlimited --cpus 48 --gpu-type l40s --gpus 4
    python bench_submit.py openmp mpi-calc-pi --nodes 3511 --partition mit_normal --qos normal --cpus 48
    python bench_submit.py gpu-burn-r8 nvidia-hpc-benchmarks --nodes 3511 3512 --partition mit_normal_gpu --qos unlimited --cpus 48 --gpu-type h200 --gpus 8
"""

import argparse
import os
import subprocess
import sys
import time

from bench_config import (
    ROOT_DIR,
    BENCHMARKS,
    generate_thread_list,
    query_nodes_in_partition,
    query_cpus_on_node,
    query_gpus_on_node,
    sbatch_submit,
    sbatch_submit_stdin,
)


# ── Submission functions ─────────────────────────────────────────────────────

def submit_openmp(nodes, partition, reservation, qos, cpus):
    """Submit OpenMP benchmark jobs. Mirrors openmp/run/run.sh."""
    print("########## openmp #########")

    output_dir = os.path.join(ROOT_DIR, "openmp", "work", partition, "output")
    os.makedirs(output_dir, exist_ok=True)

    script_dir = os.path.join(ROOT_DIR, "openmp", "src", "pi_omp")
    cores = cpus
    thread_list = generate_thread_list(cores)
    thread_list_str = " ".join(str(t) for t in thread_list)
    print(f"List of threads: {thread_list_str}")

    for node in nodes:
        host = f"node{node}"
        print(f"running on host {host}")

        script = f"""#!/bin/bash
#SBATCH -t 30
#SBATCH -p {partition}
#SBATCH -n {cores}
#SBATCH -N 1
#SBATCH -w {host}
#SBATCH -o {output_dir}/out_full.%N-%J
#SBATCH -q {qos}
#SBATCH --exclusive
#SBATCH -J openmp

IFS=' ' read -a thread_list <<< "{thread_list_str}"

echo "Benchmarking on ${{thread_list[*]}}"

for j in ${{!thread_list[@]}};
do
     export OMP_NUM_THREADS=${{thread_list[j]}}
     echo "Ran with OMP_NUM_THREADS=$OMP_NUM_THREADS"
     time {script_dir}
done
"""
        sbatch_submit_stdin(script, reservation)


def submit_mpi_calc_pi(nodes, partition, reservation, qos, cpus):
    """Submit MPI calc-pi benchmark jobs. Mirrors mpi-calc-pi/run/run.sh."""
    print("########## mpi-calc-pi #########")

    output_dir = os.path.join(ROOT_DIR, "mpi-calc-pi", "work", partition, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    script_dir = os.path.join(ROOT_DIR, "mpi-calc-pi", "src")
    cores = cpus
    thread_list = generate_thread_list(cores)
    thread_list_str = " ".join(str(t) for t in thread_list)
    print(f"List of threads: {thread_list_str}")

    for node in nodes:
        host = f"node{node}"
        print(f"running on host {host}")

        script = f"""#!/bin/bash
#SBATCH -t 30
#SBATCH -p {partition}
#SBATCH -n {cores}
#SBATCH -N 1
#SBATCH -w {host}
#SBATCH -o {output_dir}/out_full.%N-%J
#SBATCH -q {qos}
#SBATCH --exclusive
#SBATCH -J mpi-pi

IFS=' ' read -a thread_list <<< "{thread_list_str}"

echo "Benchmarking on ${{thread_list[*]}}"

module load openmpi/4.1.4

which mpirun

for j in ${{!thread_list[@]}};
do
     export NUM_THREADS=${{thread_list[j]}}
     echo "Ran with MPI_NUM_THREADS=$NUM_THREADS"
     mpirun --oversubscribe -np ${{NUM_THREADS}} {script_dir}/calc_pi_mpi
done
"""
        sbatch_submit_stdin(script, reservation)


def submit_mpi_p2p(nodes, partition, reservation, qos):
    """Submit MPI point-to-point benchmark jobs. Mirrors mpi-p2p/run/run.sh."""
    print("########## mpi-p2p #########")

    output_dir = os.path.join(ROOT_DIR, "mpi-p2p", "work", partition, "output")
    os.makedirs(output_dir, exist_ok=True)

    env_sh = os.path.join(ROOT_DIR, "mpi-p2p", "run", "env.sh")

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i < j:
                host1 = f"node{nodes[i]}"
                host2 = f"node{nodes[j]}"
                print(f"Running on hosts {host1} and {host2}")

                script = f"""#!/bin/bash
#SBATCH -p {partition}
#SBATCH -t 10
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB
#SBATCH --constraint=rocky8
#SBATCH -w {host1},{host2}
#SBATCH -o {output_dir}/out.{host1}_{host2}-%J
#SBATCH -J mpi-p2p
#SBATCH --exclusive
#SBATCH -q {qos}

source {env_sh} r8 4.1.4

echo "number of nodes = ${{SLURM_NNODES}}"
echo "total number of tasks = ${{SLURM_NTASKS}}"
echo "number of tasks per core = ${{SLURM_NTASKS_PER_CORE}}"
echo "number of cores per task = ${{SLURM_CPUS_PER_TASK}}"
echo "total number of virtual cores (hyperthreads) per node = ${{SLURM_CPUS_ON_NODE}}"
echo "total memory per node = ${{SLURM_MEM_PER_NODE}}"

echo "--- mpirun ---"
which mpirun
echo "---srun---"
srun hostname
mpirun -n ${{SLURM_NTASKS}} hostname
echo "--- osu_bw ---"
mpirun -n ${{SLURM_NTASKS}} osu_bw
echo "--- osu_latency ---"
mpirun -n ${{SLURM_NTASKS}} osu_latency
"""
                sbatch_submit_stdin(script, reservation)
                time.sleep(1)


def submit_gpu_burn(nodes, partition, reservation, qos, cpus, gpu_type, gpus):
    """Submit GPU burn benchmark jobs. Mirrors gpu-burn-r8/run/run.sh."""
    print("########## gpu-burn-r8 #########")

    bench_dir = os.path.join(ROOT_DIR, "gpu-burn-r8")
    out_dir = os.path.join(bench_dir, partition, f"output-{gpu_type}")
    os.makedirs(out_dir, exist_ok=True)

    for node in nodes:
        host = f"node{node}"

        script = f"""#!/bin/bash
#SBATCH -t 100
#SBATCH -p {partition}
#SBATCH -q {qos}
#SBATCH -N 1
#SBATCH -n {2 * gpus}
#SBATCH --gres=gpu:{gpu_type}:{gpus}
#SBATCH -w {host}
#SBATCH --mem=50GB
#SBATCH --exclusive
#SBATCH -J gpu-burn

hostname

cd {bench_dir}

echo "Running tensor core single precision"
./gpu_burn -tc 300 > {out_dir}/tc32_{host}-${{SLURM_JOB_ID}}.out
echo "Finished tensor core gpu burn. Output saved to tc32 folder"

echo "Running single precision (standard)"
./gpu_burn 300 > {out_dir}/std32_{host}-${{SLURM_JOB_ID}}.out
echo "Finished single precision gpu burn. Output saved to std32 folder"

echo "Running double precision"
./gpu_burn -d 300 > {out_dir}/d64_{host}-${{SLURM_JOB_ID}}.out
echo "Finished double precision gpu burn. Output saved to d64 folder"
"""
        sbatch_submit_stdin(script, reservation)


def submit_nccl_tests(nodes, partition, reservation, qos, cpus, gpu_type, gpus):
    """Submit NCCL test jobs. Mirrors nccl-tests/run/run.sh and run-2node.sh."""
    print("########## nccl-tests #########")

    run_dir = os.path.join(ROOT_DIR, "nccl-tests", "run")
    bench_dir = os.path.join(ROOT_DIR, "nccl-tests")

    # ── 1-node jobs ──────────────────────────────────────────────────────
    out_dir_1 = os.path.join(bench_dir, partition, "out-1node")
    os.makedirs(out_dir_1, exist_ok=True)

    flags_1 = [
        "-p", partition,
        "-q", qos,
        f"--gres=gpu:{gpu_type}:{gpus}",
        "--exclusive",
        "-o", os.path.join(out_dir_1, "%x-%N-%J"),
    ]
    if reservation and reservation != "none":
        flags_1.extend(["--reservation", reservation])

    job_script_1 = os.path.join(run_dir, "1node.sh")
    for node in nodes:
        host = f"node{node}"
        print(host)
        cmd = ["sbatch"] + flags_1 + ["-w", host, job_script_1, str(gpus)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=run_dir)
        if result.returncode != 0:
            print(f"  sbatch failed: {result.stderr.strip()}", file=sys.stderr)
        else:
            print(f"  {result.stdout.strip()}")

    # ── 2-node jobs (if more than one node) ──────────────────────────────
    if len(nodes) < 2:
        return

    out_dir_2 = os.path.join(bench_dir, partition, "out-2node")
    os.makedirs(out_dir_2, exist_ok=True)

    flags_2 = [
        "-p", partition,
        "-q", qos,
        f"--gpus-per-node={gpu_type}:1",
        "--exclusive",
        "-o", os.path.join(out_dir_2, "%x-%N-%J"),
    ]
    if reservation and reservation != "none":
        flags_2.extend(["--reservation", reservation])

    job_script_2 = os.path.join(run_dir, "2nodes-2gpus.sh")
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i < j:
                host1 = f"node{nodes[i]}"
                host2 = f"node{nodes[j]}"
                print(f"Running on hosts {host1} and {host2}")
                cmd = ["sbatch"] + flags_2 + [
                    "-w", f"{host1},{host2}",
                    job_script_2,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=run_dir)
                if result.returncode != 0:
                    print(f"  sbatch failed: {result.stderr.strip()}", file=sys.stderr)
                else:
                    print(f"  {result.stdout.strip()}")


def submit_nvidia_hpc(nodes, partition, reservation, qos, cpus, gpu_type, gpus):
    """Submit NVIDIA HPC benchmark jobs. Mirrors nvidia-hpc-benchmarks/run/run.sh."""
    print("########## nvidia-hpc-benchmarks #########")

    bench_dir = os.path.join(ROOT_DIR, "nvidia-hpc-benchmarks")
    run_dir = os.path.join(bench_dir, "run")
    output_dir = os.path.join(bench_dir, partition, f"output-{gpu_type}")
    os.makedirs(output_dir, exist_ok=True)

    image = os.path.join(run_dir, "hpc-benchmarks_25.04.sif")

    for node in nodes:
        host = f"node{node}"
        print(host)

        script = f"""#!/bin/bash
#SBATCH -o {output_dir}/{host}-%J.out
#SBATCH -t 100
#SBATCH -p {partition}
#SBATCH -q {qos}
#SBATCH --mem=1000GB
#SBATCH -N 1
#SBATCH -n {2 * gpus}
#SBATCH -w {host}
#SBATCH --gres=gpu:{gpu_type}:{gpus}
#SBATCH --exclusive
#SBATCH -J nvidia-hpc

hostname

module load apptainer/1.1.9

module list
which singularity

./execute-all.sh {image} {gpus}
"""
        sbatch_submit_stdin(script, reservation, extra_flags=None)


def submit_gpu_fryer(nodes, partition, reservation, qos, cpus, gpu_type, gpus):
    """Submit gpu-fryer benchmark jobs. Mirrors gpu-fryer/job.sh and submit.sh."""
    print("########## gpu-fryer #########")

    bench_dir = os.path.join(ROOT_DIR, "gpu-fryer")
    out_dir = os.path.join(bench_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    image = os.path.join(bench_dir, "gpu-fryer_1.1.0.sif")
    sing_cmd = f"singularity exec --nv -B /lib64:/home/shaohao/lib64 {image}"
    flags = "--nvml-lib-path /home/shaohao/lib64/libnvidia-ml.so.1"
    elapse = 300

    for node in nodes:
        host = f"node{node}"
        print(host)

        script = f"""#!/bin/bash
#SBATCH -p {partition}
#SBATCH -q {qos}
#SBATCH -t 01:00:00
#SBATCH --mem=30GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:{gpu_type}:{gpus}
#SBATCH -w {host}
#SBATCH -o {out_dir}/%N-%J.out
#SBATCH -J gpu-fryer

module load apptainer/1.4.2
which singularity

echo "Number of GPUs = $SLURM_GPUS $SLURM_GPUS_ON_NODE"

echo "======== Run with fp32 =========="
{sing_cmd} gpu-fryer --use-fp32 {flags} {elapse}
echo "======== Run with bf16 =========="
{sing_cmd} gpu-fryer --use-bf16 {flags} {elapse}
echo "======== Run with fp8  =========="
{sing_cmd} gpu-fryer --use-fp8 {flags} {elapse}
"""
        sbatch_submit_stdin(script, reservation)


def submit_megatron_lm(nodes, partition, reservation, qos, cpus, gpu_type, gpus):
    """
    Submit Megatron-LM GPT benchmark jobs. Mirrors megatron-lm/Megatron-LM/submit.sh.

    Submits one job per GPU-count configuration (1, 2, 4, ... up to gpus) for
    single-node runs, and for every node pair when multiple nodes are given.
    GBS = 128 x total_GPUs so gradient accumulation stays constant.
    """
    print("########## megatron-lm #########")

    megatron_dir = os.path.join(ROOT_DIR, "megatron-lm", "Megatron-LM")
    out_dir = os.path.join(megatron_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    job_sh = os.path.join(megatron_dir, "job.sh")

    # Build GPU-count list: powers of 2 from 1 up to gpus
    gpu_counts = []
    n = 1
    while n <= gpus:
        gpu_counts.append(n)
        n *= 2

    # ── 1-node jobs ──────────────────────────────────────────────────────────
    for node in nodes:
        host = f"node{node}"
        for n_gpus in gpu_counts:
            gbs = 128 * n_gpus
            print(f"{host}  gpus={n_gpus}  GBS={gbs}")
            flags = [
                "-N", "1", "-n", "1",
                f"--gpus-per-node={gpu_type}:{n_gpus}",
                "-p", partition,
                "-q", qos,
                "-w", host,
            ]
            if reservation and reservation != "none":
                flags += [f"--reservation={reservation}"]
            cmd = ["sbatch"] + flags + [job_sh, str(gbs)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=megatron_dir)
            if result.returncode != 0:
                print(f"  sbatch failed: {result.stderr.strip()}", file=sys.stderr)
            else:
                print(f"  {result.stdout.strip()}")

    # ── 2-node jobs (every node pair, skip 1-GPU since scaling needs ≥2) ────
    if len(nodes) < 2:
        return

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i < j:
                host1 = f"node{nodes[i]}"
                host2 = f"node{nodes[j]}"
                for n_gpus in gpu_counts:
                    gbs = 128 * 2 * n_gpus  # 2 nodes
                    print(f"{host1},{host2}  gpus_per_node={n_gpus}  GBS={gbs}")
                    flags = [
                        "-N", "2", "-n", "2",
                        f"--gpus-per-node={gpu_type}:{n_gpus}",
                        "-p", partition,
                        "-q", qos,
                        "-w", f"{host1},{host2}",
                    ]
                    if reservation and reservation != "none":
                        flags += [f"--reservation={reservation}"]
                    cmd = ["sbatch"] + flags + [job_sh, str(gbs)]
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=megatron_dir)
                    if result.returncode != 0:
                        print(f"  sbatch failed: {result.stderr.strip()}", file=sys.stderr)
                    else:
                        print(f"  {result.stdout.strip()}")


def submit_numpy(nodes, partition, reservation, qos, cpus):
    """Submit numpy matrix-multiply benchmark jobs. Mirrors numpy/job_full_all_example.sh."""
    print("########## numpy #########")

    output_dir = os.path.join(ROOT_DIR, "numpy", partition, "output")
    os.makedirs(output_dir, exist_ok=True)

    script_py = os.path.join(ROOT_DIR, "numpy", "mat_mult.py")
    cores = cpus
    thread_list = generate_thread_list(cores)
    thread_list_str = " ".join(str(t) for t in thread_list)
    print(f"List of threads: {thread_list_str}")

    for node in nodes:
        host = f"node{node}"
        print(f"running on host {host}")

        script = f"""#!/bin/bash
#SBATCH -t 30
#SBATCH -p {partition}
#SBATCH -n {cores}
#SBATCH -N 1
#SBATCH -w {host}
#SBATCH -o {output_dir}/out_full.%N-%J
#SBATCH -q {qos}
#SBATCH --exclusive
#SBATCH -J numpy

IFS=' ' read -a thread_list <<< "{thread_list_str}"

echo "Benchmarking on ${{thread_list[*]}}"

module load miniforge/23.11.0-0

for j in ${{!thread_list[@]}};
do
     export OMP_NUM_THREADS=${{thread_list[j]}}
     echo "Ran with OMP_NUM_THREADS=$OMP_NUM_THREADS"
     python {script_py}
done
"""
        sbatch_submit_stdin(script, reservation)


# ── Dispatch table ───────────────────────────────────────────────────────────

SUBMIT_DISPATCH = {
    "openmp":                submit_openmp,
    "mpi-calc-pi":           submit_mpi_calc_pi,
    "mpi-p2p":               submit_mpi_p2p,
    "gpu-burn-r8":           submit_gpu_burn,
    "nccl-tests":            submit_nccl_tests,
    "nvidia-hpc-benchmarks": submit_nvidia_hpc,
    "gpu-fryer":             submit_gpu_fryer,
    "megatron-lm":           submit_megatron_lm,
    "numpy":                 submit_numpy,
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit Slurm benchmark jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s nccl-tests --nodes 3511 3512 --partition mit_normal_gpu --qos unlimited --cpus 48 --gpu-type l40s --gpus 4
  %(prog)s openmp mpi-calc-pi --nodes 3511 --partition mit_normal --qos normal --cpus 48
  %(prog)s gpu-burn-r8 --nodes 3511 3512 --partition mit_normal_gpu --qos unlimited --cpus 48 --gpu-type h200 --gpus 8
""",
    )
    parser.add_argument(
        "benchmarks", nargs="*",
        help=f"Benchmark(s) to run (choices: {', '.join(BENCHMARKS.keys())})",
    )
    parser.add_argument("--all-bench", action="store_true",
                        help="Submit every registered benchmark")
    parser.add_argument("--nodes", nargs="+", help="Node numbers (e.g. 3511 3512)")
    parser.add_argument("--partition", help="Slurm partition")
    parser.add_argument("--reservation", default="none", help="Slurm reservation (default: none)")
    parser.add_argument("--qos", help="Slurm QoS")
    parser.add_argument("--cpus", type=int, help="Number of CPUs per node")
    parser.add_argument("--gpu-type", help="GPU type (e.g. l40s, a100, h100, h200)")
    parser.add_argument("--gpus", type=int, help="Number of GPUs per node")
    args = parser.parse_args()

    if args.all_bench and args.benchmarks:
        parser.error("--all-bench cannot be combined with explicit benchmark names")
    if not args.all_bench and not args.benchmarks:
        parser.error("provide benchmark names or --all-bench")
    if args.all_bench:
        args.benchmarks = list(BENCHMARKS.keys())
    else:
        unknown = [b for b in args.benchmarks if b not in BENCHMARKS]
        if unknown:
            parser.error(f"unknown benchmark(s): {', '.join(unknown)}")

    return args


def resolve_args(args):
    """Fill in missing arguments by querying Slurm."""
    # Partition is needed to query nodes
    if not args.partition:
        print("Error: --partition is required (or will be queried from Slurm in future)", file=sys.stderr)
        sys.exit(1)

    # Nodes: query from partition if not provided
    if not args.nodes:
        print(f"Querying nodes in partition {args.partition}...")
        node_list = query_nodes_in_partition(args.partition)
        # Extract node numbers (strip "node" prefix)
        args.nodes = [n.replace("node", "") for n in node_list]
        print(f"Found nodes: {' '.join(args.nodes)}")

    if not args.nodes:
        print("Error: no nodes found. Provide --nodes explicitly.", file=sys.stderr)
        sys.exit(1)

    # QoS: default to normal if not provided
    if not args.qos:
        args.qos = "normal"
        print(f"Using default QoS: {args.qos}")

    # CPUs: query from first node if not provided
    if not args.cpus:
        first_node = f"node{args.nodes[0]}"
        print(f"Querying CPUs on {first_node}...")
        args.cpus = query_cpus_on_node(first_node)
        print(f"Found CPUs: {args.cpus}")

    # GPU info: query from first node if not provided
    needs_gpu = any(BENCHMARKS[b] == "gpu" for b in args.benchmarks)
    if needs_gpu and (not args.gpu_type or not args.gpus):
        first_node = f"node{args.nodes[0]}"
        print(f"Querying GPU info on {first_node}...")
        detected_type, detected_count = query_gpus_on_node(first_node)
        if not args.gpu_type and detected_type:
            args.gpu_type = detected_type
            print(f"Found GPU type: {args.gpu_type}")
        if not args.gpus and detected_count:
            args.gpus = detected_count
            print(f"Found GPU count: {args.gpus}")

    if needs_gpu and (not args.gpu_type or not args.gpus):
        print("Error: GPU benchmarks require --gpu-type and --gpus", file=sys.stderr)
        sys.exit(1)

    return args


def main():
    args = parse_args()

    # When --all-bench is used and GPU info cannot be determined, drop GPU
    # benchmarks instead of failing (so the flag works on CPU-only partitions).
    if args.all_bench:
        needs_gpu = any(BENCHMARKS[b] == "gpu" for b in args.benchmarks)
        if needs_gpu and (not args.gpu_type or not args.gpus):
            first_node = f"node{args.nodes[0]}" if args.nodes else None
            detected_type, detected_count = (None, 0)
            if first_node:
                try:
                    detected_type, detected_count = query_gpus_on_node(first_node)
                except Exception:
                    pass
            if not ((args.gpu_type or detected_type) and (args.gpus or detected_count)):
                print("No GPU info available — skipping GPU benchmarks for --all-bench.",
                      file=sys.stderr)
                args.benchmarks = [b for b in args.benchmarks if BENCHMARKS[b] != "gpu"]

    args = resolve_args(args)

    for bench in args.benchmarks:
        category = BENCHMARKS[bench]
        func = SUBMIT_DISPATCH[bench]

        if category == "cpu":
            func(args.nodes, args.partition, args.reservation, args.qos, args.cpus)
        elif category == "mpi":
            func(args.nodes, args.partition, args.reservation, args.qos)
        elif category == "gpu":
            func(args.nodes, args.partition, args.reservation, args.qos,
                 args.cpus, args.gpu_type, args.gpus)


if __name__ == "__main__":
    main()
