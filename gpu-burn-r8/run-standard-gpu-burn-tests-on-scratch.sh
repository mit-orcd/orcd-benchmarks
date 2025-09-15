#!/bin/bash
set -x
# Title: run-standard-gpu-burn-tests.sh
# Date: 2024-09-18
# Author: jmurray1@mit.edu
# Purpose: Runs 3 tests on GPUs with interactive input for partition name and hostname in file names

# Prompt for the partition name
read -p "Enter the partition being tested: " partition_name

# Get the hostname of the system
host_name=$(hostname)

# Define the final output directory
output_dir="/orcd/data/orcd/002/benchmarks/gpu-burn-r8/$partition_name"
mkdir -p "$output_dir"

# Define temporary scratch directory
scratch_dir="/scratch/gpu-burn-tmp-$host_name"
mkdir -p "$scratch_dir"

# Move to the GPU-Burn directory
cd /orcd/data/orcd/002/benchmarks/gpu-burn-r8

# Run the tests and save output to scratch
echo "Running Test 1: Using tensor cores (300s)"
./gpu_burn -tc 300 > "$scratch_dir/test1_tc_${host_name}_output.txt" 2>&1

echo "Running Test 2: Standard Burn (300s)"
./gpu_burn 300 > "$scratch_dir/test2_burn_${host_name}_output.txt" 2>&1

echo "Running Test 3: Using doubles (300s)"
./gpu_burn -d 300 > "$scratch_dir/test3_detailed_${host_name}_stdout.txt" 2> "$scratch_dir/test3_detailed_${host_name}_stderr.txt"

# Move results to final output directory
mv "$scratch_dir"/* "$output_dir/"
rmdir "$scratch_dir"

# Check if test 3 had stderr output
if [ -s "$output_dir/test3_detailed_${host_name}_stderr.txt" ]; then
    echo "Test 3 completed, but there were errors. Check stderr output in $output_dir/test3_detailed_${host_name}_stderr.txt"
else
    echo "Test 3 completed successfully, output saved to $output_dir/test3_detailed_${host_name}_stdout.txt"
fi

echo "Completed 3 tests. Results saved in $output_dir"

