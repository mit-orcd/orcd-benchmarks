#!/bin/bash
# Title: run-standard-gpu-burn-tests.sh
# Date: 2024-09-18
# Author: jmurray1@mit.edu
# Purpose: Runs 3 tests on GPUs with interactive input for partition name and hostname in file names
# ssh to the node or get an interactive session
 
# Prompt for the partition name
read -p "Enter the partition being tested: " partition_name

# check if partition actually exists, if not bail out.
 
# Get the hostname of the system
host_name=$(hostname)
 
# Define the directory path for test results
output_dir="/orcd/data/orcd/001/benchmarks/gpu-burn-r8/$partition_name"
 
# Create the directory if it doesn't exist
mkdir -p "$output_dir"
 
# Move to the GPU-Burn directory
cd /orcd/data/orcd/001/benchmarks/gpu-burn-r8
 
# Run the tests and save the output in the new directory with hostname in filenames
echo "Running Test 1: Using tensor cores (300s)"
./gpu_burn -tc 300 > "$output_dir/test1_tc_${host_name}_output.txt" 2>&1
echo "Test 1 completed, output saved to $output_dir/test1_tc_${host_name}_output.txt"
 
echo "Running Test 2: Standard Burn (300s)"
./gpu_burn 300 > "$output_dir/test2_burn_${host_name}_output.txt" 2>&1
echo "Test 2 completed, output saved to $output_dir/test2_burn_${host_name}_output.txt"
 
echo "Running Test 3: Using doubles (300s)"
# Explicitly capturing stdout and stderr separately for debugging
./gpu_burn -d 300 > "$output_dir/test3_detailed_${host_name}_stdout.txt" 2> "$output_dir/test3_detailed_${host_name}_stderr.txt"
 
# Check if test 3 generated any output on stderr
if [ -s "$output_dir/test3_detailed_${host_name}_stderr.txt" ]; then
    echo "Test 3 completed, but there were errors. Check stderr output in $output_dir/test3_detailed_${host_name}_stderr.txt"
else
    echo "Test 3 completed successfully, output saved to $output_dir/test3_detailed_${host_name}_stdout.txt"
fi
 
# Final message
echo "Completed 3 tests. Results saved in $output_dir"
