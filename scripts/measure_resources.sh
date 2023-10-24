#!/bin/bash

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script runs mlir-smith, measures the time, memory and LoC, and stores them in a CSV.

# Be safe.
set -u # Disallow using undefined variables.

# Check if a path to the tools was provided.
if [ $# -ne 2 ]; then
  echo "Usage: $0 <path to mlir-smith> <path to config>"
  exit 1
fi

# The paths to the tools.
mlir_smith=$1
config=$2

# Check if the tools exist and are executable.
if [ ! -x "$mlir_smith" ]; then
  echo "Error: mlir-smith does not exist at '$mlir_smith' or is not executable."
  exit 1
fi

# Number of runs.
num_runs=10

# Create the CSV header.
echo "Execution Time (s),Memory Usage (KB),Lines of Code"

# Loop to run mlir-smith, measure, and save results in the CSV.
for ((run = 1; run <= num_runs; run++)); do
  # Temporary files to store outputs
  temp_file_loc=$(mktemp)
  temp_file_res=$(mktemp)

  # Run mlir-smith, measure execution time and memory usage, and capture the output.
  /usr/bin/time -f "%e,%M" "$mlir_smith" -c "$config" 1>"$temp_file_loc" 2>"$temp_file_res"
  IFS=',' read -r execution_time memory_usage <"$temp_file_res"

  # Count Lines of Code (LoC) in the output.
  loc=$(wc -l <"$temp_file_loc")

  # Clean up
  rm "$temp_file_loc"
  rm "$temp_file_res"

  # Append results to the CSV.
  echo "$execution_time,$memory_usage,$loc"
done