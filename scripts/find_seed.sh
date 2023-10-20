#!/bin/bash

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script uses mlir-smith to search for a seed generating a specific output.

# Be safe.
set -u # Disallow using undefined variables.

# Check if a path to the tool was provided.
if [ $# -ne 2 ]; then
  echo "Usage: $0 <path to mlir-smith> <word to grep>"
  exit 1
fi

# The path to the tool.
mlir_smith=$1

# The word to grep for.
word=$2

# Check if the tool exists and is executable.
if [ ! -x "$mlir_smith" ]; then
  echo "Error: mlir-smith does not exist at '$mlir_smith' or is not executable."
  exit 1
fi

# The range of seeds to test.
start_seed=0
end_seed=10000

# The timeout in seconds.
timeout=5

# Function to process each seed.
# shellcheck disable=SC2317  # Don't warn about unreachable commands in this function.
process_seed() {
  seed=$1
  mlir_smith=$2
  timeout=$3
  word=$4

  output=$(timeout "$timeout" ./"$mlir_smith" --seed "$seed" 2>&1)
  grep_result=$(echo "$output" | grep -c "$word")

  if [ "$grep_result" -gt 0 ]; then
    echo -e "\nFound output string '$word' with seed: $seed"
    return 0
  fi

  return 1
}

export -f process_seed

# Use GNU Parallel to run multiple threads.
seq $start_seed $end_seed | parallel --halt now,success=1 --progress process_seed {} "$mlir_smith" $timeout "$word"
