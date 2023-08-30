#!/bin/bash

# This script runs mlir-smith searches for a seed generating a specific output.

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
end_seed=1000

# The timeout in seconds.
timeout=5

for ((seed = start_seed; seed <= end_seed; seed++)); do
  echo -ne "Running mlir-smith with seed: $seed\r"

  output=$(timeout $timeout ./"$mlir_smith" --seed $seed 2>&1)
  grep_result=$(echo "$output" | grep -c "$word")

  if [ "$grep_result" -gt 0 ]; then
    echo -e "\nFound output string '$word' with seed: $seed"
    exit 0
  fi
done

echo -e "\nOutput string '$word' not found in the seed range"
exit 1
