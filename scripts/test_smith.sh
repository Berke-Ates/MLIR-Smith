#!/bin/bash

# This script runs mlir-smith with multiple seeds and reports any crashes,
# timeouts or invalid outputs.

# Be safe.
set -u # Disallow using undefined variables.

# Check if a path to the tools was provided.
if [ $# -ne 2 ]; then
  echo "Usage: $0 <path to mlir-smith> <path to mlir-opt>"
  exit 1
fi

# The paths to the tools.
mlir_smith=$1
mlir_opt=$2

# Check if the tools exist and are executable.
if [ ! -x "$mlir_smith" ]; then
  echo "Error: mlir-smith does not exist at '$mlir_smith' or is not executable."
  exit 1
fi

if [ ! -x "$mlir_opt" ]; then
  echo "Error: mlir-opt does not exist at '$mlir_opt' or is not executable."
  exit 1
fi

# The range of seeds to test.
start_seed=1
end_seed=10000

# The timeout in seconds.
timeout=5

# Function to process each seed.
# shellcheck disable=SC2317  # Don't warn about unreachable commands in this function.
process_seed() {
  seed=$1
  mlir_smith=$2
  mlir_opt=$3
  timeout=$4

  # Get the operation names from the output of mlir-smith --dump
  mapfile -t ops < <("$mlir_smith" --dump | awk -F' = ' '{print $1}' | grep '\.')

  # Create a temporary file to store the output of mlir-smith.
  temp_file=$(mktemp)

  timeout "$timeout" ./"$mlir_smith" --seed "$seed" >"$temp_file" 2>&1
  result=$?
  if [ "$result" -eq 124 ]; then
    echo "Timeout with seed: $seed"
    rm "$temp_file"
    return 1
  elif [ "$result" -ne 0 ]; then
    echo "Crash with seed: $seed"
    rm "$temp_file"
    return 1
  else
    # If mlir-smith did not crash or timeout, feed the output to mlir-opt.
    ./"$mlir_opt" <"$temp_file" >/dev/null 2>&1
    result=$?
    if [ "$result" -ne 0 ]; then
      echo "mlir-opt failed with seed: $seed"
      rm "$temp_file"
      return 1
    fi
  fi

  # Check for the occurrence of operation names.
  for op in "${ops[@]}"; do
    count=$(grep -c "$op" "$temp_file")
    if [ "$count" -gt 0 ]; then
      echo "$op: $count"
    fi
  done

  rm "$temp_file"
  return 0
}

export -f process_seed

# Create a temporary file to store the output of mlir-smith.
temp_file=$(mktemp)

# Use GNU Parallel to run multiple threads.
seq $start_seed $end_seed | parallel --halt now,fail=1 --progress process_seed {} "$mlir_smith" "$mlir_opt" $timeout >"$temp_file"

# Calculate the number of occurrences per operation.
total_ops=$(awk -F': ' '{sum += $2} END {print sum}' "$temp_file")
echo "Occurrences per op (${total_ops}):"
awk -v total_ops="$total_ops" -F': ' '{arr[$1] += $2} END {for (op in arr) printf "%-20s\t%d (%.2f%%)\n", op, arr[op], arr[op]/total_ops*100}' "$temp_file" |
  sort

# Calculate the number of occurrences per file.
total_files=$((end_seed - start_seed + 1))
echo -e "\nOccurrence per file (${total_files}):"
awk -v total_files="$total_files" -F': ' '{arr[$1]++} END {for (op in arr) printf "%-20s\t%d (%.2f%%)\n", op, arr[op], arr[op]/total_files*100}' "$temp_file" |
  sort

rm "$temp_file"
exit 0
