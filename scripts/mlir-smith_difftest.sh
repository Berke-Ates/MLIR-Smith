#!/bin/bash

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script uses MLIR-Smith to differentially test MLIR's optimization passes.

# Be safe.
set -u # Disallow using undefined variables.

# Check if a path to the tools was provided.
if [ $# -ne 4 ]; then
  echo "Usage: $0 <path to mlir-smith> <path to mlir-opt> <path to mlir-translate> <path to llc>"
  exit 1
fi

# The paths to the tools and input files.
mlir_smith=$1
mlir_opt=$2
mlir_translate=$3
llc=$4
clang=$(which clang)
result_path=./results

# Check if the tools exist and are executable.
for tool in "$mlir_smith" "$mlir_opt" "$mlir_translate" "$llc" "$clang"; do
  if [ ! -x "$tool" ]; then
    echo "Error: Tool does not exist at '$tool' or is not executable."
    exit 1
  fi
done

# Initialize counters
declare -A failure_counts
failure_counts[generation]=0
failure_counts[optimization]=0
failure_counts[translation]=0
failure_counts[llc]=0
failure_counts[clang]=0
failure_counts[timeout]=0
failure_counts[diff]=0
programs_tested=0

# Function to log the failure and save the original mlir file with annotation
log_failure() {
  local step=$1
  local file=$2
  local reason=$3

  mkdir -p "$result_path/$step"
  echo "// Reason for failure: $step step - $reason" >"$result_path/$step/$programs_tested.mlir"
  cat "$file" >>"$result_path/$step/$programs_tested.mlir"
}

# Infinite loop
while true; do
  # Sleep for a brief moment to avoid overwhelming the system
  sleep 1
  # Increase counter
  ((programs_tested++))
  # Move the cursor to the top of the screen
  echo -ne "\033[2J\033[H"
  # Inform the user that the script can be stopped with CTRL+C
  echo "The script is running and will continue to do so until you stop it with CTRL+C."
  # Report the number of programs tested so far
  echo "Programs tested so far: $programs_tested"

  # Report failures
  echo "Current failure counts:"
  for step in "${!failure_counts[@]}"; do
    echo "$step: ${failure_counts[$step]}"
  done

  # Temporary directory
  temp_dir=$(mktemp -d)

  # Generate
  if ! $mlir_smith >"$temp_dir/orig.mlir" 2>/dev/null; then
    ((failure_counts[generation]++))
    log_failure "generation" "$temp_dir/orig.mlir" "mlir_smith failed"
    rm -r "$temp_dir"
    continue
  fi

  # Optimize
  if ! $mlir_opt -cse --canonicalize --symbol-dce --loop-invariant-code-motion --inline "$temp_dir/orig.mlir" >"$temp_dir/opt.mlir" 2>/dev/null; then
    ((failure_counts[optimization]++))
    log_failure "optimization" "$temp_dir/orig.mlir" "mlir_opt failed"
    rm -r "$temp_dir"
    continue
  fi

  # Get LLVM IR through LLVM Dialect
  if ! $mlir_opt --convert-scf-to-cf --convert-func-to-llvm --convert-cf-to-llvm \
    --convert-math-to-llvm --convert-arith-to-llvm --lower-host-to-llvm \
    --reconcile-unrealized-casts \
    "$temp_dir/orig.mlir" 2>/dev/null |
    $mlir_translate --mlir-to-llvmir >"$temp_dir/orig.ll" 2>/dev/null; then
    ((failure_counts[translation]++))
    log_failure "lowering" "$temp_dir/orig.mlir" "mlir-opt/mlir-translate failed on orig.ll"
    rm -r "$temp_dir"
    continue
  fi

  if ! $mlir_opt --convert-scf-to-cf --convert-func-to-llvm --convert-cf-to-llvm \
    --convert-math-to-llvm --convert-arith-to-llvm --lower-host-to-llvm \
    --reconcile-unrealized-casts \
    "$temp_dir/opt.mlir" |
    $mlir_translate --mlir-to-llvmir >"$temp_dir/opt.ll" 2>/dev/null; then
    ((failure_counts[translation]++))
    log_failure "lowering" "$temp_dir/orig.mlir" "mlir-opt/mlir-translate failed on opt.ll"
    rm -r "$temp_dir"
    continue
  fi

  # Compile
  if ! $llc -O0 --relocation-model=pic "$temp_dir/orig.ll" -o "$temp_dir/orig.s" &>/dev/null; then
    ((failure_counts[llc]++))
    log_failure "llc" "$temp_dir/orig.mlir" "llc failed on orig.ll"
    rm -r "$temp_dir"
    continue
  fi

  if ! $clang -O0 -fPIC -march=native "$temp_dir/orig.s" -o "$temp_dir/orig.out" &>/dev/null; then
    ((failure_counts[clang]++))
    log_failure "clang" "$temp_dir/orig.mlir" "clang failed on orig.s"
    rm -r "$temp_dir"
    continue
  fi

  if ! $llc -O0 --relocation-model=pic "$temp_dir/opt.ll" -o "$temp_dir/opt.s" &>/dev/null; then
    ((failure_counts[llc]++))
    log_failure "llc" "$temp_dir/orig.mlir" "llc failed on opt.ll"
    rm -r "$temp_dir"
    continue
  fi

  if ! $clang -O0 -fPIC -march=native "$temp_dir/opt.s" -o "$temp_dir/opt.out" &>/dev/null; then
    ((failure_counts[clang]++))
    log_failure "clang" "$temp_dir/orig.mlir" "clang failed on opt.s"
    rm -r "$temp_dir"
    continue
  fi

  # Run binaries with timeout and check exit codes
  timeout --foreground 5s "$temp_dir/orig.out"
  orig_exit_code=$?
  timeout --foreground 5s "$temp_dir/opt.out"
  opt_exit_code=$?

  # Check if only one of the binaries hit the timeout
  if { [ $orig_exit_code -eq 124 ] && [ $opt_exit_code -ne 124 ]; } ||
    { [ $orig_exit_code -ne 124 ] && [ $opt_exit_code -eq 124 ]; }; then
    ((failure_counts[timeout]++))
    log_failure "execution" "$temp_dir/orig.mlir" "One binary timed out while the other did not"
  fi

  # Check if exit codes are different and not due to timeout
  if [ $orig_exit_code -ne $opt_exit_code ] &&
    [ $orig_exit_code -ne 124 ] && [ $opt_exit_code -ne 124 ]; then
    ((failure_counts[diff]++))
    log_failure "execution" "$temp_dir/orig.mlir" "Different exit codes without timeout: $orig_exit_code vs $opt_exit_code"
  fi

  # Cleanup temporaries.
  rm -r "$temp_dir"
done
