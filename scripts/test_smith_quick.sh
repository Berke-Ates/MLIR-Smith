#!/bin/bash

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script is a faster version of `test_smith.sh` without operation tracking

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
start_seed=0
end_seed=10000

# The timeout in seconds.
timeout=5

seq $start_seed $end_seed | parallel --halt now,fail=1 --progress -- "timeout $timeout $mlir_smith --seed {} 2>&1 | $mlir_opt >/dev/null 2>&1"

exit 0
