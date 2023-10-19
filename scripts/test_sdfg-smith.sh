#!/bin/bash

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script tests sdfg-smith using the DaCe verifier.

# Be safe.
set -u # Disallow using undefined variables.

# Check if a path to the tools was provided.
if [ $# -ne 3 ]; then
  echo "Usage: $0 <path to sdfg-smith> <path to sdfg-translate> <path to config>"
  exit 1
fi

# The paths to the tools.
sdfg_smith=$1
sdfg_translate=$2
config=$3

# Check if the tools exist and are executable.
if [ ! -x "$sdfg_smith" ]; then
  echo "Error: sdfg-smith does not exist at '$sdfg_smith' or is not executable."
  exit 1
fi

if [ ! -x "$sdfg_translate" ]; then
  echo "Error: sdfg-translate does not exist at '$sdfg_translate' or is not executable."
  exit 1
fi

# The range of seeds to test.
start_seed=0
end_seed=10000
total_seeds=$((end_seed - start_seed))

# Silence any warning for nicer output.
export PYTHONWARNINGS="ignore"

# Assuming mlir_to_sdfg.py is placed in the same folder.
dace_test="$(dirname "$0")/mlir_to_sdfg.py"

for ((seed = start_seed; seed <= end_seed; seed++)); do
  "$sdfg_smith" -c "$config" -seed "$seed" | $dace_test - "$sdfg_translate" /dev/null
  if [ $? -ne 0 ]; then
    echo -e "\nFailed on seed $seed"
    break
  fi

  # Calculate and display progress
  progress=$(((seed * 100) / total_seeds))
  echo -ne "\rProgress: $progress% (Seed: $seed/$total_seeds)"

done

# Print a newline at the end for clean output
echo ""
