#!/bin/bash

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script converts and translates a MLIR in builtin dialects into WASM.

# Be safe.
set -u # Disallow using undefined variables.

# Check if a path to the tools was provided.
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <path to mlir-opt> <path to mlir-translate> [<path to input file>]"
  exit 1
fi

# The paths to the tools and input files.
mlir_opt=$1
mlir_translate=$2
emcc=$(which emcc)
wasm2wat=$(which wasm2wat)

# Check if the tools exist and are executable.
for tool in "$mlir_opt" "$mlir_translate" "$emcc" "$wasm2wat"; do
  if [ ! -x "$tool" ]; then
    echo "Error: Tool does not exist at '$tool' or is not executable."
    exit 1
  fi
done

# Determine if reading from stdin or file
if [ $# -eq 2 ]; then
  input_file="-" # Read from stdin
fi

temp_dir=$(mktemp -d)

# Get LLVM IR through LLVM Dialect
$mlir_opt --convert-scf-to-cf --convert-func-to-llvm --convert-cf-to-llvm \
  --convert-math-to-llvm --convert-arith-to-llvm --lower-host-to-llvm \
  --reconcile-unrealized-casts \
  "$input_file" |
  $mlir_translate --mlir-to-llvmir >"$temp_dir/out".ll

# Compile to WASM
$emcc -O0 "$temp_dir/out".ll -o "$temp_dir/out".wasm

# Generate human-readable WAT
$wasm2wat "$temp_dir/out".wasm

# Cleanup temporaries.
rm -r "$temp_dir"
