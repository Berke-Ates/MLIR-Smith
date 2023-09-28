#!/bin/bash

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script converts and translates a MLIR in builtin dialects into WASM.

# Be safe.
set -u # Disallow using undefined variables.

# Check if a path to the tools was provided.
if [ $# -ne 3 ]; then
  echo "Usage: $0 <path to mlir-opt> <path to mlir-translate> <path to input file>"
  exit 1
fi

# The paths to the tools and input files.
mlir_opt=$1
mlir_translate=$2
input_file=$3
emcc=$(which emcc)
wasm2wat=$(which wasm2wat)

# Check if the tools exist and are executable.
if [ ! -x "$mlir_opt" ]; then
  echo "Error: mlir-opt does not exist at '$mlir_opt' or is not executable."
  exit 1
fi

if [ ! -x "$mlir_translate" ]; then
  echo "Error: mlir-translate does not exist at '$mlir_translate' or is not executable."
  exit 1
fi

if [ ! -x "$emcc" ]; then
  echo "Error: emcc does not exist at '$emcc' or is not executable."
  exit 1
fi

if [ ! -x "$wasm2wat" ]; then
  echo "Error: wasm2wat does not exist at '$wasm2wat' or is not executable."
  exit 1
fi

# Get LLVM IR through LLVM Dialect
$mlir_opt --convert-scf-to-cf --convert-func-to-llvm --convert-cf-to-llvm \
  --convert-math-to-llvm --convert-arith-to-llvm --lower-host-to-llvm \
  --reconcile-unrealized-casts \
  "$input_file" |
  $mlir_translate --mlir-to-llvmir >"$input_file".ll

# Compile to WASM
$emcc -O0 "$input_file".ll -o "$input_file".wasm

# Generate human-readable WAT
$wasm2wat "$input_file".wasm
