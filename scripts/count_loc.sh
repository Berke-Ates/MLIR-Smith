#!/bin/bash

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script counts the loc in generate functions.

# Be safe.
set -u # Disallow using undefined variables.

# Hardcoded map of name -> file
declare -A name_to_file

sdfg_dialect_dir=./mlir-dace-smith/lib/SDFG/Dialect
mlir_dialect_dir=./llvm-project-smith/mlir/lib/Dialect
mlir_ir_dir=./llvm-project-smith/mlir/lib/IR

name_to_file["SDFG_Types"]="$sdfg_dialect_dir/Dialect.cpp"
name_to_file["SDFG_Ops"]="$sdfg_dialect_dir/Ops.cpp"
name_to_file["Arith_Ops"]="$mlir_dialect_dir/Arith/IR/ArithOps.cpp"
name_to_file["Math_Ops"]="$mlir_dialect_dir/Math/IR/MathOps.cpp"
name_to_file["SCF_Ops"]="$mlir_dialect_dir/SCF/IR/SCF.cpp"
name_to_file["Memref_Types"]="$mlir_ir_dir/BuiltinTypes.cpp"
name_to_file["Memref_Ops"]="$mlir_dialect_dir/MemRef/IR/MemRefOps.cpp"
name_to_file["Func_Ops"]="$mlir_dialect_dir/Func/IR/FuncOps.cpp"

for name in "${!name_to_file[@]}"; do
  file="${name_to_file[$name]}"
  inside_generate=0
  brace_count=0
  line_count=0

  while IFS= read -r line; do
    # Trim leading and trailing whitespaces
    trimmed_line=$(echo "$line" | awk '{$1=$1};1')

    if [[ $inside_generate -eq 0 && $line =~ ::generate\(GeneratorOpBuilder[[:space:]]*\&builder\) ]]; then
      inside_generate=1
    fi

    if [[ $inside_generate -eq 1 ]]; then
      for ((i = 0; i < ${#line}; i++)); do
        char="${line:$i:1}"
        if [[ $char == "{" ]]; then
          ((brace_count++))
        elif [[ $char == "}" ]]; then
          ((brace_count--))
        fi
      done

      if [[ $brace_count -gt 0 && -n $trimmed_line ]]; then
        ((line_count++))
      fi

      if [[ $brace_count -eq 0 && $inside_generate -eq 1 ]]; then
        inside_generate=0
        ((line_count--))
      fi
    fi
  done <"$file"

  echo "$name: $line_count"
done
