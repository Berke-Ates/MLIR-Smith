#!/bin/python3

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script converts the SDFG dialect in MLIR to the SDFG IR using 
# `sdfg-translate`

import sys
import os
import subprocess
import dace
import json

if len(sys.argv) != 3:
    print("SDFG Translation Tool")
    print("Arguments:")
    print("  Input MLIR: The MLIR in the SDFG dialect to translate (- for stdin)")
    print("  SDFG-Translate: The sdfg-translate tool")
    exit(1)

input_file = sys.argv[1]
sdfg_translate = sys.argv[2]

# Check if sdfg-translate exists and is executable
if not os.path.exists(sdfg_translate):
    print(f"'{sdfg_translate}' does not exist.")
    exit(1)

if not os.access(sdfg_translate, os.X_OK):
    print(f"'{sdfg_translate}' is not executable.")
    exit(1)

# If input_file is '-', read from stdin (for bash pipelines)
if input_file == '-':
    input_data = sys.stdin.read()
else:
    with open(input_file, 'r') as f:
        input_data = f.read()

# Set ASAN_OPTIONS to ignore all errors
os.environ["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=0"

# Run sdfg-translate with --mlir-to-sdfg flag
try:
    result = subprocess.run([sdfg_translate, '--mlir-to-sdfg'], input=input_data, text=True,capture_output=True, check=True)
    sdfg = dace.SDFG.from_json(json.loads(result.stdout))
    print(sdfg.to_json())
except subprocess.CalledProcessError as e:
    print(f"Error executing '{sdfg_translate}': {e}")
    print(f"Error output:\n{e.stderr}")
    exit(1)
