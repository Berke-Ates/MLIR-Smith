# MLIR-Forge: A Random Code Generator Framework in MLIR

> TODO: Update the text below.

Welcome to MLIR-Forge, a powerful tool designed to generate random MLIR
(Multi-Level Intermediate Representation) code for a registered set of dialects.
MLIR-Forge aims to simplify the process of testing and validating MLIR compilers,
optimizers, and code transformations by providing a diverse set of MLIR test
cases. Inspired by the success of CSmith in generating random C programs,
MLIR-Forge follows a similar approach to help detect and diagnose errors in the
MLIR ecosystem.

## Building

To run the project follow the steps below. The commands have been separated for
copy and paste convenience.

### Dependencies

```sh
sudo apt update && sudo apt install git clang lld cmake ninja-build ccache
```

### Clone

```sh
git clone --recurse-submodules --depth 1 --shallow-submodules https://github.com/Berke-Ates/MLIR-Smith
```

### Build LLVM Project with MLIR-Smith

```sh
cd llvm-project-smith && mkdir build && cd build
```

```sh
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="host" -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DLLVM_CCACHE_BUILD=ON -DLLVM_USE_SANITIZER="Address;Undefined" -DLLVM_INSTALL_UTILS=ON
```

```sh
ninja
```

```sh
cd ../..
```

### Build MLIR-DaCe with LLVM

```sh
cd mlir-dace-smith && mkdir build && cd build
```

```sh
cmake -G Ninja .. -DMLIR_DIR=$PWD/../../llvm-project-smith/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$PWD/../../llvm-project-smith/build/bin/llvm-lit -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DLLVM_USE_SANITIZER="Address;Undefined"
```

> NOTE: The SDFG Translate Tool (`sdfg-translate`) currently has memory leaks.
> To use it, either build without ASAN: `-DLLVM_USE_SANITIZER="Address;Undefined"`
> or disable it by setting the environment variable: `ASAN_OPTIONS="detect_leaks=0:halt_on_error=0"`

```sh
ninja && ninja sdfg-smith
```

You'll find `mlir-smith` alongside other tools in `llvm-project-smith/build/bin`.
Similarly for the SDFG version: `mlir-dace-smith/build/bin`.

## Testing

To run the testing scripts you need to install the following additional dependencies:

```sh
sudo apt update && sudo apt install parallel
```

### SDFG-Smith

> TODO: Add fuzz testing for DaCe transformations.

### WASM-Smith

To run the WASM pipeline you need to install the following additional dependencies:

```sh
sudo apt update && sudo apt install emscripten wabt
```

Then run the pipeline to generate WASM:

```sh
./llvm-project-smith/build/bin/mlir-smith | ./scripts/mlir_to_wasm.sh ./llvm-project-smith/build/bin/mlir-opt ./llvm-project-smith/build/bin/mlir-translate
```

If issues occur due to mismatching LLVM versions, update the LLVM build and
set the environment variable as outlined [here](https://emscripten.org/docs/building_from_source/index.html).

### MLIR-Smith

To run the MLIR differential testing pipeline you need to install the following additional dependencies:

```sh
sudo apt update && sudo apt install clang
```

Then run the pipeline to differentially test MLIR optimization passes:

```sh
./scripts/mlir-smith_difftest.sh ./llvm-project-smith/build/bin/mlir-smith ./llvm-project-smith/build/bin/mlir-opt ./llvm-project-smith/build/bin/mlir-translate ./llvm-project-smith/build/bin/llc
```
