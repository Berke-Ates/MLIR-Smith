# MLIR-Smith

> TODO: Description of the project

## Building

To run the project follow the steps below. The commands have been separated for
copy and paste convenience.

### Dependencies

```sh
sudo apt update && sudo apt install git clang lld cmake ninja-build
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

```sh
ninja
```

You'll find `mlir-smith` alongside other tools in `llvm-project-smith/build/bin`.
Similarly for the SDFG version: `mlir-dace-smith/build/bin`.

## Testing

> TODO: Add fuzz testing for DaCe transformations.

> TODO: Add WASM pipeline.
