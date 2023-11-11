// mlir-opt -cse –canonicalize –symbol-dce –loop-invariant-code-motion –inline
// removes double free.

module {
  func.func @main() -> i32 {
    %alloc = memref.alloc() {_gen_dealloc} : memref<1xindex>
    memref.dealloc %alloc : memref<1xindex>
    memref.dealloc %alloc : memref<1xindex>

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
