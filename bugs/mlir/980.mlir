// The binary from the pipeline causes a munmap_chunk(): invalid pointer

module {
  func.func @main() -> i32 {
    %alloca = memref.alloca() : memref<i8>
    memref.dealloc %alloca : memref<i8>

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
