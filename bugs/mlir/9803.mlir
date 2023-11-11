// The binary from the pipeline causes a free(): invalid size

module {
  func.func @main() -> i32 {
    scf.while : () -> () {
      %alloca = memref.alloca() : memref<i8>
      %c0 = arith.constant 0 : i8
      memref.store %c0, %alloca[] : memref<i8>
      memref.dealloc %alloca : memref<i8>

      %false = arith.constant false
      scf.condition(%false)
    } do {
      scf.yield
    }

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
