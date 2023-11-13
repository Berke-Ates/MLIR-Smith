// Reason for failure: emcc step - emcc failed -O3 on orig.ll
module {
  func.func @main() -> i32 {
    %alloc = memref.alloc() : memref<1xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    memref.store %cst, %alloc[%c0] : memref<1xf16>
    %0 = math.sin %cst : f16
    %1 = math.fpowi %cst, %c0 : f16, index
    %2 = arith.negf %cst : f16
    %3 = arith.subi %c0, %c0 : index
    %4 = math.sin %cst : f16
    %5 = math.fma %1, %4, %0 : f16
    %6 = arith.minui %c0, %c0 : index
    %7 = arith.minf %0, %0 : f16
    %c1 = arith.constant 1 : index
    %8 = arith.maxsi %3, %c1 : index
    %9 = arith.remui %c0, %8 : index
    %10 = arith.index_cast %c0 : index to i8
    %11 = arith.fptosi %2 : f16 to i16
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
