// Reason for failure: emcc step - emcc failed -O3 on orig.ll
module {
  func.func @main() -> i32 {
    %alloca = memref.alloca() : memref<1xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    memref.store %cst, %alloca[%c0] : memref<1xf16>
    %0 = arith.maxsi %c0, %c0 : index
    %1 = math.log2 %cst : f16
    %2 = math.fpowi %cst, %c0 : f16, index
    %3 = math.fma %1, %2, %1 : f16
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
