// Reason for failure: emcc step - emcc failed -O3 on orig.ll
module {
  func.func @main() -> i32 {
    %c0f = arith.constant 0.0 : f16
    %c0 = arith.constant 0 : i32

    %2 = math.fpowi %c0f, %c0 : f16, i32
    %3 = math.absf  %2 : f16
    
    return %c0 : i32
  }
}
