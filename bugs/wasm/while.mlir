// Reason for failure: execution step - Different exit codes without timeout: 0 vs 7
module {
  func.func @main() -> i32 {
    %false = arith.constant false
    %11 = arith.divui %false, %false : i1 // Signed unsigned does not matter

    // Triggers bug: RuntimeError: unreachable
    scf.while : () -> () {
      scf.condition(%11)
    } do {
      scf.yield
    }

    // Both agree (without the loop) that this is 0
    %c0_i32 = arith.extui %11 : i1 to i32
    return %c0_i32 : i32
  }
}
