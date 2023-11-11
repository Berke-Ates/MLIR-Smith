// mlir-opt -cse –canonicalize –symbol-dce –loop-invariant-code-motion –inline
// removes infinite while loop. 

module {
  func.func @main() {
    %true = arith.constant true
    scf.while () : () -> () {
      scf.condition(%true) 
    } do {
      scf.yield
    }

    "launch.nukes"(){} : ()->()
    return
  }
}
