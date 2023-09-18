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
