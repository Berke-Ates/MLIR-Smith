// Auto-optimze makes this sdfg invalid

module {
  sdfg.sdfg {entry = @state_0} () -> (){
    sdfg.state @state_0{
      %2 = sdfg.alloc {init} () : !sdfg.array<i32>

      sdfg.nested_sdfg {entry = @state_1} () -> (%2 as %arg1: !sdfg.array<i32>){
        sdfg.state @state_1{
          %7 = sdfg.alloc {init} () : !sdfg.array<i32>
          sdfg.copy %7 -> %7 : !sdfg.array<i32>
        }
      }

      sdfg.copy %2 -> %2 : !sdfg.array<i32>
    }
  }
}
