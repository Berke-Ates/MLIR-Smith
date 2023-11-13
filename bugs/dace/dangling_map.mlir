// Auto-optimze makes this sdfg invalid 
// "Dangling out-connector OUT__0 (at state state_0, node mapEntry_1[_arg0=0:2])"

module {
  sdfg.sdfg {entry = @state_0} () -> (){
    sdfg.state @state_0{
      %2 = sdfg.alloc {init} () : !sdfg.array<i32>

      sdfg.map (%i) = (0) to (1) step (1){
        sdfg.copy %2 -> %2 : !sdfg.array<i32>
      }
  
      sdfg.nested_sdfg {entry = @state_3} () -> (%2 as %arg1: !sdfg.array<i32>){
        sdfg.state @state_3{
        }
      }
    }
  }
}
