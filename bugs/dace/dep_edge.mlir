// Pipeline causes LookupError: Memlet does not point to any of the nodes

module {
  sdfg.sdfg {entry = @state_0} () -> (%arg0: !sdfg.array<2x2x1x3xf32>){
    %0 = sdfg.alloc {init} () : !sdfg.array<2x2x1x3xf32>
    sdfg.state @state_0{
      %1 = sdfg.alloc {init} () : !sdfg.array<2x17x1x7x14x3x5xi64>
      %2:4 = sdfg.tasklet () -> (i16, i16, i16, i16){
        %c-17_i16 = arith.constant -17 : i16
        sdfg.return %c-17_i16, %c-17_i16, %c-17_i16, %c-17_i16 : i16, i16, i16, i16
      }
      %3 = sdfg.alloc {init} () : !sdfg.array<2xi16>
      %4:10 = sdfg.tasklet (%2#1 as %arg1: i16, %2#2 as %arg2: i16, %2#3 as %arg3: i16, %2#1 as %arg4: i16, %2#1 as %arg5: i16, %2#0 as %arg6: i16) -> (i16, i16, i16, i16, i16, i16, i16, i16, i16, i16){
        %c1_i16 = arith.constant 1 : i16
        %6 = arith.maxsi %arg6, %c1_i16 : i16
        %7 = arith.ceildivui %arg6, %6 : i16
        sdfg.return %arg6, %6, %arg1, %c1_i16, %c1_i16, %arg3, %arg4, %7, %arg4, %arg2 : i16, i16, i16, i16, i16, i16, i16, i16, i16, i16
      }
      %5 = sdfg.load %3[0] : !sdfg.array<2xi16> -> i16
      sdfg.store %4#9, %3[0] : i16 -> !sdfg.array<2xi16>
      sdfg.nested_sdfg {entry = @state_1} () -> (%1 as %arg1: !sdfg.array<2x17x1x7x14x3x5xi64>, %3 as %arg2: !sdfg.array<2xi16>){
        sdfg.edge {assign = [], condition = "1"} @state_1 -> @state_1
        sdfg.state @state_1{
        }
      }
    }
  }
}
