(module
  (type (;0;) (func (result i32)))
  (type (;1;) (func))
  (type (;2;) (func (param i32 i32) (result i32)))
  (type (;3;) (func (param i32)))
  (type (;4;) (func (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "args_sizes_get" (func (;0;) (type 2)))
  (import "wasi_snapshot_preview1" "args_get" (func (;1;) (type 2)))
  (import "wasi_snapshot_preview1" "proc_exit" (func (;2;) (type 3)))
  (func (;3;) (type 1)
    call 16)
  (func (;4;) (type 2) (param i32 i32) (result i32)
    unreachable)
  (func (;5;) (type 1)
    block  ;; label = @1
      i32.const 1
      i32.eqz
      br_if 0 (;@1;)
      call 3
    end
    call 8
    call 11
    unreachable)
  (func (;6;) (type 2) (param i32 i32) (result i32)
    local.get 0
    local.get 1
    call 4)
  (func (;7;) (type 0) (result i32)
    (local i32 i32 i32 i32)
    global.get 0
    i32.const 16
    i32.sub
    local.tee 0
    global.set 0
    block  ;; label = @1
      block  ;; label = @2
        local.get 0
        local.tee 1
        i32.const 12
        i32.add
        local.get 1
        i32.const 8
        i32.add
        call 0
        br_if 0 (;@2;)
        block  ;; label = @3
          block  ;; label = @4
            local.get 1
            i32.load offset=12
            local.tee 2
            br_if 0 (;@4;)
            i32.const 0
            local.set 2
            br 1 (;@3;)
          end
          local.get 0
          local.get 2
          i32.const 2
          i32.shl
          local.tee 2
          i32.const 19
          i32.add
          i32.const -16
          i32.and
          i32.sub
          local.tee 0
          local.tee 3
          global.set 0
          local.get 3
          local.get 1
          i32.load offset=8
          i32.const 15
          i32.add
          i32.const -16
          i32.and
          i32.sub
          local.tee 3
          global.set 0
          local.get 0
          local.get 2
          i32.add
          i32.const 0
          i32.store
          local.get 0
          local.get 3
          call 1
          br_if 2 (;@1;)
          local.get 1
          i32.load offset=12
          local.set 2
        end
        local.get 2
        local.get 0
        call 6
        local.set 0
        local.get 1
        i32.const 16
        i32.add
        global.set 0
        local.get 0
        return
      end
      i32.const 71
      call 2
      unreachable
    end
    i32.const 71
    call 2
    unreachable)
  (func (;8;) (type 0) (result i32)
    call 7)
  (func (;9;) (type 1))
  (func (;10;) (type 1)
    (local i32)
    i32.const 0
    local.set 0
    block  ;; label = @1
      i32.const 0
      i32.const 0
      i32.le_u
      br_if 0 (;@1;)
      loop  ;; label = @2
        local.get 0
        i32.const -4
        i32.add
        local.tee 0
        i32.load
        call_indirect (type 1)
        local.get 0
        i32.const 0
        i32.gt_u
        br_if 0 (;@2;)
      end
    end
    call 9)
  (func (;11;) (type 3) (param i32)
    call 9
    call 10
    call 9
    local.get 0
    call 12
    unreachable)
  (func (;12;) (type 3) (param i32)
    local.get 0
    call 2
    unreachable)
  (func (;13;) (type 0) (result i32)
    global.get 0)
  (func (;14;) (type 3) (param i32)
    local.get 0
    global.set 0)
  (func (;15;) (type 4) (param i32) (result i32)
    (local i32 i32)
    global.get 0
    local.get 0
    i32.sub
    i32.const -16
    i32.and
    local.tee 1
    global.set 0
    local.get 1)
  (func (;16;) (type 1)
    i32.const 5243920
    global.set 2
    i32.const 1028
    i32.const 15
    i32.add
    i32.const -16
    i32.and
    global.set 1)
  (func (;17;) (type 0) (result i32)
    global.get 0
    global.get 1
    i32.sub)
  (func (;18;) (type 0) (result i32)
    global.get 2)
  (func (;19;) (type 0) (result i32)
    global.get 1)
  (func (;20;) (type 0) (result i32)
    i32.const 1024)
  (table (;0;) 2 2 funcref)
  (memory (;0;) 256 256)
  (global (;0;) (mut i32) (i32.const 5243920))
  (global (;1;) (mut i32) (i32.const 0))
  (global (;2;) (mut i32) (i32.const 0))
  (export "memory" (memory 0))
  (export "_start" (func 5))
  (export "__indirect_function_table" (table 0))
  (export "__errno_location" (func 20))
  (export "emscripten_stack_init" (func 16))
  (export "emscripten_stack_get_free" (func 17))
  (export "emscripten_stack_get_base" (func 18))
  (export "emscripten_stack_get_end" (func 19))
  (export "stackSave" (func 13))
  (export "stackRestore" (func 14))
  (export "stackAlloc" (func 15))
  (elem (;0;) (i32.const 1) func 3))
